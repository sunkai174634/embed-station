"""Cross-encoder reranker engine: ONNX sequence-classification model served
through OpenVINO on the GPU, sharing the process-wide inference lock with the
embedding engine (single-container dual-model design).

Contract: score(query, text) pairs in one batched call; sigmoid activation;
results carry original indices (TEI /rerank semantics).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .inference import EmbeddingResult  # noqa: F401  (shared type re-export convenience)


class RerankResult:
    __slots__ = ("pairs", "scores")

    def __init__(self, pairs: list[tuple[int, float]]) -> None:
        self.pairs = pairs  # (original_index, sigmoid_score), sorted by score desc

    @property
    def scores_only(self) -> list[float]:
        return [s for _, s in self.pairs]


class OnnxRerankerEngine:
    """Serve an ONNX cross-encoder (e.g. hs-rerank-1) through OpenVINO GPU."""

    def __init__(
        self,
        model_dir: Path,
        model_id: str,
        gpu_lock: threading.Lock,
        cache_dir: Path | None = None,
    ) -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.model_dir = model_dir
        self.gpu_lock = gpu_lock  # shared with the embedding engine: one GPU, one queue
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX reranker not found: {onnx_path.name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        core = ov.Core()
        if "GPU" not in core.available_devices:
            raise RuntimeError("OpenVINO GPU device is unavailable; CPU fallback is disabled")
        self.device_name = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(str(onnx_path))
        self.compiled = core.compile_model(model, "GPU")
        self.request = self.compiled.create_infer_request()
        # readiness probe: must produce a finite score
        scores = self.rerank("readiness probe", ["probe document"])
        if not np.isfinite(scores).all():
            raise RuntimeError("GPU reranker probe returned non-finite scores")

    def rerank(self, query: str, texts: list[str]) -> np.ndarray:
        """Return sigmoid scores, one per text, aligned with input order."""
        if not texts:
            raise ValueError("texts must not be empty")
        enc = self.tokenizer(
            [(query, text) for text in texts], padding=True, truncation=False, return_tensors="np"
        )
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        with self.gpu_lock:
            self.request.infer(feed)
            logits = self.request.get_output_tensor(0).data.copy()
        scores = 1.0 / (1.0 + np.exp(-logits[:, 0].astype(np.float64)))
        if not np.isfinite(scores).all():
            raise RuntimeError("GPU reranker returned non-finite scores")
        return scores

    def close(self) -> None:
        with self.gpu_lock:
            pass

    def count_tokens(self, query: str, texts: list[str]) -> int:
        """Total tokens for the (query, text) pairs, computed before GPU work."""
        enc = self.tokenizer(
            [(query, text) for text in texts], padding=False, truncation=False, return_tensors=None
        )
        return sum(len(ids) for ids in enc["input_ids"])
