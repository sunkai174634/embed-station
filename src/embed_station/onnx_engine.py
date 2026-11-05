"""ONNX-backed engine: loads a tokenizer + ONNX embedding model via OpenVINO,
runs on GPU with the same serialized-inference contract as OpenVINOEngine.

This makes the service model-agnostic: any encoder exported to ONNX with
dynamic batch/seq axes (e.g. hs-embed-1) can be served, not just OpenVINO IR
exports. Mean pooling + L2 normalization follow the sentence-transformers
convention declared by the model's 1_Pooling/config.json.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .inference import EmbeddingResult


class OnnxOpenVINOEngine:
    """Serve an ONNX encoder through OpenVINO with sentence-transformers pooling."""

    def __init__(self, model_dir: Path, model_id: str) -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.model_dir = model_dir
        onnx_path = model_dir / "model.onnx"
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path.name}")

        # Pooling config: sentence-transformers writes 1_Pooling/config.json with
        # pooling_mode_cls/mask/mean flags. hs-embed-1 uses mean pooling.
        pooling_cfg = {}
        pooling_file = model_dir / "1_Pooling" / "config.json"
        if pooling_file.is_file():
            pooling_cfg = json.loads(pooling_file.read_text())
        self._mean_pooling = pooling_cfg.get("pooling_mode_mean_tokens", True)
        if pooling_cfg.get("pooling_mode_cls_token", False):
            raise ValueError("CLS pooling is not supported by this engine")
        if not self._mean_pooling:
            raise ValueError("Only mean pooling is supported by this engine")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")

        core = ov.Core()
        if "GPU" not in core.available_devices:
            raise RuntimeError("OpenVINO GPU device is unavailable; CPU fallback is disabled")
        self.device_name = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
        model = core.read_model(str(onnx_path))
        self.compiled = core.compile_model(model, "GPU")
        self.request = self.compiled.create_infer_request()
        self.infer_lock = threading.Lock()
        outputs = self.embed(["readiness probe"])
        self.dimension = len(outputs.vectors[0])
        if self.dimension <= 0:
            raise RuntimeError("GPU probe returned an empty embedding")

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            raise ValueError("input must not be empty")
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_tensors="np")
        ids = encoded["input_ids"].astype(np.int64)
        mask = encoded["attention_mask"].astype(np.int64)
        lengths = mask.sum(axis=1)
        if np.any(lengths == 0):
            raise ValueError("input produced an empty token sequence")
        with self.infer_lock:
            self.request.infer({"input_ids": ids, "attention_mask": mask})
            hidden = self.request.get_output_tensor(0).data.copy()
        mask_f = mask.astype(np.float32)
        summed = (hidden.astype(np.float32) * mask_f[..., None]).sum(axis=1)
        vectors = summed / np.clip(mask_f.sum(axis=1, keepdims=True), 1e-9, None)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms == 0):
            raise RuntimeError("GPU inference returned invalid embedding values")
        vectors /= norms
        return EmbeddingResult(vectors.tolist(), [int(length) for length in lengths])

    def close(self) -> None:
        with self.infer_lock:
            pass

    def count_tokens(self, texts: list[str]) -> int:
        """Total input tokens for a request, computed before any GPU work."""
        encoded = self.tokenizer(texts, padding=False, truncation=False, return_tensors=None)
        return sum(len(ids) for ids in encoded["input_ids"])
