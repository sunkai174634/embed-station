from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    token_counts: list[int]

    @property
    def token_count(self) -> int:
        return sum(self.token_counts)


class Engine(Protocol):
    model_id: str
    dimension: int
    device_name: str

    def embed(self, texts: list[str]) -> EmbeddingResult: ...

    def close(self) -> None: ...

    def count_tokens(self, texts: list[str]) -> int: ...


class OpenVINOEngine:
    def __init__(self, model_dir: Path, model_id: str) -> None:
        import openvino as ov
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.model_dir = model_dir
        xml = model_dir / "openvino_model.xml"
        if not xml.is_file():
            raise FileNotFoundError(f"OpenVINO IR not found: {xml.name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False
        )
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id")
        self.tokenizer.padding_side = "left"
        core = ov.Core()
        if "GPU" not in core.available_devices:
            raise RuntimeError("OpenVINO GPU device is unavailable; CPU fallback is disabled")
        self.device_name = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
        model = core.read_model(str(xml))
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
        rows = np.arange(len(texts))
        last_positions = mask.shape[1] - 1 - np.argmax(mask[:, ::-1], axis=1)
        vectors = hidden[rows, last_positions].astype(np.float32)
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
