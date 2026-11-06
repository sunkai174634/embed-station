"""Model registry: the station is a multi-model platform, not a single-model
service. Models are declared via the MODELS env (comma-separated entries)

    MODELS="id:dir:format:role:default_on[,...]"

    id         serving name used in request `model` fields
    dir        model directory (mounted into the container)
    format     openvino | onnx (embedding engines) or rerank (cross-encoder)
    role       embedding | rerank
    default_on "on"/"off" — initial toggle state at startup

Backward compatibility: when MODELS is unset, the legacy MODEL_DIR /
MODEL_ID / MODEL_FORMAT / RERANKER_* envs describe the same registry with
one embedding model (always on) and an optional reranker.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inference import Engine, OpenVINOEngine
from .onnx_engine import OnnxOpenVINOEngine
from .rerank_engine import OnnxRerankerEngine
from .settings import load_model_limits, settings


def _resolve_limits(model_dir: Path, role: str) -> dict:
    """Effective limits for one model: explicit env override wins, else the
    model's own station_limits.json declaration, else the built-in default.
    Env detection uses presence of the variable, so pinning a value for a
    model without a declaration still works."""
    declared = load_model_limits(model_dir)
    resolved = dict(declared)
    if role == "rerank":
        if os.getenv("RERANK_MAX_TOKENS"):
            resolved["max_total_tokens"] = int(os.getenv("RERANK_MAX_TOKENS", "0"))
    elif os.getenv("MAX_INPUT_TOKENS"):
        resolved["max_input_tokens"] = int(os.getenv("MAX_INPUT_TOKENS", "0"))
    return resolved


@dataclass
class ModelSpec:
    model_id: str
    model_dir: Path
    format: str  # openvino | onnx | rerank
    role: str  # embedding | rerank
    default_on: bool
    limits: dict = field(default_factory=dict)


@dataclass
class ModelSlot:
    spec: ModelSpec
    engine: Engine | OnnxRerankerEngine | None = None
    batch: Any = None  # BatchProcessor for embedding models, created by the app
    enabled: bool = False
    loading: bool = False


def _parse_entry(entry: str) -> ModelSpec | None:
    parts = [p.strip() for p in entry.split(":")]
    if len(parts) < 4:
        return None
    model_id, directory, fmt, role = parts[:4]
    default_on = (parts[4].lower() if len(parts) > 4 else "on") == "on"
    if role not in ("embedding", "rerank"):
        return None
    if fmt not in ("openvino", "onnx", "rerank"):
        return None
    if role == "rerank" and fmt != "rerank":
        # rerank role implies the ONNX cross-encoder engine
        fmt = "rerank"
    return ModelSpec(
        model_id=model_id,
        model_dir=Path(directory),
        format=fmt,
        role=role,
        default_on=default_on,
        limits=_resolve_limits(Path(directory), role),
    )


def load_specs() -> list[ModelSpec]:
    declared = os.getenv("MODELS", "")
    specs: list[ModelSpec] = []
    if declared:
        for entry in declared.split(","):
            if not entry.strip():
                continue
            spec = _parse_entry(entry)
            if spec is None:
                raise ValueError(
                    f"invalid MODELS entry: {entry!r} (expected id:dir:format:role[:on|off])"
                )
            specs.append(spec)
        ids = [s.model_id for s in specs]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate model ids in MODELS: {ids}")
        if not any(s.role == "embedding" for s in specs):
            raise ValueError("MODELS must declare at least one embedding model")
        return specs
    # Legacy single-model fallback
    specs.append(
        ModelSpec(
            model_id=settings.model_id,
            model_dir=settings.model_dir,
            format=settings.model_format,
            role="embedding",
            default_on=True,
            limits=_resolve_limits(settings.model_dir, "embedding"),
        )
    )
    if settings.reranker_model_dir:
        specs.append(
            ModelSpec(
                model_id=settings.reranker_model_id,
                model_dir=Path(settings.reranker_model_dir),
                format="rerank",
                role="rerank",
                default_on=True,
                limits=_resolve_limits(Path(settings.reranker_model_dir), "rerank"),
            )
        )
    return specs


class Registry:
    """All models on the station, with runtime enable/disable.

    All engines share one GPU lock: the N150 has a single iGPU, so inference
    is serialized platform-wide; the lock is owned by the first engine built
    and handed to the rest.
    """

    def __init__(self, specs: list[ModelSpec]) -> None:
        self.slots: dict[str, ModelSlot] = {s.model_id: ModelSlot(spec=s) for s in specs}
        self.gpu_lock = threading.Lock()
        self._lock = threading.Lock()  # guards toggle/lookup against requests
        self.default_embedding = os.getenv("DEFAULT_EMBEDDING_MODEL", "")
        if not self.default_embedding:
            for s in specs:
                if s.role == "embedding":
                    self.default_embedding = s.model_id
                    break

    def _build(self, spec: ModelSpec) -> Engine | OnnxRerankerEngine:
        ov_cache = os.getenv("OV_CACHE_DIR")
        cache_dir = Path(ov_cache) if ov_cache else None
        if spec.role == "rerank":
            return OnnxRerankerEngine(spec.model_dir, spec.model_id, self.gpu_lock, cache_dir)
        if spec.format == "onnx":
            return OnnxOpenVINOEngine(spec.model_dir, spec.model_id, cache_dir)
        return OpenVINOEngine(spec.model_dir, spec.model_id, cache_dir)

    def start_default(self) -> list[str]:
        """Build and enable every model whose default_on is true."""
        started: list[str] = []
        for slot in self.slots.values():
            if slot.spec.default_on:
                slot.engine = self._build(slot.spec)
                slot.enabled = True
                started.append(slot.spec.model_id)
        return started

    def get(self, model_id: str | None, role: str) -> ModelSlot | None:
        with self._lock:
            if model_id is None:
                if role == "embedding":
                    return self.slots.get(self.default_embedding)
                return self._first(role)
            return self.slots.get(model_id)

    def _first(self, role: str) -> ModelSlot | None:
        for slot in self.slots.values():
            if slot.spec.role == role:
                return slot
        return None

    def toggle(self, model_id: str, enable: bool) -> ModelSlot:
        """Enable (build engine) or disable (drop engine) a model at runtime."""
        with self._lock:
            slot = self.slots.get(model_id)
            if slot is None:
                raise KeyError(model_id)
            if enable and slot.engine is None:
                slot.loading = True
                try:
                    slot.engine = self._build(slot.spec)
                    slot.enabled = True
                finally:
                    slot.loading = False
            elif not enable and slot.engine is not None:
                engine = slot.engine
                slot.engine = None
                slot.enabled = False
                try:
                    engine.close()
                except Exception:  # noqa: BLE001 — closing must never break a toggle
                    pass
            return slot

    def close(self) -> None:
        for slot in self.slots.values():
            if slot.engine is not None:
                try:
                    slot.engine.close()
                except Exception:  # noqa: BLE001
                    pass
                slot.engine = None
                slot.enabled = False

    def describe(self) -> list[dict]:
        out = []
        for slot in self.slots.values():
            out.append(
                {
                    "id": slot.spec.model_id,
                    "role": slot.spec.role,
                    "format": slot.spec.format,
                    "enabled": slot.enabled,
                    "loading": slot.loading,
                    "default": (
                        self.default_embedding == slot.spec.model_id
                        and slot.spec.role == "embedding"
                    ),
                    "dimension": getattr(slot.engine, "dimension", None) if slot.engine else None,
                    "max_input_tokens": slot.spec.limits.get("max_input_tokens"),
                    "max_total_tokens": slot.spec.limits.get("max_total_tokens"),
                    "max_batch_texts": slot.spec.limits.get("max_batch_texts"),
                }
            )
        return out
