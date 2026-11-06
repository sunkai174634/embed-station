import json
import os
from dataclasses import dataclass
from pathlib import Path


def load_model_limits(model_dir: Path) -> dict:
    """Read optional station_limits.json from a model directory.

    Per-model limits live WITH the model (they are a property of the
    model+hardware combination, e.g. Qwen3-INT8's 3584-token iGPU hang
    threshold), so swapping models can never leave a stale env-configured
    limit behind. Keys: max_input_tokens, max_total_tokens (rerank budget).
    """
    path = model_dir / "station_limits.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return {k: v for k, v in data.items() if isinstance(v, int)}
    except (json.JSONDecodeError, OSError):
        return {}


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("MODEL_DIR", "/models/qwen3-embedding-0.6b-int8-ov"))
    model_id: str = os.getenv("MODEL_ID", "qwen3-embedding-0.6b-int8-ov")
    model_format: str = os.getenv("MODEL_FORMAT", "openvino")
    reranker_model_dir: str = os.getenv("RERANKER_MODEL_DIR", "")
    reranker_model_id: str = os.getenv("RERANKER_MODEL_ID", "reranker")
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    jobs: int = int(os.getenv("INFER_JOBS", "1"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    batch_window_ms: int = int(os.getenv("BATCH_WINDOW_MS", "0"))
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "3584"))
    rerank_max_tokens: int = int(os.getenv("RERANK_MAX_TOKENS", "24000"))
    max_batch_items: int = int(os.getenv("MAX_BATCH_ITEMS", "100"))
    max_queue_items: int = int(os.getenv("MAX_QUEUE_ITEMS", "512"))
    telemetry_retention_days: int = int(os.getenv("TELEMETRY_RETENTION_DAYS", "0"))
    infer_timeout_s: float = float(os.getenv("INFER_TIMEOUT_S", "30"))

    def validate(self) -> None:
        if self.jobs < 1 or self.batch_size < 1 or self.batch_window_ms < 0:
            raise ValueError("jobs/batch_size must be positive and batch_window_ms non-negative")
        if self.max_input_tokens < 1:
            raise ValueError("MAX_INPUT_TOKENS must be positive")
        if self.rerank_max_tokens < 1:
            raise ValueError("RERANK_MAX_TOKENS must be positive")
        if self.max_batch_items < 1:
            raise ValueError("MAX_BATCH_ITEMS must be positive")
        if self.max_queue_items < 1:
            raise ValueError("MAX_QUEUE_ITEMS must be positive")
        if self.telemetry_retention_days < 0:
            raise ValueError("TELEMETRY_RETENTION_DAYS must be non-negative")
        if self.infer_timeout_s <= 0:
            raise ValueError("INFER_TIMEOUT_S must be positive")
        if not self.model_id.strip():
            raise ValueError("MODEL_ID must not be empty")
        if self.model_format not in ("openvino", "onnx"):
            raise ValueError("MODEL_FORMAT must be 'openvino' or 'onnx'")


settings = Settings()
