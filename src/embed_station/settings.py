import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_dir: Path = Path(os.getenv("MODEL_DIR", "/models/qwen3-embedding-0.6b-int8-ov"))
    model_id: str = os.getenv("MODEL_ID", "qwen3-embedding-0.6b-int8-ov")
    model_format: str = os.getenv("MODEL_FORMAT", "openvino")
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8080"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    jobs: int = int(os.getenv("INFER_JOBS", "1"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    batch_window_ms: int = int(os.getenv("BATCH_WINDOW_MS", "0"))
    max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "3584"))
    max_batch_items: int = int(os.getenv("MAX_BATCH_ITEMS", "100"))
    telemetry_retention_days: int = int(os.getenv("TELEMETRY_RETENTION_DAYS", "0"))
    infer_timeout_s: float = float(os.getenv("INFER_TIMEOUT_S", "30"))

    def validate(self) -> None:
        if self.jobs < 1 or self.batch_size < 1 or self.batch_window_ms < 0:
            raise ValueError("jobs/batch_size must be positive and batch_window_ms non-negative")
        if self.max_input_tokens < 1:
            raise ValueError("MAX_INPUT_TOKENS must be positive")
        if self.max_batch_items < 1:
            raise ValueError("MAX_BATCH_ITEMS must be positive")
        if self.telemetry_retention_days < 0:
            raise ValueError("TELEMETRY_RETENTION_DAYS must be non-negative")
        if self.infer_timeout_s <= 0:
            raise ValueError("INFER_TIMEOUT_S must be positive")
        if not self.model_id.strip():
            raise ValueError("MODEL_ID must not be empty")
        if self.model_format not in ("openvino", "onnx"):
            raise ValueError("MODEL_FORMAT must be 'openvino' or 'onnx'")


settings = Settings()
