from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator

from .batch import BatchProcessor
from .inference import Engine, OpenVINOEngine
from .onnx_engine import OnnxOpenVINOEngine
from .rerank_engine import OnnxRerankerEngine
from .settings import settings
from .telemetry import Telemetry, configure_logging


def build_engine() -> Engine:
    """Select the engine by MODEL_FORMAT: 'onnx' serves any ONNX encoder,
    'openvino' (default) serves an OpenVINO IR export directory."""
    cache_dir = Path(os.getenv("OV_CACHE_DIR", "/cache/gpu")) if os.getenv("OV_CACHE_DIR") else None
    if settings.model_format == "onnx":
        return OnnxOpenVINOEngine(settings.model_dir, settings.model_id, cache_dir)
    return OpenVINOEngine(settings.model_dir, settings.model_id, cache_dir)


def build_reranker(gpu_lock) -> OnnxRerankerEngine | None:
    """Optional second model: an ONNX cross-encoder sharing the process GPU lock.
    Disabled unless RERANKER_MODEL_DIR points at a model.onnx directory."""
    if not settings.reranker_model_dir:
        return None
    cache_dir = Path(os.getenv("OV_CACHE_DIR", "/cache/gpu")) if os.getenv("OV_CACHE_DIR") else None
    return OnnxRerankerEngine(
        Path(settings.reranker_model_dir), settings.reranker_model_id, gpu_lock, cache_dir
    )


class RerankRequest(BaseModel):
    """TEI-compatible /rerank payload."""

    model_config = ConfigDict(extra="ignore")
    query: str
    texts: list[str]
    model: str | None = None
    return_text: bool = False

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must be a non-empty string")
        return value

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: list[str]) -> list[str]:
        if not value or any(not isinstance(t, str) or not t.strip() for t in value):
            raise ValueError("texts must be a non-empty list of non-empty strings")
        if len(value) > settings.max_batch_items:
            raise ValueError(
                f"{len(value)} texts exceed the per-request limit of {settings.max_batch_items}"
            )
        return value


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input: str | list[str]
    model: str | None = None

    @field_validator("input")
    @classmethod
    def validate_input(cls, value: str | list[str]) -> str | list[str]:
        texts = [value] if isinstance(value, str) else value
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("input must be a non-empty string or list of non-empty strings")
        return value


def describe_exception(exc: BaseException) -> str:
    """Compact, log-safe error summary: class + message, never input text."""
    message = str(exc).strip()
    name = type(exc).__name__
    if not message:
        return name
    kept = " ".join(message.split())
    if len(kept) > 200:
        kept = kept[:197] + "..."
    return f"{name}: {kept}"


def create_app(engine: Engine | None = None, telemetry: Telemetry | None = None) -> FastAPI:
    logger = configure_logging(settings.log_level)
    state = {
        "engine": engine,
        "batch": None,
        "telemetry": telemetry,
        "started": None,
        "ready": False,
        "device": None,
        "dimension": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.validate()
        store = telemetry or Telemetry(
            settings.data_dir / "embed-station.sqlite3",
            retention_days=settings.telemetry_retention_days,
        )
        state["telemetry"] = store
        active = engine
        if active is None:
            active = await asyncio.to_thread(build_engine)
        state["engine"] = active
        state["dimension"] = active.dimension
        state["device"] = active.device_name
        state["started"] = datetime.now(timezone.utc).isoformat()
        state["ready"] = True
        state["batch"] = BatchProcessor(
            active,
            settings.batch_size,
            settings.batch_window_ms,
            workers=settings.jobs,
        )
        state["batch"].start()
        reranker = await asyncio.to_thread(build_reranker, active.infer_lock)
        state["reranker"] = reranker
        if reranker is not None:
            store.event(
                "INFO",
                "reranker_ready",
                detail=f"model={reranker.model_id} device={reranker.device_name}",
            )
        store.event("INFO", "service_ready", detail=f"device={active.device_name}")
        yield
        state["ready"] = False
        await state["batch"].close()
        if state.get("reranker") is not None:
            await asyncio.to_thread(state["reranker"].close)
        await asyncio.to_thread(active.close)

    app = FastAPI(title="Embed Station", version="0.1.0", lifespan=lifespan)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def dashboard():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    async def health():
        active = state["engine"]
        return {
            "status": "ok",
            "ready": state["ready"],
            "model": active.model_id if active else settings.model_id,
            "device": state["device"],
            "dimension": state["dimension"],
            "reranker": state["reranker"].model_id if state.get("reranker") is not None else None,
        }

    @app.get("/ready")
    async def ready():
        if not state["ready"]:
            raise HTTPException(503, "GPU model is not ready")
        return {"status": "ready", "device": state["device"]}

    @app.get("/v1/models")
    async def models():
        if not state["ready"]:
            raise HTTPException(503, "model is not ready")
        active = state["engine"]
        return {
            "object": "list",
            "data": [{"id": active.model_id, "object": "model", "owned_by": "embed-station"}],
        }

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbeddingRequest, request: Request, response: Response):
        active = state["engine"]
        store: Telemetry = state["telemetry"]
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response.headers["x-request-id"] = request_id
        if not state["ready"]:
            raise HTTPException(503, "model is not ready")
        if body.model and body.model != active.model_id:
            raise HTTPException(400, "requested model does not match configured model")
        texts = [body.input] if isinstance(body.input, str) else body.input
        if len(texts) > settings.max_batch_items:
            raise HTTPException(
                413,
                f"{len(texts)} inputs exceed the per-request limit of {settings.max_batch_items}",
            )
        processor: BatchProcessor = state["batch"]
        if processor is not None and processor.backlog + len(texts) > settings.max_queue_items:
            # Overload guard: reject before enqueueing so ghost/timed-out work
            # cannot pile up behind a wedged GPU and starve new requests.
            raise HTTPException(
                503,
                f"service is busy ({processor.backlog} queued items); retry later",
                headers={"Retry-After": "5"},
            )
        token_total = await asyncio.to_thread(state["engine"].count_tokens, texts)
        if token_total > settings.max_input_tokens:
            raise HTTPException(
                413,
                f"input of {token_total} tokens exceeds the limit of {settings.max_input_tokens};"
                " split the input into smaller batches",
            )
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            result = await asyncio.wait_for(
                state["batch"].embed(texts), timeout=settings.infer_timeout_s
            )
            duration = (time.perf_counter() - started) * 1000
            await asyncio.to_thread(
                store.record_request,
                {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "input_count": len(texts),
                    "token_count": result.token_count,
                    "duration_ms": duration,
                    "status": "ok",
                },
            )
            await asyncio.to_thread(
                store.event,
                "INFO",
                "embedding_succeeded",
                request_id,
                f"inputs={len(texts)} tokens={result.token_count} duration_ms={duration:.2f}",
            )
            logger.info(
                "embedding_succeeded",
                extra={
                    "request_id": request_id,
                    "detail": (
                        f"inputs={len(texts)} tokens={result.token_count} "
                        f"duration_ms={duration:.2f}"
                    ),
                },
            )
            return {
                "object": "list",
                "model": active.model_id,
                "data": [
                    {"object": "embedding", "index": i, "embedding": vector}
                    for i, vector in enumerate(result.vectors)
                ],
                "usage": {"prompt_tokens": result.token_count, "total_tokens": result.token_count},
            }
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as exc:
            duration = (time.perf_counter() - started) * 1000
            detail = describe_exception(exc)
            await asyncio.to_thread(
                store.record_request,
                {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "input_count": len(texts),
                    "token_count": 0,
                    "duration_ms": duration,
                    "status": "error",
                    "error": detail,
                },
            )
            await asyncio.to_thread(store.event, "ERROR", "embedding_timeout", request_id, detail)
            logger.warning(
                "embedding_timeout",
                extra={
                    "request_id": request_id,
                    "detail": f"timeout after {settings.infer_timeout_s}s",
                },
            )
            # 503 (not 500): clients back off and retry instead of hammering a
            # busy GPU. The in-flight batch itself cannot be cancelled.
            raise HTTPException(
                503,
                f"inference timed out after {settings.infer_timeout_s}s; retry later",
                headers={"Retry-After": "5"},
            ) from exc
        except ValueError as exc:
            raise HTTPException(400, describe_exception(exc)) from exc
        except Exception as exc:
            duration = (time.perf_counter() - started) * 1000
            detail = describe_exception(exc)
            await asyncio.to_thread(
                store.record_request,
                {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "input_count": len(texts),
                    "token_count": 0,
                    "duration_ms": duration,
                    "status": "error",
                    "error": detail,
                },
            )
            await asyncio.to_thread(store.event, "ERROR", "embedding_failed", request_id, detail)
            logger.exception("embedding_failed", extra={"request_id": request_id, "detail": detail})
            raise HTTPException(500, "embedding inference failed") from exc

    @app.post("/v1/rerank")
    async def rerank(body: RerankRequest, request: Request, response: Response):
        """TEI-compatible reranking: score (query, text) pairs, return sorted results."""
        reranker = state.get("reranker")
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response.headers["x-request-id"] = request_id
        if not state["ready"]:
            raise HTTPException(503, "model is not ready")
        if reranker is None:
            raise HTTPException(404, "no reranker configured; set RERANKER_MODEL_DIR")
        if body.model and body.model != reranker.model_id:
            raise HTTPException(400, "requested model does not match configured reranker")
        token_total = await asyncio.to_thread(reranker.count_tokens, body.query, body.texts)
        if token_total > settings.rerank_max_tokens:
            raise HTTPException(
                413,
                f"input of {token_total} tokens exceeds the rerank limit"
                f" of {settings.rerank_max_tokens}; split the input into smaller batches",
            )
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        store: Telemetry = state["telemetry"]
        try:
            scores = await asyncio.wait_for(
                asyncio.to_thread(reranker.rerank, body.query, body.texts),
                timeout=settings.infer_timeout_s,
            )
            order = sorted(range(len(body.texts)), key=lambda i: -scores[i])
            duration = (time.perf_counter() - started) * 1000
            await asyncio.to_thread(
                store.record_request,
                {
                    "request_id": request_id,
                    "timestamp": timestamp,
                    "input_count": len(body.texts),
                    "token_count": token_total,
                    "duration_ms": duration,
                    "status": "ok",
                },
            )
            await asyncio.to_thread(
                store.event,
                "INFO",
                "rerank_succeeded",
                request_id,
                f"query_pairs={len(body.texts)} tokens={token_total} duration_ms={duration:.2f}",
            )
            results = [
                {
                    "index": i,
                    "score": float(scores[i]),
                    **({"text": body.texts[i]} if body.return_text else {}),
                }
                for i in order
            ]
            return {"model": reranker.model_id, "results": results}
        except asyncio.TimeoutError as exc:
            await asyncio.to_thread(
                store.event,
                "ERROR",
                "rerank_timeout",
                request_id,
                f"timeout after {settings.infer_timeout_s}s",
            )
            raise HTTPException(
                503,
                f"rerank timed out after {settings.infer_timeout_s}s; retry later",
                headers={"Retry-After": "5"},
            ) from exc
        except ValueError as exc:
            raise HTTPException(400, describe_exception(exc)) from exc
        except Exception as exc:
            detail = describe_exception(exc)
            await asyncio.to_thread(store.event, "ERROR", "rerank_failed", request_id, detail)
            raise HTTPException(500, "rerank inference failed") from exc

    @app.get("/api/status")
    async def status():
        active = state["engine"]
        summary = await asyncio.to_thread(state["telemetry"].summary) if state["telemetry"] else {}
        return {
            "ready": state["ready"],
            "device": state["device"],
            "model": active.model_id if active else settings.model_id,
            "dimension": state["dimension"],
            "started_at": state["started"],
            **summary,
        }

    @app.get("/api/requests")
    async def requests(
        limit: int = Query(25, ge=1, le=100),
        offset: int = Query(0, ge=0),
        request_id: str | None = None,
        status_filter: str | None = None,
    ):
        return await asyncio.to_thread(
            state["telemetry"].page,
            "requests",
            limit,
            offset,
            request_id=request_id,
            status=status_filter,
        )

    @app.get("/api/events")
    @app.get("/api/logs")
    async def events(
        limit: int = Query(25, ge=1, le=100),
        offset: int = Query(0, ge=0),
        request_id: str | None = None,
        level: str | None = None,
    ):
        return await asyncio.to_thread(
            state["telemetry"].page, "events", limit, offset, request_id=request_id, level=level
        )

    return app


app = create_app()
