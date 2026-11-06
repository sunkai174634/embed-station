from __future__ import annotations

import asyncio
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
from .inference import Engine
from .registry import ModelSlot, Registry, load_specs
from .settings import settings
from .telemetry import Telemetry, configure_logging


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


def create_app(registry: Registry | None = None, telemetry: Telemetry | None = None) -> FastAPI:
    logger = configure_logging(settings.log_level)

    async def attach_batch(slot) -> None:
        """Embedding slots get a BatchProcessor (BATCH_SIZE / BATCH_WINDOW_MS
        stay per-platform effective); rerankers never do. Runs on the loop."""
        if slot.spec.role != "embedding" or slot.engine is None or slot.batch is not None:
            return
        processor = BatchProcessor(
            slot.engine,
            settings.batch_size,
            settings.batch_window_ms,
            workers=settings.jobs,
        )
        processor.start()
        slot.batch = processor

    async def detach_batch(slot) -> None:
        processor = slot.batch
        slot.batch = None
        if processor is not None:
            await processor.close()

    state = {
        "registry": registry,
        "telemetry": telemetry,
        "started": None,
        "ready": False,
        "device": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.validate()
        store = telemetry or Telemetry(
            settings.data_dir / "embed-station.sqlite3",
            retention_days=settings.telemetry_retention_days,
        )
        state["telemetry"] = store
        reg = registry
        if reg is None:
            reg = await asyncio.to_thread(Registry, load_specs())
            await asyncio.to_thread(reg.start_default)
        state["registry"] = reg
        for slot in reg.slots.values():
            if slot.enabled:
                await attach_batch(slot)
        device = None
        for slot in reg.slots.values():
            if slot.engine is not None:
                device = getattr(slot.engine, "device_name", None)
                break
        state["device"] = device
        state["started"] = datetime.now(timezone.utc).isoformat()
        state["ready"] = True
        for slot in reg.slots.values():
            if slot.enabled:
                store.event(
                    "INFO",
                    "model_ready",
                    detail=(
                        f"model={slot.spec.model_id} role={slot.spec.role}"
                        f" device={getattr(slot.engine, 'device_name', '?')}"
                    ),
                )
        store.event("INFO", "service_ready", detail=f"models={len(reg.slots)} device={device}")
        yield
        state["ready"] = False
        for slot in reg.slots.values():
            await detach_batch(slot)
        await asyncio.to_thread(reg.close)

    app = FastAPI(title="Inference Station", version="0.2.0", lifespan=lifespan)
    static_dir = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/")
    async def dashboard():
        return FileResponse(static_dir / "index.html")

    @app.get("/health")
    async def health():
        reg: Registry | None = state["registry"]
        models = reg.describe() if reg else []
        return {
            "status": "ok",
            "ready": state["ready"],
            "device": state["device"],
            "models": [m["id"] for m in models if m["enabled"]],
            "all_models": models,
        }

    @app.get("/ready")
    async def ready():
        reg: Registry | None = state["registry"]
        if not state["ready"] or (reg and not any(s.enabled for s in reg.slots.values())):
            raise HTTPException(503, "no enabled model is ready")
        return {"status": "ready", "device": state["device"]}

    @app.get("/v1/models")
    async def models():
        reg: Registry | None = state["registry"]
        if not state["ready"]:
            raise HTTPException(503, "model is not ready")
        return {
            "object": "list",
            "data": [
                {"id": m["id"], "object": "model", "owned_by": "inference-station"}
                for m in (reg.describe() if reg else [])
                if m["enabled"]
            ],
        }

    @app.get("/v1/info")
    async def v1_info():
        """TEI-compatible handshake: TEI clients (e.g. Hindsight) GET this on init."""
        reg: Registry | None = state["registry"]
        if not state["ready"] or reg is None:
            raise HTTPException(503, "model is not ready")
        rr = reg.get(None, "rerank")
        model_id = rr.engine.model_id if rr and rr.engine else reg.default_embedding
        return {"model_id": model_id, "device": state["device"]}

    @app.get("/api/models")
    async def api_models():
        reg: Registry | None = state["registry"]
        if reg is None:
            return {"models": [], "default_embedding": None}
        return {"models": reg.describe(), "default_embedding": reg.default_embedding}

    @app.post("/api/models/{model_id}/toggle")
    async def api_toggle(model_id: str, enable: bool = Query(...)):
        reg: Registry | None = state["registry"]
        if reg is None or model_id not in reg.slots:
            known = sorted(reg.slots) if reg else []
            raise HTTPException(404, f"unknown model: {model_id}; known: {known}")
        slot = await asyncio.to_thread(reg.toggle, model_id, enable)
        if enable:
            await attach_batch(slot)
        else:
            await detach_batch(slot)
        state["telemetry"].event(
            "INFO",
            "model_enabled" if enable else "model_disabled",
            detail=f"model={model_id}",
        )
        return {"id": model_id, "enabled": slot.enabled}

    def _resolve_embedding(body: EmbeddingRequest) -> tuple[ModelSlot, Engine]:
        """Route to the requested embedding model; error if unknown/disabled."""
        reg: Registry | None = state["registry"]
        if reg is None:
            raise HTTPException(503, "service is not ready")
        if body.model and body.model not in reg.slots:
            raise HTTPException(
                404,
                f"unknown model: {body.model}; available: {sorted(reg.slots)}",
            )
        slot = reg.get(body.model, "embedding")
        if slot is None or slot.spec.role != "embedding":
            raise HTTPException(404, "no embedding model is registered")
        if not slot.enabled or slot.engine is None:
            raise HTTPException(
                503,
                f"model {slot.spec.model_id} is disabled; enable it via"
                f" POST /api/models/{slot.spec.model_id}/toggle?enable=true",
            )
        return slot, slot.engine

    @app.post("/v1/embeddings")
    async def embeddings(body: EmbeddingRequest, request: Request, response: Response):
        store: Telemetry = state["telemetry"]
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response.headers["x-request-id"] = request_id
        if not state["ready"]:
            raise HTTPException(503, "model is not ready")
        slot, engine = _resolve_embedding(body)
        texts = [body.input] if isinstance(body.input, str) else body.input
        if len(texts) > settings.max_batch_items:
            raise HTTPException(
                413,
                f"{len(texts)} inputs exceed the per-request limit of {settings.max_batch_items}",
            )
        processor: BatchProcessor | None = slot.batch
        if processor is not None and processor.backlog + len(texts) > settings.max_queue_items:
            # Overload guard: reject before enqueueing so ghost/timed-out work
            # cannot pile up behind a wedged GPU and starve new requests.
            raise HTTPException(
                503,
                f"service is busy ({processor.backlog} queued items); retry later",
                headers={"Retry-After": "5"},
            )
        limit = slot.spec.limits.get("max_input_tokens", settings.max_input_tokens)
        token_total = await asyncio.to_thread(engine.count_tokens, texts)
        if token_total > limit:
            raise HTTPException(
                413,
                f"input of {token_total} tokens exceeds the limit of {limit};"
                " split the input into smaller batches",
            )
        started = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        try:
            try:
                if processor is not None:
                    result = await asyncio.wait_for(
                        processor.embed(texts), timeout=settings.infer_timeout_s
                    )
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(engine.embed, texts), timeout=settings.infer_timeout_s
                    )
            except asyncio.CancelledError:
                # A toggle disabled this model while the request was queued;
                # surface it as a retryable condition instead of a bare 500.
                if not state["ready"]:
                    raise
                raise HTTPException(
                    503,
                    "model was disabled while this request was queued; retry",
                    headers={"Retry-After": "2"},
                ) from None
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
                f"model={slot.spec.model_id} inputs={len(texts)}"
                f" tokens={result.token_count} duration_ms={duration:.2f}",
            )
            logger.info(
                "embedding_succeeded",
                extra={
                    "request_id": request_id,
                    "detail": (
                        f"model={slot.spec.model_id} inputs={len(texts)}"
                        f" tokens={result.token_count} duration_ms={duration:.2f}"
                    ),
                },
            )
            return {
                "object": "list",
                "model": slot.spec.model_id,
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
        reg: Registry | None = state["registry"]
        request_id = request.headers.get("x-request-id") or str(uuid4())
        response.headers["x-request-id"] = request_id
        if not state["ready"] or reg is None:
            raise HTTPException(503, "model is not ready")
        if body.model and body.model not in reg.slots:
            raise HTTPException(
                404,
                f"unknown model: {body.model}; available: {sorted(reg.slots)}",
            )
        slot = reg.get(body.model, "rerank")
        if slot is None or slot.spec.role != "rerank":
            raise HTTPException(404, "no reranker is registered")
        if not slot.enabled or slot.engine is None:
            raise HTTPException(
                503,
                f"reranker {slot.spec.model_id} is disabled; enable it via"
                f" POST /api/models/{slot.spec.model_id}/toggle?enable=true",
            )
        reranker = slot.engine
        assert reranker is not None  # noqa: S101 — guarded by enabled check above
        batch_limit = slot.spec.limits.get("max_batch_texts", settings.max_batch_items)
        if len(body.texts) > batch_limit:
            raise HTTPException(
                413,
                f"{len(body.texts)} texts exceed the per-request limit of {batch_limit}",
            )
        rr_limit = slot.spec.limits.get("max_total_tokens", settings.rerank_max_tokens)
        token_total = await asyncio.to_thread(reranker.count_tokens, body.query, body.texts)
        if token_total > rr_limit:
            raise HTTPException(
                413,
                f"input of {token_total} tokens exceeds the rerank limit of {rr_limit};"
                " split the input into smaller batches",
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
            # TEI-compatible: the top level IS the array. Wrapping it in an
            # object breaks TEI clients (e.g. Hindsight) that iterate it
            # expecting dicts of {index, score}.
            return results
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
        reg: Registry | None = state["registry"]
        summary = await asyncio.to_thread(state["telemetry"].summary) if state["telemetry"] else {}
        default_slot = reg.get(None, "embedding") if reg else None
        return {
            "ready": state["ready"],
            "device": state["device"],
            "models": reg.describe() if reg else [],
            "default_embedding": reg.default_embedding if reg else None,
            "default_dimension": getattr(default_slot.engine, "dimension", None)
            if default_slot and default_slot.engine
            else None,
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
