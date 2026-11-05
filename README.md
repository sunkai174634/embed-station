# Embed Station

A self-hosted, OpenAI-compatible embedding service for NAS-class hardware
(Intel iGPU / small CPUs). Serves any encoder through **OpenVINO** — either an
OpenVINO IR export or an ONNX export — with a built-in dashboard, SQLite
request/event history, and hard input guards. Not tied to any single consumer:
point anything that speaks the OpenAI embeddings API at it.

**Privacy:** the service persists request metadata only (timestamps, counts,
latency, status). Input text and embedding vectors are never stored.

## Features

- `POST /v1/embeddings`, `GET /v1/models` — OpenAI-compatible
- Two engine formats, one service:
  - `MODEL_FORMAT=openvino` — OpenVINO IR exports (e.g. Qwen3-Embedding INT8)
  - `MODEL_FORMAT=onnx` — any ONNX encoder with sentence-transformers pooling
    (e.g. `hs-embed-1`); export tool included (`tools/export_onnx.py`)
- GPU-only philosophy on iGPU hosts: the process exits instead of silently
  falling back to CPU (see `docs/RUNTIME.md` for the CPU-vs-GPU trade study)
- Hard guards sized by measurement: input token cap (HTTP 413 before GPU),
  max items per request, per-request inference timeout
- Same-origin dashboard (`/`) with request history and service events
- SQLite (WAL) telemetry with optional retention

## Quick start

Requires Linux/amd64 with an Intel GPU (OpenVINO GPU runtime + drivers), or a
plain CPU host (the OpenVINO CPU plugin is used when no GPU device is present
in `MODEL_FORMAT` debugging — but the default engine intentionally refuses to
start without GPU; see "Design decisions" below).

```bash
cp .env.example .env
# edit MODEL_DIR / VIDEO_GID / RENDER_GID
docker compose up --build -d
curl -s localhost:8080/health
curl -s localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-embedding-0.6b-int8-ov","input":["hello","world"]}'
```

The compose sample binds `127.0.0.1` only; the API and dashboard are
unauthenticated — keep them on trusted networks or behind an authenticating
reverse proxy.

### Serving an ONNX model (e.g. hs-embed-1)

```bash
python3 tools/export_onnx.py --src /path/to/hs-embed-1 --out /path/to/hs-embed-1-onnx
# then set: MODEL_FORMAT=onnx MODEL_DIR=/path/to/hs-embed-1-onnx MODEL_ID=hs-embed-1
```

## Configuration

All knobs are environment variables with measured defaults; see
`.env.example` for the annotated list:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | — | Model directory (IR `openvino_model.xml` or ONNX `model.onnx` + tokenizer) |
| `MODEL_ID` | — | Served model name (echoed by `/v1/models`, checked against requests) |
| `MODEL_FORMAT` | `openvino` | `openvino` (IR) or `onnx` (ONNX + sentence-transformers pooling) |
| `INFER_JOBS` | `1` | CPU-side workers; GPU inference stays serialized |
| `BATCH_SIZE` / `BATCH_WINDOW_MS` | `4` / `20` | Micro-batching; at sparse load behaves as batch=1, auto-coalesces under load |
| `MAX_INPUT_TOKENS` | `3584` | Measured iGPU ceiling; oversize requests get 413 before GPU |
| `MAX_BATCH_ITEMS` | `100` | Max strings per request |
| `INFER_TIMEOUT_S` | `30` | Per-request inference watchdog |
| `TELEMETRY_RETENTION_DAYS` | `0` | `0` keeps history forever |

## Design decisions

- **OpenVINO over PyTorch** — one graph format covers CPU and iGPU; on the
  target N150 the CPU plugin alone is 1.6x faster than eager PyTorch, and the
  iGPU gives 2.8x batch throughput. Full study: `docs/RUNTIME.md`.
- **Serialized GPU inference** — one `InferRequest` behind a mutex; concurrent
  submission crashed reproducibly on this hardware (details in
  `docs/BENCHMARKS.md` §3).
- **Fail-fast on missing GPU** — silent CPU fallback would quietly turn a
  "fast" endpoint into a slow one; misconfiguration should be loud.
- **Metadata-only telemetry** — memory systems ingest personal content;
  storing raw text would make the telemetry DB a privacy liability.

## Documentation

- `docs/RUNTIME.md` — PyTorch CPU vs OpenVINO CPU vs OpenVINO iGPU selection
- `docs/BENCHMARKS.md` — all measured numbers, methodology, semantic-fidelity
  check, load context, A/B retrieval study
- `docs/ARCHITECTURE.md` — module layout and data flow
- `docs/ACCEPTANCE.md` — acceptance evidence log

## Development

```bash
uv sync --extra dev
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

Static checks, unit tests (fake engine), container build (linux/amd64), and a
real-GPU smoke on the target NAS are all part of the acceptance flow in
`docs/ACCEPTANCE.md`.
