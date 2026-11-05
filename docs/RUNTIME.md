# Runtime selection: PyTorch CPU vs OpenVINO CPU vs OpenVINO iGPU

Measured on the target NAS (Intel N150, 4 cores, 32GB shared memory, 24EU iGPU)
inside the service container, same inputs, warm caches. Full methodology and
raw numbers: `docs/BENCHMARKS.md`.

| Runtime | b1 (single query) | b32 (batch ingest) | Notes |
|---|---|---|---|
| PyTorch CPU (eager) | 15.6 ms | 248.6 ms | Reference; what `sentence-transformers` does by default |
| OpenVINO CPU | 9.9 ms | 211.7 ms | Graph-fused kernels beat eager PyTorch without any hardware change |
| **OpenVINO iGPU** | **8.0 ms** | **89.5 ms** | ~2.8x batch throughput over PyTorch CPU; offloads the weak CPU cores |

Why the service standardizes on OpenVINO (both devices) rather than PyTorch:

1. **One runtime, two devices.** OpenVINO's CPU and GPU plugins share the same
   graph format, so the same serving code covers CPU-only hosts and iGPU hosts.
   PyTorch would need a separate path (and a much larger image) for GPU.
2. **CPU wins are free.** Even without an iGPU, OpenVINO CPU is 1.6x faster
   than eager PyTorch on this hardware — the runtime choice pays off before
   any GPU is involved.
3. **Deterministic, serialized inference.** A single `InferRequest` behind a
   mutex is the concurrency contract that survived stress testing on this iGPU
   (see `docs/BENCHMARKS.md`); PyTorch threading under concurrent HTTP load
   was not revalidated and is not claimed.
4. **Small models don't need big iron.** A 112M-parameter encoder runs in
   single-digit milliseconds on either OpenVINO device; the batch path is where
   the iGPU pulls ahead (2.8x), which is exactly the offline-ingestion shape.

When PyTorch (CPU) remains the right choice: one-off experimentation,
HuggingFace-native pipelines, and any model whose ONNX/OpenVINO export is not
validated. The export tool (`tools/export_onnx.py`) plus the semantic check in
`docs/BENCHMARKS.md` is how we validate before switching.
