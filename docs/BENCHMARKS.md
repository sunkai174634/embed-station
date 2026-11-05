# Benchmarks

All measurements are from the real deployment target: an Intel N150 NAS
(4 E-cores, 32GB DDR5 shared with the iGPU, 24EU Intel Graphics), Linux, in the
service's own container (OpenVINO 2026.3, Python 3.10), warm caches, model on
local disk. Numbers are wall-clock per call unless stated otherwise.

## 1. Runtime comparison (hs-embed-1, 112M params, 384-dim)

Same container, same inputs, 20 iterations (b1) / 5 iterations (b32).

| Runtime | compile | b1 | b32 (batch=32) |
|---|---|---|---|
| PyTorch CPU (eager, fp32) | — | 15.6 ms | 248.6 ms |
| OpenVINO CPU | 0.3 s | 9.9 ms | 211.7 ms |
| OpenVINO iGPU | 4.0 s | 8.0 ms | **89.5 ms** |

Takeaways:
- iGPU batch throughput is ~2.8x eager-CPU; single-query latency is already
  single-digit ms everywhere, so interactive recall sees little benefit.
- The compile cost is one-time per process start (4 s); keep the process alive.

## 2. Semantic fidelity of the ONNX export

`model.onnx` served on iGPU vs official safetensors weights on CPU (PyTorch,
mean pooling + L2 on both sides), four mixed CN/EN inputs:

```
cosine: [0.99999, 1.0, 1.0, 1.0]  -> SEMANTIC_MATCH (threshold 0.999)
```

The export is not a quality gamble; it is numerically the same model.

## 3. Qwen3-Embedding-0.6B INT8 (OpenVINO IR, 1024-dim) on the same iGPU

Per-request latency at batch=1, and sustained token throughput:

| Input shape | Latency / throughput |
|---|---|
| short (~12 tok) | ~71-105 ms; ~330 tok/s sustained |
| medium (~52 tok) | batch=4 sweet spot: 7.6 texts/s, 396 tok/s |
| long (2048 tok) | P50 9.8 s, 207 tok/s |
| long (2549 tok) | P50 13.6 s, 188 tok/s |

Hard limits measured on this iGPU (see also `MAX_INPUT_TOKENS`):
- 3584 tokens: stable upper bound (~20 s)
- 3712+ tokens: hangs the iGPU worker — the service rejects oversized inputs
  with HTTP 413 before touching the GPU.

Concurrency contract: one `InferRequest` behind a process-wide mutex. Two
threads driving `AsyncInferQueue` simultaneously reproduce
`Infer Request is busy`; the serialized design passed 80-request concurrent
stress with zero errors. `INFER_JOBS` only parallelizes CPU-side work.

## 4. Production load context (why the small model wins here)

Consumption logs from the busiest recorded day (API-gateway consumption logs, 48h window):

- 166,903 tokens / 249 requests over the day
- busiest hour: 31,831 tokens = avg 8.8 tok/s vs ~330 tok/s ceiling (>35x headroom)

At this load the GPU is busy <1% of the day; runtime choice matters for batch
ingest (2.8x) and matters almost not at all for interactive recall, whose
end-to-end latency is dominated by the caller's LLM stages (tens of seconds).

## 5. A/B retrieval quality (why we serve both models)

Identical corpora and queries, both served by this service on iGPU:

- Hand-built 12-fact set: hs-embed-1 10/10 vs Qwen3-0.6B 9/10 (hit@1)
- 63-fact hard-negative set: MRR 0.938 vs 0.759
- 1,212 real memory facts (raw cosine): MRR 0.157 vs 0.095
- 200-fact end-to-end through an external memory system bank (retain + recall, same LLM and
  reranker): hit@1 6/15 vs 5/15; ingest 115.7 s vs 353.9 s

Domain-specialized small models beat general-purpose ones on their target task;
the general model stays useful for cross-consumer 1024-dim needs. The service
serves either via `MODEL_FORMAT`.
