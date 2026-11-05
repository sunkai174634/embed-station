import asyncio
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from embed_station.api import create_app
from embed_station.inference import EmbeddingResult
from embed_station.telemetry import Telemetry


class FakeEngine:
    model_id = "test-model"
    dimension = 3
    device_name = "Fake GPU"

    def embed(self, texts):
        vectors = []
        for text in texts:
            value = len(text)
            vector = np.array([value, 1, 2], dtype=np.float32)
            vectors.append((vector / np.linalg.norm(vector)).tolist())
        return EmbeddingResult(vectors, [len(text.split()) for text in texts])

    def close(self):
        pass

    def count_tokens(self, texts):
        return sum(len(text.split()) for text in texts)


def client(tmp_path: Path):
    app = create_app(FakeEngine(), Telemetry(tmp_path / "data.sqlite3"))
    return TestClient(app)


def test_openai_api_and_persistence(tmp_path):
    with client(tmp_path) as test:
        response = test.post(
            "/v1/embeddings",
            headers={"X-Request-ID": "request-42"},
            json={"model": "test-model", "input": ["one two", "three"]},
        )
        assert response.status_code == 200
        assert response.headers["x-request-id"] == "request-42"
        body = response.json()
        assert [item["index"] for item in body["data"]] == [0, 1]
        assert body["usage"]["prompt_tokens"] == 3
        assert all(abs(np.linalg.norm(item["embedding"]) - 1) < 1e-6 for item in body["data"])
        assert test.get("/ready").json()["status"] == "ready"
        history = test.get("/api/requests").json()
        assert history["total"] == 1
        assert history["data"][0]["request_id"] == "request-42"
        assert test.get("/api/logs?limit=100").json()["total"] >= 2
        repeated = test.post(
            "/v1/embeddings", headers={"X-Request-ID": "request-42"}, json={"input": "again"}
        )
        assert repeated.status_code == 200
        assert test.get("/api/requests?request_id=request-42").json()["total"] == 2
        assert test.get("/v1/models").json()["data"][0]["id"] == "test-model"


def test_validation_model_mismatch_and_pagination(tmp_path):
    with client(tmp_path) as test:
        assert test.post("/v1/embeddings", json={"input": []}).status_code == 422
        assert (
            test.post("/v1/embeddings", json={"input": "ok", "model": "other"}).status_code == 400
        )
        oversized = " ".join(["word"] * 4000)
        limited = test.post("/v1/embeddings", json={"input": oversized})
        assert limited.status_code == 413
        assert "exceeds the limit" in limited.json()["detail"]
        for _ in range(3):
            assert test.post("/v1/embeddings", json={"input": "ok"}).status_code == 200
        assert test.get("/api/requests?limit=2&offset=1").json()["total"] == 3
        assert test.get("/api/events?limit=1000").status_code == 422


def test_configured_workers_run_batches_concurrently():
    import threading
    import time

    from embed_station.batch import BatchProcessor

    class ConcurrentEngine(FakeEngine):
        def __init__(self):
            self.lock = threading.Lock()
            self.current = 0
            self.max_current = 0
            self.both_started = threading.Event()

        def embed(self, texts):
            with self.lock:
                self.current += 1
                self.max_current = max(self.max_current, self.current)
                if self.current == 2:
                    self.both_started.set()
            self.both_started.wait(timeout=1)
            time.sleep(0.01)
            with self.lock:
                self.current -= 1
            return super().embed(texts)

    async def run():
        engine = ConcurrentEngine()
        processor = BatchProcessor(engine, batch_size=1, window_ms=0, workers=2)
        processor.start()
        results = await asyncio.gather(processor.embed(["one"]), processor.embed(["two"]))
        await processor.close()
        return engine.max_current, results

    maximum, results = asyncio.run(run())
    assert maximum == 2
    assert [len(result.vectors) for result in results] == [1, 1]


def test_batch_cancellation_does_not_affect_other_waiter(tmp_path):
    from embed_station.batch import BatchProcessor

    processor = BatchProcessor(FakeEngine(), 2, 20)

    async def run():
        processor.start()
        canceled = asyncio.create_task(processor.embed(["a"]))
        survivor = asyncio.create_task(processor.embed(["longer"]))
        canceled.cancel()
        result = await survivor
        await processor.close()
        return result

    result = asyncio.run(run())
    assert result.vectors and len(result.token_counts) == 1


def test_html_errors_are_never_in_database(tmp_path):
    store = Telemetry(tmp_path / "db.sqlite3")
    store.record_request(
        {
            "request_id": "r1",
            "timestamp": "t",
            "input_count": 1,
            "token_count": 0,
            "duration_ms": 1,
            "status": "error",
            "error": "<script>alert(1)</script>",
        }
    )
    page = store.page("requests", 10, 0)
    assert page["data"][0]["error"].startswith("<script>")
