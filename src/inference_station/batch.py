from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .inference import EmbeddingResult, Engine


@dataclass
class _Item:
    text: str
    future: asyncio.Future


class BatchProcessor:
    def __init__(self, engine: Engine, batch_size: int, window_ms: int, workers: int = 1) -> None:
        self.engine = engine
        self.batch_size = batch_size
        self.window = window_ms / 1000
        self.workers = workers
        self.queue: asyncio.Queue[_Item] = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.active: list[_Item] = []
        self.closed = False

    @property
    def backlog(self) -> int:
        """Queued + in-flight items. Used as an overload guard by callers."""
        return self.queue.qsize() + len(self.active)

    def start(self) -> None:
        if self.tasks:
            raise RuntimeError("batch processor already started")
        self.tasks = [asyncio.create_task(self._run()) for _ in range(self.workers)]

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        if self.closed:
            raise RuntimeError("batch processor is closed")
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in texts]
        for text, future in zip(texts, futures, strict=True):
            await self.queue.put(_Item(text, future))
        results = await asyncio.gather(*futures)
        return EmbeddingResult([r[0] for r in results], [r[1] for r in results])

    async def _run(self) -> None:
        while True:
            first = await self.queue.get()
            items = [first]
            deadline = asyncio.get_running_loop().time() + self.window
            while len(items) < self.batch_size:
                timeout = deadline - asyncio.get_running_loop().time()
                if timeout <= 0:
                    break
                try:
                    items.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            active = [item for item in items if not item.future.cancelled()]
            if not active:
                continue
            self.active.extend(active)
            try:
                result = await asyncio.to_thread(self.engine.embed, [x.text for x in active])
                for item, vector, token_count in zip(
                    active, result.vectors, result.token_counts, strict=True
                ):
                    if not item.future.cancelled():
                        item.future.set_result((vector, token_count))
            except asyncio.CancelledError:
                for item in active:
                    if not item.future.done():
                        item.future.cancel()
                raise
            except Exception as exc:
                for item in active:
                    if not item.future.cancelled():
                        item.future.set_exception(exc)
            finally:
                for item in active:
                    if item in self.active:
                        self.active.remove(item)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        while not self.queue.empty():
            item = self.queue.get_nowait()
            if not item.future.done():
                item.future.cancel()
        for item in self.active:
            if not item.future.done():
                item.future.cancel()
        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
