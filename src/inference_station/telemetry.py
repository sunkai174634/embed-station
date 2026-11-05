from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
                "request_id": getattr(record, "request_id", None),
                "detail": getattr(record, "detail", None),
            },
            ensure_ascii=False,
        )


def configure_logging(level: str) -> logging.Logger:
    logger = logging.getLogger("inference_station")
    logger.setLevel(level.upper())
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.handlers[:] = [handler]
    return logger


class Telemetry:
    def __init__(self, path: Path, retention_days: int = 0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.retention_days = retention_days
        self.lock = threading.Lock()
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL, input_count INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    duration_ms REAL NOT NULL, status TEXT NOT NULL, error TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    level TEXT NOT NULL, event TEXT NOT NULL, request_id TEXT, detail TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_requests_timestamp ON requests(timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC);
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
            """)
            if retention_days > 0:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
                db.execute("DELETE FROM requests WHERE timestamp < ?", (cutoff,))
                db.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def record_request(self, item: dict[str, Any]) -> None:
        with self.lock, self._connect() as db:
            db.execute(
                "INSERT INTO requests VALUES(NULL, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item["request_id"],
                    item["timestamp"],
                    item["input_count"],
                    item["token_count"],
                    item["duration_ms"],
                    item["status"],
                    item.get("error"),
                ),
            )

    def event(
        self, level: str, event: str, request_id: str | None = None, detail: str | None = None
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.lock, self._connect() as db:
            db.execute(
                "INSERT INTO events(timestamp,level,event,request_id,detail) VALUES(?,?,?,?,?)",
                (timestamp, level, event, request_id, detail),
            )

    def page(self, table: str, limit: int, offset: int, **filters: str) -> dict[str, Any]:
        limit = min(max(limit, 1), 100)
        offset = max(offset, 0)
        allowed = {"requests": ("request_id", "status"), "events": ("request_id", "level", "event")}
        if table not in allowed:
            raise ValueError("unsupported table")
        clauses, values = [], []
        for key, value in filters.items():
            if key in allowed[table] and value:
                clauses.append(f"{key} = ?")
                values.append(value)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self.lock, self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM {table}{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                (*values, limit, offset),
            ).fetchall()
            total = db.execute(f"SELECT COUNT(*) FROM {table}{where}", values).fetchone()[0]
        return {
            "data": [dict(row) for row in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    def summary(self) -> dict[str, Any]:
        with self.lock, self._connect() as db:
            count, success, avg = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(status='ok'),0), AVG(duration_ms) FROM requests"
            ).fetchone()
            recent = db.execute(
                "SELECT duration_ms FROM requests ORDER BY id DESC LIMIT 100"
            ).fetchall()
        latencies = sorted(row[0] for row in recent)
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else None
        return {
            "requests": count,
            "successes": success,
            "success_rate": success / count if count else None,
            "average_ms": avg,
            "p95_ms": p95,
        }
