"""Content-addressed response cache: identical request key -> identical answers.

Backtests replay thousands of decisions; caching makes reruns free and exactly reproducible. The key
(`s1.types.request_key`) covers the backend, prompt version, emulator configuration, pinned model, bank
hash, state and questions in render order, so a cached answer is only ever reused for the same prompt.
Degraded responses (any answer with probs=None) are never stored (design_v1 §11.2).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from pathlib import Path

from jevtrader.s1.types import S1Response


class ResponseCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")  # WAL + NORMAL: durable across app crashes, fast commits
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self._db.commit()

    def get(self, key: str) -> S1Response | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        r = S1Response.from_json(json.loads(row[0]))
        r.cached = True
        r.request_key = key
        return r

    def put(self, key: str, response: S1Response) -> None:
        if response.degraded or any(a.probs is None for a in response.answers.values()):
            raise ValueError("degraded responses are never cached")
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(response.to_json(), ensure_ascii=False))
            )
            self._db.commit()

    async def aget(self, key: str) -> S1Response | None:
        return await asyncio.to_thread(self.get, key)

    async def aput(self, key: str, response: S1Response) -> None:
        await asyncio.to_thread(self.put, key, response)

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0])
