"""Persistent spend ledger with a hard cap and per-run sub-caps (the user's LLM budget is $1,000 in total).

Every paid call is recorded in a SQLite ledger (table `spend`, schema unchanged since v0). v1 (F10, F11,
design_v1 §11.2):
- a running total is kept in memory, so the cap check is O(1) instead of a SUM over the table per call;
- rows are committed in batches, off the event loop (`acharge` hands the commit to `asyncio.to_thread`);
- per-run sub-caps: a tag "<RUN>/<program>" counts against the cap of "<RUN>" (1.5x its planned USD).

A charge is always recorded before the cap is enforced: the money is already spent when the response
arrives, so the ledger must show it. Pre-call `check()` is what stops new calls. Other processes writing
the same file are picked up at each commit (the running total is re-read from the table then).
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("jevtrader.s1")


class BudgetExceeded(RuntimeError):
    pass


def run_of(tag: str) -> str:
    """Sub-cap key of a ledger tag: the part before the first '/' (tags are '<RUN>/<program>')."""
    return tag.split("/", 1)[0]


class SpendLedger:
    def __init__(
        self,
        path: Path,
        hard_cap_usd: float = 245.0,
        sub_caps: dict[str, float] | None = None,
        batch_size: int = 32,
        flush_interval_s: float = 2.0,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.hard_cap_usd = hard_cap_usd
        self.sub_caps = dict(sub_caps or {})
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._lock = threading.Lock()  # guards the in-memory state
        self._db_lock = threading.Lock()  # serializes commits from worker threads
        self._pending: list[tuple] = []
        self._last_flush = time.monotonic()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS spend (ts REAL, backend TEXT, model TEXT, tag TEXT,"
                " input_tokens INTEGER, output_tokens INTEGER, usd REAL)"
            )
        self._committed_total, self._committed_runs = self._read_totals()
        self._pending_total = 0.0
        self._pending_runs: dict[str, float] = {}
        self._uncapped_warned: set[str] = set()
        atexit.register(self.flush)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=60, isolation_level="IMMEDIATE")

    def _read_totals(self) -> tuple[float, dict[str, float]]:
        with self._conn() as c:
            rows = c.execute("SELECT tag, COALESCE(SUM(usd), 0) FROM spend GROUP BY tag").fetchall()
        runs: dict[str, float] = {}
        for tag, usd in rows:
            runs[run_of(tag or "")] = runs.get(run_of(tag or ""), 0.0) + float(usd)
        return sum(runs.values()), runs

    # ------------------------------------------------------------------ totals
    def total(self) -> float:
        with self._lock:
            return self._committed_total + self._pending_total

    def run_total(self, run: str) -> float:
        with self._lock:
            return self._committed_runs.get(run, 0.0) + self._pending_runs.get(run, 0.0)

    def check(self, projected_usd: float = 0.0, tag: str = "") -> None:
        """Raise if a call projected to cost `projected_usd` would cross the hard cap or the tag's sub-cap. A run
        without a sub-cap entry is covered by the hard cap only; that is logged once per run (dev1_report O-1)."""
        if self.total() + projected_usd > self.hard_cap_usd:
            raise BudgetExceeded(f"projected spend exceeds cap {self.hard_cap_usd:.2f} USD")
        run = run_of(tag)
        cap = self.sub_caps.get(run)
        if cap is None and run not in self._uncapped_warned:
            self._uncapped_warned.add(run)
            log.warning("spend ledger: run %r has no sub-cap in configs/backends.yaml; only the %.2f USD hard cap "
                        "applies", run, self.hard_cap_usd)
        if cap is not None and self.run_total(run) + projected_usd > cap:
            raise BudgetExceeded(f"projected spend of run {run!r} exceeds its sub-cap {cap:.2f} USD")

    # ------------------------------------------------------------------ charging
    def _record(self, backend: str, model: str, tag: str, input_tokens: int, output_tokens: int, usd: float) -> tuple[bool, str | None]:
        """Book the charge in memory; return (commit due, cap-violation message or None)."""
        run = run_of(tag)
        with self._lock:
            self._pending.append((time.time(), backend, model, tag, int(input_tokens), int(output_tokens), float(usd)))
            self._pending_total += usd
            self._pending_runs[run] = self._pending_runs.get(run, 0.0) + usd
            total = self._committed_total + self._pending_total
            run_total = self._committed_runs.get(run, 0.0) + self._pending_runs[run]
            due = len(self._pending) >= self.batch_size or time.monotonic() - self._last_flush >= self.flush_interval_s
        over = None
        if total > self.hard_cap_usd:
            over = f"spend {total:.2f} exceeds cap {self.hard_cap_usd:.2f} USD"
        cap = self.sub_caps.get(run)
        if cap is not None and run_total > cap:
            over = f"spend of run {run!r} ({run_total:.2f}) exceeds its sub-cap {cap:.2f} USD"
        return due, over

    def charge(self, backend: str, model: str, tag: str, input_tokens: int, output_tokens: int, usd: float) -> None:
        due, over = self._record(backend, model, tag, input_tokens, output_tokens, usd)
        if due or over:
            self.flush()
        if over:
            raise BudgetExceeded(over)

    async def acharge(self, backend: str, model: str, tag: str, input_tokens: int, output_tokens: int, usd: float) -> None:
        """`charge` for the event loop: the in-memory booking is synchronous (so the cap sees it at once);
        the SQLite commit runs in a worker thread."""
        due, over = self._record(backend, model, tag, input_tokens, output_tokens, usd)
        if due or over:
            await asyncio.to_thread(self.flush)
        if over:
            raise BudgetExceeded(over)

    def flush(self) -> None:
        """Commit pending rows. Flushes are serialized end to end, so the in-memory totals never go stale. If the
        commit fails (e.g. the database stays locked past the timeout) the rows go back to the pending list and
        the error is raised: a paid call is never dropped from the ledger."""
        with self._db_lock:
            with self._lock:
                rows, self._pending = self._pending, []
                self._last_flush = time.monotonic()
            if not rows:
                return
            try:
                with self._conn() as c:
                    c.executemany("INSERT INTO spend VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            except BaseException:
                with self._lock:
                    self._pending = rows + self._pending
                raise
            with self._lock:  # committed: move the rows from the pending to the committed totals
                usd = sum(r[-1] for r in rows)
                self._pending_total -= usd
                self._committed_total += usd
                for r in rows:
                    run = run_of(r[3])
                    self._pending_runs[run] -= r[-1]
                    self._committed_runs[run] = self._committed_runs.get(run, 0.0) + r[-1]
            try:
                committed, runs = self._read_totals()  # also picks up other processes' spend
            except sqlite3.Error as e:  # the rows are in; the refresh can wait for the next flush
                log.warning("spend ledger: could not refresh totals (%s)", e)
                return
            with self._lock:
                self._committed_total, self._committed_runs = committed, runs
