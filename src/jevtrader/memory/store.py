"""Multi-layer shared memory (design_v1 §7): M-work (blackboard), M-epi (episodic events), M-sem (10-K profiles).

Point-in-time and replay rules, enforced here rather than trusted to each program:
- Every record carries `available_at` (when its information became public); a view for decision t shows a
  record only if `available_at < c_t` (M-work: the same-week blackboard, `available_at <= c_t`).
- A record also carries the week `written_t` and the program (`writer`) whose run wrote it. A view for
  (t, reader) shows same-week records only from programs the reader declares as dependencies, and earlier-week
  records only from the (writer, week) units the runner names in `history`: those the reader declares
  (`memory_reads`, or its own chain if week-ordered) and waits for. So the visible memory never depends on
  scheduling: a fresh run and a resumed run see identical memory.
- Records are immutable. Rewriting an id with identical content is a no-op (resume); different content
  raises. A later or contradicting development is a new record that `supersedes` the old one (no
  text-similarity dedup; margalit2026governed p.11).
- Each layer has one writer role: M-epi the news analyst (N programs), M-sem the fundamentals analyst (FA), and
  each M-work key "<program>.<name>" its program.
- Rule R5, the write gate D6: an M-epi record is accepted only with provenance (request keys) and a gate record
  showing its cluster was kept, not quarantined and at full weight (rho = 1).
- M-work entries carry a taint (T < S < W); `MemoryView.work(key, max_taint="T")` raises for a W-tainted
  entry, which is how rule R1 (no W/S value in a T-program state) is enforced (§3.2).
- Origin-bound taint (design_v2_delta MEM-5, MEM-7): every stored record carries the taint of its origin, bound at
  write time and never lowered: an M-work entry its writer's declared taint, an M-epi record W (news text), an M-sem
  record S (10-K text). A view is built for one reader with its clearance, and the reader sees exactly
      Readable(a, t) = {m : available_at(m) < c_t  and  writer(m) in Allowed(a)  and  taint(m) <= Clear(a)}
  (M-work: available_at <= c_t, same week). Asking a view for a layer above the reader's clearance raises
  `TaintError` instead of returning nothing, so a mis-declared program fails loudly.
- Nothing learned from outcomes is written here (R4): the store only accepts typed-judgment records.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import threading
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

TAINT_RANK = {"T": 0, "S": 1, "W": 2}
EPI_WRITER = re.compile(r"^(N\d*|news)$")  # the news analyst's programs
SEM_WRITERS = frozenset({"FA"})
LAYER_TAINT = {"epi": "W", "sem": "S"}  # origin-bound taint of the M-epi and M-sem layers (M-work: per entry)


class MemoryError_(RuntimeError):
    """Base class (named to avoid shadowing the builtin MemoryError)."""


class ImmutableRecordError(MemoryError_):
    pass


class WriterError(MemoryError_):
    pass


class TaintError(MemoryError_):
    pass


class PITError(MemoryError_):
    pass


class UndeclaredMemoryRead(MemoryError_):
    """A program read M-epi/M-sem without declaring whose memory it reads (`memory_reads`/`week_ordered`)."""


class GateError(MemoryError_):
    """An M-epi write that did not pass the write gate D6 (rule R5)."""


def write_gate(kept: bool, quarantined: bool, rho: float) -> dict[str, Any]:
    """The gate record a news program attaches as `provenance["gate"]` (design_v1 §3.3 D6): M-epi takes only
    kept, non-quarantined clusters with full weight (rho = 1, i.e. no promotional member)."""
    return {"kept": bool(kept), "quarantined": bool(quarantined), "rho": float(rho)}


def passes_write_gate(provenance: Mapping[str, Any]) -> bool:
    g = provenance.get("gate") or {}
    return (g.get("kept") is True and g.get("quarantined") is False and g.get("rho") == 1.0
            and bool(provenance.get("request_keys")))


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _iso(x: Any) -> Any:
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, dict):
        return {k: _iso(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_iso(v) for v in x]
    return x


# ------------------------------------------------------------------------------------------ records
@dataclass(frozen=True)
class WorkEntry:
    """M-work: one blackboard value of week t (M answers, policy aggregates, market bucket, ...)."""

    t: pd.Timestamp
    key: str  # "<program>.<name>", e.g. "M.regime"
    value: Any  # JSON
    taint: str  # "T" | "S" | "W"
    available_at: pd.Timestamp  # <= c_t of week t

    @property
    def record_id(self) -> str:
        return f"{self.t.date()}:{self.key}"


@dataclass(frozen=True)
class EventRecord:
    """M-epi: one kept news cluster (design_v1 §7 table)."""

    event_id: str
    symbol: str
    first_seen: pd.Timestamp  # created_at of the cluster's first item
    available_at: pd.Timestamp  # = first_seen (§7); visibility also needs an earlier writing week
    item_ids: tuple[str, ...]
    width: int
    type_probs: dict[str, float] | None
    effect_probs: dict[str, float] | None
    p_important: float | None
    p_unconfirmed: float | None
    p_scheduled: float | None
    surprise: float | None
    novelty: float | None
    headline_blinded: str  # screened kept text, entities blinded and levels masked; times stay absolute and are
    # blinded by the reader with its own decision year (news.reblind_time), so a record read in a later year is right
    supersedes: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)  # bank_hash, backend, model, prompt_version, request_keys
    vs_probs: dict[str, float] | None = None  # N1.vs_expectations distribution (N3 `earlier_events` rows)

    @property
    def record_id(self) -> str:
        return self.event_id

    @property
    def top_type(self) -> str | None:
        return None if not self.type_probs else max(self.type_probs, key=self.type_probs.get)


@dataclass(frozen=True)
class ProfileRecord:
    """M-sem: the F-A profile of one 10-K; replaced at each 10-K."""

    symbol: str
    accession: str
    filed: pd.Timestamp
    usable_from: pd.Timestamp  # the session after `filed`
    flags: dict[str, float | None]
    excerpt_ids: tuple[str, ...]
    bank_hash: str
    supersedes: str | None = None  # record_id of the previous 10-K's profile

    @property
    def record_id(self) -> str:
        return f"{self.symbol}:{self.accession}"

    @property
    def available_at(self) -> pd.Timestamp:
        return self.usable_from


_KIND = {"work": WorkEntry, "epi": EventRecord, "sem": ProfileRecord}
_TS_FIELDS = {"t", "available_at", "first_seen", "filed", "usable_from"}


def _normalize(rec):
    """Timestamps to UTC and sequences to tuples, so equality of reloaded records is exact."""
    upd = {}
    for f in dataclasses.fields(rec):
        v = getattr(rec, f.name)
        if f.name in _TS_FIELDS and v is not None:
            upd[f.name] = _ts(v)
        elif f.name in ("item_ids", "excerpt_ids"):
            upd[f.name] = tuple(v)
    return dataclasses.replace(rec, **upd)


def _to_json(rec) -> dict:
    return {f.name: _iso(getattr(rec, f.name)) for f in dataclasses.fields(rec)}


def _from_json(kind: str, d: dict):
    cls = _KIND[kind]
    return _normalize(cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}}))


@dataclass(frozen=True)
class _Stored:
    kind: str
    record: Any
    writer: str
    written_t: pd.Timestamp

    @property
    def taint(self) -> str:
        """Origin-bound: M-work its entry's taint, M-epi W, M-sem S (MEM-5)."""
        return self.record.taint if self.kind == "work" else LAYER_TAINT[self.kind]


# ------------------------------------------------------------------------------------------ store
class MemoryStore:
    """The single writer path for all three layers, with optional JSONL persistence (for resumable runs)."""

    def __init__(self, root: Path | None = None, keep: Callable[[str, pd.Timestamp], bool] | None = None,
                 enforce_gate: bool = True):
        """`keep(writer, written_t)`: on reload, records of units that did not finish are dropped (the runner
        passes "has a DONE shard"), so a resumed unit rewrites its records from scratch. `enforce_gate=False` is the
        registered "write gate off" arm of the A7 persistence ablation only (design_v2_delta RB-1): M-epi then takes
        promotional clusters too, still only with provenance and a gate record (which states the real outcome)."""
        self.enforce_gate = enforce_gate
        self.root = Path(root) if root is not None else None
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str], _Stored] = {}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            for kind in _KIND:
                f = self.root / f"{kind}.jsonl"
                if not f.exists():
                    continue
                lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
                kept = []
                for line in lines:
                    d = json.loads(line)
                    written_t = _ts(d["written_t"])
                    if keep is not None and not keep(d["writer"], written_t):
                        continue
                    rec = _from_json(kind, d["record"])
                    self._rows[(kind, rec.record_id)] = _Stored(kind, rec, d["writer"], written_t)
                    kept.append(line)
                if len(kept) != len(lines):
                    tmp = f.with_suffix(".jsonl.tmp")
                    tmp.write_text("".join(ln + "\n" for ln in kept))
                    tmp.replace(f)

    # -------------------------------------------------------------- writes
    def _put(self, kind: str, rec, writer: str, written_t: pd.Timestamp) -> None:
        rec = _normalize(rec)
        key = (kind, rec.record_id)
        written_t = _ts(written_t)
        with self._lock:
            old = self._rows.get(key)
            if old is not None:
                if old.record == rec and old.writer == writer and old.written_t == written_t:
                    return  # idempotent re-write (resumed run)
                raise ImmutableRecordError(f"{kind} record {rec.record_id!r} already exists with other content")
            self._rows[key] = _Stored(kind, rec, writer, written_t)
            if self.root is not None:
                line = json.dumps({"record": _to_json(rec), "writer": writer, "written_t": written_t.isoformat()},
                                  ensure_ascii=False)
                with open(self.root / f"{kind}.jsonl", "a") as f:
                    f.write(line + "\n")

    def write_work(self, t, cutoff_utc, writer: str, taint: str, entries: Mapping[str, Any]) -> None:
        """Blackboard values of week t from `writer`; keys are namespaced "<writer>.<name>"."""
        if taint not in TAINT_RANK:
            raise ValueError(f"taint must be one of {sorted(TAINT_RANK)}")
        json.dumps(dict(entries))  # values must be JSON
        for key, value in entries.items():
            if key.split(".", 1)[0] != writer:
                raise WriterError(f"M-work key {key!r} is not owned by {writer!r}")
            self._put("work", WorkEntry(_ts(t), key, value, taint, _ts(cutoff_utc)), writer, t)

    def write_event(self, rec: EventRecord, writer: str, t, cutoff_utc) -> None:
        if not EPI_WRITER.match(writer):
            raise WriterError(f"M-epi is written only by the news analyst, not {writer!r}")
        ok = passes_write_gate(rec.provenance) if self.enforce_gate else (
            bool(rec.provenance.get("request_keys")) and isinstance(rec.provenance.get("gate"), dict))
        if not ok:
            raise GateError(f"event {rec.event_id}: M-epi takes only kept, non-quarantined clusters with rho = 1 and "
                            f"provenance (R5/D6); got {rec.provenance.get('gate')!r}")
        rec = _normalize(rec)
        if not rec.first_seen < _ts(cutoff_utc):
            raise PITError(f"event {rec.event_id}: first_seen {rec.first_seen} not before the writing cutoff")
        if rec.available_at < rec.first_seen:
            raise PITError(f"event {rec.event_id}: available before first seen")
        if rec.supersedes is not None:
            old = self._rows.get(("epi", rec.supersedes))
            if old is None or old.record.symbol != rec.symbol or old.written_t >= _ts(t):
                raise MemoryError_(f"event {rec.event_id} supersedes unknown/later/foreign record {rec.supersedes!r}")
        self._put("epi", rec, writer, t)

    def write_profile(self, rec: ProfileRecord, writer: str, t, cutoff_utc) -> None:
        if writer not in SEM_WRITERS:
            raise WriterError(f"M-sem is written only by FA, not {writer!r}")
        rec = _normalize(rec)
        if not rec.filed < _ts(cutoff_utc):
            raise PITError(f"profile {rec.record_id}: filed {rec.filed} not before the writing cutoff")
        if rec.usable_from <= rec.filed:
            raise PITError(f"profile {rec.record_id}: usable before the session after filing")
        if rec.supersedes is not None:
            old = self._rows.get(("sem", rec.supersedes))
            if old is None or old.record.symbol != rec.symbol or old.record.filed >= rec.filed:
                raise MemoryError_(f"profile {rec.record_id} supersedes unknown/later/foreign {rec.supersedes!r}")
        self._put("sem", rec, writer, t)

    def write(self, rec, writer: str, t, cutoff_utc) -> None:
        if isinstance(rec, EventRecord):
            self.write_event(rec, writer, t, cutoff_utc)
        elif isinstance(rec, ProfileRecord):
            self.write_profile(rec, writer, t, cutoff_utc)
        else:
            raise TypeError(f"not a memory record: {type(rec).__name__}")

    # -------------------------------------------------------------- reads
    def view(self, t, cutoff_utc, deps: Iterable[str] = (), history: Collection[tuple[str, Any]] | None = None,
             readable: Iterable[str] | None = None, clearance: str = "W") -> MemoryView:
        """Memory for decision t: same-week records of `deps`, and earlier-week M-epi/M-sem records of the
        (writer, written week) units in `history` (None: every earlier week, for direct use outside the runner).
        The runner always passes `history`, and waits for exactly those units before the reader runs.
        `readable`: the writers the reader declared; a reader that declared none gets an error, not an empty
        memory, when it reads M-epi or M-sem (None: no check). `clearance`: the highest taint the reader may read
        (its own trust channel; MEM-5): records above it are not in the view, and asking for them raises."""
        if clearance not in TAINT_RANK:
            raise ValueError(f"clearance must be one of {sorted(TAINT_RANK)}")
        t, cutoff = _ts(t), _ts(cutoff_utc)
        deps = frozenset(deps)
        hist = None if history is None else {(w, _ts(x)) for w, x in history}
        with self._lock:
            rows = list(self._rows.values())
        vis, above = [], []
        for r in rows:
            same_week_ok = r.written_t == t and r.writer in deps
            earlier_ok = r.written_t < t and (hist is None or (r.writer, r.written_t) in hist)
            if r.kind == "work":
                ok = r.record.t == t and same_week_ok and r.record.available_at <= cutoff
            else:
                ok = (earlier_ok or same_week_ok) and r.record.available_at < cutoff
            if ok:
                (vis if TAINT_RANK[r.taint] <= TAINT_RANK[clearance] else above).append(r)
        return MemoryView(t, cutoff, vis, None if readable is None else frozenset(readable), clearance,
                          {(r.kind, r.record.key if r.kind == "work" else None): r.taint for r in above})

    def __len__(self) -> int:
        return len(self._rows)


class MemoryView:
    """Read-only, point-in-time view of memory for one decision (and one reader)."""

    def __init__(self, t: pd.Timestamp, cutoff_utc: pd.Timestamp, rows: list[_Stored],
                 readable: frozenset[str] | None = None, clearance: str = "W",
                 withheld: Mapping[tuple[str, str | None], str] | None = None):
        self.t, self.cutoff_utc = t, cutoff_utc
        self._readable = readable
        self.clearance = clearance
        self._withheld = dict(withheld or {})  # (layer, M-work key or None) -> taint, for the TaintError message
        self._rows = list(rows)
        self._work = {r.record.key: r.record for r in rows if r.kind == "work"}
        self._epi = [r.record for r in rows if r.kind == "epi"]
        self._sem = [r.record for r in rows if r.kind == "sem"]

    @classmethod
    def empty(cls, t, cutoff_utc) -> MemoryView:
        return cls(_ts(t), _ts(cutoff_utc), [])

    # -------------------------------------------------------------- M-work
    def work(self, key: str, max_taint: str = "W") -> WorkEntry | None:
        withheld = self._withheld.get(("work", key))
        if withheld is not None:
            raise TaintError(f"M-work {key!r} is {withheld}-tainted; this reader's clearance is {self.clearance} "
                             f"(R1, MEM-5)")
        e = self._work.get(key)
        if e is not None and TAINT_RANK[e.taint] > TAINT_RANK[max_taint]:
            raise TaintError(f"M-work {key!r} is {e.taint}-tainted; this reader accepts at most {max_taint} (R1)")
        return e

    def work_keys(self) -> list[str]:
        return sorted(self._work)

    # -------------------------------------------------------------- M-epi
    def events(
        self,
        symbol: str,
        window_weeks: float = 8,
        prefer_type: str | None = None,
        limit: int | None = None,
        current_only: bool = False,
    ) -> list[EventRecord]:
        """Records of `symbol` first seen within `window_weeks` before the cutoff; same top type first when
        `prefer_type` is given, then most recent first (N2 retrieval, §5.1). `current_only` leaves out records a later
        record supersedes (N2 compares a new cluster with the latest version of each story); N3's `earlier_events`
        reads every record, so a supersession never hides an earlier development."""
        self._check_declared("events")
        self._check_clearance("epi")
        lo = self.cutoff_utc - pd.Timedelta(weeks=window_weeks)
        recs = [r for r in self._epi if r.symbol == symbol and r.first_seen >= lo]
        if current_only:
            superseded = {r.supersedes for r in self._epi if r.supersedes}
            recs = [r for r in recs if r.event_id not in superseded]
        recs.sort(key=lambda r: (prefer_type is not None and r.top_type != prefer_type, -r.first_seen.value, r.event_id))
        return recs if limit is None else recs[:limit]

    # -------------------------------------------------------------- M-sem
    def profile(self, symbol: str) -> ProfileRecord | None:
        """The profile of the latest 10-K usable before the cutoff (a newer one supersedes it only once usable)."""
        self._check_declared("profile")
        self._check_clearance("sem")
        recs = [r for r in self._sem if r.symbol == symbol]
        return max(recs, key=lambda r: (r.usable_from, r.filed, r.accession)) if recs else None

    def _check_clearance(self, layer: str) -> None:
        if TAINT_RANK[LAYER_TAINT[layer]] > TAINT_RANK[self.clearance]:
            raise TaintError(f"M-{layer} records are {LAYER_TAINT[layer]}-tainted by origin; this reader's "
                             f"clearance is {self.clearance} (MEM-5)")

    def records(self) -> list[tuple[str, Any, str, str]]:
        """Audit: every readable record as (layer, record, writer, taint), i.e. the set Readable(a, t) (MEM-5)."""
        return [(r.kind, r.record, r.writer, r.taint) for r in self._rows]

    def _check_declared(self, what: str) -> None:
        if self._readable is not None and not self._readable:
            raise UndeclaredMemoryRead(f"{what}(): this program declares no memory it reads; set `memory_reads` (or "
                                       f"`week_ordered` for its own chain) so the runner waits for the writers")

    # -------------------------------------------------------------- audit
    def fingerprint(self) -> str:
        """SHA-256 over everything visible; equal fingerprints mean identical memory inputs."""
        payload = json.dumps(
            {
                "work": [_to_json(self._work[k]) for k in sorted(self._work)],
                "epi": sorted((_to_json(r) for r in self._epi), key=lambda d: d["event_id"]),
                "sem": sorted((_to_json(r) for r in self._sem), key=lambda d: (d["symbol"], d["accession"])),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()
