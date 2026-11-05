"""Layer-A generator (design_v1 §4.2, §12.3): runs judgment programs over decision weeks and writes the
judgment tables, the only interface between Layer A and Layer B ("capture once, replay many").

    generate(spec) -> results/judgments/<gen_id>/{answers,requests,evidence,features}.parquet + MANIFEST.json

Scheduling. The unit of work is (program, week). A unit waits for the same-week units of the programs it
depends on (their blackboard entries), for its own previous week if the program is week-ordered (N2/N3 read
M-epi), and for every earlier-week unit of the programs whose memory it reads (`memory_reads`). A unit sees
exactly the memory of the units it waited for, so what it sees never depends on scheduling. Everything else
runs concurrently: all M, P1, P2, T, F-A, F-B, N0 and N1 units of every week start at once, while week-ordered
units advance week by week with all stocks of a week in parallel. The backend's semaphore bounds in-flight
requests (max_concurrency 64), and identical in-flight requests share one call.

Resumability. A finished unit is a shard directory with a DONE marker, written after its memory writes. The
marker records the unit's fingerprint (program version, backend and its configuration hash, bank hash,
variant, data freeze, and the context: the hash of the context factory's `describe()`, e.g. the builder's subset,
symbols, blinder configuration and archive identity), the hash of the names the unit's context held (U_t and the
focus) and the units it depended on. On resume a unit is kept only if its fingerprint matches (otherwise the run is
refused: rows from another backend, bank, context or archive must not be mixed in) and its dependencies are the
planned ones and are all kept; any other unit is redone, together with everything downstream of it (an inserted
earlier week re-runs the week-ordered chain after it). A context factory without `describe()` (a synthetic world)
cannot be fingerprinted, so the runner rebuilds the week's context and refuses a kept unit whose names differ. Memory
writes are idempotent, so a crash between the memory write and the shard is harmless, and the response cache makes
redoing a unit free.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import importlib
import json
import logging
import shutil
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from jevtrader.memory.store import MemoryStore, PITError
from jevtrader.programs.bank_v1 import load_bank
from jevtrader.programs.base import JudgmentProgram, ProgramOutput, WeekContext
from jevtrader.s1.types import canonical_json

log = logging.getLogger("jevtrader.runner")
ROOT = Path(__file__).resolve().parents[3]
TABLES = ("answers", "requests", "evidence", "features")
COLUMNS = {
    "answers": ["t", "symbol", "program", "stage", "qid", "template", "kind", "probs_json", "request_key"],
    "requests": ["request_key", "t", "symbol", "program", "stage", "backend", "model_pinned", "response_model",
                 "prompt_version", "bank_hash", "data_freeze", "input_tokens", "cached_tokens", "output_tokens", "usd",
                 "latency_s", "cached", "degraded", "attempts", "usage_estimated", "state_sha256", "tag", "n_questions",
                 "lint_warnings", "backend_meta"],
    "evidence": ["request_key", "eid", "available_at_utc", "kept", "quarantined", "text_sha256", "d0_flags"],
    "features": ["t", "symbol", "feature", "value", "covered", "program", "taint"],
}
SORT = {
    "answers": ["t", "program", "symbol", "stage", "request_key", "qid"],
    "requests": ["t", "program", "symbol", "stage", "request_key"],
    "evidence": ["request_key", "eid"],
    "features": ["t", "program", "symbol", "feature"],
}
COLUMN_DEFAULTS = {"requests": {"backend_meta": ""}, "evidence": {"d0_flags": None}}  # columns added after Day 1 (older shards lack them)
INCLUSIVE_EVIDENCE = ("px:", "bb:")  # evidence available at the cutoff itself (closes at d_t, same-week M-work)
PROGRAM_REGISTRY: dict[str, str] = {
    "T": "jevtrader.programs.technical:TechnicalProgram",
    "M": "jevtrader.programs.macro:MacroProgram",
    "FA": "jevtrader.programs.fundamentals:FilingProfileProgram",
    "FB": "jevtrader.programs.fundamentals:FundamentalsProgram",
    "P1": "jevtrader.programs.policy:PolicyItemProgram",
    "P2": "jevtrader.programs.policy:PolicyWeekProgram",
    "PX": "jevtrader.programs.policy:PolicyExposureProgram",
    "D": "jevtrader.programs.probes:RecallProbeProgram",
    "D2": "jevtrader.programs.probes:ReidProbeProgram",
    "CF": "jevtrader.programs.content_free:ContentFreeProgram",
}
# the programs of one weekly Layer-A run (design_v1 §4.2), in the order the registry knows them; news, sentinel and
# reader programs of other modules join through `discover_programs`. Diagnostics (D, D2, CF) are opt-in.
DIAGNOSTIC_PROGRAMS = ("D", "D2", "CF")


@dataclass
class GeneratorSpec:
    gen_id: str
    weeks: list[pd.Timestamp]
    programs: list[str]  # registry names (or keys of `program_objects`)
    run_tag: str  # spend-ledger run id, e.g. "MAIN" (per-run sub-cap)
    variant: frozenset[str] = frozenset()
    data_freeze: str = ""
    backend: str | None = None  # overrides configs/backends.yaml s1.backend
    backend_config: str | None = None
    out_root: Path = ROOT / "results" / "judgments"
    max_parallel_units: int = 16
    notes: str = ""
    # run programs although some declared same-week dependencies are not in the run (e.g. T without M renders
    # the market regime as "not available"); off by default, so that omission is never silent
    allow_missing_deps: bool = False
    # runtime objects, not part of the manifest
    context_factory: Callable[[pd.Timestamp], WeekContext] | None = field(default=None, repr=False)
    program_objects: Mapping[str, Any] | None = field(default=None, repr=False)
    s1: Any = field(default=None, repr=False)

    def to_manifest(self) -> dict:
        return {
            "gen_id": self.gen_id,
            "weeks": [str(pd.Timestamp(w).date()) for w in self.weeks],
            "programs": list(self.programs),
            "run_tag": self.run_tag,
            "variant": sorted(self.variant),
            "data_freeze": self.data_freeze,
            "backend": self.backend,
            "backend_config": self.backend_config,
            "max_parallel_units": self.max_parallel_units,
            "notes": self.notes,
            "allow_missing_deps": self.allow_missing_deps,
        }


class ResumeMismatch(ValueError):
    """A finished unit on disk was produced by another backend, bank, program version, variant, data freeze or
    context (other names, blinder or archive)."""


class _Tagged:
    """Backend proxy that prefixes every ledger tag with the run id ("MAIN/T")."""

    def __init__(self, s1, run_tag: str):
        self._s1, self.run_tag = s1, run_tag

    def __getattr__(self, k):
        return getattr(self._s1, k)

    async def ask(self, state, questions, tag: str = "", **kw):
        return await self._s1.ask(state, questions, tag=f"{self.run_tag}/{tag}" if self.run_tag else tag, **kw)


def _is_program_class(obj: Any) -> bool:
    return (isinstance(obj, type) and isinstance(getattr(obj, "name", None), str) and bool(obj.name)
            and isinstance(getattr(obj, "version", None), str) and asyncio.iscoroutinefunction(getattr(obj, "run", None)))


def discover_programs(package: str = "jevtrader.programs") -> dict[str, str]:
    """name -> 'module:Class' for every judgment-program class defined in the modules of `package` (e.g. the news,
    sentinel and reader programs of other workstreams). A module that fails to import is skipped with a warning, so
    an in-progress file elsewhere never breaks a run that does not use it. Explicit PROGRAM_REGISTRY entries win."""
    import pkgutil

    pkg = importlib.import_module(package)
    found: dict[str, str] = {}
    for m in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda m: m.name):
        mod_name = f"{package}.{m.name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:  # noqa: BLE001 (a broken module of another workstream must not stop discovery)
            log.warning("program discovery: %s failed to import (%s: %s)", mod_name, type(e).__name__, e)
            continue
        for attr in sorted(vars(mod)):
            obj = getattr(mod, attr)
            if _is_program_class(obj) and obj.__module__ == mod_name:
                found.setdefault(obj.name, f"{mod_name}:{attr}")
    return {**found, **PROGRAM_REGISTRY}


def resolve_program(name: str, objects: Mapping[str, Any] | None = None) -> JudgmentProgram:
    if objects and name in objects:
        return objects[name]
    target = PROGRAM_REGISTRY.get(name)
    if target is None:
        target = discover_programs().get(name)
    if target is None:
        raise KeyError(f"unknown program {name!r}; registered: {sorted(PROGRAM_REGISTRY)}; discovered: "
                       f"{sorted(set(discover_programs()) - set(PROGRAM_REGISTRY))}")
    mod, cls = target.split(":")
    return getattr(importlib.import_module(mod), cls)()


def _clearance(p) -> str:
    """The highest taint a program may read (MEM-5): its own trust channel (T, S or W); diagnostic programs (taint D,
    which never write memory or features) read at W."""
    from jevtrader.memory.store import TAINT_RANK

    taint = getattr(p, "taint", "W")
    return taint if taint in TAINT_RANK else "W"


def topo_order(programs: list[JudgmentProgram]) -> list[JudgmentProgram]:
    """Programs in dependency order (dependencies outside the run are ignored); raises on a cycle."""
    by = {p.name: p for p in programs}
    out, state = [], {}

    def visit(n: str, path: tuple[str, ...]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise ValueError(f"dependency cycle: {' -> '.join(path + (n,))}")
        state[n] = 1
        for d in getattr(by[n], "depends_on", ()):
            if d in by:
                visit(d, path + (n,))
        state[n] = 2
        out.append(by[n])

    for p in programs:
        visit(p.name, ())
    return out


def git_sha(root: Path = ROOT) -> str | None:
    """HEAD commit read from .git files (no git command is run)."""
    g = Path(root) / ".git"
    try:
        head = (g / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        f = g / ref
        if f.exists():
            return f.read_text().strip()
        packed = g / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    except OSError:
        pass
    return None


# ------------------------------------------------------------------------------------ units
Unit = tuple[str, pd.Timestamp]


def _uid(u: Unit) -> str:
    return f"{u[0]}/{pd.Timestamp(u[1]).date()}"


def unit_deps(p: JudgmentProgram, i: int, weeks: list[pd.Timestamp], names: set[str]) -> list[Unit]:
    """The units (program, week) that unit (p, weeks[i]) waits for: same-week dependencies, its previous week
    if week-ordered, and every earlier week of the programs whose memory it reads."""
    t = weeks[i]
    out = {(d, t) for d in getattr(p, "depends_on", ()) if d in names}
    if getattr(p, "week_ordered", False) and i > 0:
        out.add((p.name, weeks[i - 1]))
    for w in getattr(p, "memory_reads", ()):
        if w in names:
            out |= {(w, weeks[k]) for k in range(i)}
    return sorted(out, key=lambda u: (u[1], u[0]))


def memory_history(p: JudgmentProgram, i: int, weeks: list[pd.Timestamp], names: set[str]) -> set[Unit]:
    """Earlier-week units whose M-epi/M-sem records unit (p, weeks[i]) may read: its declared `memory_reads`
    and, if week-ordered, its own chain. The unit waits for all of them (`unit_deps`)."""
    writers = {w for w in getattr(p, "memory_reads", ()) if w in names}
    if getattr(p, "week_ordered", False):
        writers.add(p.name)
    return {(w, weeks[k]) for w in writers for k in range(i)}


def context_description(factory) -> dict | None:
    """The context factory's `describe()` (programs.context.ContextBuilder: subset, symbols, blinder configuration,
    archive identity), or None for a factory that cannot describe itself."""
    d = getattr(factory, "describe", None)
    return d() if callable(d) else None


def context_names_sha256(ctx: WeekContext) -> str:
    """Hash of the names a week's context holds: U_t and the names stock-level programs ask about."""
    return hashlib.sha256(canonical_json({"universe": list(ctx.universe), "focus": list(ctx.stock_names())}).encode()
                          ).hexdigest()


def unit_fingerprint(p: JudgmentProgram, s1, bank_sha256: str, spec: GeneratorSpec,
                     context: dict | None = None) -> tuple[str, dict]:
    """What a unit's rows depend on besides its week: program version, backend configuration, bank, variant, data
    freeze and the context description (`context_description`; None when the factory has none)."""
    d = s1.describe() if hasattr(s1, "describe") else {"backend": getattr(s1, "backend_id", "?")}
    payload = {
        "program": p.name, "version": getattr(p, "version", ""), "taint": getattr(p, "taint", "W"),
        "backend": {k: d.get(k) for k in ("backend", "model_pinned", "prompt_version", "cfg_hash", "bank_hash")},
        "bank_sha256": bank_sha256, "variant": sorted(spec.variant), "data_freeze": spec.data_freeze,
        "context_sha256": None if context is None else hashlib.sha256(canonical_json(context).encode()).hexdigest(),
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest(), payload


def _read_marker(unit: Path) -> dict | None:
    f = unit / "DONE"
    if not f.exists():
        return None
    try:
        m = json.loads(f.read_text())
    except json.JSONDecodeError:
        return None  # a pre-fingerprint marker: cannot be verified, so the unit is redone
    return m if isinstance(m, dict) else None


# ------------------------------------------------------------------------------------ shards
def _unit_dir(out: Path, program: str, t: pd.Timestamp) -> Path:
    return out / "shards" / program / str(pd.Timestamp(t).date())


def _frames(outs: list[ProgramOutput], taint: str, run_tag: str, data_freeze: str) -> dict[str, pd.DataFrame]:
    rows: dict[str, list[dict]] = {k: [] for k in TABLES}
    for o in outs:
        for r in o.trace:
            d = dataclasses.asdict(r)
            d["probs_json"] = None if d.pop("probs") is None else json.dumps(r.probs, sort_keys=True)
            rows["answers"].append(d)
        for r in o.requests:
            d = dataclasses.asdict(r)
            d["tag"] = f"{run_tag}/{r.tag}" if run_tag else r.tag
            d["data_freeze"] = data_freeze
            rows["requests"].append(d)
        # d0_flags may also be set on a row after it was built (news programs: the item's final status and flags)
        rows["evidence"] += [{**dataclasses.asdict(e), "d0_flags": getattr(e, "d0_flags", None)} for e in o.evidence]
        rows["features"] += [{"t": o.t, "symbol": o.entity, "feature": k, "value": None if v is None else float(v),
                              "covered": v is not None, "program": o.program, "taint": taint}
                             for k, v in o.features.items()]
    frames = {}
    for k in TABLES:
        df = pd.DataFrame(rows[k], columns=COLUMNS[k])
        if k == "features":
            df["value"] = df["value"].astype("float64")
        if k == "evidence":
            df["d0_flags"] = df["d0_flags"].astype("object")
        frames[k] = df
    return frames


def _write_shard(unit: Path, frames: dict[str, pd.DataFrame], marker: dict) -> None:
    if unit.exists():
        shutil.rmtree(unit)
    tmp = unit.with_name(unit.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for k, df in frames.items():
        df.to_parquet(tmp / f"{k}.parquet", index=False)
    (tmp / "DONE").write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n")
    tmp.rename(unit)


def _validate(outs: list[ProgramOutput], p: JudgmentProgram, ctx: WeekContext) -> None:
    allowed = set(ctx.universe) | {"MARKET"}
    for o in outs:
        if o.program != p.name or pd.Timestamp(o.t) != pd.Timestamp(ctx.t) or o.entity not in allowed:
            raise ValueError(f"{p.name}: output for {o.program}/{o.entity}/{o.t} outside its unit")
        keys = {r.request_key for r in o.requests}
        if any(r.request_key not in keys for r in o.trace):
            raise ValueError(f"{p.name}: trace row without its request row ({o.entity})")
        for e in o.evidence:
            at = pd.Timestamp(e.available_at_utc)
            # prices (a close at d_t) and the same-week blackboard are available at the cutoff itself; news, memory
            # records and filings must be strictly earlier (created_at < c_t, §2.4)
            late = at > ctx.cutoff_utc if e.eid.startswith(INCLUSIVE_EVIDENCE) else at >= ctx.cutoff_utc
            if late:
                raise PITError(f"{p.name}/{o.entity}: evidence {e.eid} available {e.available_at_utc}, not before "
                               f"the cutoff {ctx.cutoff_utc}")


def _apply_memory(store: MemoryStore, outs: list[ProgramOutput], p: JudgmentProgram, ctx: WeekContext) -> None:
    bb: dict[str, Any] = {}
    for o in outs:
        for k, v in o.blackboard.items():
            if k in bb and bb[k] != v:
                raise ValueError(f"{p.name}: conflicting blackboard values for {k!r}")
            bb[k] = v
    if bb:
        store.write_work(ctx.t, ctx.cutoff_utc, p.name, getattr(p, "taint", "W"), bb)
    for o in outs:
        for rec in o.memory_writes:
            store.write(rec, p.name, ctx.t, ctx.cutoff_utc)


# ------------------------------------------------------------------------------------ generate
async def agenerate(spec: GeneratorSpec, resume: bool = True) -> Path:
    t_start = time.time()
    if spec.context_factory is None:
        raise ValueError("GeneratorSpec.context_factory is required (it builds the point-in-time WeekContext)")
    # the freeze (bank_v1.FROZEN_RUN_TAGS): a MAIN or PRE run refuses any bank but the frozen one, before it touches
    # its output directory or calls a model
    bank_sha = load_bank(run_tag=spec.run_tag).sha256
    out = Path(spec.out_root) / spec.gen_id
    if not resume and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    programs = topo_order([resolve_program(n, spec.program_objects) for n in spec.programs])
    names = {p.name for p in programs}
    missing = {p.name: sorted(set(getattr(p, "depends_on", ())) - names) for p in programs}
    missing = {k: v for k, v in missing.items() if v}
    if missing and not spec.allow_missing_deps:
        raise ValueError(f"declared dependencies missing from the run: {missing}; add them, or set "
                         f"GeneratorSpec.allow_missing_deps=True (recorded in the MANIFEST)")
    for k, v in missing.items():
        log.warning("%s runs without its dependencies %s (allow_missing_deps)", k, v)
    weeks = sorted({pd.Timestamp(w) for w in spec.weeks})
    if spec.s1 is not None:
        s1 = spec.s1
    else:
        from jevtrader.s1.backends import make_backend

        s1 = make_backend(spec.backend_config, backend=spec.backend)
    if getattr(s1, "bank_hash", None) != bank_sha:
        raise ValueError("the backend's bank_hash must be the loaded bank's SHA-256 (it is part of every request key)")
    tagged = _Tagged(s1, spec.run_tag)
    ctx_desc = context_description(spec.context_factory)
    fps = {p.name: unit_fingerprint(p, s1, bank_sha, spec, ctx_desc) for p in programs}
    deps = {(p.name, t): unit_deps(p, i, weeks, names) for p in programs for i, t in enumerate(weeks)}
    contexts: dict[pd.Timestamp, WeekContext] = {}

    def context(t: pd.Timestamp) -> WeekContext:
        if t not in contexts:
            contexts[t] = dataclasses.replace(spec.context_factory(t), variant=frozenset(spec.variant))
        return contexts[t]

    # which finished units can be kept: same fingerprint, same planned dependencies, all dependencies kept (and, for
    # a factory that cannot describe itself, the same names in the week's context)
    kept: dict[Unit, bool] = {}
    n_invalidated = 0
    for t in weeks:
        for p in programs:
            unit, udir = (p.name, t), _unit_dir(out, p.name, t)
            marker = _read_marker(udir) if resume else None
            if marker is not None and marker.get("fingerprint") != fps[p.name][0]:
                raise ResumeMismatch(
                    f"{_uid(unit)} was produced under {marker.get('fingerprint_payload')}, not {fps[p.name][1]}; "
                    f"resume only with the same backend, bank, program version, variant, data freeze and context "
                    f"(or use resume=False / a new gen_id)")
            if marker is not None and ctx_desc is None and marker.get("names_sha256") != \
                    context_names_sha256(context(t)):
                raise ResumeMismatch(
                    f"{_uid(unit)} was produced for other names (U_t or focus) than this run's context of "
                    f"{pd.Timestamp(t).date()}; use resume=False or a new gen_id")
            ok = marker is not None and marker.get("deps") == [_uid(u) for u in deps[unit]] \
                and all(kept[u] for u in deps[unit])
            kept[unit] = ok
            if not ok and udir.exists():
                n_invalidated += marker is not None
                shutil.rmtree(udir)  # its memory records are dropped when the store loads below
    # the write gate is off only for the registered A7 "gate_off" arm (design_v2_delta RB-1; memory.store)
    store = MemoryStore(out / "memory", keep=lambda writer, t: (_unit_dir(out, writer, t) / "DONE").exists(),
                        enforce_gate="gate_off" not in set(spec.variant))
    events = {(p.name, t): asyncio.Event() for p in programs for t in weeks}
    for unit, ev in events.items():
        if kept[unit]:
            ev.set()
    n_skipped = sum(ev.is_set() for ev in events.values())
    unit_sem = asyncio.Semaphore(spec.max_parallel_units)

    async def run_unit(p: JudgmentProgram, i: int, t: pd.Timestamp) -> None:
        unit = (p.name, t)
        ev = events[unit]
        if ev.is_set():
            return
        for u in deps[unit]:
            await events[u].wait()
        # a week-ordered chain runs one unit at a time anyway and is the critical path: no slot needed
        async with contextlib.nullcontext() if getattr(p, "week_ordered", False) else unit_sem:
            ctx0 = context(t)
            view = store.view(t, ctx0.cutoff_utc, deps=[d for d in getattr(p, "depends_on", ()) if d in names],
                              history=memory_history(p, i, weeks, names),
                              readable={*getattr(p, "depends_on", ()), *getattr(p, "memory_reads", ()),
                                        *((p.name,) if getattr(p, "week_ordered", False) else ())},
                              clearance=_clearance(p))
            ctx = dataclasses.replace(ctx0, memory=view)
            outs = await p.run(ctx, tagged)
            _validate(outs, p, ctx)
            _apply_memory(store, outs, p, ctx)
            frames = _frames(outs, getattr(p, "taint", "W"), spec.run_tag, spec.data_freeze)
            marker = {"fingerprint": fps[p.name][0], "fingerprint_payload": fps[p.name][1],
                      "deps": [_uid(u) for u in deps[unit]], "names_sha256": context_names_sha256(ctx0),
                      "n_focus": len(ctx0.stock_names())}
            await asyncio.to_thread(_write_shard, _unit_dir(out, p.name, t), frames, marker)
        ev.set()

    try:
        async with asyncio.TaskGroup() as tg:
            for p in programs:
                for i, t in enumerate(weeks):
                    tg.create_task(run_unit(p, i, t))
    except BaseExceptionGroup as eg:
        raise eg.exceptions[0] from eg
    finally:
        if getattr(s1, "ledger", None) is not None:
            s1.ledger.flush()
    tables = consolidate(out, [p.name for p in programs], weeks)
    write_manifest(out, spec, s1, programs, tables, wall_s=time.time() - t_start, n_skipped=n_skipped,
                   extra={"missing_dependencies": missing, "units_invalidated": n_invalidated,
                          "requests_coalesced": getattr(s1, "coalesced", 0),
                          "unit_fingerprints": {k: v[0] for k, v in fps.items()}, "context": ctx_desc})
    return out


def generate(spec: GeneratorSpec, resume: bool = True) -> Path:
    """Run Layer A for `spec` and return results/judgments/<gen_id>. The only code path that calls models."""
    return asyncio.run(agenerate(spec, resume))


def consolidate(out: Path, programs: list[str], weeks: list[pd.Timestamp]) -> dict[str, pd.DataFrame]:
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in TABLES}
    for name in programs:
        for t in weeks:
            unit = _unit_dir(out, name, t)
            if not (unit / "DONE").exists():
                raise RuntimeError(f"unit {name}/{t.date()} did not finish")
            for k in TABLES:
                df = pd.read_parquet(unit / f"{k}.parquet")
                if len(df):
                    parts[k].append(df)
    tables = {}
    for k in TABLES:
        df = pd.concat(parts[k], ignore_index=True) if parts[k] else pd.DataFrame(columns=COLUMNS[k])
        for c, default in COLUMN_DEFAULTS.get(k, {}).items():  # shards written before a column was added
            if c not in df:
                df[c] = default
        df = df[COLUMNS[k]].sort_values(SORT[k], kind="stable").reset_index(drop=True)
        tmp = out / f"{k}.tmp.parquet"
        df.to_parquet(tmp, index=False)
        tmp.replace(out / f"{k}.parquet")
        tables[k] = df
    return tables


def write_manifest(out: Path, spec: GeneratorSpec, s1, programs, tables, wall_s: float, n_skipped: int,
                   extra: dict | None = None) -> dict:
    ans, req = tables["answers"], tables["requests"]
    n_deg = int(ans["probs_json"].isna().sum()) if len(ans) else 0
    bank = load_bank()
    m = {
        "gen_id": spec.gen_id,
        "spec": spec.to_manifest(),
        "git_sha": git_sha(),
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "bank": {"version": bank.version, "sha256": bank.sha256},
        "backend": s1.describe() if hasattr(s1, "describe") else {"backend": getattr(s1, "backend_id", "?")},
        "programs": {p.name: getattr(p, "version", "") for p in programs},
        "counts": {
            "units": len(programs) * len({pd.Timestamp(w) for w in spec.weeks}),
            "units_resumed": n_skipped,
            "requests": len(req),
            "requests_cached": int(req["cached"].sum()) if len(req) else 0,
            "answers": len(ans),
            "answers_degraded": n_deg,
            "features": len(tables["features"]),
            "evidence": len(tables["evidence"]),
        },
        "validity_rate": (1.0 - n_deg / len(ans)) if len(ans) else None,
        "spend_usd": float(req.loc[~req["cached"].astype(bool), "usd"].sum()) if len(req) else 0.0,
        "tokens": {k: int(req[k].sum()) if len(req) else 0 for k in ("input_tokens", "cached_tokens", "output_tokens")},
        "l9_warnings": {str(k): int(v) for k, v in req.groupby("program")["lint_warnings"].sum().items() if v}
        if len(req) else {},
        "wall_time_s": round(wall_s, 3),
        **(extra or {}),
    }
    (out / "MANIFEST.json").write_text(json.dumps(m, indent=1, default=str))
    return m
