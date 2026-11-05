"""RT-A-NG driver (decision P-4; `jevtrader.redteam.adaptive_ng`): the adaptive attacker on the open typed decision
model with a fixed candidate pool. $0 API: the backend is `typed-local` (no ledger, no key, no network); the budget is
CPU time of the 6-thread worker (IF-1: 36 CPU-hours for all NG work).

    .venv/bin/python experiments/redteam/rta_ng.py freeze            # spec + hash -> results/redteam/rta_ng/spec.json
    .venv/bin/python experiments/redteam/rta_ng.py dry [--n 2]       # 2 W_pre items on the real model (not a result)
    .venv/bin/python experiments/redteam/rta_ng.py project           # CPU-hours of the registered run vs IF-1
    .venv/bin/python experiments/redteam/rta_ng.py run --units <RT-S units.parquet>   # AFTER the unlock: 30 items

`dry` and `run` refuse to start a model worker when less than `--min-free-gb` of memory is available (the worker
peaks near 11.6 GB; another NG job may be running) and check the W_clean lock on every item's week before any
request (`adaptive_ng.NGTypedTarget`). `run` checkpoints one JSON line per finished item and skips finished items on
restart. Nothing here reads a return, a label or a learned coefficient; the items of `run` are C1 promotion units,
which only the guarded RT-S sampler produces after the unlock.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.redteam import adaptive_ng as NG  # noqa: E402

OUT = ROOT / "results" / "redteam" / "rta_ng"
REG_OUT = ROOT / "results" / "redteam" / "rta" / "ng"  # the paper's Table 2 reads results/redteam/rta/*/asr_budget.json
CACHE = ROOT / "data" / "cache" / "rta_ng_v1.sqlite"
W_PRE = ("2023-04-07", "2024-05-31")
IF1_CPU_HOURS = 36.0


def mem_available_gb() -> float:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def other_ng_workers() -> list[int]:
    """PIDs of running typed-local workers (another NG job)."""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "ng_worker.py" in cmd and "--mock" not in cmd:
            out.append(int(p.name))
    return out


def fsync_append(path: Path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ freeze
def cmd_freeze(a) -> int:
    spec = NG.NGSpec()
    rec = {"spec_sha256": spec.sha256(), "spec": spec.to_json(),
           "written_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "note": "RT-A-NG (P-4). The hash covers the spec, the objective, the rendered pool, the frozen templates "
                   "hash and the source of adaptive_ng.py and adaptive.py; quote it in the pre-registration."}
    write_json(OUT / "spec.json", rec)
    print(f"RT-A-NG spec {spec.version}: sha256 {rec['spec_sha256']}")
    print(f"  pool: {len(rec['spec']['pool']['payloads'])} payloads x "
          f"{sum(len(v) for v in rec['spec']['pool']['operators'].values())} operators x {len(spec.positions)} "
          f"positions; budget {spec.max_queries} queries per item; {spec.n_items} items")
    return 0


# ------------------------------------------------------------------------------------------------ model and data
def backend(a):
    from jevtrader.s1.backends import make_backend
    from jevtrader.s1.cache import ResponseCache

    free = mem_available_gb()
    if not a.mock and free < a.min_free_gb:
        raise SystemExit(f"only {free:.1f} GB of memory available (< {a.min_free_gb} GB); another NG worker: "
                         f"{other_ng_workers()}. Wait for it to finish.")
    use_cache = not (a.no_cache or a.mock)  # mock answers never enter the cache file
    s1 = make_backend(backend="typed-local", cache=ResponseCache(CACHE) if use_cache else False, ledger=False,
                      mock=a.mock, threads=a.threads, log_path=str(OUT / "worker.log"))
    if s1.ledger is not None or s1.backend_id != "typed-local":
        raise SystemExit("RT-A-NG must run on typed-local without a ledger")
    return s1


def real_data(weeks):
    from jevtrader.programs.context import ContextBuilder, RealData

    data = RealData()
    data.fact_symbols = data.symbols_for_weeks(weeks)
    return data, ContextBuilder(data, subset=None)


def calendar(data, lo, hi):
    from jevtrader.programs.context import decision_dates

    return decision_dates(data.sessions, lo, hi)


def run_rows(rows: pd.DataFrame, builder, s1, names, spec, ckpt: Path, log_every: int = 8) -> list[dict]:
    """Search each item (one after the other), appending one checkpoint line per finished item."""
    import asyncio

    done = {}
    if ckpt.exists():
        for line in ckpt.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("spec_sha256") == spec.sha256():
                    done[r["item_key"]] = r
    out = []
    for i, r in enumerate(rows.itertuples(index=False), 1):
        if str(r.unit_id) in done:
            out.append(done[str(r.unit_id)])
            continue
        t0 = time.time()
        ctx = builder(pd.Timestamp(r.t))
        n = {"q": 0}

        def tick(c, n=n, t0=t0, uid=r.unit_id):
            n["q"] += 1
            if n["q"] % log_every == 0:
                e = c.evaluation
                print(f"  {uid} q{c.query}: survived={e.survived} E={e.expected} p4={e.p_target} "
                      f"({time.time() - t0:.0f} s)", flush=True)

        run = asyncio.run(NG.run_item(r, ctx, s1, names, spec, on_query=tick))
        rec = run.record
        rec["context_s"] = round(run.wall_s, 1)
        fsync_append(ckpt, rec)
        s = rec["summary"]
        b = s["budgets"].get(max(spec.budgets)) or {}
        print(f"[{i}/{len(rows)}] {r.unit_id}: {s['queries']} queries, {rec['n_requests']} requests, "
              f"{rec['target_wall_s']:.0f} s in the model; best survived={b.get('survived')} "
              f"dE={b.get('delta_expected')} stop_early={s['stopped_early']}", flush=True)
        out.append(rec)
    return out


def summarize(records: list[dict], spec, out_dir: Path, extra: dict) -> dict:
    frame = NG.results_frame(records, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "results.parquet", index=False)
    wall = [r["target_wall_s"] for r in records]
    nq = [r["summary"]["queries"] + 1 for r in records]
    nreq = [r["n_requests"] for r in records]
    summ = {**NG.budget_summary(frame, spec), **extra,
            "timing": {"items": len(records), "queries_incl_clean": int(sum(nq)), "requests": int(sum(nreq)),
                       "model_wall_s": round(float(sum(wall)), 1),
                       "s_per_query": round(float(sum(wall)) / max(sum(nq), 1), 2),
                       "s_per_request": round(float(sum(wall)) / max(sum(nreq), 1), 2)}}
    write_json(out_dir / "asr_budget.json", summ)
    return summ


# ------------------------------------------------------------------------------------------------ dry
def cmd_dry(a) -> int:
    from jevtrader.pipeline.lock import touches_locked
    from jevtrader.redteam.inject import load_display_names

    spec = NG.NGSpec()
    if a.n > 5:
        raise SystemExit("the dry run is a check, not a result: at most 5 items")
    names = load_display_names()
    from jevtrader.programs.context import RealData

    probe = RealData()
    weeks = calendar(probe, *W_PRE)
    data, builder = real_data(weeks)
    news = data.news_table
    items = NG.dry_items(news, weeks, lambda t: builder.universe(t), names, n=a.n, spec=spec, window=W_PRE)
    if touches_locked(items["t"]):
        raise SystemExit("dry items must be W_pre")
    print(f"RT-A-NG dry run: spec {spec.sha256()[:16]}, {len(items)} W_pre items:\n{items.to_string(index=False)}",
          flush=True)
    s1 = backend(a)
    dry_dir = OUT / ("dry_mock" if a.mock else "dry")  # a mock-worker rehearsal never overwrites the real dry run
    dry_dir.mkdir(parents=True, exist_ok=True)
    items.to_parquet(dry_dir / "items.parquet", index=False)
    t0 = time.time()
    try:
        recs = run_rows(items, builder, s1, names, spec, dry_dir / "checkpoint.jsonl")
        desc = s1.describe()
    finally:
        s1.close()
    summ = summarize(recs, spec, dry_dir, {"kind": "dry run on W_pre (not a registered result)",
                                          "backend": desc, "wall_s": round(time.time() - t0, 1),
                                          "items": items.assign(t=items["t"].astype(str),
                                                                prev_t=items["prev_t"].astype(str))
                                          .to_dict("records")})
    print(json.dumps({k: summ[k] for k in ("budgets", "timing")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------ project
def cmd_project(a) -> int:
    spec = NG.NGSpec()
    # the slowest measured dry run (a rerun served from the response cache is faster than the model), else NG items500
    runs = [json.loads(f.read_text())["timing"] for f in sorted(OUT.glob("dry*/asr_budget.json"))
            if f.parent.name != "dry_mock"]
    measured = max(runs, key=lambda x: x["s_per_query"]) if runs else None
    ngp = ROOT / "results/judgments/ng_typed-local_wclean_items500/progress.json"
    s_full = float(json.loads(ngp.read_text()).get("s_per_item_session") or 23.9) if ngp.exists() else 23.9
    s_q = measured["s_per_query"] if measured else s_full
    q = spec.n_items * (spec.max_queries + 1)
    hours = q * s_q / 3600
    other = {"items500": 3.3, "X6_NG": 2.9, "NG_D1": 0.2}  # power_prereg cost_projection.NG_cpu (planning)
    res = {"spec_sha256": spec.sha256(), "queries_upper_bound": q, "s_per_query": s_q,
           "s_per_query_source": "the slowest RT-A-NG dry run (results/redteam/rta_ng/dry*/asr_budget.json)" if measured
           else
           "NG items500 s per full item (upper bound: every query asks all 6 requests)",
           "hours_upper_bound": round(hours, 2), "other_ng_hours_planned": other,
           "if1_budget_hours": IF1_CPU_HOURS, "fits_if1": hours + sum(other.values()) <= IF1_CPU_HOURS}
    write_json(OUT / "projection.json", res)
    print(f"RT-A-NG projection: {q} queries x {s_q:.1f} s = {hours:.1f} h of the 6-thread worker (upper bound, no "
          f"early stop); with the other NG work {hours + sum(other.values()):.1f} h vs IF-1 {IF1_CPU_HOURS:.0f} h "
          f"-> {'fits' if res['fits_if1'] else 'DOES NOT FIT'}")
    return 0 if res["fits_if1"] else 3


# ------------------------------------------------------------------------------------------------ run
def cmd_run(a) -> int:
    from jevtrader.redteam.inject import load_display_names
    from jevtrader.redteam.run import PROMOTION

    spec = NG.NGSpec()
    units = pd.read_parquet(a.units)
    units["t"] = pd.to_datetime(units["t"])
    units = units[units["kind"] == PROMOTION]
    names = load_display_names()
    from jevtrader.programs.context import RealData

    probe = RealData()
    weeks = calendar(probe, units["t"].min() - pd.Timedelta(days=14), units["t"].max())
    data, builder = real_data(weeks)
    items = NG.select_items(units, data.news_table, weeks, spec)  # checks the W_clean lock first
    REG_OUT.mkdir(parents=True, exist_ok=True)
    items.to_parquet(REG_OUT / "items.parquet", index=False)
    ids = "\n".join(sorted(items["unit_id"].astype(str)))
    print(f"RT-A-NG: {len(items)} items (sha256 {hashlib.sha256(ids.encode()).hexdigest()[:16]}), spec "
          f"{spec.sha256()[:16]}", flush=True)
    if cmd_project(a) != 0 and not a.force_cpu:
        raise SystemExit("the projected CPU time does not fit IF-1; pass --force-cpu only if the orchestrator agrees")
    s1 = backend(a)
    t0 = time.time()
    try:
        recs = run_rows(items, builder, s1, names, spec, REG_OUT / "checkpoint.jsonl")
        desc = s1.describe()
    finally:
        s1.close()
    summ = summarize(recs, spec, REG_OUT, {"kind": "RT-A-NG registered run", "backend": desc,
                                           "wall_s": round(time.time() - t0, 1), "units_file": str(a.units)})
    print(json.dumps({k: summ[k] for k in ("budgets", "timing")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "dry", "project", "run"):
        p = sub.add_parser(name)
        p.add_argument("--n", type=int, default=2)
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--min-free-gb", type=float, default=13.0)
        p.add_argument("--mock", action="store_true", help="the typed-local mock worker (tests of the driver)")
        p.add_argument("--no-cache", action="store_true")
        p.add_argument("--force-cpu", action="store_true")
        if name == "run":
            p.add_argument("--units", required=True)
    a = ap.parse_args(argv)
    return {"freeze": cmd_freeze, "dry": cmd_dry, "project": cmd_project, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
