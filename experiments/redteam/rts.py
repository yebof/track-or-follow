"""RT-S driver for the W_clean unlock (pre-registration draft §3.1, §13; `jevtrader.redteam.stages`, `.run`,
`.projection`). Every paid step is preceded by a FREE dry run of its exact requests (`projection.ProjectingEmulator`:
the paid emulator's request keys against the real response cache, read-only; gates open, so an upper bound) and a
printed preflight against its sub-cap; it aborts (exit 3) if the projection does not fit. Caps are read from
configs/backends.yaml and never changed here.

    # after the unlock (W_clean; MAIN's context):
    rts.py et                                # typed clean Layer B -> E_t (R^T <= K + J), guarded by the lock
    rts.py clean-readers [--run]             # H, H-spot, H-guard on E_t: dry run + preflight (H 1.50 / HSPOT 2.00 /
                                             #   HGUARD 2.00); --run pays, stage by stage, only if each fits
    rts.py sample                            # typed + H clean logs -> the registered C1 sample (guarded)
    rts.py nodef-clean [--run]               # NODEF clean on the units' stock-weeks (NODEF 3.00)
    rts.py stages                            # plan + dry run of every attacked stage, cumulative vs RT-S 12.00
    rts.py run --stage 1_typed_H             # one stage: fresh dry run, preflight, then the paid run (guarded)
    rts.py score --stage 1_typed_H           # replay scoring (clean Layer-B runs per reader) -> asr.json
    rts.py equiv                             # rehearsal only: the W_clean-equivalent of the dry-run projections

    # the mock rehearsal on W_pre (no paid call; the mock backend answers every model request):
    rts.py --gen pre --window W_pre_slice --backend mock --out data/cache/redteam/rts_wpre_mock --scale 0.05 \
        --last-weeks 8 et|clean-readers --run|sample|nodef-clean --run|stages|run --stage ...|score --stage ...

`--backend mock` never touches the ledger or the response cache. `--backend llm-logprob` (the paid emulator) is the
only paid backend accepted, and only through `--run` / `run` after a passing preflight.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "pre"))

from jevtrader.redteam import projection as P  # noqa: E402
from jevtrader.redteam import run as RT  # noqa: E402
from jevtrader.redteam import stages as S  # noqa: E402

GENS = {"main": ("main_emu_wclean_u100", "dev_run.json", "W_clean"),
        "pre": ("pre_blinded_emu_wpre_s30", "pre_run.json", "W_pre_slice")}
PAID = "llm-logprob"


# ------------------------------------------------------------------------------------------------ world
@dataclasses.dataclass
class World:
    gen: str
    gen_dir: Path
    data: object
    builder: object
    news: pd.DataFrame
    calendar: list
    window: str
    context_id: str


def world(a) -> World:
    from jevtrader.programs.context import ContextBuilder, RealData, decision_dates
    from jevtrader.s1.types import canonical_json

    gen_id, runfile, default_window = GENS[a.gen]
    gen_dir = ROOT / "results" / "judgments" / gen_id
    rec = json.loads((gen_dir / runfile).read_text())
    data = RealData()
    data.fact_symbols = data.symbols_for_weeks([pd.Timestamp(w) for w in rec["weeks"]])
    builder = ContextBuilder(data, subset=None)
    cid = hashlib.sha256(canonical_json(builder.describe()).encode()).hexdigest()[:10]
    if a.gen == "main":
        man = json.loads((gen_dir / "MANIFEST.json").read_text())
        if cid != rec["context_id"] or data.archive_id() != man["spec"]["data_freeze"]:
            raise SystemExit(f"context {cid} / archive {data.archive_id()} differ from MAIN's {rec['context_id']} / "
                             f"{man['spec']['data_freeze']}: unchanged requests would not be cache hits")
    weeks = [pd.Timestamp(w) for w in rec["weeks"]]
    cal = decision_dates(data.sessions, weeks[0] - pd.Timedelta(days=21), weeks[-1] + pd.Timedelta(days=21))
    return World(a.gen, gen_dir, data, builder, data.news_table, cal, a.window or default_window, cid)


def out_dir(a) -> Path:
    d = Path(a.out) if a.out else ROOT / "results" / "redteam" / "rts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))


def layer_b(a, w: World, cache: dict | None = None):
    """The typed clean Layer-B run of the clean generation (not registered, not written). W_pre reads prices capped
    before W_clean (experiments/pre/caproot); `evaluate` refuses a locked window before the unlock."""
    from jevtrader.pipeline.layer_b import DecisionConfig, evaluate, load_inputs, resolve_window

    if cache is not None and "typed" in cache:
        return cache["typed"]
    cfg = DecisionConfig.from_yaml(gen_id=w.gen_dir.name, name="primary", allow_missing=tuple(a.allow_missing))
    start, end = resolve_window(w.window)
    kw = {}
    if w.gen == "pre":
        import caproot

        kw["data_root"] = caproot.capped_root(Path(a.cap_root))
    inputs = load_inputs(cfg, start, end, ROOT, **kw)
    res = evaluate(cfg, w.window, inputs, root=ROOT, register=False, write=False)
    if a.focus_only:  # rehearsal only: keep the clean generation's focus names (PRE: S30), as W_clean's U100 has all
        if w.gen == "main":
            raise SystemExit("--focus-only is for the W_pre rehearsal only")
        foc = inputs.focus or {}
        p = res.log.predictions
        keep = [s in set(foc.get(pd.Timestamp(t), ())) for t, s in zip(pd.to_datetime(p["t"]), p["symbol"])]
        res.log.predictions = p[keep]
    out = (cfg, inputs, res.log)
    if cache is not None:
        cache["typed"] = out
    return out


def mock_backend():
    from jevtrader.s1.backends import make_backend

    return make_backend(backend="mock", cache=False, ledger=False)


def paid_backend():
    from jevtrader.s1.backends import make_backend

    s1 = make_backend(backend=PAID, cache=True, ledger=True)
    if s1.ledger is None or s1.cache is None:
        raise SystemExit("the paid backend must run with the spend ledger and the response cache")
    return s1


# ------------------------------------------------------------------------------------------------ et
def cmd_et(a) -> int:
    w = world(a)
    cfg, inputs, log = layer_b(a, w)
    et = S.in_band_frame(log, cfg.K, cfg.J)  # guarded by the lock (R^T is learned)
    et = last_weeks(a, et)
    od = out_dir(a)
    et.to_parquet(od / "E_t.parquet", index=False)
    write_json(od / "E_t.json", {"weeks": int(et["t"].nunique()), "stock_weeks": int(len(et)),
                                 "K": cfg.K, "J": cfg.J, "gen": w.gen_dir.name, "window": w.window,
                                 "sha256": hashlib.sha256(pd.util.hash_pandas_object(et, index=False).to_numpy()
                                                          .tobytes()).hexdigest()})
    print(f"E_t: {len(et)} stock-weeks in {et['t'].nunique()} weeks (R^T <= {cfg.K + cfg.J})")
    return 0


def last_weeks(a, frame: pd.DataFrame, col: str = "t") -> pd.DataFrame:
    if not a.last_weeks:
        return frame
    keep = sorted(frame[col].unique())[-a.last_weeks:]
    return frame[frame[col].isin(keep)].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ clean runs
def _generate(spec, s1):
    from jevtrader.programs.runner import generate

    spec = dataclasses.replace(spec, s1=s1)
    return generate(spec, resume=True)


def dry_generate(spec, od: Path, label: str) -> dict:
    """The spec's generation on the projecting backend (high, then point) into a scratch root; projected USD."""
    import shutil

    res = {}
    for mode in ("high", "point"):
        s1 = P.projecting_backend(mode)
        d = od / "dry" / mode / label.replace(" ", "_")
        if d.exists():
            shutil.rmtree(d)  # a projection is always a fresh dry run
        gen = _generate(dataclasses.replace(spec, out_root=d, run_tag=spec.run_tag), s1)
        res[mode] = P.spend_of(pd.read_parquet(gen / "requests.parquet"))
        res[mode]["real_cache_hits"] = s1.cache.real_hits
    return res


def clean_run(a, w: World, spec, od: Path, label: str) -> dict:
    """Dry run -> preflight -> (with --run) the run on the chosen backend."""
    rec = {"label": label, "run_tag": spec.run_tag, "weeks": len(spec.weeks)}
    proj = dry_generate(spec, od, label)  # free: the projecting backend never calls a provider
    pf = P.preflight(spec.run_tag, label, proj["high"]["usd"], proj["point"]["usd"])
    print(pf.line(), flush=True)
    rec.update(projection=proj, preflight=pf.to_json())
    if a.backend == "mock":
        rec["mode"] = "mock"
        if a.run:
            gen = _generate(dataclasses.replace(spec, out_root=od / "clean", run_tag=spec.run_tag), mock_backend())
            rec["gen_dir"] = str(gen)
        return rec
    if a.run:
        if not pf.ok:
            print(f"ABORT {label}: the projection does not fit", flush=True)
            rec["aborted"] = True
            return rec
        s1 = P.paid_guard(paid_backend(), pf.headroom)
        par = {"max_parallel_units": a.max_parallel_units} if getattr(a, "max_parallel_units", 0) else {}
        gen = _generate(dataclasses.replace(spec, out_root=ROOT / "results" / "judgments", **par), s1)
        rec["gen_dir"] = str(gen)
        rec["guard_spent"] = s1.spent
    return rec


def reader_gen_dir(a, reader: str) -> Path:
    tag = S.CLEAN_READERS[reader].lower()
    base = (out_dir(a) / "clean") if a.backend == "mock" else ROOT / "results" / "judgments"
    return base / f"{tag}_emu_wclean_et"


def cmd_clean_readers(a) -> int:
    w = world(a)
    od = out_dir(a)
    et = pd.read_parquet(od / "E_t.parquet")
    focus = S.focus_by_week(et)
    recs = {}
    for reader in (a.reader.split(",") if a.reader else list(S.CLEAN_READERS)):
        spec = S.clean_reader_spec(reader, focus, w.builder, None, od, data_freeze=w.data.archive_id())
        recs[reader] = clean_run(a, w, spec, od, f"clean {reader} on E_t")
        if recs[reader].get("aborted"):
            break
    write_json(od / "clean_readers.json", recs)
    return 3 if any(r.get("aborted") for r in recs.values()) else 0


def cmd_nodef_clean(a) -> int:
    w = world(a)
    od = out_dir(a)
    units = pd.read_parquet(od / "units.parquet")
    units = units[units["kind"] == RT.PROMOTION]
    spec = S.nodef_clean_spec(units, w.builder, None, od, data_freeze=w.data.archive_id())
    rec = clean_run(a, w, spec, od, "NODEF clean on the unit stock-weeks")
    write_json(od / "nodef_clean.json", rec)
    return 3 if rec.get("aborted") else 0


# ------------------------------------------------------------------------------------------------ sample
def cmd_sample(a) -> int:
    w = world(a)
    od = out_dir(a)
    cfg, inputs, log = layer_b(a, w)
    hgen = reader_gen_dir(a, "H")
    if not (hgen / "features.parquet").exists():
        raise SystemExit(f"no clean H generation at {hgen}: run clean-readers first")
    _, hres, _ = S.reader_log(cfg, "H", inputs, w.window, ROOT, [hgen], allow_missing=a.allow_missing)
    logs = {"typed": log, "H": hres.log}
    sizes = dict(S.REGISTERED_SAMPLE)
    if a.scale != 1.0:  # rehearsal only: the registered call has scale 1
        if w.gen == "main":
            raise SystemExit("--scale is for the W_pre rehearsal only")
        sizes = {k: (max(1, int(round(v * a.scale))) if isinstance(v, int) and v else v) for k, v in sizes.items()}
    if a.last_weeks:
        keep = sorted(pd.to_datetime(log.predictions["t"]).unique())[-a.last_weeks:]
        for lg in logs.values():
            lg.predictions = lg.predictions[pd.to_datetime(lg.predictions["t"]).isin(keep)]
    units = RT.sample_units(logs, cfg.K, cfg.J, **sizes)  # refuses before the unlock on W_clean
    units.to_parquet(od / "units.parquet", index=False)
    ids = "\n".join(sorted(units["unit_id"]))
    rec = {"sizes": sizes, "counts": units.groupby("kind").size().to_dict(),
           "strata": units.groupby(["kind", "stratum"]).size().rename("n").reset_index().to_dict("records"),
           "weeks": int(units["t"].nunique()), "sha256": hashlib.sha256(ids.encode()).hexdigest()}
    write_json(od / "units.json", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------------ stages
def plans(a, w: World, units: pd.DataFrame, which=None):
    from jevtrader.redteam.inject import load_display_names

    names = load_display_names()
    cfg, inputs, log = layer_b(a, w, a._cache)
    hgen = reader_gen_dir(a, "H")
    elig = None
    if (hgen / "features.parquet").exists():
        _, hres, _ = S.reader_log(cfg, "H", inputs, w.window, ROOT, [hgen], allow_missing=a.allow_missing)
        elig = RT.eligibility_oracle({"typed": log, "H": hres.log}, cfg.K, cfg.J)
    out = []
    for st in S.STAGES:
        if which and st.name != which:
            continue
        out.append(S.plan_stage(st, w.news, units, w.calendar, eligible_later=elig, names=names))
    return out


def clean_gen_of(a, st: S.Stage, w: World) -> Path | None:
    if st.clean == "nodef":
        base = (out_dir(a) / "clean") if a.backend == "mock" else ROOT / "results" / "judgments"
        g = base / "nodef_emu_wclean_units"
        return g if g.exists() else None
    return w.gen_dir


def dry_stage(plan: S.StagePlan, w: World, od: Path, s1, a) -> dict:
    d = od / "dry" / s1.mode / plan.stage.arm
    dirs = asyncio.run(S.arun_stage(plan, w.news, w.builder, s1, d.parent, clean_gen_of(a, plan.stage, w),
                                    resume=False))
    return P.frames_spend(dirs)


def cmd_stages(a) -> int:
    w = world(a)
    od = out_dir(a)
    units = pd.read_parquet(od / "units.parquet")
    ps = plans(a, w, units)
    backs = {m: P.projecting_backend(m) for m in ("high", "point")}  # shared: stage k sees stage < k's answers
    rows, committed = [], 0.0
    for pl in ps:
        S.write_plan(pl, od)
        if not pl.stage.built or not pl.batches:
            rows.append({"stage": pl.stage.name, "built": pl.stage.built, "plan": pl.to_json(), "status": "not run",
                         "note": pl.stage.note})
            print(f"STAGE {pl.stage.name}: not run ({pl.stage.note or 'no units'})")
            continue
        proj = {m: dry_stage(pl, w, od, backs[m], a) for m in backs}
        pf = P.preflight("RT-S", pl.stage.name, proj["high"]["usd"], proj["point"]["usd"], committed=committed)
        print(pf.line(), flush=True)
        if pf.ok:
            committed += proj["high"]["usd"]
        rows.append({"stage": pl.stage.name, "plan": pl.to_json(), "projection": proj, "preflight": pf.to_json(),
                     "status": "fits" if pf.ok else "does not fit"})
    write_json(od / "preflight_stages.json", {"stages": rows, "calibration": P.Calibration.load().to_json(),
                                              "written_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    return 0


def cmd_run(a) -> int:
    w = world(a)
    od = out_dir(a)
    units = pd.read_parquet(od / "units.parquet")
    (pl,) = plans(a, w, units, a.stage)
    S.write_plan(pl, od)
    if not pl.stage.built:
        raise SystemExit(f"{pl.stage.name} is not built: {pl.stage.note}")
    clean = clean_gen_of(a, pl.stage, w)
    if pl.stage.clean == "nodef" and clean is None:
        raise SystemExit("the nodef arm needs the NODEF clean generation (nodef-clean --run) first")
    t0 = time.time()
    if a.backend == "mock":
        dirs = asyncio.run(S.arun_stage(pl, w.news, w.builder, mock_backend(), od, clean))
        rec = {"stage": pl.stage.name, "mode": "mock", "batches": len(dirs), "spend": P.frames_spend(dirs)}
    else:
        proj = {m: dry_stage(pl, w, od, P.projecting_backend(m), a) for m in ("high", "point")}
        pf = P.preflight("RT-S", pl.stage.name, proj["high"]["usd"], proj["point"]["usd"])
        print(pf.line(), flush=True)
        rec = {"stage": pl.stage.name, "projection": proj, "preflight": pf.to_json()}
        if not pf.ok:
            write_json(od / pl.stage.arm / "run.json", {**rec, "aborted": True})
            print(f"ABORT {pl.stage.name}", flush=True)
            return 3
        s1 = P.paid_guard(paid_backend(), pf.headroom)
        dirs = asyncio.run(S.arun_stage(pl, w.news, w.builder, s1, od, clean))
        rec.update(batches=len(dirs), spend=P.frames_spend(dirs), guard_spent=s1.spent)
    rec["wall_s"] = round(time.time() - t0, 1)
    write_json(od / pl.stage.arm / "run.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "projection"}, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------------ score
def cmd_score(a) -> int:
    from jevtrader.learn.labels import LabelStore  # noqa: F401  (labels come with the inputs)

    w = world(a)
    od = out_dir(a)
    st = S.STAGE[a.stage]
    arm_dir = od / st.arm
    plan = json.loads((arm_dir / "plan.json").read_text())
    dirs = [arm_dir / b["batch_id"] for b in plan["batch_list"] if (arm_dir / b["batch_id"] / "MANIFEST.json").exists()]
    cfg, inputs, log = layer_b(a, w)
    runs = []
    extra = {"H": reader_gen_dir(a, "H"), "H-guard": reader_gen_dir(a, "H-guard"),
             "H-spot": reader_gen_dir(a, "H-spot")}
    scores = []
    for reader in st.readers:
        if reader == "nodef":
            continue
        gens = [extra[reader]] if reader in extra else []
        if gens and not (gens[0] / "features.parquet").exists():
            print(f"reader {reader}: no clean generation, not scored")
            continue
        rcfg, rres, rin = S.reader_log(cfg, reader, inputs, w.window, ROOT, gens, allow_missing=a.allow_missing)
        runs.append(RT.CleanRun(reader, rcfg, rres.log, rin.raw, arm=st.arm))
    for d in dirs:
        if runs:
            scores.append(RT.score_batch(d, runs, labels=inputs.labels))
        if "nodef" in st.readers:
            ng = clean_gen_of(a, st, w)
            cf = RT.attacked_features(ng)
            scores.append(S.score_nodef_batch(d, RT.CleanRun("typed", cfg, log, inputs.raw), cf))
    if not scores:
        raise SystemExit("nothing to score")
    sc = pd.concat(scores, ignore_index=True)
    sc.to_parquet(arm_dir / "scores.parquet", index=False)
    summ = RT.asr_summary(sc, n_boot=a.n_boot)
    write_json(arm_dir / "asr.json", {**summ, "stage": st.name, "batches": len(dirs),
                                      "mode": a.backend, "window": w.window})
    print(json.dumps({"pooled": summ["pooled"], "c1": summ.get("c1")}, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------------ rehearsal analysis
STOCK_LEVEL = ("N3", "H", "H-spot", "H-guard", "HS")


def attacked_only(batch_dir: Path) -> dict:
    """A dry-run batch's projected USD restricted to the requests an attack changes: item-level requests (N0, N1, N2)
    whose evidence holds an attacked item, and every stock-week-level request (N3 and the readers; an upper side).
    On W_clean the other requests are MAIN's cache hits; on the W_pre rehearsal they are misses for names outside
    PRE's S30, so this is the W_clean-equivalent of a rehearsal projection."""
    man = json.loads((batch_dir / "MANIFEST.json").read_text())
    att = {e for u in man["batch"]["units"] for e in u.get("item_eids", [])}
    req = pd.read_parquet(batch_dir / "requests.parquet")
    ev = pd.read_parquet(batch_dir / "evidence.parquet")
    keys = set(ev.loc[ev["eid"].isin(att), "request_key"])
    paid = req[~req["cached"].astype(bool)]
    hit = paid["program"].isin(STOCK_LEVEL) | paid["request_key"].isin(keys)
    return {"usd_all": float(paid["usd"].sum()), "usd_attacked": float(paid.loc[hit, "usd"].sum()),
            "unit_families": len(man["batch"]["units"])}


def cmd_equiv(a) -> int:
    od = out_dir(a)
    pre = json.loads((od / "preflight_stages.json").read_text())
    reg = {"1_typed_H": 200 * 7, "2_guard_spot": 200 * 7, "4_nodef": 200 * 7, "5_demotion": 100 * 2}
    rows, cum = {}, 0.0
    for st in pre["stages"]:
        name = st["stage"]
        if "projection" not in st:
            continue
        arm = S.STAGE[name].arm
        parts = [attacked_only(d) for d in sorted((od / "dry" / "high" / arm).iterdir()) if (d / "MANIFEST.json")
                 .exists()]
        uf = sum(x["unit_families"] for x in parts)
        usd_att = sum(x["usd_attacked"] for x in parts)
        per = usd_att / max(uf, 1)
        cum += per * reg[name]
        rows[name] = {"unit_families": uf, "usd_all": round(sum(x["usd_all"] for x in parts), 4),
                      "usd_attacked_only": round(usd_att, 4), "usd_per_unit_family": round(per, 6),
                      "registered_unit_families": reg[name], "scaled_usd": round(per * reg[name], 2),
                      "cumulative_usd": round(cum, 2)}
        print(f"{name}: {uf} unit-families, ${usd_att:.3f} attacked-only (all ${rows[name]['usd_all']:.3f}); "
              f"${per:.5f} per unit-family -> x{reg[name]} = ${per * reg[name]:.2f} (cumulative ${cum:.2f})")
    write_json(od / "wclean_equivalent.json", {"stages": rows, "note": attacked_only.__doc__,
                                               "sub_cap_RT_S": P.ledger_state("RT-S")["sub_cap"]})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen", choices=sorted(GENS), default="main")
    ap.add_argument("--window", default=None)
    ap.add_argument("--backend", choices=("mock", PAID), default=PAID)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cap-root", default=str(ROOT / "data" / "cache" / "caproot_rts"))
    ap.add_argument("--scale", type=float, default=1.0, help="rehearsal only: sample sizes x scale")
    ap.add_argument("--last-weeks", type=int, default=0, help="rehearsal only: the last n weeks")
    ap.add_argument("--focus-only", action="store_true",
                    help="rehearsal only: units and E_t among the clean generation's focus names (W_pre: S30)")
    ap.add_argument("--allow-missing", default="", help="Layer-B input checks waived for the typed and reader runs")
    ap.add_argument("--n-boot", type=int, default=10_000)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("et")
    p = sub.add_parser("clean-readers")
    p.add_argument("--run", action="store_true")
    p.add_argument("--reader", default="")
    p.add_argument("--max-parallel-units", type=int, default=0,
                   help="paid run only: fewer concurrent Layer-A units (0 = the spec's default; the guard is unchanged)")
    p = sub.add_parser("nodef-clean")
    p.add_argument("--run", action="store_true")
    sub.add_parser("sample")
    sub.add_parser("stages")
    sub.add_parser("equiv")
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, choices=sorted(S.STAGE))
    p = sub.add_parser("score")
    p.add_argument("--stage", required=True, choices=sorted(S.STAGE))
    a = ap.parse_args(argv)
    a.allow_missing = tuple(x for x in a.allow_missing.split(",") if x)
    a._cache = {}
    if a.gen == "main" and (a.scale != 1.0 or a.last_weeks or a.focus_only):
        raise SystemExit("--scale / --last-weeks / --focus-only are rehearsal options (W_pre) only")
    fn = {"et": cmd_et, "clean-readers": cmd_clean_readers, "nodef-clean": cmd_nodef_clean, "sample": cmd_sample,
          "stages": cmd_stages, "run": cmd_run, "score": cmd_score, "equiv": cmd_equiv}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
