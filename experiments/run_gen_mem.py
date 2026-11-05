"""MEM (X2 / PD3; design_v1 §13.2 MEM; design_v2_delta MEM-1; preregistration_v1 DRAFT §9): the store ablations that
need paid Layer-A requests, on S30 x W_clean (115 decisions), frozen bank v1.0, emu-v1. Raw answers only.

    .venv/bin/python experiments/run_gen_mem.py replay      # $0: MAIN's N23 chain replayed (checks + N3 states)
    .venv/bin/python experiments/run_gen_mem.py dry         # $0: both arms on the fake transport -> projection.json
    .venv/bin/python experiments/run_gen_mem.py run-epi     # preflight, then paid arm 1 (tag MEM/N3)
    .venv/bin/python experiments/run_gen_mem.py run-work    # preflight fixes k, then paid arm 2 (tag MEM/T)
    .venv/bin/python experiments/run_gen_mem.py validate

**Arm 1, M-epi off (N3 re-ask).** `programs.news.NewsEventProgram` with the registered variant `mepi_off`: no N2
(novelty 1), no M-epi reads or writes, `earlier_events` empty; everything else as MAIN. N01 is not re-asked: its
blackboard rows (`N01.items/errors/sentinel.<s>`) are MAIN's, replayed (`ReplayN01`), so the clusters and item
states are exactly MAIN's (N23 re-checks every item's `n1_sha256`). Paid requests: the N3 requests only.
**Arm 2, M-work off (T re-ask).** `programs.technical.TechnicalProgram` without M in the run: the T state's
"market regime" (the M-work blackboard field) renders "not available"; everything else is MAIN's T state. Draft §9:
arm 1 runs first on all 115 weeks; arm 2 runs on as many of the LAST W_clean weeks as the remaining MEM sub-cap
allows, newest first, the week count k fixed by the preflight before the run (k = the largest k whose point
projection x 1.10 fits the remaining MEM headroom).

**Replay check ($0, `replay`).** The same N01 replay + N23 with M-epi ON, answered by MAIN's own answers table
(`MainReplay`: an unknown request raises), must reproduce MAIN's N2 and N3 request keys for every S30 stock-week: it
shows the replayed N01 rows and the rebuilt contexts give exactly MAIN's news chain, so arm 1 differs from MAIN only
by the variant. It also stores MAIN's N3 states (for FID's V2 floor and the N3 token calibration).

Outputs: results/judgments/mem_emu_mepi-off_wclean_s30/, results/judgments/mem_emu_mwork-off_wclean-last<k>_s30/
(the runner's tables and MANIFEST.json), results/judgments/mem_emu_mepi-off_wclean_s30/{replay_check.json,
projection.json, validation.json}. No W_clean outcome is read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gen_common as C  # noqa: E402

from jevtrader.programs.bank_v1 import FROZEN_SHA256, load_bank  # noqa: E402
from jevtrader.programs.context import subset_names, universe_asof  # noqa: E402
from jevtrader.programs.news import VARIANT_MEPI_OFF, NewsEventProgram  # noqa: E402
from jevtrader.programs.runner import GeneratorSpec, generate  # noqa: E402
from jevtrader.programs.technical import TechnicalProgram  # noqa: E402

log = logging.getLogger("run_gen_mem")
ROOT = C.ROOT
RUN = "MEM"
EPI_ID = "mem_emu_mepi-off_wclean_s30"
EPI = ROOT / "results" / "judgments" / EPI_ID
SCRATCH = Path("/tmp/jevtrader-scratch/paid")
FREEZE = "archive-a377a4d6abd3bfe3"
MARGIN = 1.10


def work_id(k: int) -> str:
    return f"mem_emu_mwork-off_wclean-last{k}_s30"


def s30_by_week(data, weeks) -> dict[pd.Timestamp, set[str]]:
    return {t: set(subset_names(universe_asof(data.membership, t, 100), 30, "S30-www27")) for t in weeks}


def epi_spec(gen_id: str, weeks, builder, data, s1, out_root: Path, variant: frozenset[str],
             names: dict | None = None) -> GeneratorSpec:
    bank = load_bank(run_tag=RUN)
    rep = C.ReplayN01(C.MAIN_DIR, weeks, names if names is not None else s30_by_week(data, weeks))
    return GeneratorSpec(gen_id=gen_id, weeks=list(weeks), programs=["N01", "N23"], run_tag=RUN, variant=variant,
                         data_freeze=FREEZE, backend="llm-logprob", out_root=out_root, max_parallel_units=16,
                         notes=f"experiments/run_gen_mem.py; S30; N01 replayed from {C.MAIN_DIR.name}; "
                               f"variant={sorted(variant)}",
                         context_factory=builder, program_objects={"N01": rep, "N23": NewsEventProgram(bank)}, s1=s1)


def work_spec(gen_id: str, weeks, builder, s1, out_root: Path) -> GeneratorSpec:
    bank = load_bank(run_tag=RUN)
    return GeneratorSpec(gen_id=gen_id, weeks=list(weeks), programs=["T"], run_tag=RUN,
                         variant=frozenset({"mwork_off"}), data_freeze=FREEZE, backend="llm-logprob",
                         out_root=out_root, max_parallel_units=16, allow_missing_deps=True,
                         notes="experiments/run_gen_mem.py; S30; T without M (M-work market regime off)",
                         context_factory=builder, program_objects={"T": TechnicalProgram(bank)}, s1=s1)


# ------------------------------------------------------------------------------------------ replay check ($0)
def cmd_replay(a) -> int:
    """M-epi is a per-name chain over every week the name is in U_t (MAIN ran U100), and S30(U_t) changes with U_t:
    the replay therefore runs every name that is ever in S30 over all the weeks it is in U_t, and compares the S30
    stock-weeks."""
    weeks = C.main_weeks()
    _, data0 = C.main_context("S30")
    s30 = s30_by_week(data0, weeks)
    names = set().union(*s30.values())
    builder, data = C.main_context(None, symbols=sorted(names))
    focus = {t: set(universe_asof(data.membership, t, 100)) & names for t in weeks}
    s1 = C.MainReplay(["N2", "N3"], names)
    s1.dump = SCRATCH / "mem_replay_missing.jsonl"
    out_root = SCRATCH / "mem_replay"
    t0 = time.time()
    gen = generate(epi_spec("mem_replay_mepi-on_s30union", weeks, builder, data, s1, out_root, frozenset(), focus),
                   resume=True)
    req = pd.read_parquet(gen / "requests.parquet")
    main = pd.read_parquet(C.MAIN_DIR / "requests.parquet", columns=["request_key", "t", "symbol", "program"])
    main = main[main.program.isin(["N2", "N3"])]
    main["d"] = pd.to_datetime(main.t).dt.normalize()
    main = main[[(d in s30 and s in s30[d]) for d, s in zip(main.d, main.symbol)]]
    req["d"] = pd.to_datetime(req.t).dt.normalize()
    req = req[[(d in s30 and s in s30[d]) for d, s in zip(req.d, req.symbol)]]
    chk = {}
    for prog in ("N2", "N3"):
        a_ = set(zip(req.d[req.program == prog], req.symbol[req.program == prog], req.request_key[req.program == prog]))
        b_ = set(zip(main.d[main.program == prog], main.symbol[main.program == prog],
                     main.request_key[main.program == prog]))
        chk[prog] = {"replay": len(a_), "main": len(b_), "equal": a_ == b_, "only_replay": len(a_ - b_),
                     "only_main": len(b_ - a_)}
    n3 = {k: st for k, st, qs, tag in s1.captured if tag.endswith("N3")}
    EPI.mkdir(parents=True, exist_ok=True)
    keyinfo = req[req.program == "N3"].drop_duplicates("request_key").set_index("request_key")  # S30 rows only
    n3 = {k: v for k, v in n3.items() if k in keyinfo.index}
    with gzip.open(EPI / "main_n3_states.jsonl.gz", "wt") as fh:
        for k, st in n3.items():
            fh.write(json.dumps({"request_key": k, "t": str(keyinfo.at[k, "d"].date()), "symbol": keyinfo.at[k, "symbol"],
                                 "state": st}, ensure_ascii=False) + "\n")
    res = {"checks": chk, "pass": all(v["equal"] for v in chk.values()), "missing_keys": s1.missing,
           "n3_states_saved": len(n3), "weeks": len(weeks), "stock_weeks": sum(len(v) for v in s30.values()),
           "wall_s": round(time.time() - t0, 1), "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (EPI / "replay_check.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, default=str), flush=True)
    return 0 if res["pass"] else 1


# ------------------------------------------------------------------------------------------ calibration
def calibration() -> dict:
    """tokens/byte of emu-v1 N3 prompts (MAIN's N3 states from the replay vs MAIN's measured input tokens) and T
    prompts (PROBE plan's rebuilt MAIN/PRE T states vs their measured tokens)."""
    from jevtrader.s1.backends import LogprobEmulator

    bank = load_bank(run_tag=RUN)
    emu = LogprobEmulator(client=object(), bank_hash=bank.sha256, prompt_cache=False)
    req = pd.read_parquet(C.MAIN_DIR / "requests.parquet", columns=["request_key", "program", "input_tokens"])
    tok = req[(req.program == "N3") & (req.input_tokens > 0)].groupby("request_key").input_tokens.first().to_dict()
    qs = {s.name: s.to_question() for s in bank.program("N3").values()}
    s3 = []
    with gzip.open(EPI / "main_n3_states.jsonl.gz", "rt") as fh:
        for line in fh:
            x = json.loads(line)
            if x["request_key"] in tok:
                s3.append((x["state"], qs, int(tok[x["request_key"]])))
    qt = {s.name: s.to_question() for s in bank.program("T").values()}
    st = []
    with gzip.open(ROOT / "results/judgments/probe_emu_d1d2/plan/calibration.jsonl.gz", "rt") as fh:
        for line in fh:
            x = json.loads(line)
            if x["program"] == "T":
                st.append((x["state"], qt, int(x["tokens"])))
    return {"N3": C.calibrate(emu, s3[:2000]) | {"source": "MAIN N3 rows (replayed states)"},
            "T": C.calibrate(emu, st) | {"source": "MAIN/PRE T rows rebuilt by the PROBE plan"}}


def ratio_fn(cal: dict):
    r3, rt = cal["N3"]["ratio"], cal["T"]["ratio"]
    return lambda system, user: r3 if '"this_week_events"' in user else rt


# ------------------------------------------------------------------------------------------ dry
def cmd_dry(a) -> int:
    weeks = C.main_weeks()
    builder, data = C.main_context("S30")
    cal = calibration()
    res = {"calibration": cal}
    for arm in ("epi", "work"):
        scratch = SCRATCH / f"mem_dry_{arm}_{int(time.time())}"
        fake = C.FakeOpenAI(ratio=ratio_fn(cal))
        s1 = C.make_s1("llm-logprob", True, scratch, fake)
        g = C.guarded_raw(s1, float("inf"))
        if arm == "epi":
            spec = epi_spec("dry_" + EPI_ID, weeks, builder, data, g, scratch, frozenset({VARIANT_MEPI_OFF}))
        else:
            spec = work_spec("dry_work", weeks, builder, g, scratch)
        spec.s1 = g
        t0 = time.time()
        gen = generate(spec, resume=False)
        s1.ledger.flush()
        req = pd.read_parquet(gen / "requests.parquet")
        paid = req[~req.cached.astype(bool)]
        by_week = paid.groupby(pd.to_datetime(paid.t).dt.strftime("%Y-%m-%d")).usd.sum()
        res[arm] = {"requests": len(req), "paid": len(paid), "projected_usd": round(float(paid.usd.sum()), 4),
                    "high_usd": round(sum(x["usd_high"] for x in fake.calls), 4), "fake": fake.summary(),
                    "real_cache_hits": s1.cache.hits_real, "by_week_usd": by_week.round(6).to_dict(),
                    "dry_dir": str(gen), "wall_s": round(time.time() - t0, 1)}
        print(arm, json.dumps({k: v for k, v in res[arm].items() if k != "by_week_usd"}), flush=True)
    res["created_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    EPI.mkdir(parents=True, exist_ok=True)
    (EPI / "projection.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------ paid
def cmd_run_epi(a) -> int:
    proj = json.loads((EPI / "projection.json").read_text())
    p = proj["epi"]
    # the preflight shows arm 1 alone; arm 2 then gets what is left of the sub-cap (k fixed at its own preflight)
    pf = C.preflight(RUN, p["projected_usd"], p["high_usd"], f"arm 1 M-epi off: {p['paid']} N3 requests, S30 x 115")
    if not pf.ok:
        return 3
    weeks = C.main_weeks()
    builder, data = C.main_context("S30")
    s1 = C.make_s1("llm-logprob", False)
    g = C.guarded_raw(s1, pf.guard)  # the runner tags MEM/<program>
    t0 = time.time()
    try:
        gen = generate(epi_spec(EPI_ID, weeks, builder, data, g, EPI.parent, frozenset({VARIANT_MEPI_OFF})))
    except C.BudgetExceeded as e:
        s1.ledger.flush()
        print(f"BUDGET STOP: {e}; spent ${g.spent:.4f}", flush=True)
        C.launch_record(EPI, {"arm": "epi", "status": "budget_stop", "error": str(e), "spent": g.spent})
        return 3
    s1.ledger.flush()
    C.launch_record(EPI, {"arm": "epi", "status": "complete", "process_spent_usd": g.spent, "calls": g.calls,
                          "refused": g.refused, "wall_s": round(time.time() - t0, 1)})
    print(f"arm 1 done: {gen}; process spent ${g.spent:.4f}", flush=True)
    return 0


def cmd_run_work(a) -> int:
    proj = json.loads((EPI / "projection.json").read_text())
    p = proj["work"]
    per_week = {pd.Timestamp(k): float(v) for k, v in p["by_week_usd"].items()}
    weeks = sorted(per_week, reverse=True)  # newest first
    cap = C.sub_cap(RUN)
    total, runs = C.ledger_totals()
    head = min(cap - runs.get(RUN, 0.0), C.STOP_LEDGER - total, C.load_cfg()["ledger"]["hard_cap_usd"] - total)
    k, acc = 0, 0.0
    for w in weeks:
        if (acc + per_week[w]) * MARGIN > head:
            break
        acc += per_week[w]
        k += 1
    if a.weeks is not None:
        k = min(k, a.weeks)
        acc = sum(per_week[w] for w in weeks[:k])
    hi = p["high_usd"] * acc / max(p["projected_usd"], 1e-9)
    print(f"arm 2: k = {k} of {len(weeks)} weeks (newest first) fit the MEM headroom ${head:.2f} at projection x "
          f"{MARGIN:.2f}", flush=True)
    pf = C.preflight(RUN, acc, hi, f"arm 2 M-work off: T re-ask, S30 x the last {k} W_clean weeks")
    if not pf.ok or k == 0:
        return 3
    sel = sorted(weeks[:k])
    builder, _ = C.main_context("S30")
    s1 = C.make_s1("llm-logprob", False)
    g = C.guarded_raw(s1, pf.guard)  # the runner tags MEM/<program>
    t0 = time.time()
    out = ROOT / "results" / "judgments"
    try:
        gen = generate(work_spec(work_id(k), sel, builder, g, out))
    except C.BudgetExceeded as e:
        s1.ledger.flush()
        print(f"BUDGET STOP: {e}; spent ${g.spent:.4f}", flush=True)
        C.launch_record(out / work_id(k), {"arm": "work", "status": "budget_stop", "error": str(e), "spent": g.spent})
        return 3
    s1.ledger.flush()
    C.launch_record(gen, {"arm": "work", "k": k, "weeks": [str(w.date()) for w in sel], "status": "complete",
                          "process_spent_usd": g.spent, "calls": g.calls, "refused": g.refused,
                          "projection_usd": acc, "wall_s": round(time.time() - t0, 1)})
    print(f"arm 2 done: {gen}; process spent ${g.spent:.4f}", flush=True)
    return 0


# ------------------------------------------------------------------------------------------ validation
def cmd_validate(a) -> int:
    weeks = C.main_weeks()
    from jevtrader.programs.context import RealData

    data = RealData()
    s30 = s30_by_week(data, weeks)
    res = {"arms": {}}
    main = pd.read_parquet(C.MAIN_DIR / "requests.parquet", columns=["t", "symbol", "program", "request_key"])
    main["d"] = pd.to_datetime(main.t).dt.normalize()
    for arm, gdir in [("epi", EPI)] + [("work", p) for p in sorted((ROOT / "results/judgments").glob(
            "mem_emu_mwork-off_wclean-last*_s30"))]:
        man = json.loads((gdir / "MANIFEST.json").read_text())
        req = pd.read_parquet(gdir / "requests.parquet")
        ans = pd.read_parquet(gdir / "answers.parquet")
        req["d"] = pd.to_datetime(req.t).dt.normalize()
        prog = "N3" if arm == "epi" else "T"
        mine = req[req.program == prog]
        ws = sorted({pd.Timestamp(w) for w in man["spec"]["weeks"]})
        m = main[(main.program == prog) & main.d.isin(ws)]
        m = m[[s in s30[d] for d, s in zip(m.d, m.symbol)]]
        paid = ~mine.cached.astype(bool)
        chk = {
            "weeks": ws == (weeks if arm == "epi" else sorted(weeks)[-len(ws):]),
            # arm 1: one N3 per stock-week that had one in MAIN (same clusters); arm 2: one T per S30 stock-week
            "stock_weeks_match_main": set(zip(mine.d, mine.symbol)) == set(zip(m.d, m.symbol)),
            "no_n2": arm != "epi" or not (req.program == "N2").any(),
            # a state identical to MAIN's (no earlier event and no N2-checked cluster in MAIN either) keeps MAIN's
            # key and is answered from the response cache: every shared key must be such a cache hit
            "keys_shared_with_main_are_cache_hits": bool(mine[mine.request_key.isin(set(m.request_key))].cached
                                                         .astype(bool).all()),
            "pinned": bool((mine.model_pinned == C.MODEL).all() and (mine.response_model[paid] == C.MODEL).all()
                           and (mine.bank_hash == FROZEN_SHA256).all() and (mine.prompt_version == "emu-v1").all()),
            "validity_ge_98pct": bool(1 - ans[ans.program == prog].probs_json.isna().mean() >= 0.98),
            "tags": sorted(mine.tag.unique()) == [f"MEM/{prog}"],  # table tags (arm 1's ledger rows: MEM/MEM/N3)
        }
        res["arms"][arm] = {"gen": gdir.name, "checks": chk, "pass": all(chk.values()), "requests": len(mine),
                            "keys_shared_with_main": int(mine.request_key.isin(set(m.request_key)).sum()),
                            "attempts_gt1": int((mine.attempts > 1).sum()),
                            "paid": int(paid.sum()), "paid_usd_rows": float(mine.usd[paid].sum()),
                            "degraded": int((mine.degraded > 0).sum()), "weeks": len(ws), "manifest": man["counts"]}
    res["ledger"] = C.ledger_run_rows(RUN)
    res["pass"] = all(v["pass"] for v in res["arms"].values()) and res["ledger"]["usd"] + 1e-9 >= sum(
        v["paid_usd_rows"] for v in res["arms"].values())
    res["created_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    (EPI / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"pass": res["pass"], "arms": {k: {"checks": v["checks"], "paid": v["paid"]} for k, v in
                                                    res["arms"].items()}, "ledger": res["ledger"]["usd"]}, default=str))
    return 0 if res["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["replay", "dry", "run-epi", "run-work", "validate"])
    ap.add_argument("--weeks", type=int, default=None, help="run-work: at most this many weeks")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return {"replay": cmd_replay, "dry": cmd_dry, "run-epi": cmd_run_epi, "run-work": cmd_run_work,
            "validate": cmd_validate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
