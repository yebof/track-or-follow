"""FAITH driver (S4; `jevtrader.redteam.faith`): faithfulness interventions with null arms, sub-cap FAITH $3.00.

    .venv/bin/python experiments/faith/faith.py freeze          # module hash -> results/faith/spec.json
    .venv/bin/python experiments/faith/faith.py capture-t3      # T3 states: exact replays of MAIN's N3 (free)
    .venv/bin/python experiments/faith/faith.py project         # per-arm projection, cut-from-the-end, preflight
    .venv/bin/python experiments/faith/faith.py run --arm T3 [--mock]    # AFTER the commit: paid only if it fits
    .venv/bin/python experiments/faith/faith.py run --arm T4-TAL [--mock]
    .venv/bin/python experiments/faith/faith.py t2 [--run]       # AFTER the unlock: sample, dry run, preflight, run

Projection (free): T3 by a dry run of its exact requests on the captured states (the paid emulator's identity, the
real cache read-only, gates-open fake answers, calibrated tokens: `redteam.projection`); T4-TAL from the MEASURED cost
of the same stock-weeks' bull, bear, trader and risk calls in TAL's requests table (x 1.1 for `high`); T2 and T4-H
from MAIN's / DEV #2's measured per-request costs x their request counts (planning: their samples need learned
quantities, so they are drawn after the unlock; `t2` then dry-runs T2's exact requests and preflights them again).
The arms are cut from the end of the registered list until the rest fits the FAITH headroom (sub-cap minus the
ledger's FAITH spend); a `run` of an arm that was cut, or whose own preflight does not fit, aborts with exit 3.

Reads Layer-A tables and requests only (T3 states, TAL outputs, ledger unit costs); no return, label, IC or learned
coefficient before the unlock. `--mock` runs on the mock backend (no ledger, no cache writes) and writes under
results/faith/mock/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.redteam import faith as F  # noqa: E402
from jevtrader.redteam import projection as P  # noqa: E402

OUT = ROOT / "results" / "faith"
STATE_CACHE = ROOT / "data" / "cache" / "faith"  # captured W_clean N3 states (Layer A; not committed)
MAIN = ROOT / "results" / "judgments" / "main_emu_wclean_u100"
TAL = ROOT / "results" / "gens" / "tal_ta-lite_wclean-last60_s30"


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))


def ledger_unit_costs() -> dict:
    con = sqlite3.connect(f"file:{ROOT / 'data/cache/spend.sqlite'}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("select tag, usd from spend where tag like 'MAIN/N%' or tag like 'DEV2/H%'", con)
    finally:
        con.close()
    return {t: float(g["usd"].mean()) for t, g in df.groupby("tag")}


# ------------------------------------------------------------------------------------------------ freeze
def cmd_freeze(a) -> int:
    rec = {"faith_sha256": F.faith_sha256(), "version": F.FAITH_VERSION, "arms": list(F.ARMS), "sizes": F.SIZES,
           "seeds": F.SEEDS, "neutral_report": F.NEUTRAL_REPORT,
           "written_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    write_json(OUT / "spec.json", rec)
    print(f"FAITH {F.FAITH_VERSION}: sha256 {rec['faith_sha256']}")
    return 0


# ------------------------------------------------------------------------------------------------ T3 states
def main_world():
    import hashlib

    from jevtrader.programs.context import ContextBuilder, RealData
    from jevtrader.s1.types import canonical_json

    rec = json.loads((MAIN / "dev_run.json").read_text())
    data = RealData()
    data.fact_symbols = data.symbols_for_weeks([pd.Timestamp(w) for w in rec["weeks"]])
    builder = ContextBuilder(data, subset=None)
    cid = hashlib.sha256(canonical_json(builder.describe()).encode()).hexdigest()[:10]
    if cid != rec["context_id"]:
        raise SystemExit(f"context {cid} differs from MAIN's {rec['context_id']}")
    return data, builder


def cmd_capture_t3(a) -> int:
    req = pd.read_parquet(MAIN / "requests.parquet", columns=["t", "symbol", "program", "request_key"])
    order = F.sample_t3(req)
    order.to_parquet(OUT / "t3_order.parquet", index=False)
    head = order.head(int(a.n * a.oversample))
    data, builder = main_world()
    replay = P.projecting_backend("high")  # the real cache read-only; a miss gets a fake answer (then the key differs)
    t0 = time.time()

    def tick(week, n):
        print(f"  {week}: {n} exact states so far ({time.time() - t0:.0f} s)", flush=True)

    recs = asyncio.run(F.capture_t3(head, data.news_table, builder, replay, MAIN, a.n, on_week=tick))
    STATE_CACHE.mkdir(parents=True, exist_ok=True)
    with open(STATE_CACHE / "t3_states.pkl", "wb") as f:
        pickle.dump(recs, f)
    info = {"n_exact": len(recs), "n_tried": int(len(head)), "fake_calls_in_replay": replay.fake_calls,
            "real_cache_hits": replay.cache.real_hits, "weeks": int(pd.Series([r["t"] for r in recs]).nunique()),
            "order_file": "results/faith/t3_order.parquet", "wall_s": round(time.time() - t0, 1)}
    write_json(OUT / "t3_capture.json", info)
    print(json.dumps(info, indent=1))
    return 0 if len(recs) >= a.n else 2


def load_t3() -> list[dict]:
    p = STATE_CACHE / "t3_states.pkl"
    if not p.exists():
        raise SystemExit("no captured T3 states: run capture-t3 first")
    with open(p, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------------------------------------------ projection
def project_t3(n: int) -> F.ArmProjection:
    recs = load_t3()[:n]
    out = {}
    for mode in ("high", "point"):
        s1 = P.projecting_backend(mode)
        null = P.projecting_backend(mode, cache=False)
        rows = asyncio.run(F.ask_t3(recs, s1, null))
        out[mode] = float(sum(r.get(f"usd_{k}", 0.0) for r in rows for k in ("mirror", "complement", "null")))
        out[f"{mode}_requests"] = int(sum(1 for r in rows for k in ("mirror", "complement", "null")
                                          if f"key_{k}" in r and not r[f"cached_{k}"]))
    return F.ArmProjection("T3", round(out["high"], 4), round(out["point"], 4),
                           "dry run of the exact requests on the captured states (redteam.projection)", len(recs),
                           {"paid_requests_high": out["high_requests"]})


def project_t4_tal(n: int) -> F.ArmProjection:
    ans = pd.read_parquet(TAL / "answers.parquet")
    req = pd.read_parquet(TAL / "requests.parquet")
    out = pd.read_parquet(TAL / "outputs.parquet")
    smp = F.sample_t4_tal(ans, n)
    req["t"] = pd.to_datetime(req["t"])
    down = req[req["role"].isin(["bull", "bear", "trader", "risk"])]
    paid = down[~down["cached"].astype(bool)]
    mean_role = paid.groupby("role")["usd"].mean().to_dict()
    total, n_inv = 0.0, 0
    for u in smp.itertuples(index=False):
        g = down[(down["t"] == pd.Timestamp(u.t)) & (down["symbol"] == u.symbol)]
        per = sum(float(r.usd) if not r.cached else mean_role[r.role] for r in g.itertuples(index=False))
        rep = F.tal_reports(out, req, u.t, u.symbol)
        inv = F.invert_stance(rep.get("news", "")) is not None
        n_inv += inv
        total += per * (2 + inv)  # neutral + null always; invert when the stance is not neutral
    return F.ArmProjection("T4-TAL", round(total * 1.1, 4), round(total, 4),
                           "TAL's measured cost of the same stock-weeks' bull, bear, trader and risk calls x runs "
                           "(high = x 1.1)", len(smp), {"units_with_inversion": n_inv, "mean_role_usd": mean_role})


def project_planning(arm: str, unit: dict) -> F.ArmProjection:
    if arm == "T2":
        n3, n2 = unit["MAIN/N3"], unit["MAIN/N2"]
        per = 6 * n3 + 5 * 1.0 * n2  # 5 deletion runs + the null N3; one changed N2 per deletion run (upper side)
        return F.ArmProjection("T2", round(F.SIZES["T2"] * per * 1.1, 4), round(F.SIZES["T2"] * per, 4),
                               "planning: MAIN's measured USD per N3 and N2 request x (6 N3 + 5 N2) per stock-week; "
                               "dry run after the unlock", F.SIZES["T2"], {"usd_per_unit": per})
    h = unit.get("DEV2/H", 0.0)
    return F.ArmProjection("T4-H", round(F.SIZES["T4-H"] * 3 * h * 1.1, 4), round(F.SIZES["T4-H"] * 3 * h, 4),
                           "planning: DEV #2's measured USD per H request x 3 per stock-week; dry run after the unlock",
                           F.SIZES["T4-H"], {"usd_per_request": h})


def cmd_project(a) -> int:
    unit = ledger_unit_costs()
    arms = {"T3": project_t3(F.SIZES["T3"]), "T2": project_planning("T2", unit),
            "T4-TAL": project_t4_tal(F.SIZES["T4-TAL"]), "T4-H": project_planning("T4-H", unit)}
    st = P.ledger_state(F.RUN_TAG)
    headroom = (st["sub_cap"] or 0.0) - st["run_spent"]
    kept, cut = P.cut_from_end([(k, arms[k].high) for k in F.ARMS], headroom)
    lines, committed = [], 0.0
    for k in F.ARMS:
        pr = arms[k]
        pf = P.preflight(F.RUN_TAG, k, pr.high, pr.point, committed=committed)
        lines.append({**pf.to_json(), "basis": pr.basis, "units": pr.units, "kept": k in kept})
        ln = pf.line()
        if k not in kept:
            ln = ln.rsplit(" -> ", 1)[0] + " -> NOT RUN (cut from the end of the registered list)"
        print(ln, flush=True)
        if k in kept:
            committed += pr.high
    res = {"faith_sha256": F.faith_sha256(), "arms": {k: v.to_json() for k, v in arms.items()},
           "headroom": round(headroom, 4), "kept": kept, "cut": cut, "preflight": lines,
           "ledger": st, "calibration": P.Calibration.load().to_json(),
           "written_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")}
    write_json(OUT / "preflight.json", res)
    print(f"FAITH: headroom ${headroom:.2f}; runs {kept or 'nothing'}; not run (budget): {cut or 'none'}")
    return 0


# ------------------------------------------------------------------------------------------------ run
def cmd_run(a) -> int:
    if a.arm not in ("T3", "T4-TAL"):
        raise SystemExit(f"{a.arm}: its sample needs the unlock; run it from the post-unlock path (docs)")
    base = OUT / ("mock" if a.mock else "") / a.arm
    base.mkdir(parents=True, exist_ok=True)
    if not a.mock:
        pre = json.loads((OUT / "preflight.json").read_text())
        if a.arm not in pre["kept"]:
            raise SystemExit(f"ABORT {a.arm}: cut from the end by the FAITH preflight ({pre['cut']})")
        arm = pre["arms"][a.arm]
        committed = sum(pre["arms"][k]["high"] for k in pre["kept"][: pre["kept"].index(a.arm)]
                        if not (OUT / k / "summary.json").exists())
        pf = P.preflight(F.RUN_TAG, a.arm, arm["high"], arm["point"], committed=committed)
        print(pf.line(), flush=True)
        if not pf.ok:
            print(f"ABORT {a.arm}", flush=True)
            return 3
    t0 = time.time()
    if a.arm == "T3":
        recs = load_t3()[: a.n or F.SIZES["T3"]]
        if a.mock:
            from jevtrader.s1.backends import make_backend

            s1 = make_backend(backend="mock", cache=False, ledger=False)
            null = make_backend(backend="mock", cache=False, ledger=False)
        else:
            from jevtrader.s1.backends import make_backend

            ledger = P.config_ledger()
            guard = P.SharedGuard(pf.headroom)  # one budget for the intervention and the null requests
            s1 = guard.wrap(make_backend(backend="llm-logprob", cache=True, ledger=ledger))
            null = guard.wrap(make_backend(backend="llm-logprob", cache=False, ledger=ledger))
        rows = pd.DataFrame(asyncio.run(F.ask_t3(recs, s1, null)))
        rows.to_parquet(base / "rows.parquet", index=False)
        summ = {"arm": "T3", **F.t3_summary(rows), "mode": "mock" if a.mock else "paid",
                "spend_usd": float(rows.filter(like="usd_").sum().sum())}
    else:
        summ = run_t4_tal(a, base)
    summ["wall_s"] = round(time.time() - t0, 1)
    summ["faith_sha256"] = F.faith_sha256()
    write_json(base / "summary.json", summ)
    print(json.dumps(summ, indent=1, default=str))
    return 0


def run_t4_tal(a, base: Path) -> dict:
    from jevtrader.baselines.llm import GenLLM, _bank_prompt

    ans = pd.read_parquet(TAL / "answers.parquet")
    req = pd.read_parquet(TAL / "requests.parquet")
    out = pd.read_parquet(TAL / "outputs.parquet")
    smp = F.sample_t4_tal(ans, a.n or F.SIZES["T4-TAL"])
    p = _bank_prompt("ta-lite-v1")
    if a.mock:
        sys.path.insert(0, str(ROOT / "tests"))
        llm = GenLLM(client=_MockGenClient(), cache=False, tag="FAITH/T4-TAL")
        null = GenLLM(client=_MockGenClient(), cache=False, tag="FAITH/T4-TAL-null")
    else:
        pre = json.loads((OUT / "preflight.json").read_text())
        ledger = P.config_ledger()
        headroom = P.preflight(F.RUN_TAG, "T4-TAL", pre["arms"]["T4-TAL"]["high"], 0.0).headroom
        llm = GenLLM(ledger=ledger, cache=True, tag="FAITH/T4-TAL", budget_usd=headroom / 2)
        null = GenLLM(ledger=ledger, cache=False, tag="FAITH/T4-TAL-null", budget_usd=headroom / 2)
    ans["t"] = pd.to_datetime(ans["t"])
    orig = ans.set_index(["t", "symbol"])

    async def one(u):
        rep = F.tal_reports(out, req, u.t, u.symbol)
        o = orig.loc[(pd.Timestamp(u.t), u.symbol)]
        jobs = {"neutral": F.ta_downstream(p, {**rep, "news": F.NEUTRAL_REPORT}, llm),
                "null": F.ta_downstream(p, rep, null)}
        inv = F.invert_stance(rep.get("news", ""))
        if inv is not None:
            jobs["invert"] = F.ta_downstream(p, {**rep, "news": inv}, llm)
        res = dict(zip(jobs, await asyncio.gather(*jobs.values())))
        row = {"t": u.t, "symbol": u.symbol, "decision_orig": o["decision"], "p_orig": o["p_outperform"]}
        for k, r in res.items():
            row.update({f"decision_{k}": r["decision"], f"p_{k}": r["p"], f"usd_{k}": r["usd"]})
        return row

    async def all_():
        return [await one(u) for u in smp.itertuples(index=False)]

    rows = pd.DataFrame(asyncio.run(all_()))
    rows.to_parquet(base / "rows.parquet", index=False)
    priced = float(rows.filter(like="usd_").sum().sum())
    summ = {"arm": "T4-TAL", "n": int(len(rows)), "mode": "mock" if a.mock else "paid",
            "spend_usd": 0.0 if a.mock else priced}
    if a.mock:  # the canned client's fixed usage priced at list price: no call was made, no ledger row written
        summ["mock_priced_usd"] = priced
    for k in ("neutral", "invert", "null"):
        if f"decision_{k}" in rows:
            ok = rows[f"decision_{k}"].notna()
            summ[f"flip_{k}"] = float((rows.loc[ok, f"decision_{k}"] != rows.loc[ok, "decision_orig"]).mean())
            summ[f"n_{k}"] = int(ok.sum())
    return summ


# ------------------------------------------------------------------------------------------------ T2 (after the unlock)
def cmd_t2(a) -> int:
    """T2 after the unlock: the typed clean Layer-B run of MAIN (refuses before the unlock), the registered sample,
    a dry run of every unit (projection), the preflight against the FAITH headroom left after the arms before it, and
    with --run the paid run under one shared guard."""
    from jevtrader.pipeline.layer_b import DecisionConfig, evaluate, load_inputs, resolve_window
    from jevtrader.programs.news import card_evidence
    from jevtrader.redteam.run import CleanRun

    allow = tuple(x for x in a.allow_missing.split(",") if x)
    cfg = DecisionConfig.from_yaml(gen_id=MAIN.name, name="primary", allow_missing=allow)
    start, end = resolve_window("W_clean")
    inputs = load_inputs(cfg, start, end, ROOT)
    log = evaluate(cfg, "W_clean", inputs, root=ROOT, register=False, write=False).log
    cards = card_evidence(MAIN)
    counts = {(pd.Timestamp(d), s): int(c.get("screening", {}).get("n_items", 0)) for (d, s), c in cards.items()}
    smp = F.sample_t2(log, counts, a.n or F.SIZES["T2"])
    smp.to_parquet(OUT / "t2_sample.parquet", index=False)
    data, builder = main_world()
    typed = CleanRun("typed", cfg, log, inputs.raw)

    def rows_with(s1, null):
        async def go():
            return [await F.t2_unit(u.t, u.symbol, cards[(str(pd.Timestamp(u.t).date()), u.symbol)],
                                    data.news_table, builder, s1, null, MAIN, typed)
                    for u in smp.itertuples(index=False)]
        return pd.DataFrame(asyncio.run(go()))

    proj = {}
    for mode in ("high", "point"):
        r = rows_with(P.projecting_backend(mode), P.projecting_backend(mode, cache=False))
        proj[mode] = float(r.filter(like="usd_").sum().sum())
    pre = json.loads((OUT / "preflight.json").read_text())
    committed = sum(pre["arms"][k]["high"] for k in F.ARMS[: F.ARMS.index("T2")]
                    if not (OUT / k / "summary.json").exists())
    pf = P.preflight(F.RUN_TAG, "T2 (dry run)", proj["high"], proj["point"], committed=committed)
    print(pf.line(), flush=True)
    write_json(OUT / "T2" / "projection.json", {"projection": proj, "preflight": pf.to_json(), "n": len(smp)})
    if not a.run:
        return 0
    if not pf.ok:
        print("ABORT T2", flush=True)
        return 3
    from jevtrader.s1.backends import make_backend

    ledger = P.config_ledger()
    guard = P.SharedGuard(pf.headroom)
    rows = rows_with(guard.wrap(make_backend(backend="llm-logprob", cache=True, ledger=ledger)),
                     guard.wrap(make_backend(backend="llm-logprob", cache=False, ledger=ledger)))
    (OUT / "T2").mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUT / "T2" / "rows.parquet", index=False)
    summ = {"arm": "T2", **F.t2_summary(rows), "spend_usd": guard.spent, "faith_sha256": F.faith_sha256()}
    write_json(OUT / "T2" / "summary.json", summ)
    print(json.dumps(summ, indent=1, default=str))
    return 0


class _MockGenClient:
    """A canned OpenAI-like client for the T4-TAL mock run (no network): valid trader and risk JSON."""

    def __init__(self):
        from types import SimpleNamespace

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, model, messages, temperature, max_completion_tokens):
        import hashlib
        from types import SimpleNamespace

        h = int(hashlib.sha256(messages[-1]["content"].encode()).hexdigest(), 16)
        dec = ("BUY", "HOLD", "SELL")[h % 3]
        text = json.dumps({"reason": "mock", "decision": dec, "p_outperform": round((h % 100) / 100, 2),
                           "cited": [], "action": "approve"})
        return SimpleNamespace(model=model, usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20,
                                                                  prompt_tokens_details=None),
                               choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    p = sub.add_parser("capture-t3")
    p.add_argument("--n", type=int, default=F.SIZES["T3"])
    p.add_argument("--oversample", type=float, default=1.15)
    sub.add_parser("project")
    p = sub.add_parser("t2")
    p.add_argument("--run", action="store_true")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--allow-missing", default="", help="Layer-B input checks waived for the typed clean run")
    p = sub.add_parser("run")
    p.add_argument("--arm", required=True, choices=F.ARMS)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--n", type=int, default=0)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return {"freeze": cmd_freeze, "capture-t3": cmd_capture_t3, "project": cmd_project, "run": cmd_run,
            "t2": cmd_t2}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
