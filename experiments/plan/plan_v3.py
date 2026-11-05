"""Planning model v3: run costs re-projected from DEV #1's measured tokens and the measured prompt caching
(docs/impl/dev2_ops.md §2; dev1_report §4, P-1 / RP-6). Planning estimates, NOT paper results: the paper's cost table
is regenerated from `requests.parquet` rows of the actual runs (CLAUDE.md rule 2).

What changes against plan_v1 (results/plan/plan_v1.json, design_v1 §13.2) and design_v2_delta §6:
- per-request input/output tokens and provider-cached shares of every S1 program are DEV #1's paid calls
  (results/judgments/dev1_emu_wdev10_s50/requests.parquet, 12,521 calls on 2021 W_dev weeks), instead of the
  prototype's o200k proxy counts;
- request rates are the plan's for W_clean (news 6.679 items per stock-week, P1 0.6, F-A 0.096 in a contiguous run),
  except F-B, which runs every week (1.0; DEV #1 and WS-B), and the reader programs (H, H-spot, H-guard, HS), which run
  on the 89.9% of W_clean stock-weeks with news;
- caching: four scenarios, because the 50-request DEV2OPS measurement (results/dev2_ops/prompt_cache_measure.json)
  and the concurrent n1 stream's keyed traffic did not show that the guide's routing reaches N1's ceiling:
    dev1_measured   every program at DEV #1's measured cached share (N1 32.8%)  <- the base case
    n1_burst        N1 at DEV2OPS's burst realization (arms A+B: 70.8% of the ceiling), the rest as DEV #1
    n1_ceiling      N1 at its ceiling (the 1,792-token cached prefix on every request: 85.3%), the rest as DEV #1
    uncached        no provider caching at all (the upper bound, e.g. a state-first N1 layout plus no other hits)
- rows whose unit costs are not S1-emulator requests (TA-lite, Mono-CoT, S2, faithfulness, fidelity, probes, live) keep
  their plan values and are marked `basis: plan`;
- N1 layout options of the n1 stream (docs/impl/dev2_n1.md) are priced from that stream's measured tokens when its
  arm files exist (experiments/dev/n1_fix/out/arms/*.jsonl, read-only, hashed into the output).

Prices: gpt-4.1-mini $0.40 in / $0.10 cached in / $1.60 out per 1M tokens (jevtrader.s1.pricing; docs/pricing/).

Run: .venv/bin/python experiments/plan/plan_v3.py  -> results/plan/plan_v3.json
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.s1.pricing import CACHED_INPUT, PRICES

MODEL = "gpt-4.1-mini-2025-04-14"
P_IN, P_OUT = (p / 1e6 for p in PRICES[MODEL])
P_CACHED = CACHED_INPUT[MODEL] / 1e6
PLAN_V1 = ROOT / "results/plan/plan_v1.json"
DEV1 = ROOT / "results/judgments/dev1_emu_wdev10_s50/requests.parquet"
MEASURE = ROOT / "results/dev2_ops/prompt_cache_measure.json"
N1_ARMS = ROOT / "experiments/dev/n1_fix/out/arms"
BACKENDS = ROOT / "configs/backends.yaml"
LEDGER = ROOT / "data/cache/spend.sqlite"
OUT = ROOT / "results/plan/plan_v3.json"

SW_CLEAN = 115 * 100  # W_clean x U100 stock-weeks
SW_MAIN = SW_CLEAN + 200  # plus the pseudo-forward weeks 09-18 and 09-25
S50, S30 = 115 * 50, 115 * 30
DECISION = ("N0", "N1", "N2", "N3", "T", "FB", "FA", "P1", "P2", "M")
# design_v2_delta §6: RB-5 H-spot = $16 = a clean U100 run (H's $8.9) + attacked units at H's $0.00077 per request
HSPOT_ATTACKED_REQUESTS = (16.0 - 8.9) / 0.00077
# plan values of the design_v2_delta §6 rows (plan_v1.json has none of them); with plan_v1's $329.6 they give ~$390
DELTA_PLAN = {"delta: RB-5 H-spot, clean U100 (tag HSPOT)": 8.9, "delta: RB-5 H-spot, attacked units (tag RT-S)": 7.1,
              "delta: RB-15 H-guard, clean U100 (tag HGUARD)": 8.5,
              "delta: RB-15 H-guard, ~4,500 attacked-unit requests (tag RT-S)": 3.5,
              "delta: LA-2 LAP on MAIN stock-weeks": 1.3, "delta: LA-3 option permutation (FIDX)": 2.2,
              "delta: RB-1, RB-2, RB-3, RB-11 attack families": 8.2, "delta: VC-3 content-free states (CF)": 0.1,
              "delta: IF-2 VERB-C1i": 3.5, "delta: RB-17 Mono-CoT under attack": 0.6,
              "delta: QB-9 reason-first TA-style prompt": 1.3, "delta: P1 extras (RB-10 10, other 5)": 15.0}
READERS = ("H", "H-spot", "H-guard", "HS")
SCENARIOS = ("dev1_measured", "n1_burst", "n1_ceiling", "uncached")


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def unit_profile(req: pd.DataFrame) -> dict:
    """Per program over DEV #1's paid calls: tokens per request, cached share, the cached-prefix size of a hit (the
    static prefix rounded down to 128 tokens: the mode of cached_tokens over hits) and the ceiling share."""
    out = {}
    for prog, g in req[~req["cached"]].groupby("program"):
        hits = g.loc[g["cached_tokens"] > 0, "cached_tokens"]
        prefix = int(hits.mode().iloc[0]) if len(hits) else 0
        out[prog] = {"paid_requests": len(g), "in_per_req": float(g["input_tokens"].mean()),
                     "out_per_req": float(g["output_tokens"].mean()),
                     "cached_share_dev1": float(g["cached_tokens"].sum() / g["input_tokens"].sum()),
                     "hit_rate_dev1": float((g["cached_tokens"] > 0).mean()), "cached_prefix_tokens": prefix,
                     "ceiling_share": min(1.0, prefix / float(g["input_tokens"].mean())) if prefix else 0.0,
                     "usd_per_paid_call_dev1": float(g["usd"].mean())}
    return out


def usd_req(p: dict, share: float) -> float:
    return p["in_per_req"] * ((1 - share) * P_IN + share * P_CACHED) + p["out_per_req"] * P_OUT


def shares(units: dict, scenario: str, r_burst: float) -> dict[str, float]:
    s = {k: v["cached_share_dev1"] for k, v in units.items()}
    if scenario == "n1_burst":
        s["N1"] = max(s["N1"], r_burst * units["N1"]["ceiling_share"])
    elif scenario == "n1_ceiling":
        s["N1"] = max(s["N1"], units["N1"]["ceiling_share"])
    elif scenario == "uncached":
        s = {k: 0.0 for k in s}
    return s


def main() -> dict:
    plan = json.loads(PLAN_V1.read_text())
    tp = plan["token_profile_by_request_clean"]
    news = plan["news_volume_measured"]
    req = pd.read_parquet(DEV1)
    units = unit_profile(req)
    # P2 did not run in DEV #1 (off by default): the plan's proxy tokens, uncached
    units["P2"] = {"paid_requests": 0, "in_per_req": float(tp["P2"]["emu_in_per_request"]),
                   "out_per_req": float(tp["P2"]["emu_out_per_request"]), "cached_share_dev1": 0.0, "hit_rate_dev1": 0.0,
                   "cached_prefix_tokens": 0, "ceiling_share": 0.0, "usd_per_paid_call_dev1": None, "basis": "plan_v1"}
    meas = json.loads(MEASURE.read_text()) if MEASURE.exists() else None
    r_burst = 0.0
    if meas:
        ab = [meas["arms"][a] for a in ("A", "B")]
        r_burst = sum(a["cached_tokens"] for a in ab) / sum(a["n"] * meas["static_prefix"]["dev1_cached_prefix_tokens"]
                                                            for a in ab)
    share_item = news["W_clean"]["share_with_item"]
    rates_clean = {"N0": tp["N0"]["rate_per_stock_week"], "N1": tp["N1"]["rate_per_stock_week"],
                   "N2": tp["N2"]["rate_per_stock_week"], "N3": tp["N3"]["rate_per_stock_week"], "T": 1.0, "FB": 1.0,
                   "FA": tp["FA"]["rate_per_stock_week"], "P1": tp["P1"]["rate_per_stock_week"],
                   "P2": tp["P2"]["rate_per_stock_week"], "M": tp["M"]["rate_per_stock_week"]}
    items_pre = news["W_pre_slice"]["items_per_stock_week"]
    rates_pre = {**rates_clean, "N0": items_pre, "N1": items_pre}

    def per_sw(rates: dict, sh: dict) -> dict:
        tin = sum(r * units[k]["in_per_req"] for k, r in rates.items())
        tout = sum(r * units[k]["out_per_req"] for k, r in rates.items())
        tc = sum(r * units[k]["in_per_req"] * sh[k] for k, r in rates.items())
        return {"requests": sum(rates.values()), "in": tin, "out": tout, "cached": tc,
                "usd": sum(r * usd_req(units[k], sh[k]) for k, r in rates.items())}

    # the dev1_report's check: DEV #1 tokens at the plan's own rates (F-B 0.327) -> $0.00835 per stock-week
    plan_rates = {**rates_clean, "FB": tp["FB"]["rate_per_stock_week"]}
    check = per_sw(plan_rates, shares(units, "dev1_measured", r_burst))

    rows_v1 = plan["cost_rows_usd"]
    spent_by_run: dict[str, float] = {}
    if LEDGER.exists():
        with sqlite3.connect(LEDGER) as c:
            for tag, usd in c.execute("SELECT tag, SUM(usd) FROM spend GROUP BY tag").fetchall():
                run = (tag or "").split("/", 1)[0]
                spent_by_run[run] = spent_by_run.get(run, 0.0) + float(usd)
    dev1_run = float(req.loc[~req["cached"], "usd"].sum())  # DEV #1's Layer-A run (the diagnostics excluded)
    dev2_side = sum(v for k, v in spent_by_run.items() if k.startswith("DEV2") and k != "DEV2")
    out_rows: dict[str, dict] = {}
    profiles = {}
    for sc in SCENARIOS:
        sh = shares(units, sc, r_burst)
        pc, pp = per_sw(rates_clean, sh), per_sw(rates_pre, sh)
        profiles[sc] = {"clean": pc, "pre_slice": pp}
        u = {k: usd_req(units[k], sh[k]) for k in units}
        h = {k: u[k] for k in READERS}
        n1, n0, n2, n3 = u["N1"], u["N0"], u["N2"], u["N3"]
        typed_attack = n0 + n1 + n2 + n3
        nodef_attack = n1 + 1500 * P_IN + 20 * P_OUT  # N1 on raw text + N3 on raw headlines (plan_v1's assumption)
        talite_sw = 52.56 / S50
        verb_n1 = units["N1"]["in_per_req"] * ((1 - sh["N1"]) * P_IN + sh["N1"] * P_CACHED) + 13 * 5 * 9 * P_OUT
        v3 = {
            "DEV: 2 x 500 W_dev stock-weeks (bank + prompt shake-down)":
                (spent_by_run.get("DEV1", 0.0) + max(dev1_run, spent_by_run.get("DEV2", 0.0)) + dev2_side,
                 "ledger: DEV1 actual + DEV2 as DEV #1's Layer-A run (or its actual, if larger) + DEV2* side streams"),
            "MAIN: W_clean 115 wk x U100 + 2 pseudo-forward weeks": (SW_MAIN * pc["usd"], "measured"),
            "W_pre slice: S30 x 60 wk blinded + unblinded; S30 W_clean unblinded":
                (2 * 30 * 60 * 0.95 * pp["usd"] + S30 * 0.95 * pc["usd"], "measured"),
            "Mono-S1 holistic (one request, same evidence), U100": (SW_CLEAN * share_item * h["H"], "measured"),
            "Headline sentiment (S1 Choice per headline), U100": (SW_CLEAN * share_item * h["HS"], "measured"),
            "TA-lite NL multi-agent trader, S50 (capped outputs)": (rows_v1["TA-lite NL multi-agent trader, S50 (capped outputs)"], "plan"),
            "Typed pipeline without defenses (clean), S50":
                (S50 * (rates_clean["N1"] * n1 + 0.9 * (1500 * P_IN)), "measured N1; plan N3-raw tokens"),
            "C1 static red-team: promotion 400 sw x A1-A7 x {typed, nodef, holistic, sentiment}":
                (400 * 7 * (typed_attack + nodef_attack + h["H"] + h["HS"]) * 1.15, "measured"),
            "C1 bound check (rank > K+J): 100 sw x A1-A7": (100 * 7 * (typed_attack + h["H"] + h["HS"]), "measured"),
            "C1 demotion: 200 held sw x {A2-neg, A6-neg, A8}": (200 * 3 * (typed_attack + h["H"] + h["HS"]), "measured"),
            "C1 TA-lite under attack: 100 S50 sw x 4 families": (100 * 4 * talite_sw, "plan"),
            "C1 adaptive black-box: 150 items x 64 queries, typed + holistic, proposer":
                (150 * 64 * (n0 + n1 + h["H"] + 500 * P_IN + 60 * P_OUT), "measured (proposer: plan tokens)"),
            "Faithfulness T2/T3 (+ null arms), TA-lite T4, Mono-S1 deletion": (8.0, "plan"),
            "Emulator fidelity V1 (one question per call) + V2 re-ask floor": (5.0, "plan"),
            "Probes D1 (monthly window rule + all weeks) / D2 re-identification": (3.0, "plan"),
            "Memory ablations: M-epi off (N3 re-ask), L0 off (T re-ask)": (SW_CLEAN * (0.84 * n3 + u["T"]), "measured"),
            "Live + shadow books (5 weeks)": (8.0, "plan"),
            "P1: S2 (gpt-4.1-mini CoT) + V verification, all S50 stock-weeks":
                (rows_v1["P1: S2 (gpt-4.1-mini CoT) + V verification, all S50 stock-weeks"], "plan"),
            "P1: verbalized backend on a 6,000-item N1 sample": (6000 * verb_n1, "measured N1 input; plan output"),
            "P1: Mono-CoT single LLM, S50": (8.8, "plan"),
            "P1: X-Web prevalence, 10k pre-cutoff items (N0 + guard Nouls)": (10_000 * (n0 + 500 * P_IN), "measured N0"),
            # design_v2_delta §6 additions
            "delta: RB-5 H-spot, clean U100 (tag HSPOT)": (SW_CLEAN * share_item * h["H-spot"], "measured"),
            "delta: RB-5 H-spot, attacked units (tag RT-S)": (HSPOT_ATTACKED_REQUESTS * h["H-spot"], "measured"),
            "delta: RB-15 H-guard, clean U100 (tag HGUARD)": (SW_CLEAN * share_item * h["H-guard"], "measured"),
            "delta: RB-15 H-guard, ~4,500 attacked-unit requests (tag RT-S)": (4500 * h["H-guard"], "measured"),
            "delta: LA-2 LAP on MAIN stock-weeks": (SW_MAIN * u["D"], "measured (D)"),
            "delta: LA-3 option permutation (FIDX)": (2.2, "plan"),
            "delta: RB-1, RB-2, RB-3, RB-11 attack families": (8.2, "plan"),
            "delta: VC-3 content-free states (CF)": (0.1, "plan"),
            "delta: IF-2 VERB-C1i": (3.5, "plan"),
            "delta: RB-17 Mono-CoT under attack": (0.6, "plan"),
            "delta: QB-9 reason-first TA-style prompt": (1.3, "plan"),
            "delta: P1 extras (RB-10 10, other 5)": (15.0, "plan"),
        }
        for name, (usd, basis) in v3.items():
            row = out_rows.setdefault(name, {"plan": rows_v1.get(name, DELTA_PLAN.get(name)),
                                             "plan_source": "plan_v1" if name in rows_v1 else "design_v2_delta §6",
                                             "basis": basis, "v3": {}})
            row["v3"][sc] = round(usd, 2)
    totals = {sc: round(sum(r["v3"][sc] for r in out_rows.values()), 1) for sc in SCENARIOS}
    main_rerun = {sc: round(SW_MAIN * profiles["uncached"]["clean"]["usd"], 1) for sc in SCENARIOS}

    # N1 layout options (the n1 stream's decision): measured tokens of its arm files, priced per N1 item
    layouts = None
    if N1_ARMS.exists() and any(N1_ARMS.glob("*.jsonl")):
        rows = []
        files = {}
        for f in sorted(N1_ARMS.glob("*.jsonl")):
            files[f.name] = sha(f)
            for line in f.read_text().splitlines():
                r = json.loads(line)
                rows.append({k: r.get(k) for k in ("arm", "group", "layout", "questions", "cached", "input_tokens",
                                                     "cached_tokens", "output_tokens")})
        d = pd.DataFrame(rows)
        d["nq"] = d["questions"].map(len)
        d["option"] = d["arm"].str.split(".").str[0]  # an arm's request groups, e.g. B = B.tri + B.attr
        paid = d[~d["cached"].astype(bool)]

        def hit_prefix(x):
            h = x[x > 0]
            return int(h.mode().iloc[0]) if len(h) else 0

        g = d.groupby(["option", "arm", "group", "layout", "nq"]).agg(
            n=("input_tokens", "size"), inp=("input_tokens", "mean"), out=("output_tokens", "mean")).reset_index()
        pre = paid.groupby(["option", "arm", "group"])["cached_tokens"].agg(hit_prefix).rename("cached_prefix")
        g = g.merge(pre.reset_index(), on=["option", "arm", "group"], how="left").fillna({"cached_prefix": 0})
        n1 = units["N1"]

        def item_usd(parts, share_fn) -> float:
            return sum(p["inp"] * ((1 - share_fn(p)) * P_IN + share_fn(p) * P_CACHED) + p["out"] * P_OUT for p in parts)

        opts = {"joint_qf_13": [{"inp": n1["in_per_req"], "out": n1["out_per_req"], "layout": "qf", "nq": 13,
                                 "cached_prefix": n1["cached_prefix_tokens"]}]}
        for opt in sorted(set(g["option"])):
            opts[f"option_{opt}"] = g[g["option"] == opt].to_dict("records")
        realization = n1["cached_share_dev1"] / n1["ceiling_share"] if n1["ceiling_share"] else 0.0

        def share_dev1(p):
            # every request realizes DEV #1's N1 fraction of its own ceiling: its cached prefix (measured on the n1
            # stream's hits, else the 13-question prefix scaled by its question count) over its input; a prefix below
            # 1,024 tokens and a state-first layout cache nothing
            if p.get("layout") != "qf":
                return 0.0
            prefix = p.get("cached_prefix") or (n1["cached_prefix_tokens"] * p["nq"] / 13.0) // 128 * 128
            return realization * prefix / p["inp"] if prefix >= 1024 else 0.0

        layouts = {"files": files, "realization_of_ceiling_dev1": realization,
                   "note": "options are the n1 stream's arms (experiments/dev/n1_fix; docs/impl/dev2_n1.md defines them); "
                           "each priced per N1 item at DEV #1's caching realization; requests of an option are its "
                           "request groups", "options": {}}
        for name, parts in opts.items():
            per_item = item_usd(parts, share_dev1)
            layouts["options"][name] = {
                "requests_per_item": len(parts), "tokens_in_per_item": round(sum(p["inp"] for p in parts), 1),
                "usd_per_item": per_item, "usd_per_item_uncached": item_usd(parts, lambda p: 0.0),
                "main_delta_usd_vs_joint": round(SW_MAIN * rates_clean["N1"] * (per_item - item_usd(opts["joint_qf_13"], share_dev1)), 2),
                "parts": [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in p.items()} for p in parts]}

    caps = yaml.safe_load(BACKENDS.read_text())["ledger"]["sub_caps_usd"]
    keymap = {"MAIN": "MAIN: W_clean 115 wk x U100 + 2 pseudo-forward weeks",
              "PRE": "W_pre slice: S30 x 60 wk blinded + unblinded; S30 W_clean unblinded",
              "H": "Mono-S1 holistic (one request, same evidence), U100",
              "HS": "Headline sentiment (S1 Choice per headline), U100",
              "NODEF": "Typed pipeline without defenses (clean), S50",
              "RT-A": "C1 adaptive black-box: 150 items x 64 queries, typed + holistic, proposer",
              "MEM": "Memory ablations: M-epi off (N3 re-ask), L0 off (T re-ask)",
              "VERB": "P1: verbalized backend on a 6,000-item N1 sample",
              "XWEB": "P1: X-Web prevalence, 10k pre-cutoff items (N0 + guard Nouls)",
              "HSPOT": "delta: RB-5 H-spot, clean U100 (tag HSPOT)",
              "HGUARD": "delta: RB-15 H-guard, clean U100 (tag HGUARD)",
              "LAP": "delta: LA-2 LAP on MAIN stock-weeks"}
    # redteam/run.py tags every static arm "RT-S/<arm>" (readers' attacked units included)
    rts = ("C1 static red-team: promotion 400 sw x A1-A7 x {typed, nodef, holistic, sentiment}",
           "C1 bound check (rank > K+J): 100 sw x A1-A7", "C1 demotion: 200 held sw x {A2-neg, A6-neg, A8}",
           "C1 TA-lite under attack: 100 S50 sw x 4 families", "delta: RB-1, RB-2, RB-3, RB-11 attack families",
           "delta: RB-5 H-spot, attacked units (tag RT-S)",
           "delta: RB-15 H-guard, ~4,500 attacked-unit requests (tag RT-S)")
    groups = {run: (row,) for run, row in keymap.items()} | {"RT-S": rts}
    subcap = {}
    for run, rows_of in groups.items():
        v = {sc: round(sum(out_rows[k]["v3"][sc] for k in rows_of), 2) for sc in SCENARIOS}
        cap = caps.get(run)
        rule = math.ceil(1.5 * v["dev1_measured"] / 0.05) * 0.05  # the config's rule: 1.5 x planned USD
        subcap[run] = {"sub_cap": cap, "v3": v, "fits_dev1_measured": cap is not None and v["dev1_measured"] <= cap,
                       "fits_uncached": cap is not None and v["uncached"] <= cap,
                       "rule_1.5x_v3": round(rule, 2), "recommended": round(max(cap or 0.0, rule), 2)}
    spent = {"total_usd": round(sum(spent_by_run.values()), 4),
             "by_run_usd": {k: round(v, 4) for k, v in sorted(spent_by_run.items())}} if spent_by_run else None

    out = {
        "status": "planning estimates, not paper results",
        "generated_by": "experiments/plan/plan_v3.py",
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in (PLAN_V1, DEV1, MEASURE, BACKENDS)},
        "prices_usd_per_1M": {"in": P_IN * 1e6, "cached_in": P_CACHED * 1e6, "out": P_OUT * 1e6, "model": MODEL},
        "unit_profile_dev1": units,
        "prompt_cache_measurement": None if meas is None else {
            "run_tag": meas["run_tag"], "n_requests": meas["n_requests"], "ledger_spend_usd": meas["ledger_spend_usd"],
            "arms": {a: {k: meas["arms"][a][k] for k in ("n", "cached_share", "hit_rate", "realization", "keys")}
                     for a in meas["arms"]},
            "burst_realization_A_plus_B": r_burst, "answers_agree_with_dev1": meas["answers_agree_with_dev1"]},
        "caching_scenarios": {sc: {"n1_cached_share": shares(units, sc, r_burst)["N1"],
                                   "decision_run_cached_share_clean": profiles[sc]["clean"]["cached"] / profiles[sc]["clean"]["in"]}
                              for sc in SCENARIOS},
        "rates_per_stock_week_clean": rates_clean, "reader_rate_per_stock_week_clean": share_item,
        "check_dev1_report_usd_per_sw_at_plan_rates": {"v3": round(check["usd"], 5), "dev1_report": 0.00835},
        "profile_per_stock_week": {sc: {w: {k: round(v, 5 if k == "usd" else 1) for k, v in p.items()}
                                        for w, p in profiles[sc].items()} for sc in SCENARIOS},
        "cost_rows_usd": out_rows,
        "total_usd": totals,
        "total_plan_usd": round(sum(r["plan"] for r in out_rows.values()), 1),
        "total_plan_v1_usd": plan["total_planned_usd"], "total_design_v2_delta_usd": 390,
        "total_x1.3_usd": {sc: round(1.3 * t, 0) for sc, t in totals.items()},
        "plus_one_uncached_MAIN_rerun_usd": {sc: round(1.3 * totals[sc] + main_rerun[sc], 0) for sc in SCENARIOS},
        "MAIN_requests": round(SW_MAIN * profiles["dev1_measured"]["clean"]["requests"]),
        "MAIN_hours_at_dev1_throughput": round(SW_MAIN * profiles["dev1_measured"]["clean"]["requests"] / 49.3 / 3600, 2),
        "n1_layout_options": layouts,
        "subcap_check": subcap,
        "ledger_spent": spent,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("check_dev1_report_usd_per_sw_at_plan_rates", "caching_scenarios", "total_usd",
                                          "total_x1.3_usd", "plus_one_uncached_MAIN_rerun_usd", "MAIN_requests",
                                          "MAIN_hours_at_dev1_throughput", "subcap_check")}, indent=1, default=float))
    print({sc: profiles[sc]["clean"]["usd"] for sc in SCENARIOS})
    return out


if __name__ == "__main__":
    main()
