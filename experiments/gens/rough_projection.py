"""Rough cost projections of the reduced VERB, PROBE, FID and MEM generations from measured ledger costs per paid
request (MAIN and DEV #2 tags) and the request counts their reduced designs imply. Planning numbers for the budget
decision only: a paid launch of any of these runs still needs its own free dry-run projection (as TAL and MCOT had).
Nothing here reads a W_clean outcome.

    .venv/bin/python experiments/gens/rough_projection.py --out results/gens/rough_projection.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
P_OUT = 1.60e-6  # gpt-4.1-mini-2025-04-14 output, USD per token (s1/pricing.py)
P_IN = 0.40e-6


def per_request(c, tag: str) -> dict:
    n, usd, i, o = c.execute("SELECT COUNT(*), SUM(usd), SUM(input_tokens), SUM(output_tokens) FROM spend WHERE tag = ?",
                             (tag,)).fetchone()
    return {"tag": tag, "rows": n, "usd_per_request": usd / n, "in_per_request": i / n, "out_per_request": o / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/gens/rough_projection.json"))
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/backends.yaml").read_text())
    c = sqlite3.connect(f"file:{ROOT / cfg['ledger']['path']}?mode=ro", uri=True)
    m = {k: per_request(c, f"MAIN/{k}") for k in ("N1", "N3", "T")}
    d = {k: per_request(c, f"DEV2/{k}") for k in ("D", "D2")}
    live = json.loads((ROOT / "experiments/dev/n1_fix/out/cost_plan.json").read_text())["plans"]["live"]["n1_item"]
    subcaps = cfg["ledger"]["sub_caps_usd"]
    res = {"basis": {"MAIN": m, "DEV2": d, "n1_item_live": {k: live[k] for k in ("requests", "in", "out")}}}

    # VERB: 2,000 N1 items, verbalized (verb-v1): the logprob item cost plus ~9 output tokens per option
    # (plan_v1.py's pilot figure): stage-1 single-question Nouls (2 options) and the 13-question stage-2 request
    # (plan_v1: 13 x 5 options), each replacing the logprob answer's measured output
    logprob_item = live["requests"] * m["N1"]["usd_per_request"]
    stage2 = [g for g in live["groups"] if len(g["ask"]) > 1][0]
    extra_out = sum(g["weight"] * (2 * 9 - g["out"]) for g in live["groups"] if len(g["ask"]) == 1) + \
        stage2["weight"] * (13 * 5 * 9 - stage2["out"])
    verb_item = logprob_item + extra_out * P_OUT
    stage2_only = stage2["in"] * P_IN + 13 * 5 * 9 * P_OUT
    res["VERB"] = {"items": 2000, "usd_per_item_all_requests": round(verb_item, 6),
                   "projected_all_requests": round(2000 * verb_item, 2),
                   "projected_stage2_only": round(2000 * stage2_only, 2), "sub_cap": subcaps["VERB"]}

    # PROBE: D1 18 months x 10 names x 2 weeks = 360 units; D2 600 states; requests per unit from DEV #2
    # (D: 1,010 requests; D2: 999 requests on DEV #2's 500 stock-weeks -> ~2 each)
    d1_req, d2_req = 360 * 2, 600 * 2
    res["PROBE"] = {"D1_requests": d1_req, "D2_requests": d2_req,
                    "projected": round(d1_req * d["D"]["usd_per_request"] + d2_req * d["D2"]["usd_per_request"], 2),
                    "sub_cap": subcaps["PROBE"]}

    # FID: V1 on 150 stock-weeks: T's 6 questions one per call; N1's stage-2 13 questions one per call for every
    # asked item (MAIN: 6.16 asked items per stock-week, stage-2 rate 0.757); V2: 100 requests x 3 repeats
    t_single = 6 * (m["T"]["in_per_request"] - 5 * 60) * P_IN + 6 * 4 * P_OUT  # ~60 tokens per extra question text
    n1_single = 13 * (stage2["in"] - 12 * 110) * P_IN + 13 * 3 * P_OUT  # ~110 tokens per extra question text
    v1_t, v1_n1 = 150 * t_single, 150 * 6.16 * 0.757 * n1_single
    v2 = 300 * m["N1"]["usd_per_request"]
    res["FID"] = {"V1_T": round(v1_t, 2), "V1_N1_all_items": round(v1_n1, 2), "V2": round(v2, 2),
                  "projected": round(v1_t + v1_n1 + v2, 2),
                  "projected_one_item_per_stock_week": round(v1_t + 150 * n1_single + v2, 2), "sub_cap": subcaps["FID"]}

    # MEM: S30 x 115 W_clean weeks: M-epi off = N3 re-asked without priors (MAIN 0.611 N3 per stock-week; the
    # state loses its M-epi history, priced at PRE's first-weeks N3 cost as a floor and MAIN's mean as a ceiling);
    # M-work off = T re-asked without the blackboard market field
    sw = 30 * 115
    n3 = sw * 0.611
    res["MEM"] = {"stock_weeks": sw, "N3_requests": round(n3), "T_requests": sw,
                  "projected_low": round(n3 * per_request(c, "PRE/B/N3")["usd_per_request"] * 0.8
                                         + sw * m["T"]["usd_per_request"], 2),
                  "projected": round(n3 * m["N3"]["usd_per_request"] + sw * m["T"]["usd_per_request"], 2),
                  "sub_cap": subcaps["MEM"]}
    c.close()
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "basis"}, indent=1))


if __name__ == "__main__":
    main()
