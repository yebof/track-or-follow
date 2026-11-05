#!/usr/bin/env python3
"""D2 re-identification rates (prereg §8; design_v1 §13.7 D2), from the PROBE generation's raw answers.

    .venv/bin/python experiments/analysis/d2_reid.py                          # registered: both periods
    .venv/bin/python experiments/analysis/d2_reid.py --window W_pre_slice     # the W_pre period only (development)

**Input.** `probe_emu_d1d2` (experiments/run_gen_probe.py): jobs.parquet (one row per D2 request: period, blinding
level, source, the true registrant name in `meta_json`) and answers.parquet (by request key and stage). Questions:
`D.reid_company` (the true name + 4 same-FF12 adjacent-size decoys + cannot_tell; chance .2 among the five names) on
every state, and `D.reid_year` (y2019..y2026 + cannot_tell; chance 1/8) on every N1 state.

**Statistics** per (period, level, question), per (period, level, source, question), and pooled over periods per
(level, question): n; accuracy = share of states whose top answer is the true label (the true registrant name; for
the year, "y" + the calendar year of the decision date t), with its Wilson 95% CI and the exact one-sided binomial p
against chance; the share answering cannot_tell; the mean probability on the true label. Descriptive; no test enters
a family. The paper's D3 statement is allowed only after these rates are reported (§8).

**Lock.** D2 reads no market outcome, but its W_clean states are part of the confirmatory window, so the rates of
W_clean rows are computed only after the unlock like every other W_clean statistic (common.Guard on their dates).
Output: results/analysis/d2.json (registered) or results/analysis/<window>/d2.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats as S

NAME = "d2"
DEFAULT_PROBE = "probe_emu_d1d2"
CHANCE = {"reid_company": 0.2, "reid_year": 1 / 8}
CANNOT = "cannot_tell"
PERIODS = {"W_pre_slice": "W_pre", C.REGISTERED_WINDOW: None}  # None = both periods


def load_rows(gen_dir: Path) -> pd.DataFrame:
    """One row per D2 request: period, level, source, question, t, symbol, true label and the answer's probs."""
    jobs = pd.read_parquet(Path(gen_dir) / "jobs.parquet", columns=["uid", "request_key", "stage", "t", "symbol",
                                                                    "meta_json"])
    jobs = jobs[jobs["stage"].astype(str).str.startswith("D2")].copy()
    meta = [json.loads(x) for x in jobs["meta_json"]]
    jobs["period"] = [m["period"] for m in meta]
    jobs["level"] = [m["level"] for m in meta]
    jobs["source"] = [m["source"] for m in meta]
    jobs["question"] = ["reid_year" if str(s).startswith("D2y") else "reid_company" for s in jobs["stage"]]
    jobs["true_name"] = [m.get("true_name") for m in meta]
    # identical states (T and F-B carry no free text, so their `names` state is the `full` one) share a request key
    # and were answered once: join on the key and the question, not the stage
    ans = pd.read_parquet(Path(gen_dir) / "answers.parquet", columns=["request_key", "qid", "probs_json"])
    ans = ans.drop_duplicates(["request_key", "qid"]).rename(columns={"qid": "question"})
    j = jobs.merge(ans, on=["request_key", "question"], how="left")
    j["t"] = pd.to_datetime(j["t"])
    j["true"] = [f"y{t.year}" if q == "reid_year" else n for q, t, n in zip(j["question"], j["t"], j["true_name"])]
    j["probs"] = [json.loads(x) if isinstance(x, str) else None for x in j["probs_json"]]
    return j.drop(columns=["probs_json", "meta_json"])


def score_rows(j: pd.DataFrame) -> pd.DataFrame:
    top = [max(p, key=p.get) if p else None for p in j["probs"]]
    return j.assign(top=top, correct=[t is not None and t == y for t, y in zip(top, j["true"])],
                    cannot_tell=[t == CANNOT for t in top],
                    p_true=[float((p or {}).get(y, 0.0)) if p else np.nan for p, y in zip(j["probs"], j["true"])],
                    answered=[p is not None for p in j["probs"]])


def cell(g: pd.DataFrame, question: str) -> dict:
    from scipy import stats as st

    g = g[g["answered"]]
    n, k = len(g), int(g["correct"].sum())
    chance = CHANCE[question]
    return {"n": n, "correct": k, "accuracy": k / n if n else None, "ci": S.wilson(k, n) if n else None,
            "chance": chance, "p_vs_chance_one_sided": float(st.binomtest(k, n, chance, "greater").pvalue) if n
            else None, "cannot_tell_share": float(g["cannot_tell"].mean()) if n else None,
            "mean_p_true": float(g["p_true"].mean()) if n else None}


def summarize(s: pd.DataFrame) -> dict:
    out = {"by_period_level": [], "by_period_level_source": [], "pooled_by_level": []}
    for (per, lev, q), g in s.groupby(["period", "level", "question"]):
        out["by_period_level"].append({"period": per, "level": lev, "question": q, **cell(g, q)})
    for (per, lev, src, q), g in s.groupby(["period", "level", "source", "question"]):
        out["by_period_level_source"].append({"period": per, "level": lev, "source": src, "question": q,
                                              **cell(g, q)})
    for (lev, q), g in s.groupby(["level", "question"]):
        out["pooled_by_level"].append({"level": lev, "question": q, **cell(g, q)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", default=DEFAULT_PROBE)
    ap.add_argument("--window", default=C.REGISTERED_WINDOW, help="W_clean: both periods; W_pre_slice: W_pre only")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    d = Path(a.probe) if Path(a.probe).is_dir() else root / "results/judgments" / a.probe
    rows = load_rows(d)
    if a.window not in PERIODS:
        raise SystemExit(f"D2: window must be one of {sorted(PERIODS)}")
    if PERIODS[a.window]:
        rows = rows[rows["period"] == PERIODS[a.window]]
    guard.check(rows["t"].unique(), "D2 rates")  # W_clean rows only after the unlock (module doc)
    s = score_rows(rows)
    res = {**summarize(s), "n_requests": int(len(s)), "n_unanswered": int((~s["answered"]).sum()),
           "definitions": {"accuracy": "top answer = the true label; cannot_tell counts as wrong",
                           "true_year": "'y' + the calendar year of the decision date t", "chance": CHANCE}}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, res, script=__file__, window=a.window, guard=guard,
                   inputs={"jobs": d / "jobs.parquet", "answers": d / "answers.parquet"}, argv=argv)
    print(f"D2 {a.window}: {len(s)} requests -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
