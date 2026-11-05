#!/usr/bin/env python3
"""D1 window-start rule (prereg §7.2; design_v1 §2.5, adapted to the reduced probe).

    .venv/bin/python experiments/analysis/d1_window_start.py --schedule        # the probe schedule (no outcome)
    .venv/bin/python experiments/analysis/d1_window_start.py                   # the rule (after the unlock; reads
                                                                               # the registered PROBE generations)
    .venv/bin/python experiments/analysis/d1_window_start.py --window W_pre_slice --center 2023-07   # development

**Schedule (seeds P-5).** For every month of the probe period (2024-01..2025-06): the 2 weekly decisions of the month
with the smallest sha256("D1-www27|" + ISO date), and at each of them the 10 names of U_t with the smallest
sha256("D1-www27" + symbol) (`pos` = the name's place 0..9 in that order). A weekly decision is the last session of
an ISO week on the FULL session calendar (built through the end of W_clean), and it belongs to the month of its own
date: a week that straddles two months gives one decision, in the month of its last session, never a month-end
pseudo-decision cut at the period edge. `--schedule` writes the (t, symbol) list; it reads only the session calendar
and the universe. The same schedule over the W_pre slice's months (decisions inside the slice) is the W_pre run.

**Statistic.** P(beat) = the `yes` probability of `D.recall_beats_median` (probe program D, stage D1, unblinded, no
evidence) against the realized T1 label y = 1[r > m_t] of the same stock-week (learn.labels.LabelStore over U_t).
For each month m, the AUC over the pooled 3-month window {m-1, m, m+1} (about 60 probes in 6 decision weeks) and
its two-sided NAME-PERMUTATION test of AUC = 0.5 (stats.name_permutation_auc: one permutation of the 10 name
positions applied in every week of the window, B = 10,000, seed 0, p = (1 + #{|AUC_b - .5| >= |AUC - .5|}) /
(1 + B)). The window "differs from 0.5" iff p < ALPHA = .05, and "covers 0.5" otherwise. Why a permutation test: the
window has only about 6 week clusters and the same 10 names every week, and a percentile week-block bootstrap
interval over 6 clusters is too narrow (audit 2026-09-28: 15.3% of null windows excluded 0.5 at a nominal 5%); the
name permutation keeps each week's cross-section and each name's persistence and is exact under the null that
P(beat) carries no information about the realized labels (null false-move rate by simulation:
experiments/analysis/sim_null_rates.py). The 95% week-block bootstrap interval (B = 10,000, seed 0) is still written
for each window, as a description only; it decides nothing. A window at the edge of the probe period pools the
months that exist and is flagged.

**Rule.** If the window centred on 2024-07 differs from 0.5 (permutation p < .05), the confirmatory window starts at
the first decision of the first month m >= 2024-07 whose window covers 0.5 (p >= .05); otherwise at the first
decision of W_clean (2024-07-05). A window whose AUC is undefined (one class only) covers nothing. If no month up to
the end of the probe period covers 0.5, the rule is EXHAUSTED: the start is the first weekly decision of the month
after the probe period (2025-07-03, from the full calendar) and the output says so. If the probe was not run (P-3),
the rule is not applied, W_clean starts on 2024-07-05, and the output says so.

**Probe answers (registered).** The union of the PROBE generation `probe_emu_d1d2` and its re-ask of the scheduled
probes it had asked on month-end pseudo-dates instead, `probe_emu_d1_reask` (experiments/run_gen_probe_reask.py);
a (t, symbol) answered twice with different values stops the script. **Completeness.** Every scheduled probe must
have an answer; otherwise the script stops before reading any label (pass --allow-missing to compute on the answered
probes, which the output records; never for the registered run). Answers off the schedule are ignored and listed.

**Timing.** Windows touching 2024-06 onward read W_clean labels: the rule runs right after the commit and before any
H1-H3 statistic; the lock check covers every probed decision date. Output: results/analysis/d1.json (read by H2, H3
and the report scripts through common.d1_start).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NAME = "d1"
SEED_TEXT = "D1-www27"
WEEKS_PER_MONTH, NAMES_PER_WEEK = 2, 10
PROBE_MONTHS = ("2024-01", "2025-06")  # registered probe period
CENTER = "2024-07"  # the rule's reference month
N_BOOT, N_PERM, SEED = 10_000, 10_000, 0
ALPHA = 0.05  # a window differs from 0.5 iff its name-permutation p < ALPHA
PROGRAM, STAGE, QID = "D", "D1", "recall_beats_median"
REGISTERED_PROBES = ("probe_emu_d1d2", "probe_emu_d1_reask")  # the PROBE generation and its re-ask (module doc)


def month_key(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m")


def months_between(a: str, b: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(a, b, freq="M")]


def schedule(calendar, universe, months) -> pd.DataFrame:
    """The registered probe schedule (module doc): t, symbol, month, pos (the name's place in the hash order).
    `calendar` must be true weekly decisions (`rule_calendar`), not a calendar cut at a month edge."""
    months = set(months)
    by_m: dict[str, list[pd.Timestamp]] = {}
    for t in calendar:
        m = month_key(t)
        if m in months:
            by_m.setdefault(m, []).append(pd.Timestamp(t))
    rows = []
    for m, ts in sorted(by_m.items()):
        pick = sorted(ts, key=lambda d: hashlib.sha256(f"{SEED_TEXT}|{d.date().isoformat()}".encode()).hexdigest())
        for t in sorted(pick[:WEEKS_PER_MONTH]):
            names = sorted(universe[t], key=lambda s: hashlib.sha256(f"{SEED_TEXT}{s}".encode()).hexdigest())
            rows += [{"t": t, "symbol": s, "month": m, "pos": k} for k, s in enumerate(names[:NAMES_PER_WEEK])]
    return pd.DataFrame(rows, columns=["t", "symbol", "month", "pos"])


def probe_answers(gen_dir: Path) -> pd.DataFrame:
    """P(beat) per (t, symbol) from a probe generation's answers.parquet."""
    a = pd.read_parquet(Path(gen_dir) / "answers.parquet")
    q = a[(a["program"] == PROGRAM) & (a["stage"] == STAGE)
          & (a["qid"].astype(str).str.replace("D.", "", regex=False) == QID)].copy()
    q["p"] = [json.loads(x).get("yes") if isinstance(x, str) else np.nan for x in q["probs_json"]]
    q["t"] = pd.to_datetime(q["t"])
    return q[["t", "symbol", "p"]].drop_duplicates(["t", "symbol"], keep="last")


def union_answers(gen_dirs) -> pd.DataFrame:
    """The answers of several probe generations; a (t, symbol) answered with different values in two stops."""
    parts = [probe_answers(d).assign(source=Path(d).name) for d in gen_dirs]
    a = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["t", "symbol", "p", "source"])
    both = a[a.duplicated(["t", "symbol"], keep=False)]
    if len(both):
        spread = both.groupby(["t", "symbol"])["p"].agg(lambda x: float(np.nanmax(x) - np.nanmin(x)))
        if (spread > 1e-12).any():
            raise SystemExit(f"D1: {int((spread > 1e-12).sum())} probes answered differently in two generations")
    return a.drop_duplicates(["t", "symbol"], keep="first").reset_index(drop=True)


def window_auc(frame: pd.DataFrame, months: list[str], n_boot: int = N_BOOT, seed: int = SEED,
               n_perm: int = N_PERM) -> dict:
    """AUC of p vs y over the rows of `months`, its name-permutation p (the rule's test) and a descriptive
    week-block bootstrap interval."""
    import stats as S
    from jevtrader.eval.metrics import auc

    g = frame[frame["month"].isin(months)].dropna(subset=["p", "y"])
    weeks = sorted(g["t"].unique())
    est = auc(g["p"], g["y"]) if len(g) else float("nan")
    out = {"months": months, "n": len(g), "weeks": len(weeks), "positives": int(g["y"].sum()) if len(g) else 0,
           "auc": est, "p_perm": float("nan"), "ci": [float("nan"), float("nan")]}
    if len(weeks) < 2 or not np.isfinite(est):
        return out
    pt = S.name_permutation_auc(g, "t", "pos", "p", "y", n_perm=n_perm, seed=seed)
    out["p_perm"], out["n_perm"] = pt["p_two_sided"], n_perm
    bs = S.week_block_bootstrap_xy(g, "t", "p", "y", auc, n_boot=n_boot, seed=seed)
    out["ci"] = bs["ci"]  # descriptive only
    out["n_boot"], out["seed"] = n_boot, seed
    return out


def monthly_windows(frame: pd.DataFrame, months: list[str], n_boot: int = N_BOOT, seed: int = SEED,
                    n_perm: int = N_PERM) -> list[dict]:
    rows = []
    for i, m in enumerate(months):
        pool = months[max(0, i - 1): i + 2]
        r = window_auc(frame, pool, n_boot, seed, n_perm)
        r["center"] = m
        r["edge"] = len(pool) < 3
        rows.append(r)
    return rows


def covers(window: dict, alpha: float = ALPHA) -> bool | None:
    """True if the window's AUC is not detectably different from 0.5 (name-permutation p >= alpha), False if it is,
    None if undefined."""
    p = window.get("p_perm")
    if p is None or not np.isfinite(p):
        return None
    return bool(p >= alpha)


def apply_rule(windows: list[dict], calendar, center: str = CENTER, default_start=None) -> dict:
    """The start-month rule (module doc) on the monthly windows."""
    by = {w["center"]: w for w in windows}
    cal = sorted(pd.Timestamp(t) for t in calendar)
    def first_of(m: str):
        return next((t for t in cal if month_key(t) == m), None)

    default_start = pd.Timestamp(default_start) if default_start is not None else first_of(center)
    c = by.get(center)
    if c is None:
        return {"applied": False, "start": default_start, "reason": f"no probe window centred on {center}"}
    cov = covers(c)
    if cov is None:
        return {"applied": False, "start": default_start, "reason": f"the {center} window's test is undefined",
                "center_window": c}
    if cov:
        return {"applied": True, "moved": False, "start": default_start, "center_window": c,
                "reason": f"the {center} window covers 0.5 (permutation p >= {ALPHA})"}
    later = [w for w in windows if w["center"] >= center]
    for w in later:
        if covers(w):
            return {"applied": True, "moved": w["center"] != center, "start": first_of(w["center"]),
                    "start_month": w["center"], "center_window": c,
                    "reason": f"the {center} window differs from 0.5 (permutation p < {ALPHA}); first covering "
                              f"month {w['center']}"}
    last = later[-1]["center"] if later else center
    nxt = (pd.Period(last, "M") + 1).strftime("%Y-%m")
    start = first_of(nxt)
    if start is None:
        raise ValueError(f"D1: the calendar has no decision in {nxt}; build it past the probe period (rule_calendar)")
    return {"applied": True, "moved": True, "exhausted": True, "start": start, "start_month": nxt,
            "center_window": c, "reason": f"no window from {center} to {last} covers 0.5 (rule exhausted)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", default=C.REGISTERED_WINDOW, help="W_clean: the rule; another window: AUCs only")
    ap.add_argument("--months", default=None, help="probe months 'YYYY-MM:YYYY-MM' (default: the registered period "
                                                   "for W_clean, the window's months otherwise)")
    ap.add_argument("--probe", nargs="*", default=list(REGISTERED_PROBES),
                    help="probe generation ids or dirs (default: the registered PROBE generation and its re-ask; "
                         "'' = PROBE not run)")
    ap.add_argument("--center", default=CENTER)
    ap.add_argument("--schedule", action="store_true", help="write the probe schedule only (no outcome)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--allow-missing", action="store_true",
                    help="compute the rule even if scheduled probes lack an answer (recorded in the output)")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    if a.months:
        m0, m1 = a.months.split(":")
    elif a.window == C.REGISTERED_WINDOW:
        m0, m1 = PROBE_MONTHS
    else:
        w0, w1 = C.resolve_window(a.window, root)
        m0, m1 = month_key(w0), month_key(w1)
    months = months_between(m0, m1)
    lo, hi = pd.Timestamp(f"{m0}-01"), pd.Period(m1, "M").end_time.normalize()
    if a.window != C.REGISTERED_WINDOW and not a.months:  # a development window: decisions inside the window only
        lo = max(lo, C.resolve_window(a.window, root)[0])
    w_clean0 = C.windows(root)["W_clean"][0]
    cal = rule_calendar(root, lo)
    uni = _universe(root, cal)
    sched = schedule(cal, uni, months)
    if a.schedule:
        path = C.out_path(f"{NAME}_schedule", a.window, a.out_dir, root)
        C.write_result(path, {"months": months, "n": len(sched), "seed": SEED_TEXT,
                              "rule": f"{WEEKS_PER_MONTH} decisions per month by sha256('{SEED_TEXT}|'+ISO date); "
                                      f"{NAMES_PER_WEEK} names of U_t by sha256('{SEED_TEXT}'+symbol)",
                              "schedule": sched.assign(t=sched["t"].dt.date.astype(str)).to_dict("records")},
                       script=__file__, window=a.window, guard=guard, inputs={"universe": root / C.UNIVERSE_FILE},
                       argv=argv)
        print(f"D1 schedule: {len(sched)} probes over {len(months)} months -> {C.rel(path, root)}")
        return 0
    refs = [x for x in (a.probe or []) if x]
    probe_dirs = [Path(x) if Path(x).is_dir() else root / "results/judgments" / x for x in refs]
    absent = [str(d) for d in probe_dirs if not (d / "answers.parquet").exists()]
    if refs and absent:
        raise SystemExit(f"D1: probe generation(s) without answers: {absent}")
    if not refs:
        rule = {"applied": False, "start": w_clean0, "reason": "PROBE not run (P-3): the rule is not applied; W_clean "
                                                               "starts on its first decision, W_release stays the "
                                                               "registered look-ahead sensitivity"}
        payload = {"rule": rule, "schedule": {"months": months, "n": len(sched)}}
        path = C.out_path(NAME, a.window, a.out_dir, root)
        C.write_result(path, payload, script=__file__, window=a.window, guard=guard, argv=argv)
        print(f"D1: not applied ({rule['reason'][:40]}...) -> {C.rel(path, root)}")
        return 0
    ans = union_answers(probe_dirs)
    frame = sched.merge(ans[["t", "symbol", "p"]], on=["t", "symbol"], how="left")
    missing = int(frame["p"].isna().sum())
    extra = ans.merge(sched[["t", "symbol"]], on=["t", "symbol"], how="left", indicator=True)
    extra = extra[extra["_merge"] == "left_only"]
    if missing and not a.allow_missing:  # checked on judgments only, before any label is read
        by_m = frame[frame["p"].isna()].groupby("month").size().to_dict()
        raise SystemExit(f"D1: {missing} scheduled probes have no answer (by month {by_m}); the probe holds "
                         f"{len(extra)} answers off the registered schedule (e.g. "
                         f"{sorted({str(t.date()) for t in extra['t']})[:6]}). Re-ask the scheduled probes, or pass "
                         "--allow-missing to compute the rule on the answered ones (recorded in the output)")
    guard.check(frame["t"].unique(), "D1 labels")
    px = C.load_prices(guard, lo, hi)
    labels = C.load_labels(px, sorted(frame["t"].unique()), {t: uni[t] for t in sorted(frame["t"].unique())})
    frame = frame.merge(labels.table[["t", "symbol", "y"]], on=["t", "symbol"], how="left")
    wins = monthly_windows(frame, months, a.n_boot, SEED, a.n_perm)
    payload = {"schedule": {"months": months, "n": len(sched), "answers_missing": missing,
                            "allow_missing": bool(a.allow_missing),
                            "extra_answers_outside_schedule": int(len(extra)),
                            "extra_dates": sorted({str(t.date()) for t in extra["t"]})},
               "monthly_windows": wins,
               "settings": {"n_perm": a.n_perm, "n_boot": a.n_boot, "seed": SEED, "alpha": ALPHA,
                            "test": "two-sided name-permutation test of AUC = 0.5 (stats.name_permutation_auc)",
                            "interval": "week-block bootstrap percentile interval (descriptive only)",
                            "center": a.center, "pool": "{m-1, m, m+1}", "price_cap": px.cap}}
    if a.window == C.REGISTERED_WINDOW or a.center != CENTER:
        payload["rule"] = apply_rule(wins, cal, a.center, w_clean0 if a.center == CENTER else None)
    else:
        payload["rule"] = {"applied": False, "start": None, "reason": "AUCs only (not the registered window)"}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard,
                   inputs={**{f"probe answers {d.name}": d / "answers.parquet" for d in probe_dirs},
                           "universe": px.root / C.UNIVERSE_FILE}, argv=argv)
    print(f"D1 {a.window}: rule {payload['rule'].get('reason')}; start {payload['rule'].get('start')} -> "
          f"{C.rel(path, root)}")
    return 0


def rule_calendar(root: Path, lo) -> list[pd.Timestamp]:
    """The weekly decisions from `lo` through the end of W_clean (configs): true ISO-week ends only, so no month is
    offered a pseudo-decision cut at the probe period's edge, and the month after the probe period has its first
    decision (the exhausted rule's start)."""
    return _calendar(root, lo, C.windows(root)["W_clean"][1])


def _calendar(root: Path, lo, hi) -> list[pd.Timestamp]:
    """Weekly decision dates from the session calendar (the date column of the ETF bars; no price is read)."""
    from jevtrader.backtest.engine import weekly_decision_dates

    f = root / C.BARS_FILES[1]
    if not f.exists():
        f = C.ROOT / C.BARS_FILES[1]
    b = pd.read_parquet(f, columns=["symbol", "date"])
    ses = pd.DatetimeIndex(sorted(pd.to_datetime(b.loc[b["symbol"] == "SPY", "date"]).unique()))
    return weekly_decision_dates(ses, str(pd.Timestamp(lo).date()), str(pd.Timestamp(hi).date()))


def _universe(root: Path, cal) -> dict:
    from jevtrader.pipeline.layer_b import universe_by_date

    f = root / C.UNIVERSE_FILE
    if not f.exists():
        f = C.ROOT / C.UNIVERSE_FILE
    return universe_by_date(pd.read_parquet(f), cal)


if __name__ == "__main__":
    sys.exit(main())
