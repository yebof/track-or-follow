#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E8: stress tests of the timing and reference-reader findings of E6/E7, and two
frame sensitivities of the registered H2 result ($0; local CPU; no model API call).

    .venv/bin/python experiments/exploratory/e8_timing_robustness.py   # -> results/exploratory/e8_timing_robustness.json

Why exploratory. Specified on 2026-09-29 after the unlock, after the registered H2 outcome, after E6/E7, and after the
round-2 paper reviews (docs/reviews/paper_v2_r2_ac.md W3, W8; paper_v2_r2_devil.md B1, M5, M6, m1), which had printed
read-only diagnostics for several of these cells (docs/reviews/paper_v2_r2_devil_checks.py D1-D5). Nothing here decides
a registered claim: every output is a descriptive estimate with a two-sided 95% interval; no p-value is computed; no
configuration, threshold or reader is chosen from it.

Frame. The registered H2 frame, rebuilt read-only exactly as E6 does (e6_publication_timing.build, which checks the
weekly statistic against results/analysis/h2.json to 1e-12), with E6's publication-aligned criterion AR_pub (same
function, same cached vendor minute bars) and E5's price-report classifier of record (the regex).

1. Reference readers against three criteria. For emu's E[effect], FinBERT tone (E7's scores: P(positive) -
   P(negative)) and Loughran-McDonald tone (E5's lm_tone: minus the net-negative share): the weekly statistic (mean over
   weeks of the Spearman correlation, as H2) against AR[0,+1], the placebo AR[-5,-1] and AR_pub; the paired weekly
   difference AR[0,+1] minus placebo per reader; and the paired weekly differences emu - FinBERT and emu - dictionary on
   each criterion, on the units both readers scored. NW(4) two-sided 95% intervals.
2. Timestamps. (a) SEC clock: for frame items whose top event type is "earnings results", the nearest Item 2.02 8-K of
   the focal firm accepted within [created_at - 2 days, created_at + 1 day] (the registered 8-K label window); the
   share of items created after that acceptance and the median signed gap (item minus filing, minutes). SEC
   acceptance times do not come from the news vendor. (b) Edits: the judged text is the item's last version before the
   decision cutoff, whereas E6 dates publication by the vendor's created_at; the share of frame items whose updated_at
   exceeds created_at by more than 1 and more than 60 minutes. (c) The after-close-or-non-session timing group of E6:
   the share created on a non-session day, and the median minutes from the last regular close to the item for those
   created after the close of a session (the group's AR[0] starts at that close).
3. Byline classes. The FEED-1 classes (jevtrader.data.provenance.byline_class with the question bank's desk bylines):
   desk, named contributor, other (wire, e-mail, organization or empty byline). Per class: share of frame units,
   price-report share, the weekly statistic against AR[0,+1] and against AR_pub (undefined for a class too small to
   give weekly correlations).
4. Frame sensitivities of the registered statistic (reviews W8/m1: the registered frame is selected by emu's own
   answers). Weekly statistic of E[effect] vs AR[0,+1] on (a) every item-firm pair of the window (from the D1 start)
   that has an effect answer, kept or not and of any event type; (b) the kept pairs of every event type, including
   "stock activity or opinion"; (c) (a) without price-report items.

Every weekly statistic is h2_validity.weekly_test's statistic (at least 4 units per week); intervals are two-sided
Newey-West(4) 95% (stats.nw_test). Texts are read from the licensed news table and never written to results/.
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
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402  (registered, read-only)
import e5_construct_checks as E5  # noqa: E402  (exploratory frame, text, classifier, dictionary; read-only)
import e6_publication_timing as E6  # noqa: E402  (exploratory frame and publication-aligned criterion; read-only)
import e7_reference_readers as E7  # noqa: E402  (FinBERT scores path; read-only)
import h2_validity as H2  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402  (registered, read-only)

NAME = "e8_timing_robustness"
GEN = "main_emu_wclean_u100"
READERS = {"emu": "E_effect", "finbert": "tone", "dictionary": "lm_pos"}
CRITERIA = {"ar_0_1": "ar", "placebo": "ar_placebo", "ar_pub": "ar_pub"}
EARNINGS = "earnings_results"
WINDOW_BEFORE, WINDOW_AFTER = pd.Timedelta(days=2), pd.Timedelta(days=1)  # registered 8-K label window (h2 sec_labels)
EDIT_MIN = (1, 60)
CLASS_GROUPS = {"desk": ("desk",), "named": ("named",), "other": ("wire", "other_email", "other_org", "other_empty")}


def nw(x: np.ndarray | pd.Series) -> dict:
    r = S.nw_test(pd.Series(x, dtype=float).dropna().to_numpy(), S.NW_LAGS, "two-sided")
    return {"mean": r["mean"], "ci": r["ci"], "se": r["se"], "n_weeks": int(r["n"])}


def weekly(f: pd.DataFrame, x: str, y: str) -> pd.Series:
    return S.weekly_spearman(f.dropna(subset=[x, y]), x, y, "t", H2.MIN_UNITS).set_index("t")["rho"]


def wk(f: pd.DataFrame, x: str = "E_effect", y: str = "ar") -> dict:
    return {**nw(weekly(f, x, y)), "n_units": int(f.dropna(subset=[x, y]).shape[0])}


def paired(f: pd.DataFrame, x1: str, x2: str, y: str) -> dict:
    h = f.dropna(subset=[x1, x2, y])
    d = (weekly(h, x1, y) - weekly(h, x2, y)).dropna()
    return {**nw(d), "n_units": int(len(h)), "definition": f"mean over weeks of rho({x1}, {y}) - rho({x2}, {y}), "
                                                          "same units; NW(4) two-sided"}


def main_minus_placebo(f: pd.DataFrame, x: str) -> dict:
    h = f.dropna(subset=[x, "ar", "ar_placebo"])
    return nw((weekly(h, x, "ar") - weekly(h, x, "ar_placebo")).dropna())


# ------------------------------------------------------------------------------------------------ 1. readers
def readers(g: pd.DataFrame) -> dict:
    out: dict = {"per_reader": {}, "paired_differences": {}}
    for r, x in READERS.items():
        out["per_reader"][r] = {c: wk(g, x, y) for c, y in CRITERIA.items()}
        out["per_reader"][r]["main_minus_placebo"] = main_minus_placebo(g, x)
        out["per_reader"][r]["coverage_units"] = int(g[x].notna().sum())
    for other in ("finbert", "dictionary"):
        out["paired_differences"][f"emu_minus_{other}"] = {c: paired(g, "E_effect", READERS[other], y)
                                                           for c, y in CRITERIA.items()}
    return out


# ------------------------------------------------------------------------------------------------ 2. timestamps
def sec_clock(root: Path, g: pd.DataFrame) -> dict:
    from jevtrader.data.earnings import load_8k_items

    sub = root / C.SUBMISSIONS
    gaps = []
    h = g[g["top_type"] == EARNINGS]
    for sym, hs in h.groupby("symbol"):
        k = load_8k_items(sub, sym)
        if not len(k):
            continue
        acc = k[k["items"].map(lambda xs: "2.02" in xs)]["accepted_utc"]
        ts = np.sort(acc.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]"))
        cs = hs["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
        for c in cs:
            w = ts[(ts >= c - WINDOW_BEFORE.to_timedelta64()) & (ts <= c + WINDOW_AFTER.to_timedelta64())]
            if len(w):
                gaps.append((c - w[np.argmin(np.abs(w - c))]) / np.timedelta64(1, "m"))
    gaps = np.asarray(gaps, dtype=float)
    after = gaps[gaps > 0]
    return {"earnings_units": int(len(h)), "units_with_2_02_in_window": int(len(gaps)),
            "share_created_after_acceptance": float((gaps > 0).mean()) if len(gaps) else None,
            "median_gap_minutes_signed": float(np.median(gaps)) if len(gaps) else None,
            "median_gap_minutes_among_after": float(np.median(after)) if len(after) else None,
            "share_within_60_min_after": float(((gaps > 0) & (gaps <= 60)).mean()) if len(gaps) else None,
            "definition": "earnings-type frame units; nearest Item 2.02 8-K accepted in [created - 2d, created + 1d]; "
                          "gap = created_at - acceptance (minutes)"}


def news_meta() -> pd.DataFrame:
    nt = pd.read_parquet(E5.NEWS, columns=["id", "created_at", "updated_at", "author"]).drop_duplicates("id")
    nt["eid"] = ["news:" + hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in nt["id"]]
    return nt.drop_duplicates("eid")[["eid", "updated_at", "author"]]


def edits(m: pd.DataFrame) -> dict:
    lag = (pd.to_datetime(m["updated_at"], utc=True) - pd.to_datetime(m["created_at"], utc=True)).dt.total_seconds() / 60
    ok = lag.notna()
    return {"matched_share": float(ok.mean()),
            **{f"share_edited_gt_{k}_min": float((lag[ok] > k).mean()) for k in EDIT_MIN},
            "definition": "updated_at - created_at of the news table's record (the judged text is the last version "
                          "before the cutoff; E6 dates publication by created_at)"}


def after_close_group(g: pd.DataFrame, sessions) -> dict:
    h = g[g["timing"] == "after_close_or_nonsession"]
    ses = pd.DatetimeIndex(sessions)
    et = h["created_at"].dt.tz_convert(C.ET)
    day = et.dt.tz_localize(None).dt.normalize()
    on_session = day.isin(ses)
    after = h[on_session.to_numpy()]
    close = pd.to_datetime(after["created_at"].dt.tz_convert(C.ET).dt.strftime("%Y-%m-%d") + " " + E6.REG_CLOSE)
    mins = (after["created_at"].dt.tz_convert(C.ET).dt.tz_localize(None) - close).dt.total_seconds() / 60
    return {"n_units": int(len(h)), "share_nonsession_day": float((~on_session).mean()) if len(h) else None,
            "share_after_session_close": float(on_session.mean()) if len(h) else None,
            "median_minutes_after_close": float(mins.median()) if len(mins) else None,
            "ar_pub_after_session_close": wk(after, y="ar_pub"),
            "ar_0_1_after_session_close": wk(after)}


# ------------------------------------------------------------------------------------------------ 3. bylines
def bylines(g: pd.DataFrame) -> dict:
    from jevtrader.data.provenance import byline_class, desk_bylines

    desk = desk_bylines()
    cls = g["author"].fillna("").map({a: byline_class(a, desk) for a in g["author"].fillna("").unique()})
    out = {}
    for name, members in CLASS_GROUPS.items():
        h = g[cls.isin(members).to_numpy()]
        out[name] = {"share_units": float(len(h) / len(g)), "n_units": int(len(h)),
                     "price_report_share": float(h["price"].mean()) if len(h) else None,
                     "ar_0_1": wk(h), "ar_pub": wk(h, y="ar_pub")}
    out["matched_author_share"] = float(g["author"].notna().mean())
    return out


# ------------------------------------------------------------------------------------------------ 4. frames
def frames(root: Path, guard, px, d1: dict) -> dict:
    spec, _ = E5.load_spec()
    start, end = guard.check_window("W_clean", "E8 frame sensitivity")
    it = H2.item_rows(root / "results" / "judgments" / GEN, start, end)
    it = it[it["t"] >= pd.Timestamp(d1["start"])].reset_index(drop=True)
    guard.check(it["t"].unique(), "E8 frame sensitivity labels")
    eff = it[np.isfinite(it["E_effect"].to_numpy(dtype=float))].reset_index(drop=True)
    eff = H2.add_abnormal_returns(eff, px.book)
    eff = E5.classify(E5.attach_text(eff), spec)
    kept = eff[eff["kept"]]
    reg = H2.c2_frame(it)
    return {"pairs_in_window": int(len(it)), "pairs_with_effect_answer": int(len(eff)),
            "kept_pairs": int(it["kept"].sum()), "registered_frame_units": int(len(reg)),
            "all_with_effect": wk(eff), "kept_all_types": wk(kept),
            "all_with_effect_no_price": wk(eff[~eff["price"]]),
            "share_price_all_with_effect": float(eff["price"].mean()),
            "definition": "(a) every item-firm pair from the D1 start with an effect answer, kept or not, any event "
                          "type; (b) kept pairs of every type; (c) (a) without price-report items (E5 regex)"}


# ------------------------------------------------------------------------------------------------ main
def cmd_run(root: Path, argv) -> int:
    guard = C.Guard(root)
    g, px, reg, start, end, d1 = E6.build(root, guard)
    g = g.merge(E6.publication_aligned(g, px.book), on="unit", how="left")
    sc = pd.read_csv(E7.SCORES).drop_duplicates("eid")
    sc["tone"] = sc["p_positive"] - sc["p_negative"]
    g = g.merge(sc[["eid", "tone"]], on="eid", how="left")
    g["lm_pos"] = E5.lm_tone(g)
    g = g.merge(news_meta(), on="eid", how="left")
    check = weekly(g, "E_effect", "ar").mean()
    if abs(check - reg["confirmatory"]["mean"]) > E6.TOL:
        raise SystemExit(f"E8: frame does not reproduce h2.json ({check})")
    payload = {"question": "stress tests of the publication-timing and reference-reader findings; frame sensitivities "
                           "of the registered H2 statistic",
               "frame": {"n_units": int(len(g)), "n_weeks": int(g["t"].nunique()), "h2_mean_recomputed": float(check),
                         "h2_mean_registered": reg["confirmatory"]["mean"],
                         "ar_pub_units": int(g["ar_pub"].notna().sum())},
               "readers": readers(g),
               "timestamps": {"field": "created_at of the vendor record (publication clock of E6)",
                              "sec_clock": sec_clock(root, g), "edits": edits(g),
                              "after_close_group": after_close_group(g, px.book.sessions)},
               "bylines": bylines(g),
               "frame_sensitivity": frames(root, guard, px, d1),
               "settings": {"nw_lags": S.NW_LAGS, "ci": "two-sided NW 95%", "min_units_per_week": H2.MIN_UNITS,
                            "readers": READERS, "criteria": CRITERIA, "edit_thresholds_min": list(EDIT_MIN),
                            "byline_groups": {k: list(v) for k, v in CLASS_GROUPS.items()}}}
    inputs = {"work.jsonl": root / "results/judgments" / GEN / "memory" / "work.jsonl",
              "h2": root / "results/analysis/h2.json", "finbert scores": E7.SCORES, "news table": E5.NEWS,
              "e5 spec": E5.SPEC}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs, argv=argv,
                why="stress tests specified after the H2 outcome, E6/E7 and the round-2 paper reviews")
    rd = payload["readers"]
    for r in READERS:
        v = rd["per_reader"][r]
        print(f"{r:10s} " + "  ".join(f"{c} {v[c]['mean']:+.4f} {[round(x, 3) for x in v[c]['ci']]}" for c in CRITERIA))
    for k, v in rd["paired_differences"].items():
        print(f"{k:22s} " + "  ".join(f"{c} {v[c]['mean']:+.4f} {[round(x, 3) for x in v[c]['ci']]}" for c in CRITERIA))
    print("sec clock", payload["timestamps"]["sec_clock"])
    print("edits", payload["timestamps"]["edits"])
    print("after-close group", {k: v for k, v in payload["timestamps"]["after_close_group"].items()
                                if not isinstance(v, dict)})
    for k, v in payload["bylines"].items():
        if isinstance(v, dict):
            print(f"byline {k:8s} share {v['share_units']:.4f} price {v['price_report_share']} "
                  f"ar {v['ar_0_1']['mean']} pub {v['ar_pub']['mean']}")
    fs = payload["frame_sensitivity"]
    print(f"frames: all-with-effect {fs['pairs_with_effect_answer']} {fs['all_with_effect']['mean']:+.4f}; kept "
          f"{fs['kept_pairs']} {fs['kept_all_types']['mean']:+.4f}; no price {fs['all_with_effect_no_price']['mean']:+.4f}")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("run",), nargs="?", default="run")
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    return cmd_run(Path(a.root), argv)


if __name__ == "__main__":
    sys.exit(main())
