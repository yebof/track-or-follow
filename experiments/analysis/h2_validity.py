#!/usr/bin/env python3
"""H2 (code C2): concept validity of typed item judgments, graded by later market reactions (prereg §3.2).

    .venv/bin/python experiments/analysis/h2_validity.py                       # registered: MAIN on W_clean
    .venv/bin/python experiments/analysis/h2_validity.py --frame-only          # the item frame only (no outcome)
    .venv/bin/python experiments/analysis/h2_validity.py --gen pre_blinded_emu_wpre_s30 --window W_pre_slice

**Item frame (Layer-A answers only; allowed before the unlock).** Every N01 item row of the generation's M-work
blackboard (`N01.items.<symbol>` in memory/work.jsonl) whose decision t lies in the window and that is C2-eligible by
the implementation's own answers: `status.kept` (P(about_focal) >= .5, P(new_development) >= .5, not quarantined,
attributes answered) and the top `event_type` (argmax; ties to the first label in bank order) is not
`stock_activity_or_opinion`. One row per (t, focal symbol, evidence id), the rule of experiments/ng/items500.py.
E[effect] = sum_k k P(effect = k) over the five levels 0..4.

**Label.** AR[0,+1] = sum over sessions d0, d0+1 of r_i - beta_i r_SPY (learn.labels.abnormal_returns): d0 is the session
of `created_at` if before 16:00 ET, else the next session; beta_i the trailing-252-session beta ending the session
before the window; close-to-close adjusted returns. Placebo AR[-5,-1] with the same function. Evaluation only.

**Confirmatory statistic.** For each week t, the Spearman correlation (average ranks) of E[effect] with AR[0,+1] over
that week's units (weeks with fewer than MIN_UNITS pairs or a constant side are undefined and counted); the mean over
weeks; one-sided (greater) Newey-West(4) t with a t(n-1) reference. Units sharing a (focal, d0) share their label and
stay separate rows. The window starts at the D1 rule's start (results/analysis/d1.json, prereg §7.2).

**Reported alongside (never in the Holm family).** Placebo AR[-5,-1]; controls: items created before 09:30 ET only,
the prior-look stock-weeks excluded (results/prereg/prior_look_stock_weeks.csv); sensitivities: W_release, and the
first 13 weeks dropped as a sub-series from the date common.drop_start (2024-10-04 for W_clean, from the configs,
unchanged by the D1 rule). Secondary: 8-K item agreement (AUC, Brier, ECE, resolution first; eval.metrics.score_table):
P(earnings_results) vs an Item 2.02 8-K of the focal firm accepted in [created - 2 days, created + 1 day];
P(deal_or_investment) vs Items 1.01/2.01; P(management_or_labor) vs Item 5.02; and `vs_expectations`: among units whose
top answer is `above` or `below`, P(above) / (P(above) + P(below)) vs 1[AR[0,+1] > 0]. Agent level (VC-14): sentinel
h (risk.parquet) vs the T3 tail event of the same decision's holding week (AUC).

**Per implementation (VC-13; descriptive).** For NG (experiments/ng/items500.py: items.parquet, or checkpoint.jsonl
while it runs) and VERB (experiments/run_gen_verb.py: answers.parquet + jobs.parquet + sample.parquet; `verb_items`),
on the implementation's sample units: pooled Spearman of its E[effect] with AR (week-block bootstrap CI and Fisher-z
CI; about 4.5 units per week make weekly means unestimable), the same statistic for emu on the same units,
emu-vs-implementation agreement (Spearman of E[effect]; Cohen's kappa of top event types with the agreement share's
CI), the implementation's own C2 eligibility rate, and 8-K agreement on its units. An implementation's own C2
eligibility is the production triage (programs.news.triage) on ITS answers with emu's N0 decision (NG and VERB read
the N1 state MAIN built from emu's N0 decision), and its top event type is not stock_activity_or_opinion. The
registered call reads NG `ng_typed-local_wclean_items500` and VERB `verb_llm-verbalized_wclean_items2000`;
`--frame-only` also parses both implementations (Layer-A answers only) and writes their counts and a SHA-256 of the
parsed item table.

Lock: the frame is built from judgments; before any price, 8-K or risk label is read, the lock is checked on every
decision date of the frame (common.Guard).
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

NAME = "h2"
DEFAULT_GEN = "main_emu_wclean_u100"
DEFAULT_NG = "ng_typed-local_wclean_items500"
DEFAULT_VERB = "verb_llm-verbalized_wclean_items2000"
VERB_GROUPS = ("about_focal", "instructs_reader", "promotional", "new_development", "attributes")
ITEMS_KEY = "N01.items."  # programs.news.ITEMS_KEY
EXCLUDED_TYPE = "stock_activity_or_opinion"
MIN_UNITS = 4  # a week's Spearman needs at least this many finite pairs (the power analysis's n_t >= 4)
ALTERNATIVE = "greater"
AR_WINDOW, PLACEBO_WINDOW = (0, 1), (-5, -1)
PRE_OPEN = "09:30"  # control: items created before 09:30 ET only
SEC_WINDOW = (pd.Timedelta(days=2), pd.Timedelta(days=1))  # [created - 2 days, created + 1 day]
SEC_CONCEPTS = {  # event_type option -> 8-K item codes (design_v1 §13.1 C2 secondary)
    "earnings_results": ("2.02",),
    "deal_or_investment": ("1.01", "2.01"),
    "management_or_labor": ("5.02",),
}
N_BOOT, SEED = 2000, 0
DROP_FIRST = 13


# ------------------------------------------------------------------------------------------------ frame
def top_label(probs: dict | None) -> str | None:
    return None if not probs else max(probs, key=probs.get)  # first maximal key: bank order


def expected_effect(probs: dict | None) -> float:
    if not probs:
        return float("nan")
    return float(sum(int(k) * float(p) for k, p in probs.items()))


def item_rows(gen_dir: Path, start, end) -> pd.DataFrame:
    """Every N01 item row of the generation's blackboard with t in [start, end] (Layer-A answers only)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows, seen, dup = [], set(), 0
    tag = f'"key": "{ITEMS_KEY}'
    with open(Path(gen_dir) / "memory" / "work.jsonl") as f:
        for line in f:
            if tag not in line:
                continue
            rec = json.loads(line)["record"]
            if not rec["key"].startswith(ITEMS_KEY):
                continue
            t = pd.Timestamp(rec["t"][:10])
            if not (start <= t <= end):
                continue
            sym = rec["key"][len(ITEMS_KEY):]
            for r in rec["value"]:
                uid = (t, sym, r["eid"])
                if uid in seen:
                    dup += 1
                    continue
                seen.add(uid)
                st = r.get("status") or {}
                et = r.get("event_type")
                vs = r.get("vs_expectations")
                rows.append({"t": t, "symbol": sym, "eid": r["eid"], "created_at": r.get("created_at"),
                             "kept": bool(st.get("kept")), "top_type": top_label(et),
                             "E_effect": expected_effect(r.get("effect")),
                             "p_type": et or {}, "vs": vs or {}})
    df = pd.DataFrame(rows, columns=["t", "symbol", "eid", "created_at", "kept", "top_type", "E_effect", "p_type",
                                     "vs"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df.attrs["duplicates_skipped"] = dup
    return df


def c2_frame(items: pd.DataFrame) -> pd.DataFrame:
    """The C2-eligible units (module doc)."""
    ok = items["kept"] & items["top_type"].notna() & (items["top_type"] != EXCLUDED_TYPE)
    f = items[ok].copy()
    f["unit"] = [f"{t.date()}|{s}|{e}" for t, s, e in zip(f["t"], f["symbol"], f["eid"])]
    return f.reset_index(drop=True)


def frame_counts(f: pd.DataFrame, items: pd.DataFrame | None = None) -> dict:
    n_t = f.groupby("t").size()
    out = {"units": len(f), "weeks": int(n_t.size),
           "units_per_week": {"min": int(n_t.min()), "median": float(n_t.median()), "max": int(n_t.max())}
           if len(n_t) else {}, "symbols": int(f["symbol"].nunique())}
    if items is not None:
        out["items_in_window"] = len(items)
        out["kept"] = int(items["kept"].sum())
        out["duplicates_skipped"] = int(items.attrs.get("duplicates_skipped", 0))
    return out


# ------------------------------------------------------------------------------------------------ labels
def add_abnormal_returns(f: pd.DataFrame, book) -> pd.DataFrame:
    from jevtrader.learn.labels import abnormal_returns

    ev = f[["symbol", "created_at"]]
    a = abnormal_returns(book, ev, window=AR_WINDOW)
    p = abnormal_returns(book, ev, window=PLACEBO_WINDOW)
    out = f.copy()
    out["d0"] = pd.to_datetime(a["d0"])
    out["ar"] = a["ar"].to_numpy(dtype=float)
    out["ar_placebo"] = p["ar"].to_numpy(dtype=float)
    return out


def sec_labels(f: pd.DataFrame, sub_dir: Path) -> pd.DataFrame:
    """1[an 8-K of the focal firm reporting one of the concept's items was accepted in [created - 2d, created + 1d]]."""
    from jevtrader.data.earnings import load_8k_items

    out = f.copy()
    times: dict[tuple[str, str], np.ndarray] = {}
    no_archive = set()
    for sym in sorted(set(f["symbol"])):
        k = load_8k_items(sub_dir, sym)
        if not len(k) and not (Path(sub_dir) / f"{sym}.json").exists():
            no_archive.add(sym)
        for concept, codes in SEC_CONCEPTS.items():
            hit = k[k["items"].map(lambda xs, c=codes: any(x in c for x in xs))] if len(k) else k
            times[(sym, concept)] = np.sort(hit["accepted_utc"].to_numpy(dtype="datetime64[ns]")) if len(hit) else \
                np.array([], dtype="datetime64[ns]")
    created = f["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    lo_all = created - np.timedelta64(SEC_WINDOW[0])
    hi_all = created + np.timedelta64(SEC_WINDOW[1])
    for concept in SEC_CONCEPTS:
        col = np.full(len(f), np.nan)
        for i, sym in enumerate(f["symbol"]):
            if np.isnat(created[i]) or sym in no_archive:
                continue
            ts = times[(sym, concept)]
            j = np.searchsorted(ts, lo_all[i], side="left")
            col[i] = float(j < len(ts) and ts[j] <= hi_all[i])
        out[f"sec_{concept}"] = col
    out.attrs["sec_symbols_without_archive"] = sorted(no_archive)
    return out


# ------------------------------------------------------------------------------------------------ statistics
def weekly_test(f: pd.DataFrame, x: str = "E_effect", y: str = "ar") -> dict:
    ws = S.weekly_spearman(f, x, y, "t", MIN_UNITS)
    out = S.weekly_mean_test(ws, "rho", S.NW_LAGS, ALTERNATIVE)
    out["statistic"] = f"mean over weeks of Spearman({x}, {y})"
    out["distinct_focal_d0"] = int(f.dropna(subset=[y])[["symbol", "d0"]].drop_duplicates().shape[0]) if "d0" in f else None
    out["by_week"] = {str(pd.Timestamp(t).date()): {"n": int(n), "rho": r} for t, n, r in ws.itertuples(index=False)}
    return out


def pooled_spearman(f: pd.DataFrame, x: str, y: str = "ar", n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    g = f[np.isfinite(f[x].to_numpy(dtype=float)) & np.isfinite(f[y].to_numpy(dtype=float))]
    r = S.spearman(g[x], g[y])
    bs = S.week_block_bootstrap_xy(g, "t", x, y, S.spearman, n_boot=n_boot, seed=seed)
    return {"rho": r, "n": len(g), "weeks": int(g["t"].nunique()), "ci_week_block": bs["ci"],
            "ci_fisher": S.fisher_ci(r, len(g)), "n_boot": n_boot, "seed": seed}


def sec_agreement(f: pd.DataFrame, p_type_col: str = "p_type", n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """8-K concept agreement per concept (score_table) with a week-block bootstrap 95% CI of the AUC."""
    from jevtrader.eval.metrics import auc, score_table

    out = {}
    for concept in SEC_CONCEPTS:
        lab = f[f"sec_{concept}"].to_numpy(dtype=float)
        p = np.array([float((d or {}).get(concept, np.nan)) for d in f[p_type_col]])
        ok = np.isfinite(lab) & np.isfinite(p)
        row = {**score_table(p[ok], lab[ok]), "prevalence": float(lab[ok].mean()) if ok.any() else None,
               "auc_ci": [float("nan"), float("nan")]}
        if ok.any() and 0 < lab[ok].sum() < ok.sum() and "t" in f:
            g = pd.DataFrame({"t": f["t"].to_numpy()[ok], "p": p[ok], "y": lab[ok]})
            row["auc_ci"] = S.week_block_bootstrap_xy(g, "t", "p", "y", auc, n_boot=n_boot, seed=seed)["ci"]
        out[f"{concept}_vs_{'/'.join(SEC_CONCEPTS[concept])}"] = row
    return out


REGISTERED_IMPLEMENTATIONS = ("emu", "NG", "VERB")


def sec_claim(per_impl: dict) -> dict:
    """The SEC clause of the H2 sentence ("agree with SEC item codes on each implementation"): True iff every
    registered implementation ran and, on its units, every 8-K concept's AUC CI lies entirely above 0.5."""
    by = {}
    for name in REGISTERED_IMPLEMENTATIONS:
        d = per_impl.get(name) or {}
        sec = d.get("sec_8k") if name != "emu" else d.get("sec_8k_full_frame")
        if d.get("status") != "ok" or not sec:
            by[name] = {"status": d.get("status", "not run"), "above": False}
            continue
        lows = {k: (v.get("auc_ci") or [float("nan")])[0] for k, v in sec.items()}
        by[name] = {"status": "ok", "auc_ci_low": lows,
                    "above": all(isinstance(x, float) and np.isfinite(x) and x > 0.5 for x in lows.values())}
    return {"each_implementation_auc_ci_above_0.5": all(v["above"] for v in by.values()), "by_implementation": by,
            "rule": "every registered implementation (emu, NG, VERB) ran and every 8-K concept's week-block AUC CI "
                    "lies above 0.5"}


def vs_expectations_vs_sign(f: pd.DataFrame, vs_col: str = "vs") -> dict:
    from jevtrader.eval.metrics import score_table

    rows = []
    for d, ar in zip(f[vs_col], f["ar"]):
        d = d or {}
        if top_label(d) not in ("above", "below") or not np.isfinite(ar):
            continue
        a, b = float(d.get("above", 0.0)), float(d.get("below", 0.0))
        if a + b <= 0:
            continue
        rows.append((a / (a + b), float(ar > 0)))
    if not rows:
        return {"n": 0}
    p, y = map(np.asarray, zip(*rows))
    return {**score_table(p, y), "definition": "units with top vs_expectations in {above, below}; "
            "score P(above)/(P(above)+P(below)); label 1[AR[0,+1] > 0]"}


def sentinel_t3(risk: pd.DataFrame, labels, weeks) -> dict:
    """VC-14: sentinel h at decision t vs the T3 tail event z of the same decision's holding week (AUC)."""
    from jevtrader.eval.metrics import auc

    lab = labels.table[["t", "symbol", "z"]]
    r = risk[["t", "symbol", "h"]].copy()
    r["t"] = pd.to_datetime(r["t"])
    j = r[r["t"].isin(set(weeks))].merge(lab, on=["t", "symbol"], how="inner")
    j = j[np.isfinite(j["h"].to_numpy(dtype=float)) & np.isfinite(j["z"].to_numpy(dtype=float))]
    if not len(j) or j["z"].nunique() < 2:
        return {"n": len(j), "auc": None}
    bs = S.week_block_bootstrap_xy(j, "t", "h", "z", auc, n_boot=N_BOOT, seed=SEED)
    return {"n": len(j), "tails": int(j["z"].sum()), "auc": auc(j["h"], j["z"]), "ci_week_block": bs["ci"]}


# ------------------------------------------------------------------------------------------------ implementations
def verb_items(gen_dir: Path) -> pd.DataFrame:
    """VERB's item-level answers (experiments/run_gen_verb.py layout; no items table): every sample unit's five N1
    requests (jobs.parquet: uid 't|symbol|eid|group', meta_json with the unit, the group and the questions the group
    `use`s) joined to their answers (answers.parquet by request key; identical states across items share a key and so
    an answer), each request contributing only the questions it is used for, as production reads them. Per unit:
    E_effect, top_type, p_type, vs, the production triage (programs.news.triage with emu's N0 decision: the item was
    not quarantined by N0, since its N1 state is MAIN's) and c2_eligible = kept and top type != the excluded type.
    Units missing a group are kept with `complete` False and are never C2-eligible."""
    from jevtrader.programs.news import triage
    from jevtrader.s1.types import Answer

    gen_dir = Path(gen_dir)
    jobs = pd.read_parquet(gen_dir / "jobs.parquet", columns=["uid", "request_key", "meta_json"])
    meta = [json.loads(x) for x in jobs["meta_json"]]
    jobs = jobs.assign(unit=[m["unit"] for m in meta], group=[m["group"] for m in meta],
                       use=[tuple(m["use"]) for m in meta])
    ans = pd.read_parquet(gen_dir / "answers.parquet", columns=["request_key", "qid", "kind", "probs_json"])
    ans["qid"] = ans["qid"].astype(str).str.replace("N1.", "", regex=False)
    ans = ans.drop_duplicates(["request_key", "qid"])
    j = jobs[["unit", "group", "use", "request_key"]].merge(ans, on="request_key", how="left")
    j = j[[q in u for q, u in zip(j["qid"], j["use"])]]
    per: dict[str, dict[str, Answer]] = {}
    groups: dict[str, set] = {}
    for u, g, q, k, pj in j[["unit", "group", "qid", "kind", "probs_json"]].itertuples(index=False):
        probs = json.loads(pj) if isinstance(pj, str) else None
        per.setdefault(u, {})[q] = Answer(str(k), {str(a): float(b) for a, b in probs.items()} if probs else None)
        groups.setdefault(u, set()).add(g)
    sample = pd.read_parquet(gen_dir / "sample.parquet", columns=["unit", "t", "symbol", "eid"])
    rows = []
    for u, t, sym, eid in sample.itertuples(index=False):
        a = per.get(u, {})
        dist = {q: (a[q].probs if q in a else None) for q in ("event_type", "effect", "vs_expectations")}
        tri = triage(a, n0_quarantined=False)
        top = top_label(dist["event_type"])
        complete = groups.get(u, set()) == set(VERB_GROUPS)
        rows.append({"unit": u, "t": pd.Timestamp(t), "symbol": sym, "eid": eid,
                     "E_effect": expected_effect(dist["effect"]), "top_type": top, "p_type": dist["event_type"] or {},
                     "vs": dist["vs_expectations"] or {}, "kept": bool(tri.kept), "triage_reason": tri.reason,
                     "complete": complete,
                     "c2_eligible": bool(complete and tri.kept and top is not None and top != EXCLUDED_TYPE)})
    return pd.DataFrame(rows)


def implementation_items(gen_dir: Path) -> pd.DataFrame | None:
    """An implementation's item-level answers on its sample: unit, t, symbol, eid, E_effect, top_type, p_type, vs,
    c2_eligible. NG (experiments/ng/items500.py): items.parquet once finalized, else the checkpoint's item records.
    VERB (experiments/run_gen_verb.py): answers + jobs + sample (`verb_items`). None if neither layout exists."""
    gen_dir = Path(gen_dir)
    recs = []
    if not (gen_dir / "items.parquet").exists() and not (gen_dir / "checkpoint.jsonl").exists() and all(
            (gen_dir / f).exists() for f in ("answers.parquet", "jobs.parquet", "sample.parquet")):
        return verb_items(gen_dir)
    if (gen_dir / "items.parquet").exists():
        it = pd.read_parquet(gen_dir / "items.parquet")
        for r in it.to_dict("records"):
            recs.append({k: (json.loads(v) if isinstance(v, str) and v[:1] in "{[" else v) for k, v in r.items()})
    elif (gen_dir / "checkpoint.jsonl").exists():
        with open(gen_dir / "checkpoint.jsonl") as f:
            recs = [json.loads(x)["item"] for x in f if x.strip()]
    else:
        return None
    prefix = next((p for p in ("ng_", "verb_", "impl_") if recs and f"{p}effect" in recs[0]), "")
    rows = []
    for r in recs:
        et = r.get(f"{prefix}event_type")
        rows.append({"unit": r["unit"], "t": pd.Timestamp(r["t"]), "symbol": r["symbol"], "eid": r["eid"],
                     "E_effect": expected_effect(r.get(f"{prefix}effect")), "top_type": top_label(et),
                     "p_type": et or {}, "vs": r.get(f"{prefix}vs_expectations") or {},
                     "c2_eligible": bool(r.get(f"{prefix}c2_eligible", False))})
    return pd.DataFrame(rows)


def implementation_frame(it: pd.DataFrame) -> dict:
    """Counts of an implementation's parsed item table and its SHA-256 (Layer-A answers only): the canonical CSV of
    unit, E_effect (12 significant digits), top_type and c2_eligible, sorted by unit."""
    canon = it[["unit", "E_effect", "top_type", "c2_eligible"]].sort_values("unit").copy()
    canon["E_effect"] = [f"{x:.12g}" for x in canon["E_effect"]]
    return {"units": int(len(it)), "weeks": int(it["t"].nunique()) if len(it) else 0,
            "c2_eligible": int(it["c2_eligible"].sum()) if len(it) else 0,
            "complete": int(it["complete"].sum()) if "complete" in it else None,
            "sha256": C.sha256_text(canon.to_csv(index=False))}


def implementation_inputs(label: str, d: Path) -> dict:
    if (d / "items.parquet").exists():
        return {f"{label} items": d / "items.parquet"}
    if (d / "checkpoint.jsonl").exists():
        return {f"{label} items": d / "checkpoint.jsonl"}
    return {f"{label} {f}": d / f"{f}.parquet" for f in ("answers", "jobs", "sample")}


def per_implementation(name: str, impl: pd.DataFrame | None, emu: pd.DataFrame) -> dict:
    if impl is None or not len(impl):
        return {"status": "not run" if impl is None else "no finished units"}
    e = emu.set_index("unit")
    missing = int((~impl["unit"].isin(e.index)).sum())
    impl = impl[impl["unit"].isin(e.index)].copy()
    lab = e.loc[impl["unit"], ["ar", "ar_placebo", "d0", *[c for c in e.columns if c.startswith("sec_")]]]
    impl = pd.concat([impl.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
    both = impl.assign(E_emu=e.loc[impl["unit"], "E_effect"].to_numpy(),
                       top_emu=e.loc[impl["unit"], "top_type"].to_numpy())
    return {"status": "ok", "n_units": len(impl), "units_not_in_emu_frame": missing,
            "c2_eligible_own_rate": float(impl["c2_eligible"].mean()),
            "c2_eligible_own_ci": S.wilson(int(impl["c2_eligible"].sum()), len(impl)),
            "spearman_ar": pooled_spearman(impl, "E_effect"),
            "spearman_ar_placebo": pooled_spearman(impl, "E_effect", "ar_placebo"),
            "emu_same_units": pooled_spearman(both, "E_emu"),
            "agreement": {"spearman_E_effect": S.spearman(both["E_emu"], both["E_effect"]),
                          "top_event_type": S.cohen_kappa(both["top_emu"], both["top_type"])},
            "sec_8k": sec_agreement(impl), "vs_expectations_vs_ar_sign": vs_expectations_vs_sign(impl)}


# ------------------------------------------------------------------------------------------------ main
def analyse(f: pd.DataFrame, *, start: pd.Timestamp, release: pd.Timestamp, prior: set, impls: dict,
            risk: pd.DataFrame | None = None, labels=None, drop_from: pd.Timestamp | None = None) -> dict:
    """Every statistic of the module doc on a frame that already carries ar, ar_placebo, d0 and sec_* columns.
    `start` is the confirmatory start (D1 rule); `drop_from` the drop-13 sensitivity's first week (common.drop_start:
    a date from the configs, unchanged by the D1 rule; None = the frame's 14th week, synthetic tests only)."""
    prim = f[f["t"] >= start]
    local = prim["created_at"].dt.tz_convert(C.ET)
    pre_open = prim[(local.dt.strftime("%H:%M") < PRE_OPEN).to_numpy()]
    no_prior = prim[[(t, s) not in prior for t, s in zip(prim["t"], prim["symbol"])]]
    if drop_from is None:  # synthetic frames: the frame's own 14th week (none: every week is dropped)
        rest = C.first_n_weeks_dropped(prim["t"].unique(), DROP_FIRST)
        drop_from = rest[0] if rest else prim["t"].max() + pd.Timedelta(days=1)
    drop13 = pd.Timestamp(drop_from)
    emu_sec = sec_agreement(prim)
    out = {
        "confirmatory": {"hypothesis": "H2", "code": "C2", **weekly_test(prim),
                         "ar_missing": int(prim["ar"].isna().sum())},
        "placebo_ar_m5_m1": weekly_test(prim, y="ar_placebo"),
        "controls": {"pre_open_only": weekly_test(pre_open), "prior_look_excluded": weekly_test(no_prior),
                     "prior_look_units_removed": int(len(prim) - len(no_prior))},
        "sensitivities": {"W_release": weekly_test(prim[prim["t"] >= release]),
                          f"drop_first_{DROP_FIRST}": {**weekly_test(prim[prim["t"] >= drop13]),
                                                       "from": drop13,
                                                       "kind": "sub-series of the primary frame (weeks >= from)"}},
        "sec_8k": emu_sec,
        "vs_expectations_vs_ar_sign": vs_expectations_vs_sign(prim),
        "per_implementation": {"emu": {"status": "ok", "role": "confirmatory row (full frame)",
                                       "spearman_ar_pooled": pooled_spearman(prim, "E_effect"),
                                       "sec_8k_full_frame": emu_sec},
                               **{k: per_implementation(k, v, prim) for k, v in impls.items()}},
    }
    out["sec_claim"] = sec_claim(out["per_implementation"])
    if risk is not None and labels is not None:
        out["agent_level"] = {"sentinel_h_vs_T3": sentinel_t3(risk, labels, prim["t"].unique())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen", default=DEFAULT_GEN, help="generation id or directory (emu, the confirmatory reader)")
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--ng", default=DEFAULT_NG, help="NG generation id or dir ('' = none)")
    ap.add_argument("--verb", default=DEFAULT_VERB, help="VERB generation id or dir ('' = not run)")
    ap.add_argument("--frame-only", action="store_true", help="build and count the item frame only (no outcome)")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    gen = Path(a.gen) if Path(a.gen).is_dir() else root / "results/judgments" / a.gen
    start, end = C.resolve_window(a.window, root)
    d1 = C.d1_start(root) if a.window == C.REGISTERED_WINDOW else {"start": start, "applied": False,
                                                                   "reason": "not the registered window"}
    items = item_rows(gen, start, end)
    frame = c2_frame(items)
    counts = frame_counts(frame, items)
    inputs = {"work.jsonl": gen / "memory" / "work.jsonl"}
    impls, impl_frames = {}, {}
    for label, ref in (("NG", a.ng), ("VERB", a.verb)):  # Layer-A answers only (no outcome)
        impls[label] = None
        if not ref:
            continue
        d = Path(ref) if Path(ref).is_dir() else root / "results/judgments" / ref
        it = implementation_items(d) if d.exists() else None
        if it is not None:
            it = it[(it["t"] >= start) & (it["t"] <= end)].reset_index(drop=True)
            impl_frames[label] = implementation_frame(it)
            inputs.update(implementation_inputs(label, d))
        impls[label] = it
    if a.frame_only:
        path = C.out_path(f"{NAME}_frame", a.window, a.out_dir, root)
        C.write_result(path, {"frame": counts, "rule": "status.kept and top event_type != " + EXCLUDED_TYPE,
                              "implementations": impl_frames},
                       script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv)
        print(json.dumps({"frame": counts, "implementations": impl_frames}, indent=1))
        return 0
    guard.check(frame["t"].unique(), "H2 labels (AR, 8-K, T3)")
    for label, it in impls.items():
        if it is not None:
            guard.check(it["t"].unique(), f"H2 {label} rows")
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    frame = add_abnormal_returns(frame, px.book)
    frame = sec_labels(frame, px.root / C.SUBMISSIONS)
    risk, labels = None, None
    if (gen / "risk.parquet").exists():
        risk = pd.read_parquet(gen / "risk.parquet")
        weeks = sorted(frame["t"].unique())
        labels = C.load_labels(px, [w for w in px.dates if w in set(weeks)], with_sigma=True)
        inputs["risk.parquet"] = gen / "risk.parquet"
    res = analyse(frame, start=pd.Timestamp(d1["start"]), release=C.release_start(root),
                  prior=C.prior_look_keys(root), impls=impls, risk=risk, labels=labels,
                  drop_from=C.drop_start(a.window, root))
    payload = {"window_start_rule": {k: v for k, v in d1.items()}, "frame": counts,
               "implementation_frames": impl_frames, **res,
               "settings": {"min_units_per_week": MIN_UNITS, "nw_lags": S.NW_LAGS, "alternative": ALTERNATIVE,
                            "ar_window": AR_WINDOW, "placebo_window": PLACEBO_WINDOW, "pre_open_before": PRE_OPEN,
                            "sec_window_days": [-2, 1], "n_boot": N_BOOT, "seed": SEED, "price_cap": px.cap}}
    inputs.update({"universe": px.root / C.UNIVERSE_FILE, "prior looks": root / C.PRIOR_LOOKS})
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv)
    c = res["confirmatory"]
    print(f"H2 {a.window}: mean weekly Spearman {c['mean']:.4f} CI {c['ci']} t {c['t']:.2f} p {c['p']:.4g} "
          f"({c['n_weeks']} weeks, {c.get('n_units')} units) -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
