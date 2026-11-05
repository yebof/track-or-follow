"""Mechanical checks of Layer B on the W_pre slice (design_v1 §13.10; design_v2_delta BG-3c, BG-5, OPS-3).

    .venv/bin/python experiments/pre/mechanical_checks.py --gen-id pre_blinded_emu_wpre_s30 --cap-root <scratch>/caproot \
        --work <scratch>/mech --out results/pre/mechanical_checks.json

They run whatever the IC is (validity M11): each has a pass criterion fixed in this file (CRITERIA) before it is run.
W_pre is not W_clean, so outcomes on it may be read; prices are capped before W_clean (`caproot`), so no return after a
W_clean cutoff is computed. Nothing here is registered in results/trials.jsonl or written under results/eval/.

M1 synthetic-signal injection. From adjusted opens, independently of learn.labels (own session calendar, own
   e_t/e_{t+1}), rel_{i,t} = O(e_{t+1})/O(e_t) - 1 minus its U_t median; the planted score is
   s = Phi^-1((rank(rel) - 1/2)/n) + sigma * eps (eps iid N(0,1), seeded), whose expected weekly Spearman IC is
   (6/pi) asin(rho/2), rho = 1/sqrt(1 + sigma^2). It enters Layer B as a trusted code feature ("syn_inj").
   M1a isolated (the only feature; IC_th = 0.10): every learned week has beta > 0 learned from matured labels only,
   Layer B's weekly IC equals the direct IC of the score (|diff| <= 1e-9), and the mean IC is within 3 NW(4) SE of
   IC_th. M1b full (all primary features + syn_inj; IC_th = 0.30): mean Layer-B IC over learned weeks >= 0.8 x the
   direct IC of the score on the same weeks; the IC_th = 0.10 run is reported.
M2 label-shift placebos. (a) The M1a score (built from rel_t) scored against Layer B's labels of t-1 and t+1:
   |mean IC| < 3 NW SE for both, while the aligned IC passes M1a. (b) Scores built from rel_{t-1} and rel_{t+1}, through
   the isolated Layer B, scored against rel_t: the t+1 score (a label the learner never sees in time) has
   |mean IC| < 3 NW SE. The primary system's L against r_{t-1}, r_t, r_{t+1} is reported (descriptive).
M3 classical-baseline reproduction. The classical baselines' signals (baselines.classic: momentum 12-1, 1-week
   reversal, low volatility on features.technical.technical_snapshot at each decision) and Layer B's code features
   (code_features.parquet via load_judgment_features: mom_12_1, rev_1w, vol_60d) give per-week ICs against the T1
   relative return that agree (correlation of weekly ICs >= 0.99, |mean difference| <= 0.005, signs as the baselines
   rank: +mom, -ret_1w = +rev_1w, -vol); through Layer B (each alone), |IC(L)| equals |IC(x~)| of the logged
   standardized feature every learned week (<= 1e-9). The 14 code features Layer B reads equal the classical path's
   values (technical_snapshot + fundamentals) on every (t, symbol) (relative difference <= 1e-9).
M4 shuffled-score null. The M1a score permuted across names within each week, through the isolated Layer B:
   |mean IC| < 3 NW SE. The primary system's L permuted within week (1,000 seeded permutations): the mean over
   permutations of the mean IC is within 0.002 of 0 (the IC machinery has no built-in bias); the primary's percentile
   is reported (descriptive).
M5 BG-3c anchor mechanics. The trusted price anchor recomputed by a direct loop over every (t, U_t) (design rule,
   every d0 of the week's news window) equals the panel's; the filing anchors are counted; under each exclusion rule
   (anchored = primary, bylines, items, anchored_collapsed) the logged multipliers equal an independent implementation
   of BG-3 on the risk table and anchors, and the anchored rules never exclude without A = 1. Counts of exclusions and
   halvings per rule are reported (mechanics only: no performance read).
M6 band-guard bound check (BG-5 worst-case reader) on the primary run: 0 entry and 0 protection violations, trusted
   set all T, and at least one check per learned week.
M7 replay determinism. evaluate() on the primary cell in fresh processes with PYTHONHASHSEED 1, 2 and 3: identical
   SHA-256 of weights, cards, metrics, predictions, sizing logs, heads, daily returns, snapshots and multipliers, and of
   the bound check.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "pre"))

import caproot  # noqa: E402

from jevtrader.eval import bound as B  # noqa: E402
from jevtrader.eval import metrics as M  # noqa: E402
from jevtrader.learn import features as LF  # noqa: E402
from jevtrader.pipeline import layer_b as LB  # noqa: E402
from jevtrader.pipeline.registry import _plain, canonical_json  # noqa: E402

WINDOW = "W_pre_slice"
INJ = "syn_inj"
SEED = 20260928
CRITERIA = {
    "M1a": "beta>0 on every learned week; |IC_L - IC_direct| <= 1e-9 every learned week; |mean IC - IC_th| <= 3 NW SE "
           "(IC_th = 0.10)",
    "M1b": "mean Layer-B IC >= 0.8 x mean direct IC on the same learned weeks (IC_th = 0.30)",
    "M2": "|mean IC| < 3 NW SE vs r_{t-1} and vs r_{t+1} for the M1a Layer-B score; |mean IC| < 3 NW SE vs r_t for a "
          "score built from r_{t+1} through the isolated Layer B",
    "M3": "per classical signal: corr(weekly IC) >= 0.99 and |mean IC diff| <= 0.005; through Layer B "
          "||IC_L| - |IC_x~|| <= 1e-9 every learned week; code-feature values equal (rel diff <= 1e-9)",
    "M4": "shuffled M1a score through Layer B: |mean IC| < 3 NW SE; permutation null of the primary L: "
          "|mean of permutation mean ICs| <= 0.002",
    "M5": "price anchor recomputed == panel on every (t, U_t); logged multipliers == independent BG-3 rule for all four "
          "rules; no anchored-rule exclusion with A = 0",
    "M6": "0 entry violations, 0 protection violations, trusted set all T, >= 1 check per learned week",
    "M7": "identical SHA-256 of every Layer-B output across PYTHONHASHSEED 1, 2, 3",
}


# ------------------------------------------------------------------------------------ helpers
def nw(x: pd.Series) -> dict:
    x = pd.Series(x).dropna()
    t = M.t_stat_newey_west(x, 4)
    se = float(x.mean() / t) if t and np.isfinite(t) and t != 0 else float("nan")
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "t_nw4": float(t),
            "se_nw4": abs(se)}


def expected_ic(sigma: float) -> float:
    rho = 1.0 / math.sqrt(1.0 + sigma * sigma)
    return 6.0 / math.pi * math.asin(rho / 2.0)


def sigma_for(ic: float) -> float:
    rho = 2.0 * math.sin(ic * math.pi / 6.0)
    return math.sqrt(1.0 / (rho * rho) - 1.0)


def load(gen_id: str, cap: Path):
    cfg = LB.DecisionConfig.from_yaml(gen_id=gen_id, name="primary")
    start, end = LB.resolve_window(WINDOW)
    inputs = LB.load_inputs(cfg, start, end, ROOT, data_root=cap)
    assert inputs.prices.sessions.max() <= caproot.PRICE_CAP
    return cfg, inputs


def rel_pivot(labels) -> pd.DataFrame:
    lab = labels.table
    return lab.assign(rel=lab["r"] - lab["m"]).pivot(index="t", columns="symbol", values="rel")


def ic_series(score: pd.DataFrame, rel: pd.DataFrame) -> pd.Series:
    return M.information_coefficient(score, rel)


def learned(res) -> pd.DataFrame:
    p = res.log.predictions
    return p[~p["cold"]]


def run_eval(cfg, inputs, work: Path):
    return LB.evaluate(cfg, WINDOW, inputs, root=work, register=False, write=False)


def with_feature(inputs, name: str, values: pd.DataFrame):
    """A copy of the inputs with one extra raw feature (t, symbol, value); covered where finite."""
    add = values.assign(feature=name)
    add["covered"] = np.isfinite(add["value"].to_numpy(dtype=float))
    raw = pd.concat([inputs.raw, add[LB.RAW_COLUMNS]], ignore_index=True)
    raw.attrs = dict(inputs.raw.attrs)
    return dataclasses.replace(inputs, raw=raw)


def only(cfg, trusted, untrusted=(), name="chk"):
    a = dataclasses.replace(cfg.allocator, trusted=tuple(trusted), untrusted=tuple(untrusted))
    return dataclasses.replace(cfg, allocator=a, name=name, heads=False)


# ------------------------------------------------------------------------------------ independent returns
def independent_rel(cap: Path, dates: list[pd.Timestamp], universe, shift: int = 0) -> pd.DataFrame:
    """rel_{i,t} from adjusted opens with an own calendar: decision = last session of each ISO week; e_t = the session
    after d_t; e_{t+1} = the session after the next decision; r = O(e_{t+1})/O(e_t) - 1; rel = r - median over U_t.
    `shift` = k uses the holding period of the decision k weeks from t (k = -1: the previous week's period; +1: the
    next one) for the names of U_t, demeaned over them (the placebo sources)."""
    bars = pd.concat([pd.read_parquet(cap / f, columns=["symbol", "date", "open"]) for f in caproot.BAR_FILES[:2]])
    bars["date"] = pd.to_datetime(bars["date"])
    op = bars.pivot_table(index="date", columns="symbol", values="open", aggfunc="first").sort_index()
    sess = op.index
    iso = pd.Series(sess, index=sess).groupby([sess.isocalendar().year, sess.isocalendar().week]).max()
    dec = sorted(pd.DatetimeIndex(iso.to_numpy()))
    out = {}
    for t in dates:
        k = dec.index(t) + shift
        e_t = sess[sess > dec[k]][0]
        e_n = sess[sess > dec[k + 1]][0]
        if e_n > caproot.PRICE_CAP:
            raise ValueError(f"holding period of {dec[k].date()} ends after the price cap")
        syms = [s for s in universe[t] if s in op.columns]
        r = op.loc[e_n, syms] / op.loc[e_t, syms] - 1.0
        out[t] = r - r.median()
    return pd.DataFrame(out).T.sort_index()


def gaussian_rank(rel: pd.DataFrame) -> pd.DataFrame:
    def row(x: pd.Series) -> pd.Series:
        v = x.dropna()
        u = (v.rank(method="average") - 0.5) / len(v)
        return pd.Series(stats.norm.ppf(u), index=v.index).reindex(x.index)

    return rel.apply(row, axis=1)


def long(frame: pd.DataFrame) -> pd.DataFrame:
    s = frame.stack(future_stack=True).rename("value").reset_index()
    s.columns = ["t", "symbol", "value"]
    return s


def planted(rel_src: pd.DataFrame, sigma: float, seed: int, dates) -> pd.DataFrame:
    z = gaussian_rank(rel_src)
    rng = np.random.default_rng(seed)
    eps = pd.DataFrame(rng.standard_normal(z.shape), index=z.index, columns=z.columns)
    s = z + sigma * eps
    return s.reindex(dates)


# ------------------------------------------------------------------------------------ M1, M2, M4
def iso_check(cfg, inputs, score: pd.DataFrame, rel: pd.DataFrame, work: Path, tag: str) -> dict:
    """Score as the only trusted feature; Layer B's weekly IC vs the direct IC on the same names."""
    inj = with_feature(inputs, INJ, long(score))
    res = run_eval(only(cfg, (INJ,), name=f"chk-{tag}"), inj, work)
    pr = learned(res)
    L = pr.pivot(index="t", columns="symbol", values="L")
    both = L.notna() & score.reindex_like(L).notna()
    ic_L = ic_series(L.where(both), rel)
    ic_d = ic_series(score.reindex_like(L).where(both), rel)
    beta = pd.Series({pd.Timestamp(s["t"]): s["beta"].get(INJ, 0.0) for s in res.log.snapshots})
    beta = beta.reindex(ic_L.index)
    return {"res": res, "L": L, "ic_L": ic_L, "ic_direct": ic_d, "beta": beta}


def shifted_rel(rel: pd.DataFrame, k: int) -> pd.DataFrame:
    """rel of the decision k weeks later mapped onto week t (k = -1: the previous decision's label)."""
    idx = list(rel.index)
    out = {}
    for i, t in enumerate(idx):
        j = i + k
        if 0 <= j < len(idx):
            out[t] = rel.iloc[j]
    return pd.DataFrame(out).T.reindex(columns=rel.columns)


def m1_m2_m4(cfg, inputs, cap: Path, work: Path) -> dict:
    dates = sorted(inputs.universe)
    rel = rel_pivot(inputs.labels)
    ind = independent_rel(cap, dates, inputs.universe)
    common = ind.stack().index.intersection(rel.stack().dropna().index)
    d = (ind.stack().reindex(common) - rel.stack().reindex(common)).abs()
    label_agree = {"pairs": int(len(common)), "max_abs_diff": float(d.max()), "n_diff_gt_1e-9": int((d > 1e-9).sum()),
                   "labelstore_only": int(len(rel.stack().dropna().index.difference(ind.stack().dropna().index))),
                   "independent_only": int(len(ind.stack().dropna().index.difference(rel.stack().dropna().index)))}
    out: dict = {"independent_vs_labelstore": label_agree}
    # ---- M1a
    s10 = planted(ind, sigma_for(0.10), SEED, dates)
    a = iso_check(cfg, inputs, s10, rel, work, "m1a")
    lw = a["ic_L"].index
    diff = (a["ic_L"] - a["ic_direct"].reindex(lw)).abs()
    st = nw(a["ic_L"])
    m1a = {"ic_th": 0.10, "sigma": sigma_for(0.10), "learned_weeks": int(len(lw)),
           "beta_positive_weeks": int((a["beta"] > 0).sum()), "max_abs_ic_diff": float(diff.max()),
           "ic_layer_b": st, "ic_direct": nw(a["ic_direct"].reindex(lw)),
           "z_vs_th": float((st["mean"] - 0.10) / st["se_nw4"])}
    m1a["pass"] = bool(m1a["beta_positive_weeks"] == len(lw) and m1a["max_abs_ic_diff"] <= 1e-9
                       and abs(m1a["z_vs_th"]) <= 3.0)
    out["M1a"] = m1a
    # ---- M1b (full configuration + the planted score)
    m1b = {}
    for ic_th in (0.30, 0.10):
        s = planted(ind, sigma_for(ic_th), SEED + int(ic_th * 100), dates)
        inj = with_feature(inputs, INJ, long(s))
        c = dataclasses.replace(cfg, allocator=dataclasses.replace(cfg.allocator,
                                                                   trusted=cfg.allocator.trusted + (INJ,)),
                                name=f"chk-m1b-{ic_th}", heads=False)
        res = run_eval(c, inj, work)
        L = learned(res).pivot(index="t", columns="symbol", values="L")
        both = L.notna() & s.reindex_like(L).notna()
        ic_L = ic_series(L.where(both), rel)
        ic_d = ic_series(s.reindex_like(L).where(both), rel)
        first = ic_L.sort_index().head(4)
        m1b[str(ic_th)] = {"ic_th": ic_th, "ic_layer_b": nw(ic_L), "ic_direct": nw(ic_d.reindex(ic_L.index)),
                           "ratio": float(ic_L.mean() / ic_d.reindex(ic_L.index).mean()),
                           "first_learned_weeks": {k.date().isoformat(): float(v) for k, v in first.items()},
                           "beta_syn_last": float(res.log.snapshots[-1]["beta"].get(INJ, float("nan")))}
    m1b["pass"] = bool(m1b["0.3"]["ratio"] >= 0.8)
    out["M1b"] = m1b
    # ---- M2 placebos
    L = a["L"]
    pl = {}
    for k, name in ((-1, "r_t-1"), (0, "r_t"), (1, "r_t+1")):
        pl[name] = nw(ic_series(L, shifted_rel(rel, k)))
    thr = {}
    for k, name in ((-1, "score_from_r_t-1"), (1, "score_from_r_t+1")):
        src = independent_rel(cap, dates, inputs.universe, shift=k)
        s = planted(src, sigma_for(0.10), SEED + 7 + k, dates)
        b = iso_check(cfg, inputs, s, rel, work, f"m2{k:+d}")
        thr[name] = {"ic_layer_b_vs_r_t": nw(b["ic_L"]), "beta_last": float(b["beta"].dropna().iloc[-1])}
    m2 = {"m1a_score_vs_shifted_labels": pl, "shifted_scores_through_layer_b": thr}
    m2["pass"] = bool(abs(pl["r_t-1"]["t_nw4"]) < 3 and abs(pl["r_t+1"]["t_nw4"]) < 3 and m1a["pass"]
                      and abs(thr["score_from_r_t+1"]["ic_layer_b_vs_r_t"]["t_nw4"]) < 3)
    out["M2"] = m2
    # ---- M4 shuffled score through Layer B
    rng = np.random.default_rng(SEED + 99)
    sh = s10.apply(lambda row: pd.Series(rng.permutation(row.to_numpy()), index=row.index), axis=1)
    c = iso_check(cfg, inputs, sh, rel, work, "m4")
    out["M4_shuffled_through_layer_b"] = {"ic_layer_b": nw(c["ic_L"])}
    return out


def primary_placebo_and_null(res, rel: pd.DataFrame, n_perm: int = 1000) -> dict:
    pr = learned(res)
    L = pr.pivot(index="t", columns="symbol", values="L")
    out = {"primary_L_vs_labels": {name: nw(ic_series(L, shifted_rel(rel, k)))
                                   for k, name in ((-1, "r_t-1"), (0, "r_t"), (1, "r_t+1"))}}
    rng = np.random.default_rng(SEED + 5)
    actual = ic_series(L, rel).mean()
    # Spearman = Pearson of ranks; permuting the scores among a week's names permutes their ranks
    weeks = []
    for t in L.index:
        a = pd.concat([L.loc[t], rel.loc[t] if t in rel.index else pd.Series(dtype=float)], axis=1).dropna()
        if len(a) >= 5:
            rs = stats.rankdata(a.iloc[:, 0]) - (len(a) + 1) / 2.0
            rr = stats.rankdata(a.iloc[:, 1]) - (len(a) + 1) / 2.0
            weeks.append((rs, rr / np.sqrt(rr @ rr), np.sqrt(rs @ rs)))
    means = np.empty(n_perm)
    for b in range(n_perm):
        means[b] = np.mean([rng.permutation(rs) @ rr / nrs for rs, rr, nrs in weeks])
    out["permutation_null"] = {"n_perm": n_perm, "mean_of_means": float(means.mean()), "sd_of_means": float(means.std()),
                               "primary_mean_ic": float(actual),
                               "primary_percentile": float((means < actual).mean())}
    return out


# ------------------------------------------------------------------------------------ M3
def m3(cfg, inputs, work: Path) -> dict:
    from jevtrader.data import fundamentals as F
    from jevtrader.data.prices import load_price_panels
    from jevtrader.features import technical as T

    dates = sorted(inputs.universe)
    panels = load_price_panels()
    sector = T.SectorMap(panels.adjusted)
    uni = {t: list(inputs.universe[t]) for t in dates}
    fund = F.FundamentalsPanel(ROOT / "data/raw/sec/facts", sorted({s for t in dates for s in uni[t]}), panels)
    snaps, classic_raw = {}, {}
    for t in dates:
        snap = T.technical_snapshot(panels.adjusted, t, uni[t], sector_etf=sector.asof(t, uni[t]), raw=panels.raw)
        snaps[t] = snap
        classic_raw[t] = T.code_features(snap).join(fund.features(t, uni[t]), how="outer").reindex(uni[t])
    rel = rel_pivot(inputs.labels)
    raw = inputs.raw
    lb = raw[raw["feature"].isin(T.CODE_FEATURES + F.CODE_FEATURES)]
    # (i) the values Layer B reads equal the classical path's
    cr = pd.concat([df.rename_axis("symbol").reset_index().melt(id_vars="symbol", var_name="feature").assign(t=t)
                    for t, df in classic_raw.items()], ignore_index=True)
    cr = cr[cr["feature"].isin(T.CODE_FEATURES + F.CODE_FEATURES)]
    j = lb.merge(cr, on=["t", "symbol", "feature"], how="outer", suffixes=("_lb", "_cl"))
    a, b = j["value_lb"].astype(float), j["value_cl"].astype(float)
    both_nan = a.isna() & b.isna()
    one_nan = a.isna() ^ b.isna()
    rd = ((a - b).abs() / np.maximum(1.0, b.abs())).where(~a.isna() & ~b.isna())
    values = {"rows": int(len(j)), "both_missing": int(both_nan.sum()), "one_side_missing": int(one_nan.sum()),
              "max_rel_diff": float(rd.max()), "n_rel_diff_gt_1e-9": int((rd > 1e-9).sum())}
    # (ii) weekly ICs of the classical signals vs Layer B's code features
    pairs = {"momentum_12_1": ("mom_12_1", "ret_12_1", 1.0, 1.0),
             "reversal_1w": ("rev_1w", "ret_1w", 1.0, -1.0),
             "low_vol": ("vol_60d", "vol_60d", -1.0, -1.0)}
    ics = {}
    for name, (feat, col, s_lb, s_cl) in pairs.items():
        cl = pd.DataFrame({t: snaps[t][col] for t in dates}).T * s_cl
        f = lb[lb["feature"] == feat].pivot(index="t", columns="symbol", values="value") * s_lb
        ic_cl, ic_lb = ic_series(cl, rel), ic_series(f, rel)
        k = ic_cl.index.intersection(ic_lb.index)
        ics[name] = {"classical": nw(ic_cl), "layer_b_feature": nw(ic_lb), "weeks": int(len(k)),
                     "corr_weekly": float(np.corrcoef(ic_cl[k], ic_lb[k])[0, 1]),
                     "mean_diff": float(ic_lb[k].mean() - ic_cl[k].mean())}
        ics[name]["pass"] = bool(ics[name]["corr_weekly"] >= 0.99 and abs(ics[name]["mean_diff"]) <= 0.005)
    # (iii) through Layer B, each alone: |IC(L)| == |IC(x~)| of the logged standardized feature
    thru = {}
    for name, (feat, *_rest) in pairs.items():
        res = run_eval(only(cfg, (feat,), name=f"chk-m3-{feat}"), inputs, work)
        L = learned(res).pivot(index="t", columns="symbol", values="L")
        led = res.log.ledger.frame()
        x = led.pivot(index="t", columns="symbol", values=feat).reindex_like(L)
        ic_L, ic_x = ic_series(L, rel), ic_series(x, rel)
        dd = (ic_L.abs() - ic_x.reindex(ic_L.index).abs()).abs()
        beta = pd.Series({pd.Timestamp(s["t"]): s["beta"].get(feat, 0.0) for s in res.log.snapshots})
        thru[name] = {"max_abs_diff": float(dd.max()), "weeks": int(len(dd)),
                      "beta_negative_weeks": int((beta.reindex(ic_L.index) < 0).sum()),
                      "ic_layer_b": nw(ic_L)}
    ok = values["n_rel_diff_gt_1e-9"] == 0 and values["one_side_missing"] == 0 and all(v["pass"] for v in ics.values()) \
        and all(v["max_abs_diff"] <= 1e-9 for v in thru.values())
    return {"code_feature_values": values, "signal_ics": ics, "through_layer_b": thru, "pass": bool(ok)}


# ------------------------------------------------------------------------------------ M5
def price_anchor_direct(close: pd.DataFrame, dates, universe, k: float = 2.0, window: int = 60, minp: int = 40):
    """A^price by a direct loop: for every d0 in (d_{t-1}, d_t], r_ex from the close of d0-1 to the close of d_t, n
    sessions d0..d_t, sd of daily excess returns over the 60 sessions ending d0-1 (>= 40 of them)."""
    close = close.sort_index()
    sess = close.index
    iso = pd.Series(sess, index=sess).groupby([sess.isocalendar().year, sess.isocalendar().week]).max()
    cal = sorted(pd.DatetimeIndex(iso.to_numpy()))
    rets = close.pct_change(fill_method=None)
    ex = rets.sub(rets["SPY"], axis=0)
    out = {}
    for t in dates:
        prev = [d for d in cal if d < t][-1]
        d0s = sess[(sess > prev) & (sess <= t)]
        for s in universe[t]:
            hit = False
            if s in close.columns:
                for d0 in d0s:
                    i0 = sess.get_loc(d0)
                    if i0 < 1:
                        continue
                    b = sess[i0 - 1]
                    win = ex[s].loc[:b].iloc[-window:]
                    if win.notna().sum() < minp:
                        continue
                    sd = win.std(ddof=1)
                    n = sess.get_loc(t) - i0 + 1
                    r_ex = (close.at[t, s] / close.at[b, s] - 1.0) - (close.at[t, "SPY"] / close.at[b, "SPY"] - 1.0)
                    if np.isfinite(r_ex) and np.isfinite(sd) and r_ex <= -k * sd * math.sqrt(n):
                        hit = True
                        break
            out[(t, s)] = hit
    return pd.Series(out)


def bg3_rule(risk_rows: pd.DataFrame, A: pd.Series, rule: str, key: str, kinds, corro: int = 2) -> pd.DataFrame:
    """Independent BG-3 (design_v2_delta): sentinel exclusion 0 iff >= corro keys, an exclusion kind and (anchored
    rules) A = 1; the same without an anchor x0.5; no anchor needed under the bylines/items rules."""
    n = pd.to_numeric(risk_rows.get(key), errors="coerce").fillna(0.0)
    kind = risk_rows.get("problem_kind")
    ek = kind.isin(list(kinds)) if kind is not None else pd.Series(False, index=risk_rows.index)
    corr = n >= corro
    a = A.reindex(risk_rows.index).fillna(0.0) > 0
    gate = a if rule == "anchored" else pd.Series(True, index=risk_rows.index)
    excl = corr & ek & gate
    half = corr & ek & ~gate
    return pd.DataFrame({"excl": excl, "unanchored_half": half})


def m5(cfg, inputs, work: Path) -> dict:
    dates = sorted(inputs.universe)
    anc = inputs.anchors.assign(t=pd.to_datetime(inputs.anchors["t"]))
    close = inputs.prices.close
    direct = price_anchor_direct(close, dates, inputs.universe, cfg.anchors.k, cfg.anchors.sd_window,
                                 cfg.anchors.sd_min_periods)
    panel = anc.set_index(["t", "symbol"])["anchor_price"].astype(bool)
    k = direct.index.intersection(panel.index)
    mism = [f"{t.date()}:{s}" for (t, s) in k if bool(direct[(t, s)]) != bool(panel[(t, s)])]
    anchors = {"stock_weeks": int(len(k)), "missing_in_panel": int(len(direct.index.difference(panel.index))),
               "price_anchor_hits": int(panel.reindex(k).sum()), "mismatches": len(mism), "examples": mism[:5],
               "filing_anchor_hits": int(anc["anchor_filing"].astype(bool).sum()),
               "filing_source": bool(anc.attrs.get("filing_source", True))}
    from jevtrader.pipeline.variants import variant_config

    rules = {"anchored": (cfg, "anchored", "n_prob"),
             "bylines": (variant_config(cfg, "V-excl-bylines"), "bylines", "n_prob"),
             "items": (variant_config(cfg, "V-excl-items"), "items", "n_prob_items"),
             "anchored_collapsed": (variant_config(cfg, "V-excl-collapsed"), "anchored", "n_prob_collapsed")}
    risk = inputs.risk.assign(t=pd.to_datetime(inputs.risk["t"]))
    per_rule, ok = {}, True
    for name, (c, rule, key) in rules.items():
        c = dataclasses.replace(c, heads=False, name=f"chk-m5-{name}")
        res = run_eval(c, inputs, work)
        mult = res.log.multipliers.assign(t=lambda d: pd.to_datetime(d["t"]))
        bad, n_ex, n_half_unanch, anch_excl_without_A = 0, 0, 0, 0
        for t, g in mult.groupby("t"):
            g = g.set_index("symbol")
            rr = risk[risk["t"] == t].set_index("symbol").reindex(g.index)
            A = anc[anc["t"] == t].set_index("symbol")["anchor"].astype(float).reindex(g.index)
            want = bg3_rule(rr, A, rule, key, c.risk.exclusion_kinds, c.risk.corroboration)
            got_ex = g["sentinel_corroborated"] <= 0
            got_half = g["sentinel_unanchored"] < 1
            bad += int((got_ex != want["excl"]).sum() + (got_half != want["unanchored_half"]).sum())
            n_ex += int(got_ex.sum())
            n_half_unanch += int(got_half.sum())
            if rule == "anchored":
                anch_excl_without_A += int((got_ex & ~(A.fillna(0) > 0)).sum())
        rep = B.exclusion_rule_report({name: res}, inputs.labels)[name]
        per_rule[name] = {"mismatches_vs_independent_rule": bad, "sentinel_exclusions": n_ex,
                          "unanchored_halvings": n_half_unanch, "exclusions_without_anchor": anch_excl_without_A,
                          "n_exclusions_all_causes": rep["n_exclusions"], "n_halvings_all_causes": rep["n_halvings"]}
        ok = ok and bad == 0 and (rule != "anchored" or anch_excl_without_A == 0)
    eligible = {name: int(((pd.to_numeric(risk[key], errors="coerce").fillna(0) >= 2)
                           & risk["problem_kind"].isin(list(cfg.risk.exclusion_kinds))
                           & risk["t"].isin(dates)).sum())
                for name, (_, _, key) in rules.items() if key in risk}
    return {"anchors": anchors, "rules": per_rule, "eligible_risk_rows": eligible,
            "pass": bool(ok and anchors["mismatches"] == 0 and anchors["missing_in_panel"] == 0)}


# ------------------------------------------------------------------------------------ M7
DET_FILES = ("weights.parquet", "cards.jsonl", "metrics.json", "predictions.parquet", "sizing.jsonl",
             "sizing_weeks.jsonl", "heads.parquet", "daily_returns.parquet", "snapshots.jsonl", "multipliers.parquet")
DET_SCRIPT = """
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, {src!r}); sys.path.insert(0, {pre!r})
import caproot
from jevtrader.pipeline import layer_b as LB
from jevtrader.eval.bound import bound_check
from jevtrader.pipeline.registry import canonical_json
root, cap, gen = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cfg = LB.DecisionConfig.from_yaml(gen_id=gen, name="primary")
start, end = LB.resolve_window("W_pre_slice")
inputs = LB.load_inputs(cfg, start, end, root, data_root=cap)
res = LB.evaluate(cfg, "W_pre_slice", inputs, root=root, register=False, write=True)
out = {{f: hashlib.sha256((res.out_dir / f).read_bytes()).hexdigest() for f in {files!r}}}
out["bound"] = hashlib.sha256(canonical_json(bound_check(res.log, cfg)).encode()).hexdigest()
out["code_sha"] = res.metrics["code_sha"]
print(json.dumps(out, sort_keys=True))
"""


def m7(gen_id: str, cap: Path, work: Path) -> dict:
    code = DET_SCRIPT.format(src=str(ROOT / "src"), pre=str(ROOT / "experiments" / "pre"), files=DET_FILES)
    outs = {}
    for seed in ("1", "2", "3"):
        root = work / f"det{seed}"
        (root / "results").mkdir(parents=True, exist_ok=True)
        link = root / "results" / "judgments"
        if not link.exists():
            link.symlink_to(ROOT / "results" / "judgments")
        r = subprocess.run([sys.executable, "-c", code, str(root), str(cap), gen_id], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": seed}, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-3000:])
        outs[seed] = json.loads(r.stdout.strip().splitlines()[-1])
    keys = sorted(outs["1"])
    diff = sorted(k for k in keys if len({outs[s][k] for s in outs}) != 1)
    return {"seeds": sorted(outs), "files": keys, "differ": diff, "sha256_seed1": outs["1"], "pass": not diff}


# ------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", required=True)
    ap.add_argument("--cap-root", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated subset of M1..M7 (default: all)")
    a = ap.parse_args(argv)
    want = set(a.only.split(",")) if a.only else {"M1", "M2", "M3", "M4", "M5", "M6", "M7"}
    t0 = time.time()
    cap = caproot.capped_root(Path(a.cap_root))
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    LF.CATALOG[INJ] = LF.FeatureSpec(INJ, "technical", "T", "code")  # the planted score, in this process only
    cfg, inputs = load(a.gen_id, cap)
    out: dict = {"gen_id": a.gen_id, "window": WINDOW, "price_cap": str(caproot.PRICE_CAP.date()),
                 "criteria": CRITERIA, "seed": SEED}
    prim = None
    if want & {"M4", "M6", "M2"}:
        prim = run_eval(dataclasses.replace(cfg, name="chk-primary"), inputs, work)
    if want & {"M1", "M2", "M4"}:
        r = m1_m2_m4(cfg, inputs, cap, work)
        out.update({k: v for k, v in r.items() if k in ("independent_vs_labelstore", "M1a", "M1b", "M2")})
        out["M4"] = {"shuffled_through_layer_b": r["M4_shuffled_through_layer_b"]}
        pp = primary_placebo_and_null(prim, rel_pivot(inputs.labels))
        out["M2"]["primary_L_vs_labels_descriptive"] = pp["primary_L_vs_labels"]
        out["M4"]["permutation_null_primary"] = pp["permutation_null"]
        sh = r["M4_shuffled_through_layer_b"]["ic_layer_b"]
        out["M4"]["pass"] = bool(abs(sh["t_nw4"]) < 3 and abs(pp["permutation_null"]["mean_of_means"]) <= 0.002)
        print(f"M1/M2/M4 done ({time.time() - t0:.0f}s)", flush=True)
    if "M3" in want:
        out["M3"] = m3(cfg, inputs, work)
        print(f"M3 done ({time.time() - t0:.0f}s)", flush=True)
    if "M5" in want:
        out["M5"] = m5(cfg, inputs, work)
        print(f"M5 done ({time.time() - t0:.0f}s)", flush=True)
    if "M6" in want:
        bc = B.bound_check(prim.log, cfg)
        learned_weeks = int((~prim.log.predictions.drop_duplicates("t")["cold"]).sum())
        out["M6"] = {**bc, "learned_weeks": learned_weeks,
                     "pass": bool(bc["passed"] and bc["n_weeks"] == learned_weeks
                                  and bc["n_entry_checks"] >= learned_weeks and bc["n_protection_checks"] >= 1)}
    if "M7" in want:
        out["M7"] = m7(a.gen_id, cap, work)
        print(f"M7 done ({time.time() - t0:.0f}s)", flush=True)
    out["summary"] = {k: out[k]["pass"] for k in ("M1a", "M1b", "M2", "M3", "M4", "M5", "M6", "M7") if k in out}
    out["wall_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(_plain(out), indent=1, sort_keys=True, default=str))
    print(json.dumps(out["summary"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
