#!/usr/bin/env python3
"""EXPLORATORY v2, items 1+2 (docs/design/v2_exploratory_spec.md): trusted code-feature scores x book construction
x rebalance horizon, simulated daily with the registered engine's accounting. $0.

    .venv/bin/python experiments/v2/v2_portfolio.py --window W_pre_slice      # development / plumbing
    .venv/bin/python experiments/v2/v2_portfolio.py --window W_clean          # the one evaluation pass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

K, KEEP, IND_MAX = 20, 40, 6
FM_TRAIL, FM_MIN, FM_SHRINK = 104, 26, 0.5
PRIMARY = "UAW-BUF-R4"


def ranked_panel(code: pd.DataFrame) -> pd.DataFrame:
    return code.groupby(level="t", group_keys=False).apply(lambda g: g.apply(C.rank_pm1))


def uaw_weights() -> pd.Series:
    signs = yaml.safe_load((C.ROOT / "configs/learning_v1.yaml").read_text())["variants"]["uaw_signs"]
    w = pd.Series({f: float(signs[f]) for f in C.CODE_FEATURES})
    return w / (w != 0).sum()


def fm_slopes(X: pd.DataFrame, labels: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """Per-label-week cross-sectional OLS slopes of the relative return on the ranked features, and each week's
    maturity time (the latest available_at of its rows)."""
    r, m, av = ("r4", "m4", "available_at4") if horizon == 4 else ("r", "m", "available_at")
    lab = labels.dropna(subset=[r, m]).set_index(["t", "symbol"])
    j = X.join((lab[r] - lab[m]).rename("y"), how="inner").join(lab[av].rename("av"))
    rows, mat = {}, {}
    for t, g in j.groupby(level="t"):
        if len(g) < 30:
            continue
        A = np.c_[np.ones(len(g)), g[list(C.CODE_FEATURES)].to_numpy()]
        beta, *_ = np.linalg.lstsq(A, g["y"].to_numpy(), rcond=None)
        rows[t], mat[t] = beta[1:], pd.Timestamp(g["av"].max())
    return pd.DataFrame(rows, index=list(C.CODE_FEATURES)).T.sort_index(), pd.Series(mat).sort_index()


def scores_for(t, Xt: pd.DataFrame, kind: str, slopes=None) -> pd.Series:
    if kind == "MOM":
        return Xt["mom_12_1"]
    uaw = uaw_weights()
    if kind == "UAW":
        return Xt @ uaw
    sl, mat = slopes[kind]
    ok = mat.index[mat < C.cutoff_utc(t)]
    if len(ok) < FM_MIN:
        return Xt @ uaw
    b = sl.loc[ok[-FM_TRAIL:]].mean() * FM_SHRINK
    return Xt @ b


def book(kind: str, s: pd.Series, held: set, industry: dict) -> dict:
    s = s.sort_values(ascending=False)
    n = len(s)
    if kind == "TILT":
        z = C.rank_pm1(s)
        w = (1 + z) / n
        return (w / w.sum()).to_dict()
    rank = pd.Series(np.arange(1, n + 1), index=s.index)
    keep = [x for x in s.index if x in held and rank[x] <= KEEP] if kind == "BUF" else []
    chosen, cnt = [], {}
    for x in keep + [x for x in s.index if x not in keep]:
        if len(chosen) == K:
            break
        g = industry.get(x, "?")
        if kind == "BUF" and cnt.get(g, 0) >= IND_MAX:
            continue
        chosen.append(x)
        cnt[g] = cnt.get(g, 0) + 1
    return {x: 1.0 / len(chosen) for x in chosen}


def targets(d: C.Data, X: pd.DataFrame, score: str, bk: str, every: int, slopes) -> dict:
    out, held = {}, set()
    for i, t in enumerate(d.dates):
        if i % every:
            continue
        u = list(d.inputs.universe[t])
        Xt = X.loc[t].reindex(u).fillna(0.0) if t in X.index.get_level_values(0) else None
        if Xt is None:
            continue
        w = book(bk, scores_for(t, Xt, score, slopes), held, d.inputs.industry or {})
        out[t], held = w, set(w)
    return out


def run_targets(d: C.Data, tg: dict) -> C.SimResult:
    p = d.inputs.prices
    fills = {p.next_session(t): w for t, w in tg.items()}
    last = p.next_session(d.dates[-1])
    end = p.sessions[min(p.sessions.searchsorted(last) + 4, len(p.sessions) - 1)]  # last holding week
    return C.simulate(p, min(fills), end, lambda k, w: fills.get(k), d.inputs.cash_rate_daily,
                      d.inputs.delisting_returns, d.inputs.ma_exits)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, choices=[*C.GENS, "W_hist"])
    ap.add_argument("--boot", type=int, default=999)
    a = ap.parse_args(argv)
    d = C.load(a.window)
    X = ranked_panel(d.code)
    # FM training rows: the evaluation window plus, for W_clean, the earlier W_pre panel (past data only)
    Xl, L = X, d.labels
    past = {"W_clean": ("W_hist", "W_pre_slice"), "W_pre_slice": ("W_hist",)}.get(a.window, ())  # addendum v2.1
    if past:
        ds = [C.load(w) for w in past]
        Xl = pd.concat([*(ranked_panel(x.code) for x in ds), X])
        L = pd.concat([*(x.labels for x in ds), d.labels])
        Xl = Xl[~Xl.index.duplicated(keep="last")]
    slopes = {"FM4": fm_slopes(Xl, L, 4), "FM1": fm_slopes(Xl, L, 1)}
    ew = run_targets(d, {t: {s: 1 / len(d.inputs.universe[t]) for s in d.inputs.universe[t]} for t in d.dates})
    rows = {"EW-U100": C.metrics(ew, None, a.boot)}
    series = {"EW-U100": ew.returns}
    for sc in ("MOM", "UAW", "FM4", "FM1"):
        for bk in ("TOP", "BUF", "TILT"):
            for every, rn in ((1, "R1"), (4, "R4")):
                name = f"{sc}-{bk}-{rn}"
                r = run_targets(d, targets(d, X, sc, bk, every, slopes))
                rows[name] = C.metrics(r, ew, a.boot)
                series[name] = r.returns
                m = rows[name]
                print(f"{name:14s} ex {m['ann_excess']:+.3f} SR {m['sharpe']['sharpe']:+.2f} "
                      f"active {m['active_ann']:+.3f} IR {m['ir']:+.2f} TO/yr {m['turnover_per_year']:.2f}", flush=True)
    m = rows["EW-U100"]
    print(f"{'EW-U100':14s} ex {m['ann_excess']:+.3f} SR {m['sharpe']['sharpe']:+.2f}")
    C.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(series).to_parquet(C.OUT / f"v2_portfolio_{a.window}_returns.parquet")
    p = C.write(f"v2_portfolio_{a.window}", {"window": a.window, "primary": PRIMARY, "n_variants": len(rows) - 1,
                                             "rows": rows, "fm_weeks": {k: len(v[0]) for k, v in slopes.items()}})
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
