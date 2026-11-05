"""EXPLORATORY v2 (docs/design/v2_exploratory_spec.md): shared loaders, a daily book simulator and metrics.

Registered code is imported read-only; nothing here writes to results/analysis, results/eval or trials.jsonl."""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments" / "analysis")]

from jevtrader.backtest.engine import PriceBook, _DelistingPolicy, _step  # noqa: E402
from jevtrader.pipeline.layer_b import DecisionConfig, load_inputs, resolve_window  # noqa: E402

SPEC = ROOT / "docs/design/v2_exploratory_spec.md"
OUT = ROOT / "results/v2"
CACHE = ROOT / "data/cache/v2"  # gitignored cache dir (regenerable)
GENS = {"W_pre_slice": "pre_blinded_emu_wpre_s30", "W_clean": "main_emu_wclean_u100"}
# W_hist: trusted code features only (scripts/build_code_features.py into the gitignored cache); no judgments
HIST = ("2019-06-07", "2023-03-31")
HIST_DIR = ROOT / "data/cache/v2/hist"
CODE_FEATURES = ("mom_12_1", "rev_1w", "rev_1m", "vol_60d", "dist_52w_high", "rel_sector_3m", "earnings_yield",
                 "sales_yield", "gross_profitability", "operating_margin", "rev_growth_ttm", "share_change_1y",
                 "fcf_yield", "accruals")
COST_BPS = 15.0
ET = "America/New_York"


def spec_sha() -> str:
    return hashlib.sha256(SPEC.read_bytes()).hexdigest()


@dataclass
class Data:
    window: str
    inputs: object  # LayerBInputs
    code: pd.DataFrame  # index (t, symbol), columns CODE_FEATURES (raw values)
    labels: pd.DataFrame
    dates: list


def load(window: str) -> Data:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{window}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    if window == "W_hist":
        cfg = DecisionConfig.from_yaml()
        s, e = (pd.Timestamp(x) for x in HIST)
        inp = load_inputs(cfg, s, e, gen_dir=HIST_DIR, anchors=pd.DataFrame(), evidence={}, focus=None)
    else:
        cfg = DecisionConfig.from_yaml(gen_id=GENS[window])
        s, e = resolve_window(window)
        inp = load_inputs(cfg, s, e)
    raw = inp.raw[inp.raw["feature"].isin(CODE_FEATURES)]
    code = raw.pivot_table(index=["t", "symbol"], columns="feature", values="value").reindex(columns=CODE_FEATURES)
    dates = [pd.Timestamp(t) for t in sorted(inp.universe) if s <= pd.Timestamp(t) <= e]
    d = Data(window, inp, code, inp.labels.table.copy(), dates)
    f.write_bytes(pickle.dumps(d))
    return d


def cutoff_utc(t) -> pd.Timestamp:
    """c_t: 16:00 ET on decision date t."""
    return pd.Timestamp(t).tz_localize(None).replace(hour=16).tz_localize(ET).tz_convert("UTC")


def rank_pm1(x: pd.Series) -> pd.Series:
    """Cross-sectional rank mapped to [-1, 1]; missing -> 0."""
    r = x.rank(method="average")
    n = r.notna().sum()
    return ((r - 1) / (n - 1) * 2 - 1).fillna(0.0) if n > 1 else pd.Series(0.0, index=x.index)


# ------------------------------------------------------------------------------------------------ simulator
@dataclass
class SimResult:
    returns: pd.Series
    rf: pd.Series
    turnover: pd.Series  # one-way traded fraction per trade day (sum |dw| / 2 is not used: this is sum |dw|)
    costs: pd.Series
    weights: dict


def simulate(prices: PriceBook, first_day, last_day, decide: Callable[[pd.Timestamp, dict], dict | None],
             cash_rate: pd.Series | None, delisting_returns=None, ma_exits=frozenset(),
             cost_bps: float = COST_BPS) -> SimResult:
    """Daily book with the registered engine's accounting (engine._step): at each session k the book moves from
    close(k-1) to open(k); `decide(k, w_pretrade)` may return target weights traded at open(k) (cost from cash);
    the book then moves open(k) -> close(k). Long-only, fully invested at most (checked)."""
    delist = _DelistingPolicy(dict(delisting_returns or {}), frozenset(ma_exits))
    days = prices.sessions[(prices.sessions >= pd.Timestamp(first_day)) & (prices.sessions <= pd.Timestamp(last_day))]
    w: dict = {}
    prev = None
    rets, rfs, tos, costs, wlog = {}, {}, {}, {}, {}
    for k in days:
        r_cash = float(cash_rate.get(k, 0.0)) if cash_rate is not None else 0.0
        r_on = 0.0
        if prev is not None and w:
            w, r_on = _step(w, prices.close.loc[prev], prices.open.loc[k], 0.0, delist, k)
        tgt = decide(k, dict(w))
        if tgt is not None:
            ok = prices.open.loc[k]
            tgt = {s: float(x) for s, x in tgt.items() if x > 1e-12 and np.isfinite(ok.get(s, np.nan))}
            if any(x < 0 for x in tgt.values()) or sum(tgt.values()) > 1 + 1e-9:
                raise ValueError(f"long-only / cash-only violated on {k.date()}")
            traded = sum(abs(tgt.get(s, 0.0) - w.get(s, 0.0)) for s in sorted(set(w) | set(tgt)))
            cost = traded * cost_bps / 1e4
            tos[k], costs[k], wlog[k] = traded, cost, tgt
            w, r_day = _step(tgt, prices.open.loc[k], prices.close.loc[k], r_cash, delist, k, extra_cash=-cost)
        elif prev is None:  # nothing held yet
            r_day = r_cash
        else:
            w, r_day = _step(w, prices.open.loc[k], prices.close.loc[k], r_cash, delist, k) if w else ({}, r_cash)
        rets[k] = r_on + r_day + r_on * r_day
        rfs[k] = r_cash
        prev = k
    return SimResult(pd.Series(rets), pd.Series(rfs), pd.Series(tos, dtype=float), pd.Series(costs, dtype=float), wlog)


def metrics(res: SimResult, bench: SimResult | None = None, boot: int = 999) -> dict:
    import stats as S  # experiments/analysis/stats.py (registered, read-only)

    ex = res.returns - res.rf
    nav = (1 + res.returns).cumprod()
    years = len(ex) / 252
    out = {"n_days": len(ex), "ann_excess": float((1 + ex).prod() ** (1 / years) - 1),
           "ann_vol": float(ex.std(ddof=1) * np.sqrt(252)),
           "sharpe": S.sharpe_ci(ex, 5, boot, 0), "max_dd": float((nav / nav.cummax() - 1).min()),
           "turnover_per_year": float(res.turnover.sum() / 2 / years), "cost_drag": float(res.costs.sum() / years)}
    out["sharpe"] = {k: out["sharpe"][k] for k in ("sharpe", "ci")}
    if bench is not None:
        a = (res.returns - bench.returns.reindex(res.returns.index)).dropna()
        out["active_ann"] = float(a.mean() * 252)
        out["te"] = float(a.std(ddof=1) * np.sqrt(252))
        out["ir"] = out["active_ann"] / out["te"] if out["te"] > 0 else None
    return out


def write(name: str, payload: dict) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"label": "EXPLORATORY v2 (post hoc; not pre-registered)", "spec_sha256": spec_sha(), **payload}
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(payload, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return p
