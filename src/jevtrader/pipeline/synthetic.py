"""Synthetic market and judgment tables for Layer-B tests and the synthetic-signal injection check (design §13.10).

Prices follow a one-factor market model with idiosyncratic noise; a planted signal is the standardized realized
next-week relative return plus noise ("score = label + noise"), so a correct, point-in-time Layer B must recover a
positive IC through matured labels only. The calendar drops a Friday and a Monday to create holiday weeks, one name
is delisted mid-sample and one is listed late.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jevtrader.backtest.engine import PriceBook, weekly_decision_dates
from jevtrader.features.anchors import AnchorConfig, anchor_panel
from jevtrader.learn.features import CATALOG
from jevtrader.learn.labels import LabelStore, idio_vol_weekly
from jevtrader.pipeline.layer_b import LayerBInputs

FF12 = ("NoDur", "Durbl", "Manuf", "Enrgy", "Chems", "BusEq", "Telcm", "Utils", "Shops", "Hlth", "Money", "Other")
SYN_TRUSTED = ("mom_12_1", "x_trend", "t_breakout")
SYN_UNTRUSTED = ("f_imp", "no_news", "h")
DELISTING = {"S001": -0.3}  # the synthetic delisted name (explicit return, V-m14)


def synthetic_bars(n_stocks: int = 40, n_days: int = 420, seed: int = 0, start: str = "2020-06-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    fri = next(d for d in days[150:] if d.weekday() == 4)
    mon = next(d for d in days[211:] if d.weekday() == 0)
    days = days.drop([fri, mon])  # holiday weeks: one without its Friday, one without its Monday
    n = len(days)
    mkt = rng.normal(0.0003, 0.01, n)
    rows = []
    for j in range(n_stocks):
        sym = f"S{j:03d}"
        beta, ivol = rng.uniform(0.6, 1.4), rng.uniform(0.008, 0.025)
        lr = beta * mkt + rng.normal(0, ivol, n)
        close = 50.0 * np.exp(np.cumsum(lr))
        opn = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
        life = slice(0, n)
        if j == 1:
            life = slice(0, n - 120)  # delisted
        elif j == 2:
            life = slice(90, n)  # listed late
        for d, o, c in zip(days[life], opn[life], close[life]):
            rows.append((sym, d, o, c))
    spy = 300.0 * np.exp(np.cumsum(mkt))
    spy_o = np.r_[spy[0], spy[:-1]]
    rows += [("SPY", d, o, c) for d, o, c in zip(days, spy_o, spy)]
    return pd.DataFrame(rows, columns=["symbol", "date", "open", "close"])


def synthetic_inputs(
    n_stocks: int = 40, n_days: int = 420, seed: int = 0, first_decision: int = 30, trusted_signal: float = 0.6,
    text_signal: float = 0.6, coverage: float = 0.6, start: str = "2020-06-01", cash_rate: float = 0.03,
) -> LayerBInputs:
    """LayerBInputs over every weekly decision from the `first_decision`-th on (enough history for sigma). Every
    input the Layer-B input checks require is present (industry map, risk table, cash rate `cash_rate`/252, and the
    trusted anchor panel computed from the synthetic prices with no filing source configured)."""
    rng = np.random.default_rng(seed + 1)
    bars = synthetic_bars(n_stocks, n_days, seed, start)
    prices = PriceBook(bars)
    stocks = sorted(s for s in bars["symbol"].unique() if s != "SPY")
    cal = weekly_decision_dates(prices.sessions, prices.sessions[0], prices.sessions[-1])
    dates = cal[first_decision:-1]
    alive = {t: [s for s in stocks if np.isfinite(prices.open.at[t, s])] for t in dates}
    universe = {t: tuple(alive[t]) for t in dates}
    sigma = idio_vol_weekly(prices.close, ewm_min_periods=10, min_periods=20)
    labels = LabelStore.build(prices, dates, universe, sigma, delisting_returns=DELISTING)
    lab = labels.table.set_index(["t", "symbol"])
    rows = []
    for t in dates:
        uni = universe[t]
        rel = pd.Series({s: lab["r"].get((t, s), np.nan) - lab["m"].get((t, s), np.nan) for s in uni})
        z = ((rel - rel.mean()) / rel.std()).fillna(0.0)  # the planted "label + noise" signal
        past = prices.close.loc[:t].iloc[-60:]
        mom = (past.iloc[-1] / past.iloc[0] - 1.0).reindex(uni)
        cov = rng.uniform(size=len(uni)) < coverage
        for k, s in enumerate(uni):
            rows.append((t, s, "mom_12_1", float(mom[s]), bool(np.isfinite(mom[s]))))
            rows.append((t, s, "x_trend", float(trusted_signal * z[s] + rng.normal()), True))
            rows.append((t, s, "t_breakout", float(np.clip(rng.normal(0, 2), -4, 4)), bool(rng.uniform() > 0.02)))
            rows.append((t, s, "f_imp", float(text_signal * z[s] + rng.normal()) if cov[k] else np.nan, bool(cov[k])))
            rows.append((t, s, "no_news", float(not cov[k]), True))
            rows.append((t, s, "h", float(rng.beta(0.5, 5)), bool(cov[k])))
    raw = pd.DataFrame(rows, columns=["t", "symbol", "feature", "value", "covered"])
    industry = {s: FF12[i % len(FF12)] for i, s in enumerate(stocks)}
    risk_rows = []
    for t in dates:
        for s in universe[t]:
            if rng.uniform() < 0.05:
                risk_rows.append((t, s, int(rng.integers(0, 3)), float(rng.uniform(0.5, 1.0)), int(rng.integers(0, 3)),
                                  str(rng.choice(["financial_distress", "product_safety", "litigation"]))))
    risk = pd.DataFrame(risk_rows, columns=["t", "symbol", "sessions_to_earnings", "h", "n_prob", "problem_kind"])
    risk = _complete_risk(risk, raw, seed)
    cash = pd.Series(cash_rate / 252.0, index=prices.sessions)
    anchors = anchor_panel(prices.close, dates, universe, None, AnchorConfig(use_filing=False), calendar=cal)
    return LayerBInputs(raw=raw, universe=universe, labels=labels, sigma=sigma, industry=industry, risk=risk,
                        prices=prices, cash_rate_daily=cash, delisting_returns=DELISTING, anchors=anchors)


def _complete_risk(risk: pd.DataFrame, raw: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The risk table in Layer A's layout (pipeline/inputs.py): the drawn problem rows get the BG-3c key columns
    (`n_prob_items` >= `n_prob` >= `n_prob_collapsed`), and every other stock-week with news (a covered `f_imp`) gets
    the sentinel row Layer A writes for it: h = the week's sentinel feature, n_prob = 1 if h >= .8 else 0, no kind.
    A separate generator keeps the drawn rows and every other synthetic input unchanged."""
    rng = np.random.default_rng(seed + 2)
    n = risk["n_prob"].to_numpy()
    risk = risk.assign(n_prob_items=n + rng.integers(0, 2, len(risk)),
                       n_prob_collapsed=np.minimum(n, rng.integers(1, 3, len(risk))))
    news = raw[(raw["feature"] == "f_imp") & raw["covered"]][["t", "symbol"]]
    h = raw[raw["feature"] == "h"].set_index(["t", "symbol"])["value"]
    have = set(zip(risk["t"], risk["symbol"]))
    add = [(t, s) for t, s in zip(news["t"], news["symbol"]) if (t, s) not in have]
    hv = np.array([float(h.get((t, s), np.nan)) for t, s in add])
    k = (hv >= 0.8).astype(int)
    fill = pd.DataFrame({"t": [t for t, _ in add], "symbol": [s for _, s in add], "sessions_to_earnings": np.nan,
                         "h": hv, "n_prob": k, "problem_kind": None, "n_prob_items": k, "n_prob_collapsed": k})
    out = pd.concat([risk, fill], ignore_index=True)
    return out.sort_values(["t", "symbol"], kind="mergesort").reset_index(drop=True)


def write_judgment_tables(inputs: LayerBInputs, gen_dir: Path) -> None:
    """Judgment tables in Layer A's layout (programs/runner.py): `t_breakout` only as T.breakout Noul answers in
    answers.parquet (qid "T.breakout", some degraded), every other feature in features.parquet under
    program-prefixed names with null values for abstentions and no `covered` column."""
    gen_dir = Path(gen_dir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    raw = inputs.raw
    tb = raw[raw["feature"] == "t_breakout"]
    probs = [None if not c else json.dumps({"yes": float(1 / (1 + np.exp(-v))), "no": float(1 - 1 / (1 + np.exp(-v)))})
             for v, c in zip(tb["value"], tb["covered"])]
    ans = pd.DataFrame({"t": tb["t"].to_numpy(), "symbol": tb["symbol"].to_numpy(), "program": "T", "stage": "T",
                        "qid": "T.breakout", "template": "T.breakout", "kind": "noul", "probs_json": probs,
                        "request_key": "synthetic"})
    ans.to_parquet(gen_dir / "answers.parquet")
    prefix = {"x_trend": "T.x_trend", "f_imp": "N.f_imp", "no_news": "N.no_news", "h": "N1.h"}
    rest = raw[raw["feature"] != "t_breakout"]
    feats = pd.DataFrame({
        "t": rest["t"].to_numpy(), "symbol": rest["symbol"].to_numpy(), "program": "syn",
        "feature": rest["feature"].map(lambda f: prefix.get(f, f)).to_numpy(),
        "value": rest["value"].where(rest["covered"]).to_numpy(),
        "taint": rest["feature"].map(lambda f: CATALOG[f].channel).to_numpy(),  # the emitting program's channel
    })
    feats.to_parquet(gen_dir / "features.parquet")
