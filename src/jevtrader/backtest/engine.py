"""Weekly rebalancing backtest on adjusted daily bars.

Timing convention (no look-ahead):
- A decision made on session `d` may use information available up to d's close (16:00 ET).
- Its target weights are filled at the OPEN of the next session `e` (the execution session).
- Between fills the book drifts with prices; cash earns the daily rate `cash_rate_daily` (DFF/252 from FRED, PIT;
  `features.macro.MacroStore.cash_rate_daily`). Sharpe ratios use `excess_returns` over that rate (V-m11).
- Costs are charged on traded notional sum|w_target - w_drifted| at each fill (bps per side), paid from cash.

Each day is attributed exactly once, from adjusted prices:
  fill day e:        close_{e-1} -> open_e on the drifted book, rebalance, then open_e -> close_e
  other days k:      close_{k-1} -> close_k
Inside a symbol's trading life, missing bars are forward-filled (a halt is a zero return). A symbol's life ends at
its last bar (build the book from gap-trimmed bars, data/prices.py, so a reused ticker does not extend it).

Two design_v1 rules (section 2.1, 9.2):
- An unfillable target (no open at e: the name's life has ended, or it has no bars) is NOT an error: its weight
  stays in cash and the event is logged in `meta["unfilled"]` (V-M2). Callers must never filter targets on
  next-session tradability, which is future information.
- A held name whose life ends is exited at its last close times (1 + delisting return). The return must be given
  explicitly unless the exit is a known M&A exit (`ma_exits`, cash at the last close); any other delisting of a held
  name raises `DelistingError` (V-m14), so a silent 0% return can never hide a bankruptcy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class DelistingError(RuntimeError):
    pass


@dataclass
class BacktestResult:
    nav: pd.Series  # daily NAV at close, starts at 1.0 before the first fill
    daily_returns: pd.Series
    weights: pd.DataFrame  # target weights applied at each execution session (after unfillable names -> cash)
    turnover: pd.Series  # one-way traded notional / NAV at each fill
    costs: pd.Series  # cost as a fraction of NAV at each fill
    exposure: pd.Series  # gross weight after each fill
    rf: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # daily cash rate on each day
    meta: dict = field(default_factory=dict)

    @property
    def excess_returns(self) -> pd.Series:
        """Daily returns in excess of the cash rate (the Sharpe/IR/CER convention of design_v1 section 9.4)."""
        return self.daily_returns - self.rf.reindex(self.daily_returns.index).fillna(0.0)


class PriceBook:
    """Wide adjusted open/close tables indexed by session, forward-filled within each symbol's life."""

    def __init__(self, bars: pd.DataFrame):
        b = bars.copy()
        b["date"] = pd.to_datetime(b["date"])
        close = b.pivot(index="date", columns="symbol", values="close").sort_index()
        opn = b.pivot(index="date", columns="symbol", values="open").reindex(close.index)
        self.first = close.apply(lambda c: c.first_valid_index())
        self.last = close.apply(lambda c: c.last_valid_index())
        alive = pd.DataFrame(
            {s: (close.index >= self.first[s]) & (close.index <= self.last[s]) for s in close.columns},
            index=close.index,
        )
        self.close = close.ffill().where(alive)
        # a missing open inside the symbol's life trades at the previous close
        self.open = opn.fillna(close.ffill().shift(1)).where(alive)
        self.sessions = close.index

    def next_session(self, d: pd.Timestamp) -> pd.Timestamp | None:
        i = self.sessions.searchsorted(pd.Timestamp(d), side="right")
        return self.sessions[i] if i < len(self.sessions) else None

    def tradable(self, day: pd.Timestamp) -> set[str]:
        """Names with an open on `day`. For the engine's own use at the fill; never a decision input."""
        row = self.open.loc[day]
        return set(row.index[np.isfinite(row.values)])


def weekly_decision_dates(sessions: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """Last session of each ISO week within [start, end]."""
    s = pd.Series(sessions[(sessions >= start) & (sessions <= end)])
    iso = s.dt.isocalendar()
    return list(s.groupby([iso.year.values, iso.week.values]).max())


def run_backtest(
    prices: PriceBook,
    targets: dict[pd.Timestamp, dict[str, float]],
    cost_bps: float = 15.0,
    cash_rate_daily: pd.Series | None = None,
    delisting_returns: dict[str, float] | None = None,
    end: pd.Timestamp | None = None,
    ma_exits: set[str] | frozenset[str] = frozenset(),
) -> BacktestResult:
    """`targets[d]` = weights decided at the close of session d (sum of weights <= 1; rest is cash). Long-only and
    cash-only (design_v1 section 9.2 hard limits): a negative weight or a sum above 1 raises ValueError."""
    delist = _DelistingPolicy(dict(delisting_returns or {}), frozenset(ma_exits))
    fills: dict[pd.Timestamp, tuple[pd.Timestamp, dict[str, float]]] = {}
    for d in sorted(targets):
        _check_long_only(d, targets[d])
        e = prices.next_session(d)
        if e is not None:
            fills[e] = (pd.Timestamp(d), targets[d])
    if not fills:
        raise ValueError("no executable decisions")
    first_fill = min(fills)
    last = pd.Timestamp(end) if end is not None else prices.sessions[-1]
    days = prices.sessions[(prices.sessions >= first_fill) & (prices.sessions <= last)]

    w: dict[str, float] = {}
    nav = 1.0
    navs, rets, rfs, wlog, tlog, clog, elog = {}, {}, {}, {}, {}, {}, {}
    unfilled: list[tuple[pd.Timestamp, pd.Timestamp, str, float]] = []
    prev = None
    for k in days:
        r_cash = float(cash_rate_daily.get(k, 0.0)) if cash_rate_daily is not None else 0.0
        if k in fills:
            r_on = 0.0
            if prev is not None and w:
                w, r_on = _step(w, prices.close.loc[prev], prices.open.loc[k], 0.0, delist, k)
            decided, raw_tgt = fills[k]
            ok = prices.tradable(k)
            tgt = {}
            for s, x in raw_tgt.items():
                if abs(x) <= 1e-12:
                    continue
                if s in ok:
                    tgt[s] = float(x)
                else:
                    unfilled.append((decided, k, s, float(x)))  # stays in cash (V-M2)
            # sorted: a set's order follows the per-process string hash seed, and float sums depend on order, so
            # an unsorted sum made replays differ in the last bits across processes (replay determinism, OPS-3)
            traded = sum(abs(tgt.get(s, 0.0) - w.get(s, 0.0)) for s in sorted(set(w) | set(tgt)))
            cost = traded * cost_bps / 1e4
            tlog[k], clog[k], wlog[k] = traded, cost, pd.Series(tgt, dtype=float)
            elog[k] = sum(abs(x) for x in tgt.values())
            # weights are fractions of pre-cost NAV; the cost is paid out of cash
            w, r_day = _step(tgt, prices.open.loc[k], prices.close.loc[k], r_cash, delist, k, extra_cash=-cost)
            day_ret = r_on + r_day + r_on * r_day  # (1 + r_on)(1 + r_day) - 1, without cancellation
        elif w:
            w, day_ret = _step(w, prices.close.loc[prev], prices.close.loc[k], r_cash, delist, k)
        else:
            day_ret = r_cash
        nav *= 1.0 + day_ret
        navs[k], rets[k], rfs[k] = nav, day_ret, r_cash
        prev = k
    return BacktestResult(
        nav=pd.Series(navs),
        daily_returns=pd.Series(rets),
        weights=pd.DataFrame(wlog).T.fillna(0.0),
        turnover=pd.Series(tlog, dtype=float),
        costs=pd.Series(clog, dtype=float),
        exposure=pd.Series(elog, dtype=float),
        rf=pd.Series(rfs, dtype=float),
        meta={"cost_bps": cost_bps, "n_fills": len(fills),
              "unfilled": pd.DataFrame(unfilled, columns=["decision", "fill", "symbol", "weight"]),
              "delisted": pd.DataFrame(delist.log, columns=["day", "symbol", "kind", "return"])},
    )


def _check_long_only(d: pd.Timestamp, w: dict[str, float], tol: float = 1e-9) -> None:
    short = {s: x for s, x in w.items() if not x >= -tol}  # also catches NaN
    total = sum(w.values())
    if short or not total <= 1.0 + tol:
        raise ValueError(f"targets decided on {pd.Timestamp(d).date()} are not long-only and cash-only "
                         f"(negative or NaN: {sorted(short)[:5]}, sum {total:.6f})")


@dataclass
class _DelistingPolicy:
    returns: dict[str, float]
    ma_exits: frozenset[str]
    log: list = field(default_factory=list)

    def exit_return(self, symbol: str, day: pd.Timestamp) -> float:
        if symbol in self.returns:
            r, kind = self.returns[symbol], "explicit"
        elif symbol in self.ma_exits:
            r, kind = 0.0, "mna"
        else:
            raise DelistingError(f"held symbol {symbol} stopped trading before {day.date()} and is not a known M&A "
                                 "exit; supply an explicit delisting return (design_v1 section 9.2, V-m14)")
        self.log.append((day, symbol, kind, r))
        return r


def _step(
    w: dict[str, float],
    p0: pd.Series,
    p1: pd.Series,
    r_cash: float,
    delist: _DelistingPolicy,
    day: pd.Timestamp,
    extra_cash: float = 0.0,
) -> tuple[dict[str, float], float]:
    """Move the book from price vector p0 to p1. Returns (new weights as fractions of the new NAV,
    portfolio return). A symbol priced at p0 but not at p1 has ended its life: it is exited at
    p0 * (1 + delisting return) and its value becomes cash. The return is accumulated from its parts (not as
    total value - 1), so an all-cash day returns the cash rate exactly and has zero excess return."""
    val, cash, ret = {}, 1.0 - sum(w.values()) + extra_cash, extra_cash
    for s, x in w.items():
        a, b = p0.get(s, np.nan), p1.get(s, np.nan)
        if not np.isfinite(a):
            raise ValueError(f"held symbol {s} has no price at leg start")
        if np.isfinite(b):
            val[s] = x * b / a
            ret += x * (b / a - 1.0)
        else:
            r = delist.exit_return(s, day)
            cash += x * (1.0 + r)
            ret += x * r
    ret += cash * r_cash
    total = 1.0 + ret
    return {s: v / total for s, v in val.items()}, ret
