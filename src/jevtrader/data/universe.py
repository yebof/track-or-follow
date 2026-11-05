"""Point-in-time, survivorship-bias-free trading universe.

Membership comes from the historical S&P 500 component file (one ticker snapshot per change date).
Tickers in that file are the symbols in use on each date, so a company that renamed (FB -> META)
appears under different tickers over time, and an old ticker can later be reused by an unrelated
issuer. Alpaca attaches a renamed company's full price history to its *current* symbol, so every
point-in-time ticker is resolved to the Alpaca symbol that carries its history by following the
CUSIP-linked name-change chain from Alpaca's corporate-actions feed.

Fixes of design_v1 section 2.2 (V-B2, V-m2):
- ranking on RAW dollar volume (raw close x raw volume);
- a symbol's life ends at its first gap of more than 10 sessions (ticker reuse: APC, FB, NFX), and a member must
  have traded within the 10 sessions before the reconstitution date; `check_no_gaps` asserts both on the result;
- a rename chain is not followed through a ticker that was reassigned to another issuer in between (LLL: L3
  Technologies left the index in 2019; the later LLL of 2021-2022 was an unrelated company, now JXG);
- one share class per company (GOOG/GOOGL, FOX/FOXA, NWS/NWSA, DISCA/DISCK, UA/UAA);
- a point-in-time member whose resolved (renamed) symbol was never downloaded is ranked on its point-in-time
  ticker's own bars (ADS->BFH, FBHS->FBIN, GPS->GAP, HFC->DINO: the old tickers' bars exist up to the rename);
  `members_without_bars` reports these and any member with no bars at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Changes missing from Alpaca's name-change feed (checked by hand against the bar files).
_MANUAL_RENAMES = [
    ("BHGE", "BKR", "2019-10-18"),
]
# Share classes of one company among S&P 500 members since 2019 (secondary -> kept class). The pairs are exactly
# the tickers that share an SEC CIK in data/raw/sec/company_tickers.json, plus DISCA/DISCK (Discovery, delisted
# 2022); tests/test_universe.py re-derives the CIK groups. The kept class is fixed (the class-A/voting line), so the
# choice uses no market data and never flips between months.
SHARE_CLASSES = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "DISCK": "DISCA", "UA": "UAA"}
MAX_GAP = 10  # sessions; see data/prices.py


@dataclass(frozen=True)
class Rename:
    old: str
    new: str
    on: date  # first session under the new symbol


def load_renames(path: Path, membership: pd.DataFrame | None = None) -> list[Rename]:
    """Ticker renames of the same company. Alpaca's name-change feed also carries CUSIP-only changes
    (old == new) and share-line unifications (e.g. CUK -> CCL, where CCL had been trading all along);
    with `membership` given, a rename is dropped when the new ticker was itself an index member during
    the year before the change, since then it was not the company's new name."""
    rows = json.loads(Path(path).read_text())
    out = [Rename(r["old_symbol"], r["new_symbol"], date.fromisoformat(r["process_date"])) for r in rows]
    out += [Rename(o, n, date.fromisoformat(d)) for o, n, d in _MANUAL_RENAMES]
    out = [r for r in out if r.old != r.new]
    if membership is not None:
        dates_by_ticker = membership.groupby("ticker").date.apply(list).to_dict()

        def was_member(t: str, lo: date, hi: date) -> bool:
            return any(lo <= d < hi for d in dates_by_ticker.get(t, []))

        out = [r for r in out if not was_member(r.new, r.on - timedelta(days=365), r.on)]
    return sorted(out, key=lambda r: r.on)


class SymbolResolver:
    """Map (ticker as used on `asof`) -> Alpaca symbol holding the company's continuous history."""

    def __init__(self, renames: list[Rename]):
        self._by_old: dict[str, list[Rename]] = {}
        self._by_new: dict[str, list[Rename]] = {}
        for r in renames:
            self._by_old.setdefault(r.old, []).append(r)
            self._by_new.setdefault(r.new, []).append(r)

    def resolve(self, ticker: str, asof: date) -> str:
        sym, t, seen = ticker, asof, set()
        while True:
            nxt = [r for r in self._by_old.get(sym, []) if r.on > t]
            if not nxt or (sym, t) in seen:
                return sym
            seen.add((sym, t))
            r = min(nxt, key=lambda x: x.on)
            # the ticker was given to another issuer before this rename: the rename is not ours (LLL -> JXJT)
            if any(t < x.on < r.on for x in self._by_new.get(sym, [])):
                return sym
            sym, t = r.new, r.on

    def history(self, symbol: str, start: date, end: date) -> list[tuple[str, date, date]]:
        """Tickers the company currently trading as `symbol` used over [start, end), as (ticker, from, to)
        intervals. News is tagged with the ticker in use when published, so news queries follow these."""
        out, sym, hi = [], symbol, end
        while hi > start:
            prior = [r for r in self._by_new.get(sym, []) if start < r.on < hi]
            if not prior:
                out.append((sym, start, hi))
                break
            r = max(prior, key=lambda x: x.on)
            out.append((sym, r.on, hi))
            sym, hi = r.old, r.on
        return out[::-1]


def load_membership(path: Path, start: str) -> pd.DataFrame:
    """Long frame (date, ticker) of S&P 500 membership snapshots from `start` (plus the snapshot in force at `start`)."""
    m = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    before = m[m.date < start].tail(1)
    m = pd.concat([before, m[m.date >= start]])
    rows = [(d.date(), t) for d, tick in zip(m.date, m.tickers) for t in tick.split(",")]
    return pd.DataFrame(rows, columns=["date", "ticker"])


def members_on(membership: pd.DataFrame, asof: date) -> list[str]:
    snap_dates = membership.date[membership.date <= asof]
    if snap_dates.empty:
        raise ValueError(f"no membership snapshot on or before {asof}")
    d = snap_dates.max()
    return sorted(membership.ticker[membership.date == d])


def build_universe(
    bars: pd.DataFrame,
    membership: pd.DataFrame,
    resolver: SymbolResolver,
    rebalance_dates: list[date],
    top_n: int = 100,
    lookback_days: int = 63,
    min_history_days: int = 126,
    max_gap: int = MAX_GAP,
    share_classes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """For each reconstitution date, the top_n point-in-time members by median dollar volume over the
    `lookback_days` sessions strictly before that date. Requires `min_history_days` sessions of history and a bar
    within the last `max_gap` sessions; bars after a symbol's first gap of more than `max_gap` sessions are ignored.
    Only one share class per company is kept. `bars` must be UNADJUSTED (as traded): adjusted closes embed later
    dividends and spin-offs."""
    from jevtrader.data.prices import sessions_from, trim_at_first_gap

    share_classes = SHARE_CLASSES if share_classes is None else share_classes
    bars = bars[["symbol", "date", "close", "volume"]].copy()
    bars["date"] = pd.to_datetime(bars["date"])
    sessions = sessions_from(bars)
    bars = trim_at_first_gap(bars, sessions, max_gap)
    bars["dv"] = bars.close * bars.volume
    by_sym = {s: g.set_index("date").dv.sort_index() for s, g in bars.groupby("symbol")}
    out = []
    for d in rebalance_dates:
        ts = pd.Timestamp(d)
        prior = sessions[sessions < ts]
        recent = prior[-max_gap] if len(prior) >= max_gap else (prior[0] if len(prior) else ts)
        cands = {}
        for t in members_on(membership, d):
            sym = resolver.resolve(t, d)
            dv = by_sym.get(sym)
            if dv is None and t in by_sym:  # renamed symbol not downloaded: the PIT ticker's own bars (before d)
                sym, dv = t, by_sym[t]
            if dv is None:
                continue
            hist = dv[dv.index < ts]
            if len(hist) < min_history_days or hist.index[-1] < recent:
                continue
            cands[t] = (t, sym, float(hist.iloc[-lookback_days:].median()))
        for secondary, primary in share_classes.items():
            if secondary in cands and primary in cands:
                del cands[secondary]
        ranked = sorted(cands.values(), key=lambda x: -x[2])
        for rank, (t, sym, mdv) in enumerate(ranked[:top_n]):
            out.append({"date": d, "ticker": t, "symbol": sym, "median_dollar_volume": mdv, "rank": rank})
    return pd.DataFrame(out)


def members_without_bars(membership: pd.DataFrame, resolver: SymbolResolver, symbols_with_bars: set[str],
                         dates: list[date]) -> pd.DataFrame:
    """PIT members whose resolved symbol has no bars, by date, with `status`: 'resolved symbol not downloaded' when
    the point-in-time ticker has bars (build_universe then ranks it on them) or 'no bars' (it cannot enter the
    universe; the paper says "survivorship-free except for these members")."""
    rows = []
    for d in dates:
        for t in members_on(membership, d):
            sym = resolver.resolve(t, d)
            if sym not in symbols_with_bars:
                status = "resolved symbol not downloaded" if t in symbols_with_bars else "no bars"
                rows.append({"date": d, "ticker": t, "symbol": sym, "status": status})
    return pd.DataFrame(rows, columns=["date", "ticker", "symbol", "status"])


def check_no_gaps(universe: pd.DataFrame, bars: pd.DataFrame, lookback_days: int = 63, max_gap: int = MAX_GAP) -> None:
    """Assert that every universe member traded without a gap of more than `max_gap` sessions over the lookback
    window before its reconstitution date and up to it (the no-gap assertion of V-m2)."""
    from jevtrader.data.prices import sessions_from

    b = bars[["symbol", "date"]].copy()
    b["date"] = pd.to_datetime(b["date"])
    sessions = sessions_from(b)
    pos = pd.Series(range(len(sessions)), index=sessions)
    dates = {s: pos.reindex(g.values).dropna().to_numpy() for s, g in b.groupby("symbol")["date"]}
    bad = []
    for d, sym in zip(pd.to_datetime(universe.date), universe.symbol):
        k = int(sessions.searchsorted(d, side="left"))  # first session >= d
        p = dates.get(sym, np.array([]))
        w = p[(p >= k - lookback_days) & (p < k)]
        # missing sessions: window start -> first bar, between bars, last bar -> d
        pts = np.concatenate([[k - lookback_days - 1], w, [k]])
        if len(w) == 0 or (np.diff(pts) - 1).max() > max_gap:
            bad.append((d.date(), sym))
    if bad:
        raise AssertionError(f"{len(bad)} universe memberships with a gap > {max_gap} sessions, e.g. {bad[:5]}")


# Held symbols whose life ends inside the backtest: an exit is M&A when the company filed an 8-K reporting a change in
# control (Item 5.01) or a completed acquisition with a delisting notice (Items 2.01 + 3.01) within [-10, +30] days
# of its last bar. The engine exits these at the last close; any other exit needs an explicit delisting return.
def classify_exit(items_8k: pd.DataFrame, last_bar: pd.Timestamp) -> str:
    """'mna' or 'other' from a frame of 8-K filings (accepted_utc, items: list[str]) of the delisted company."""
    if items_8k is None or items_8k.empty:
        return "other"
    t = pd.to_datetime(items_8k.accepted_utc, utc=True).dt.tz_convert(None)
    last_bar = pd.Timestamp(last_bar)
    near = items_8k[(t >= last_bar - pd.Timedelta(days=10)) & (t <= last_bar + pd.Timedelta(days=30))]
    for items in near["items"]:
        s = set(items)
        if "5.01" in s or {"2.01", "3.01"} <= s:
            return "mna"
    return "other"


def ma_exits(symbols: list[str], bars: pd.DataFrame, sub_dir: Path, max_gap: int = MAX_GAP) -> dict[str, str]:
    """symbol -> exit class ('mna' | 'other') for every symbol whose (gap-trimmed) life ends before the last
    session of `bars`. Pass `{s for s, k in ma_exits(...).items() if k == 'mna'}` to `run_backtest(ma_exits=...)`;
    'other' exits need an explicit delisting return."""
    from jevtrader.data.earnings import load_8k_items
    from jevtrader.data.prices import life_ends, sessions_from

    b = bars[bars.symbol.isin(set(symbols))][["symbol", "date"]].copy()
    b["date"] = pd.to_datetime(b["date"])
    sessions = sessions_from(bars.assign(date=pd.to_datetime(bars["date"])))
    ends = life_ends(b, sessions, max_gap)
    last = b.groupby("symbol").date.max()
    out = {}
    for s in sorted(set(b.symbol)):
        end = ends.get(s) if not pd.isna(ends.get(s)) else last[s]
        if end < sessions[-1]:
            out[s] = classify_exit(load_8k_items(sub_dir, s), end)
    return out


FF12_FILE = Path(__file__).resolve().parents[3] / "data/raw/industry/Siccodes12.txt"


def load_ff12(sic_file: Path = FF12_FILE) -> list[tuple[int, int, str]]:
    """(lo, hi, FF12 code) SIC ranges from Ken French's Siccodes12.txt; SICs in no range are 'Other'."""
    import re

    out, code = [], None
    for line in Path(sic_file).read_text().splitlines():
        m = re.match(r"\s*\d+\s+(\w+)\s", line)
        if m:
            code = m.group(1)
            continue
        r = re.match(r"\s*(\d{4})-(\d{4})", line)
        if r and code:
            out.append((int(r.group(1)), int(r.group(2)), code))
    return out


def ff12_map(symbols: list[str], sub_dir: Path, sic_file: Path = FF12_FILE) -> dict[str, str]:
    """symbol -> FF12 code from the company's CURRENT SEC SIC (submissions `sic`). Not point in time: a declared
    limitation (design_v1 section 2.4, V-m6). Symbols without a submissions file are left out."""
    ranges = load_ff12(sic_file)
    out = {}
    for s in symbols:
        f = Path(sub_dir) / f"{s}.json"
        if not f.exists():
            continue
        sic = json.loads(f.read_text()).get("sic")
        sic = int(sic) if sic not in (None, "") else -1
        out[s] = next((c for lo, hi, c in ranges if lo <= sic <= hi), "Other")
    return out
