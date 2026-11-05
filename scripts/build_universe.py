"""Monthly point-in-time top-N universe (by trailing median RAW dollar volume among S&P 500 members).

Applies the design_v1 section 2.2 fixes (V-B2, V-m2): raw dollar volume; symbol life ends at the first gap of more
than 10 sessions; one share class per company; members must have traded in the 10 sessions before the date; the
LLL rename is not followed. Prints the membership change against the previous file, the PIT members without bars,
the members whose life ends inside the data (with their exit class from 8-K items), and asserts no gaps.

Run: .venv/bin/python scripts/build_universe.py [top_n]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.data.earnings import load_8k_items  # noqa: E402
from jevtrader.data.prices import life_ends, sessions_from  # noqa: E402
from jevtrader.data.universe import (SymbolResolver, build_universe, check_no_gaps, classify_exit,  # noqa: E402
                                     load_membership, load_renames, members_without_bars)

# Dollar volume must come from as-traded prices: adjusted history embeds later dividends/spin-offs (look-ahead).
bars = pd.read_parquet(ROOT / "data/raw/bars/all_daily_raw_2018_2026.parquet")
bars["date"] = pd.to_datetime(bars.date)
membership = load_membership(ROOT / "data/raw/universe/sp500_hist_components.csv", "2019-01-01")
resolver = SymbolResolver(load_renames(ROOT / "data/raw/universe/alpaca_name_changes.json", membership))
sessions = sessions_from(bars)
s = pd.Series(sessions)
firsts = s.groupby([s.dt.year, s.dt.month]).min()
reb = [d.date() for d in firsts if pd.Timestamp("2019-06-01") <= d <= pd.Timestamp("2026-09-30")]
top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
uni = build_universe(bars, membership, resolver, reb, top_n=top_n)
check_no_gaps(uni, bars)

out = ROOT / "data/processed/universe_monthly_top100.parquet"
if out.exists():
    old = pd.read_parquet(out)
    a = set(zip(old.date.astype(str), old.symbol))
    b = set(zip(uni.date.astype(str), uni.symbol))
    print(f"membership change vs previous file: {len(a - b)} removed, {len(b - a)} added of {len(b)} "
          f"({len(a ^ b) / 2 / len(b):.1%} of memberships)")
    print("  removed:", sorted({s for _, s in a - b}))
    print("  added:  ", sorted({s for _, s in b - a}))
miss = members_without_bars(membership, resolver, set(bars.symbol), reb)
print("PIT members whose resolved symbol has no bars ('resolved symbol not downloaded': ranked on the PIT ticker's "
      "own bars; 'no bars': cannot enter the universe):",
      miss.groupby(["ticker", "symbol", "status"]).date.agg(["min", "max", "count"]).to_string())
ends = life_ends(bars, sessions).dropna()
last = bars.groupby("symbol").date.max().where(lambda x: x < sessions[-1]).dropna()
exits = {}
for sym in sorted(set(uni.symbol)):
    end = ends.get(sym, last.get(sym))
    if end is not None and not pd.isna(end):
        exits[sym] = (pd.Timestamp(end).date(), classify_exit(load_8k_items(ROOT / "data/raw/sec/submissions", sym), end))
print("universe symbols whose life ends inside the data (last bar, exit class):", exits)
out.parent.mkdir(parents=True, exist_ok=True)
tmp = out.with_suffix(".tmp.parquet")
uni.to_parquet(tmp, index=False)
tmp.replace(out)
print(uni.groupby("date").size().describe())
print("union symbols:", uni.symbol.nunique(), " union PIT tickers:", uni.ticker.nunique())
print("mapped tickers (ticker != symbol):", sorted(set(uni.loc[uni.ticker != uni.symbol, "ticker"] + "->" + uni.loc[uni.ticker != uni.symbol, "symbol"])))
