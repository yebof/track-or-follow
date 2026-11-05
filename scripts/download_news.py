"""Download Benzinga news (headline + summary) for every universe company over its ticker history, plus
market-wide ETFs (macro / policy flow). Batched by shared interval, cached per batch so reruns resume."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevtrader.data.alpaca_data import AlpacaData, cached  # noqa: E402
from jevtrader.data.universe import SymbolResolver, load_membership, load_renames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
START, END = date(2019, 6, 1), date(2026, 9, 24)
MARKET = ["SPY", "QQQ", "DIA", "IWM"]

uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
membership = load_membership(ROOT / "data/raw/universe/sp500_hist_components.csv", "2018-01-01")
resolver = SymbolResolver(load_renames(ROOT / "data/raw/universe/alpaca_name_changes.json", membership))
jobs: dict[tuple[date, date], set[str]] = {}
for sym in sorted(set(uni.symbol)):
    for ticker, lo, hi in resolver.history(sym, START, END):
        jobs.setdefault((lo, hi), set()).add(ticker)
for m in MARKET:
    jobs.setdefault((START, END), set()).add(m)

alp = AlpacaData()
out_dir = ROOT / "data/raw/news/batches"
n_batches = 0
for (lo, hi), tickers in sorted(jobs.items()):
    tickers = sorted(tickers)
    step = 10
    for i in range(0, len(tickers), step):
        chunk = tickers[i : i + step]
        name = f"{lo}_{hi}_{'-'.join(chunk)}"[:180]
        df = cached(out_dir / f"{name}.parquet", lambda: alp.news(chunk, f"{lo}T00:00:00Z", f"{hi}T00:00:00Z"))
        n_batches += 1
        print(f"batch {n_batches}: {lo}..{hi} {chunk} -> {len(df)} items", flush=True)
print("done")
