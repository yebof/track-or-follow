"""Unadjusted (as-traded) daily bars for the same symbols as the adjusted file; used for price levels."""
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")
from jevtrader.data.alpaca_data import AlpacaData  # noqa: E402

syms = sorted(set(pd.read_parquet(ROOT / "data/raw/bars/sp500_daily_2018_2026.parquet", columns=["symbol"]).symbol))
syms += sorted(set(pd.read_parquet(ROOT / "data/raw/bars/etf_daily_2018_2026.parquet", columns=["symbol"]).symbol))
alp = AlpacaData()
frames = [alp.daily_bars(syms[i : i + 50], "2018-01-01", "2026-09-24", adjustment="raw") for i in range(0, len(syms), 50)]
raw = pd.concat(frames, ignore_index=True)
raw.to_parquet(ROOT / "data/raw/bars/all_daily_raw_2018_2026.parquet", index=False)
print(len(raw), raw.symbol.nunique())
