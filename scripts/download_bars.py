"""Download adjusted daily bars for every ticker that was an S&P 500 member at any point since 2018-06."""
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevtrader.data.alpaca_data import AlpacaData  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

members = pd.read_csv(ROOT / "data/raw/universe/sp500_hist_components.csv", parse_dates=["date"])
members = members[members.date >= "2018-06-01"]
tickers = sorted({t for row in members.tickers for t in row.split(",")})
alp = AlpacaData()
out_dir = ROOT / "data/raw/bars"
out_dir.mkdir(parents=True, exist_ok=True)
frames = []
for i in range(0, len(tickers), 50):
    chunk = [t.replace(".", ".") for t in tickers[i : i + 50]]
    df = alp.daily_bars(chunk, "2018-01-01", "2026-09-24")
    frames.append(df)
    print(f"{i + len(chunk)}/{len(tickers)} tickers, rows so far {sum(len(f) for f in frames)}", flush=True)
bars = pd.concat(frames, ignore_index=True)
bars.to_parquet(out_dir / "sp500_daily_2018_2026.parquet", index=False)
missing = sorted(set(tickers) - set(bars.symbol))
Path(out_dir / "missing_tickers.txt").write_text("\n".join(missing))
print("done", len(bars), "rows;", bars.symbol.nunique(), "symbols; missing", len(missing))
