"""Build data/processed/news_table.parquet from the raw Benzinga batches (canonical symbols + MARKET rows).

The table carries `updated_at` (V-m10): every point-in-time query requires created_at < c_t AND updated_at < c_t,
because the archive stores each item's last edit. It also carries `author` and `url_section` (the URL's first path
segment), the SAN-5 provenance-key and FEED-1 inputs (design_v2_delta); the full URL is dropped (blinding).
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.data.news import build_news_table  # noqa: E402
from jevtrader.data.universe import SymbolResolver, load_membership, load_renames  # noqa: E402

m = load_membership(ROOT / "data/raw/universe/sp500_hist_components.csv", "2018-01-01")
res = SymbolResolver(load_renames(ROOT / "data/raw/universe/alpaca_name_changes.json", m))
syms = sorted(set(pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet").symbol))
t = build_news_table(ROOT / "data/raw/news/batches", res, syms, date(2019, 6, 1), date(2026, 9, 24))
out = ROOT / "data/processed/news_table.parquet"
tmp = out.with_suffix(".tmp.parquet")
t.to_parquet(tmp, index=False)
tmp.replace(out)
print(len(t), "rows;", t.id.nunique(), "unique items;", t.symbol.nunique(), "symbols")
items = t.drop_duplicates("id")
lag_h = (items.updated_at - items.created_at).dt.total_seconds() / 3600
print(f"edited after creation: >1h {(lag_h > 1).mean():.4f}, >24h {(lag_h > 24).mean():.4f} of {len(items)} items")
wk = t.assign(week=t.created_at.dt.tz_convert("America/New_York").dt.to_period("W")).groupby(["symbol", "week"]).size()
print("items per symbol-week (company, n_tags<=3):")
t3 = t[(t.symbol != "MARKET") & (t.n_tags <= 3)]
w3 = t3.assign(week=t3.created_at.dt.tz_convert("America/New_York").dt.to_period("W")).groupby(["symbol", "week"]).size()
print(w3.describe().round(1).to_string())
print("MARKET per week:", wk.loc["MARKET"].describe().round(1).to_dict())
print(f"author non-empty {(items.author != '').mean():.4f}; "
      f"url_section non-empty {(items.url_section != '').mean():.4f}")
print("top url sections:", items.url_section.value_counts().head(10).to_dict())
