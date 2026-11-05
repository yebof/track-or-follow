"""Point-in-time news store over the downloaded Benzinga items.

Each item is tagged with the tickers in use when it was published; tags are mapped to the canonical
(current Alpaca) symbol with the date-aware ticker history, so FB-tagged 2021 news belongs to META.
Queries return only items with `created_at` inside [start, end) in UTC; the agents use the window
ending at the decision cutoff c_t (16:00 America/New_York on the decision date).

The archive stores each item's LAST edit, so an item edited at or after c_t shows text that did not exist at c_t.
Every query therefore also requires `updated_at < cutoff` (design_v1 section 2.4, V-m10); the cutoff defaults to
the window end. Unrecorded retractions and backfills remain unknowable (stated as a limitation).

Provenance (design_v2_delta SAN-5, FEED-1). Every archived item has `source` in {"benzinga", ""} and a URL on
www.benzinga.com, so "distinct source" is undefined. The table therefore carries the item's `author` byline and
`url_section`, the first path segment of its URL ("news", "markets", "trading-ideas", "node" for old-style
`/node/<id>` links, a two-digit year for undated-section links). The full URL is not carried: its slug repeats the
company name and would defeat blinding. `provenance_key(author)` is the SAN-5 key (NFKC, lower-case, whitespace
collapsed); an empty byline is its own key "".
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from jevtrader.data.universe import SymbolResolver

MARKET_TAGS = ("SPY", "QQQ", "DIA", "IWM")
COLUMNS = ["id", "symbol", "created_at", "updated_at", "headline", "summary", "source", "n_tags", "author",
           "url_section"]
_WS = re.compile(r"\s+")


def url_section(url: str | None) -> str:
    """First path segment of an item URL ("" if none)."""
    if not isinstance(url, str) or not url:
        return ""
    parts = urlparse(url).path.strip("/").split("/")
    return parts[0] if parts else ""


def provenance_key(author: str | None) -> str:
    """SAN-5 provenance key of a byline: NFKC, lower-case, whitespace collapsed and stripped."""
    if not isinstance(author, str):
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFKC", author).lower()).strip()


def build_news_table(batch_dir: Path, resolver: SymbolResolver, symbols: list[str], start: date,
                     end: date) -> pd.DataFrame:
    """Deduplicated long table: one row per (item, canonical symbol) plus market-tag rows (symbol='MARKET')."""
    frames = [pd.read_parquet(p) for p in sorted(Path(batch_dir).glob("*.parquet"))]
    items = pd.concat([f for f in frames if len(f)], ignore_index=True).drop_duplicates("id")
    for c in ("author", "url", "source"):
        if c not in items:
            items[c] = ""
    items["author"] = items["author"].fillna("").astype(str)
    items["url_section"] = items["url"].map(url_section)
    items["created_at"] = pd.to_datetime(items.created_at, utc=True)
    items["updated_at"] = pd.to_datetime(items.updated_at, utc=True).fillna(items["created_at"])
    # an edit can never predate creation; guard against clock noise in the feed
    items["updated_at"] = items[["created_at", "updated_at"]].max(axis=1)
    # (ticker, from, to) -> canonical symbol
    intervals = []
    for sym in symbols:
        for ticker, lo, hi in resolver.history(sym, start, end):
            intervals.append((ticker, pd.Timestamp(lo, tz="UTC"), pd.Timestamp(hi, tz="UTC"), sym))
    by_ticker: dict[str, list] = {}
    for t, lo, hi, s in intervals:
        by_ticker.setdefault(t, []).append((lo, hi, s))
    rows = []
    for it in items.itertuples(index=False):
        tags = [t for t in (it.symbols or "").split(",") if t]
        n_tags = len(tags)
        seen = set()
        for t in tags:
            if t in MARKET_TAGS and "MARKET" not in seen:
                seen.add("MARKET")
                rows.append((it.id, "MARKET", it.created_at, it.updated_at, it.headline, it.summary, it.source, n_tags,
                             it.author, it.url_section))
            for lo, hi, s in by_ticker.get(t, []):
                if lo <= it.created_at < hi and s not in seen:
                    seen.add(s)
                    rows.append((it.id, s, it.created_at, it.updated_at, it.headline, it.summary, it.source, n_tags,
                                 it.author, it.url_section))
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values(["symbol", "created_at"]).reset_index(drop=True)


class NewsStore:
    def __init__(self, table: pd.DataFrame):
        if "updated_at" not in table:
            raise ValueError("news table lacks updated_at; rebuild it with scripts/build_news_table.py (V-m10)")
        self.t = table
        self._by_sym = {s: g.sort_values("created_at", kind="stable").reset_index(drop=True)
                        for s, g in table.groupby("symbol")}

    def window(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp,
               cutoff: pd.Timestamp | None = None) -> pd.DataFrame:
        """Items of `symbol` created in [start, end) whose last edit is before `cutoff` (default: `end`)."""
        g = self._by_sym.get(symbol)
        if g is None:
            return self.t.iloc[0:0]
        lo = g.created_at.searchsorted(start, side="left")
        hi = g.created_at.searchsorted(end, side="left")
        w = g.iloc[lo:hi]
        return w[w.updated_at < (end if cutoff is None else cutoff)]


def decision_cutoff(d: pd.Timestamp) -> pd.Timestamp:
    """16:00 America/New_York on decision date d, in UTC."""
    return pd.Timestamp(pd.Timestamp(d).date()).tz_localize("America/New_York").replace(hour=16).tz_convert("UTC")
