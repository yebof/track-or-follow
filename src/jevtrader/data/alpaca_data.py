"""Read-only Alpaca market-data client (daily bars + Benzinga news) with an on-disk parquet cache.

These credentials belong to a live paper account used by another project, so this module only
ever talks to data.alpaca.markets. It never imports or constructs a trading client.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

DATA_URL = "https://data.alpaca.markets"
_MAX_RETRIES = 6
LIVE_FEED = "iex"
ASK_BAND, QUOTE_PAD = 0.02, 0.002


def prices_from_snapshots(snaps: dict) -> dict[str, float]:
    """`AlpacaData.latest_prices` on a snapshots response (pure; tested offline)."""
    out = {}
    for sym, s in (snaps or {}).items():
        trade = ((s or {}).get("latestTrade") or {}).get("p")
        if not trade or not trade > 0:
            continue
        ask = ((s or {}).get("latestQuote") or {}).get("ap")
        px = float(trade)
        if ask and ask > px and ask <= px * (1 + ASK_BAND):
            px = float(ask)
        out[sym] = px * (1 + QUOTE_PAD)
    return out


class AlpacaData:
    def __init__(self, key: str | None = None, secret: str | None = None, min_interval_s: float = 0.32):
        key = key or os.environ["ALPACA_DATA_API_KEY"]
        secret = secret or os.environ["ALPACA_DATA_SECRET_KEY"]
        self._session = requests.Session()
        self._session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        # Free data plan allows 200 requests/min; stay just under it.
        self._min_interval_s = min_interval_s
        self._last_call = 0.0

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(_MAX_RETRIES):
            wait = self._min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                r = self._session.get(DATA_URL + path, params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2**attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2**attempt * 2))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Alpaca GET {path} failed after {_MAX_RETRIES} attempts")

    def daily_bars(self, symbols: Iterable[str], start: str, end: str, adjustment: str = "all") -> pd.DataFrame:
        """Daily SIP bars. adjustment="all" (split+dividend+spin-off adjusted) is for returns only;
        adjustment="raw" gives prices as traded, needed for levels (dollar volume, market value, P/E),
        because adjusted history embeds later corporate actions. Long frame: symbol, date, o,h,l,c,v,vw,n."""
        rows = []
        symbols = list(symbols)
        for i in range(0, len(symbols), 50):
            chunk = symbols[i : i + 50]
            token = None
            while True:
                params = {
                    "symbols": ",".join(chunk),
                    "timeframe": "1Day",
                    "start": start,
                    "end": end,
                    "adjustment": adjustment,
                    "feed": "sip",
                    "limit": 10000,
                }
                if token:
                    params["page_token"] = token
                j = self._get("/v2/stocks/bars", params)
                for sym, bars in (j.get("bars") or {}).items():
                    for b in bars:
                        rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"], b.get("vw"), b.get("n")))
                token = j.get("next_page_token")
                if not token:
                    break
        df = pd.DataFrame(rows, columns=["symbol", "t", "open", "high", "low", "close", "volume", "vwap", "trades"])
        if df.empty:
            return df
        # Daily bars are stamped at 04:00/05:00 UTC of the session date; the ET calendar date is what we key on.
        df["date"] = pd.to_datetime(df["t"]).dt.tz_convert("America/New_York").dt.date
        return df.drop(columns="t").sort_values(["symbol", "date"]).reset_index(drop=True)

    def snapshots(self, symbols: Iterable[str], feed: str = LIVE_FEED) -> dict:
        """Latest trade and quote per symbol (`/v2/stocks/snapshots`). The default feed is IEX, the real-time feed of
        the free data plan (recent SIP data is delayed there)."""
        out: dict = {}
        symbols = list(symbols)
        for i in range(0, len(symbols), 100):
            j = self._get("/v2/stocks/snapshots", {"symbols": ",".join(symbols[i:i + 100]), "feed": feed})
            out.update(j.get("snapshots", j) if isinstance(j, dict) else {})
        return out

    def latest_prices(self, symbols: Iterable[str], feed: str = LIVE_FEED) -> dict[str, float]:
        """A conservative current BUY price per symbol for live order sizing: the latest trade, raised to the latest
        ask when the ask is within ASK_BAND of the trade (a wider ask on a single venue is ignored), plus QUOTE_PAD.
        Symbols without a latest trade are omitted (never priced from a stale close)."""
        return prices_from_snapshots(self.snapshots(symbols, feed))

    def news(self, symbols: Iterable[str], start: str, end: str, include_content: bool = False) -> pd.DataFrame:
        """All Benzinga news items tagged with any of `symbols` in [start, end), paginated oldest-first."""
        rows = []
        token = None
        while True:
            params = {
                "symbols": ",".join(symbols),
                "start": start,
                "end": end,
                "limit": 50,
                "sort": "asc",
                "include_content": "true" if include_content else "false",
                "exclude_contentless": "false",
            }
            if token:
                params["page_token"] = token
            j = self._get("/v1beta1/news", params)
            for n in j.get("news", []):
                rows.append(
                    {
                        "id": n["id"],
                        "created_at": n["created_at"],
                        "updated_at": n.get("updated_at"),
                        "headline": n.get("headline", ""),
                        "summary": n.get("summary", ""),
                        "content": n.get("content", "") if include_content else "",
                        "author": n.get("author", ""),
                        "source": n.get("source", ""),
                        "symbols": ",".join(n.get("symbols", [])),
                        "url": n.get("url", ""),
                    }
                )
            token = j.get("next_page_token")
            if not token:
                break
        return pd.DataFrame(rows)


def cached(path: Path, fetch, refresh: bool = False) -> pd.DataFrame:
    """Load `path` if present, otherwise call `fetch()` and persist its result as parquet."""
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return df
