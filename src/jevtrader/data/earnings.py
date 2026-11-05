"""Point-in-time earnings calendar from SEC 8-K Item 2.02 ("Results of Operations") filings, an 8-K item-code
loader for the C2 concept-agreement labels (design_v1 section 13.1; evaluation only, never a trading input), and the
trusted filing anchor of the anchor-gated exclusion rule (design_v2_delta BG-3, bank QB-13).

Item codes. Every code used here is on the official Form 8-K item list archived in `docs/sec/form8k_items.json`
(parsed from https://www.sec.gov/files/form8-k.pdf by `scripts/check_8k_items.py`, which also checks that the EDGAR
submissions archive carries item codes for every 8-K: `docs/sec/8k_item_check.json`).

Filing anchor (BG-3). A^filing_{i,t} = 1 iff an 8-K or 8-K/A of the issuer accepted in [c_t - 7 calendar days, c_t)
lists any item of ANCHOR_ITEMS: 1.03 bankruptcy or receivership; 2.04 triggering events that accelerate a direct
financial obligation; 2.06 material impairments; 3.01 delisting notice or failure to satisfy a listing rule; 4.01
change of certifying accountant; 4.02 non-reliance on previously issued financial statements. It is a trusted-channel
value: EDGAR acceptance times and item codes only, never news text.

Past releases are exact (EDGAR acceptance timestamps of 8-K, not 8-K/A, Item 2.02 filings). Several 2.02 filings
within 20 days are one release; the LAST one is kept, because the earlier ones are mostly preliminary updates
(SBUX 2024-10-22 before the 10-30 release, ILMN 2019-07-11 before 07-29). This collapsing is done on the releases
before the cutoff only, so a preliminary update counts as the latest release until the real one is filed.

The *next* release date is not knowable from filings without look-ahead, so it is forecast from past releases only.
Companies keep a stable weekday cadence, so the forecast is the release that followed the same quarter one year
earlier, plus 364 days (two years back if needed; else the last release plus 91 days). Two corrections:
- the latest release came at least EARLY_DAYS before its quarter's usual date (last year's date + 364): it was a
  preliminary update and the release is still due on the usual date;
- the forecast date passed at most OVERDUE_DAYS ago without a release: the release is imminent (forecast = now).
On W_pre x U100 (2020-01..2024-05; 8-K dates only, no returns) the rule `sessions_to_next <= 2` flags 982 stock-weeks
with precision 0.51 and recall 0.74 against 2.02 filings in (c_t, second session after d_t], and 9 of them (0.9%)
within 20 days after a release; the previous "same quarter + 52 weeks" rule had 0.37 / 0.57 and 104 (10%). The two
margins were set on W_pre (tests/test_earnings.py pins precision and recall). A company that moves its release two
or more weeks earlier than a year before looks like a preliminary update (UPS 2024-07-23 vs 2023-08-08).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ITEMS_JSON = ROOT / "docs/sec/form8k_items.json"


C2_ITEMS = {
    "2.02": "results of operations (earnings release)",
    "1.01": "entry into a material definitive agreement",
    "2.01": "completion of acquisition or disposition of assets",
    "5.02": "departure/appointment of directors or officers",
}


# BG-3 filing anchor (bank code_constants.exclusion_anchor.filing_anchor; tests check both lists agree)
ANCHOR_ITEMS = ("1.03", "2.04", "2.06", "3.01", "4.01", "4.02")
ANCHOR_WINDOW = pd.Timedelta(days=7)


def load_8k_item_list(path: Path = ITEMS_JSON) -> dict[str, str]:
    """Official Form 8-K item codes -> titles (the archived SEC list)."""
    return {x["code"]: x["title"] for x in json.loads(Path(path).read_text())["items"]}


def _utc(acc: str) -> pd.Timestamp:
    t = pd.Timestamp(acc)
    return t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")


def load_8k_items(sub_dir: Path, symbol: str, items: tuple[str, ...] | None = None) -> pd.DataFrame:
    """All 8-K / 8-K/A filings of `symbol` (incl. predecessor-entity and paginated submission files) as rows
    (accepted_utc, form, accession, items: list of item codes). With `items`, only filings reporting any of them.
    `acceptanceDateTime` is true UTC (validity review section 4), so an 8-K counts as public from `accepted_utc`."""
    sub_dir = Path(sub_dir)
    files = [sub_dir / f"{symbol}.json"] + sorted(sub_dir.glob(f"{symbol}__*.json"))
    rows = []
    for f in files:
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        block = j["filings"]["recent"] if "filings" in j else j  # paginated files hold the arrays at top level
        for form, acc, accn, it in zip(block.get("form", []), block.get("acceptanceDateTime", []),
                                       block.get("accessionNumber", []), block.get("items", [])):
            if form not in ("8-K", "8-K/A"):
                continue
            codes = [c.strip() for c in (it or "").split(",") if c.strip()]
            if items is None or any(c in items for c in codes):
                rows.append((_utc(acc), form, accn, codes))
    df = pd.DataFrame(rows, columns=["accepted_utc", "form", "accession", "items"])
    df = df.drop_duplicates("accession").sort_values("accepted_utc").reset_index(drop=True)
    df["accepted_utc"] = pd.to_datetime(df["accepted_utc"], utc=True)
    return df


def item_events(sub_dir: Path, symbol: str, item: str, cutoff: pd.Timestamp | None = None) -> pd.Series:
    """Sorted UTC acceptance times of 8-Ks reporting `item`; with `cutoff`, only those accepted before it."""
    df = load_8k_items(sub_dir, symbol, (item,))
    t = df.accepted_utc
    if cutoff is not None:
        t = t[t < cutoff]
    return t.reset_index(drop=True)


EARLY_DAYS = 14  # a release this much before its quarter's usual date is a preliminary update
OVERDUE_DAYS = 7  # a forecast date passed by at most this much without a release: the release is imminent
CLUSTER_DAYS = 20  # 2.02 filings closer than this are one release
_YEAR, _EMPTY = pd.Timedelta(days=364), pd.Series(dtype="datetime64[ns, UTC]")


def load_releases(sub_dir: Path, symbol: str) -> pd.Series:
    """Sorted UTC acceptance timestamps of every Item 2.02 8-K (not 8-K/A; incl. predecessor-entity and paginated
    files). Not collapsed: `collapse_releases` must only ever see releases before a cutoff (point in time)."""
    df = load_8k_items(sub_dir, symbol, ("2.02",))
    df = df[df.form == "8-K"]
    return pd.Series(sorted(set(df.accepted_utc)), dtype="datetime64[ns, UTC]")


def collapse_releases(ts: pd.Series, gap_days: int = CLUSTER_DAYS) -> pd.Series:
    """One release per run of 2.02 filings less than `gap_days` apart: the last one (see module doc)."""
    keep: list = []
    for t in ts:
        if keep and t - keep[-1] <= pd.Timedelta(days=gap_days):
            keep[-1] = t
        else:
            keep.append(t)
    return pd.Series(keep, dtype="datetime64[ns, UTC]")


class EarningsCalendar:
    def __init__(self, sub_dir: Path, symbols: list[str]):
        self.rel = {s: load_releases(sub_dir, s) for s in symbols}  # all 2.02 8-Ks (not collapsed)

    def past(self, symbol: str, cutoff: pd.Timestamp) -> pd.Series:
        """Releases known before `cutoff`, collapsed using only those filings."""
        r = self.rel.get(symbol, _EMPTY)
        return collapse_releases(r[r < cutoff])

    def days_since_last(self, symbol: str, cutoff: pd.Timestamp) -> float | None:
        p = self.past(symbol, cutoff)
        return None if p.empty else (cutoff - p.iloc[-1]).total_seconds() / 86400

    def released_between(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        r = self.rel.get(symbol, _EMPTY)
        return bool(((r >= start) & (r < end)).any())

    def next_estimate(self, symbol: str, cutoff: pd.Timestamp) -> pd.Timestamp | None:
        """Forecast of the next release at or after `cutoff`, using only releases before `cutoff` (module doc)."""
        p = self.past(symbol, cutoff)
        if p.empty:
            return None
        last, est = p.iloc[-1], None
        for k in (1, 2):
            lag = k * _YEAR
            near = p[(p - (last - lag)).abs() <= pd.Timedelta(days=30)]
            if near.empty:
                continue
            analog = near.iloc[int((near - (last - lag)).abs().argmin())]  # the same quarter k years earlier
            due = analog + lag
            if last < due - pd.Timedelta(days=EARLY_DAYS) and due >= cutoff:
                return due  # `last` was a preliminary update; the release is still due
            nxt = p[(p > analog + pd.Timedelta(days=45)) & (p < analog + pd.Timedelta(days=130))]
            if not nxt.empty:
                est = nxt.iloc[0] + lag
                break
        if est is None:
            est = last + pd.Timedelta(days=91)
        if est < cutoff:
            if est >= cutoff - pd.Timedelta(days=OVERDUE_DAYS):
                return cutoff  # overdue by a few days: imminent
            while est < cutoff:
                est += pd.Timedelta(days=91)
        return est

    def sessions_to_next(self, symbol: str, cutoff: pd.Timestamp, sessions: pd.DatetimeIndex) -> int | None:
        """Sessions after the cutoff's day up to the forecast release day (0 = later on the cutoff's day)."""
        est = self.next_estimate(symbol, cutoff)
        if est is None:
            return None
        day = est.tz_convert("America/New_York").normalize().tz_localize(None)
        after = sessions[sessions > cutoff.tz_convert("America/New_York").normalize().tz_localize(None)]
        return int((after <= day).sum())


def sessions_to_next_panel(cal: EarningsCalendar, decisions: list[pd.Timestamp],
                           universe: dict[pd.Timestamp, list[str]], sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows (t, symbol, sessions_to_earnings) for the risk officer's earnings rule (design_v1 section 9.2; the column
    name `decide.risk` reads): sessions until the forecast next release, from releases accepted before c_t only."""
    from jevtrader.data.news import decision_cutoff

    rows = []
    for t in decisions:
        c = decision_cutoff(t)
        for s in universe[t]:
            rows.append((pd.Timestamp(t), s, cal.sessions_to_next(s, c, sessions)))
    return pd.DataFrame(rows, columns=["t", "symbol", "sessions_to_earnings"])


def filing_anchor_events(sub_dir: Path, symbol: str, cutoff: pd.Timestamp, window: pd.Timedelta = ANCHOR_WINDOW,
                         items: tuple[str, ...] = ANCHOR_ITEMS, filings: pd.DataFrame | None = None) -> pd.DataFrame:
    """8-K / 8-K/A filings of `symbol` accepted in [cutoff - window, cutoff) that list any anchor item (BG-3).
    Pass `filings` (the output of `load_8k_items(..., items)`) to avoid re-reading the submission files."""
    df = load_8k_items(sub_dir, symbol, items) if filings is None else filings
    m = (df.accepted_utc >= cutoff - window) & (df.accepted_utc < cutoff)
    return df[m].reset_index(drop=True)


def filing_anchor_panel(sub_dir: Path, decisions: list[pd.Timestamp], universe: dict[pd.Timestamp, list[str]],
                        window: pd.Timedelta = ANCHOR_WINDOW) -> pd.DataFrame:
    """Rows (t, symbol, filing_anchor, anchor_items, anchor_accessions) for every member of U_t: the trusted filing
    anchor of BG-3 at cutoff c_t (8-Ks accepted before c_t only)."""
    from jevtrader.data.news import decision_cutoff

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for t in decisions:
        c = decision_cutoff(t)
        for s in universe[t]:
            if s not in cache:
                cache[s] = load_8k_items(sub_dir, s, ANCHOR_ITEMS)
            ev = filing_anchor_events(sub_dir, s, c, window, filings=cache[s])
            codes = sorted({x for its in ev["items"] for x in its if x in ANCHOR_ITEMS})
            rows.append((pd.Timestamp(t), s, bool(len(ev)), ",".join(codes), ",".join(ev.accession)))
    return pd.DataFrame(rows, columns=["t", "symbol", "filing_anchor", "anchor_items", "anchor_accessions"])
