"""Real-data point-in-time `WeekContext` builder (design_v1 §4.2 step 1, §12.3; design_v2_delta AV-1).

    data = RealData.load()                       # heavy stores, loaded once per process
    build = ContextBuilder(data, subset="S50")   # a callable t -> WeekContext (GeneratorSpec.context_factory)
    ctx = build(pd.Timestamp("2021-03-05"))

What a context holds at decision t (cutoff c_t = 16:00 America/New_York on d_t):
- `universe` = U_t, the monthly PIT top-100 in force at d_t (latest reconstitution date <= d_t; rankings strictly
  before it, `data.universe.build_universe`). Cross-sectional buckets (percentiles "in the universe", size ranks) are
  always taken over the full U_t. `focus` (S50/S30 or an explicit list) only selects the names whose stock-level
  requests a run makes, so a subset run asks exactly the requests a full run would ask for those names.
- `prices`/`raw_prices`: adjusted and raw panels cut at each symbol's life end (V-m2); programs read rows <= d_t.
- `news` (NewsStore: created_at and updated_at < cutoff), `facts` (FundamentalsPanel: filed < d_t), `macro`
  (MacroStore: vintages, lags), `earnings` (EarningsCalendar: 8-K acceptance < c_t), `filings` (TenKStore: filed
  < d_t, usable the session after), `blinder` (a `PitBlinder`, see below), `firms` (FF12, size rank by raw market
  value in U_t, trailing-correlation sector ETF), `sessions`, `prev_cutoff_utc` = c_{t-1};
- `study_members(c)`: the study firms that had been U members at a reconstitution on or before the date of cutoff c
  (c capped at c_t); `firm_asof(symbol, day)`: a firm's FF12 and size rank as a run deciding on an earlier `day`
  renders them (F-A renders a 10-K's focal firm at the week the filing became usable, so its requests do not
  depend on the week that first reads it); `unavailable`: sources that do not cover the week (non-strict runs only).

**Point-in-time masking.** The study set is every firm that is ever in U (2019-2026). Masking a firm before it
joins U would leak its future membership into every blinded text ("Uber" masked in 2021, "Lyft" named). The
week's `PitBlinder` therefore masks, in a text blinded at cutoff c, only the firms that had been U members at a
reconstitution on or before c (monotone: a firm stays masked after it leaves U), plus the text's focal firm. The
universe file starts on 2019-06-03; a text dated earlier uses that first reconstitution's members (logged).
10-K text (F-A; FILING_MASKING, docs/impl/dev2_blind.md §3 and its audit response) is blinded against the members at
the week that reads the filing (`PitBlinder.filing_view`), plus the focal firm, with alias claims, executives and
times as of the filing's usable time. The members at the filing date (v2) left a firm that joined U between the
filing and the reading week named ("Marriott" in a 2019 10-K read in 2021, DEV #1 P0-4); the full study set (v3)
masked firms that join U only after the reading week, so masked-versus-named told the model which competitors would
become heavily traded (DEV #2 audit). With the reading week's members a 10-K's excerpt changes only when a firm it
names joins U between two reading weeks, so it keeps nearly one request key per filing.

Each source's value-independent availability term is declared once in `AVAILABILITY` (AV-1); the loader that
applies it is named next to it, and `tests/test_availability_declared.py` checks every loader against its
declaration at the boundary. `ContextBuilder.check(t)` reports hard failures (no bar on d_t, t not a decision date,
no universe, an unblindable member), stale sources (a store that ends before the cutoff, no news item in
[c_{t-1}, c_t), macro series or filings that stopped updating: an outage must not look like a quiet week) and soft
gaps (members without facts or a 10-K). A strict builder (the default) raises `DataAvailabilityError` on a hard
failure or a stale source; a non-strict one logs them as warnings and lists the stale sources in
`WeekContext.unavailable`, so the programs that read them abstain.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jevtrader.data.news import decision_cutoff
from jevtrader.programs.base import FirmInfo, WeekContext

log = logging.getLogger("jevtrader.context")
ROOT = Path(__file__).resolve().parents[3]
TOP_N = 100
MARKET_SYMBOL = "SPY"
SUBSETS = {"S50": ("S50-www27", 50), "S30": ("S30-www27", 30)}  # design_v1 §2.2 (configs/system_v1.yaml subsets)
W_DEV = ("2021-01-01", "2022-12-31", 20, "W_dev-www27")  # design_v1 §2.5: 20 random weeks in 2021-01..2022-12
TENK_MIN_CHARS = 3000  # an Item 1 extraction shorter than this failed (scripts/download_10k_business.py)
NEXT_SESSION_OPEN = "09:30"  # a 10-K is usable from the open of the session after its filing date
# version of what a context contains (enters every unit fingerprint through `ContextBuilder.describe`): v2 = PIT
# masking of study firms, stale-source checks, F-A focal firm at the filing's first week
CONTEXT_VERSION = "ctx-v2"
MASKING_RULE = "pit-members-v1"  # a study firm is masked from the first reconstitution at which it is a U member
FILING_MASKING = "reading-week-v1"  # 10-K text (F-A): the U members at the reading week, plus the focal firm
# stale-source thresholds (an outage must not render as a quiet or unchanged week)
NEWS_STALE = pd.Timedelta(days=1)  # the news archive ends more than this before the cutoff
FRED_STALE_DAYS, EPU_STALE_DAYS, GPR_STALE_DAYS = 10, 14, 21  # beyond each series' publication lag
FACTS_STALE_DAYS = 30  # no XBRL filing by any U_t member for this long (the largest gap over the 381 decision
# weeks 2019-06..2026-09 of the archive is 15 days)
# the archive files and stores a context reads (their identity is the run's data freeze, `RealData.archive_id`)
ARCHIVE_FILES = ("data/processed/universe_monthly_top100.parquet", "data/processed/news_table.parquet")
ARCHIVE_DIRS = ("data/raw/bars", "data/raw/macro", "data/raw/sec/facts", "data/raw/sec/submissions",
                "data/raw/sec/business", "data/raw/sec/class_shares", "data/raw/universe")

# AV-1: the value-independent availability term of every source a WeekContext (or Layer B) reads, and the loader
# that applies it. Generated into configs/availability_v1.yaml (appendix table) and tested at the boundary.
AVAILABILITY: dict[str, dict[str, str]] = {
    "prices_adjusted": {"column": "session date (close)", "rule": "session <= d_t",
                        "loader": "features.technical.technical_snapshot (px.close.loc[:d])"},
    "prices_raw": {"column": "session date (close)", "rule": "session <= d_t",
                   "loader": "data.prices.PricePanels.raw_close_asof"},
    "news": {"column": "created_at, updated_at", "rule": "created_at in [c_{t-1}, c_t) and updated_at < c_t",
             "loader": "data.news.NewsStore.window"},
    "xbrl_facts": {"column": "filed", "rule": "filed < d_t (usable the session after filing)",
                   "loader": "data.fundamentals.FundamentalsPanel.snapshot_asof"},
    "form_8k": {"column": "acceptanceDateTime (UTC)", "rule": "accepted_utc < c_t",
                "loader": "data.earnings.EarningsCalendar.past"},
    "form_10k_item1": {"column": "filed", "rule": "filed < d_t; usable from the next session's open",
                       "loader": "programs.context.TenKStore.latest"},
    "alfred_vintages": {"column": "realtime_start", "rule": "realtime_start <= d_t",
                        "loader": "features.macro.MacroStore.vintage_series"},
    "fred_daily": {"column": "observation date", "rule": "date <= d_t - 1 day (DTWEXBGS, DCOILWTICO: 7 days)",
                   "loader": "features.macro.MacroStore.daily_asof"},
    "epu": {"column": "index date", "rule": "date <= d_t - 2 days", "loader": "features.macro.MacroStore.epu_asof"},
    "gpr": {"column": "index date", "rule": "date <= d_t - 7 days", "loader": "features.macro.MacroStore.gpr_asof"},
    "fomc_schedule": {"column": "announced", "rule": "announced < d_t",
                      "loader": "features.macro.MacroStore.fed_decision_within"},
    "universe": {"column": "reconstitution date", "rule": "latest reconstitution date <= d_t (ranked before it)",
                 "loader": "programs.context.universe_asof"},
    "memory_records": {"column": "available_at", "rule": "available_at < c_t (M-work: <= c_t, same week)",
                       "loader": "memory.store.MemoryStore.view"},
    "labels": {"column": "available_at", "rule": "available_at < c_t (Layer B)",
               "loader": "learn.labels.LabelStore.mature"},
}


class DataAvailabilityError(RuntimeError):
    """A decision week cannot be built from the archive (hard failure of `ContextBuilder.check`)."""


# ------------------------------------------------------------------------------------ calendar helpers
def decision_dates(sessions: pd.DatetimeIndex, start=None, end=None) -> list[pd.Timestamp]:
    """Last session of each ISO week (the engine's `weekly_decision_dates`), optionally within [start, end]."""
    s = pd.Series(pd.DatetimeIndex(sessions))
    if start is not None:
        s = s[s >= pd.Timestamp(start)]
    if end is not None:
        s = s[s <= pd.Timestamp(end)]
    iso = s.dt.isocalendar()
    return sorted(pd.Timestamp(x) for x in s.groupby([iso.year.values, iso.week.values]).max())


def previous_decision(sessions: pd.DatetimeIndex, t: pd.Timestamp) -> pd.Timestamp | None:
    """The decision date of the ISO week before t's (None at the start of the calendar)."""
    t = pd.Timestamp(t)
    y, w, _ = t.isocalendar()
    before = pd.DatetimeIndex(sessions)[pd.DatetimeIndex(sessions) < t]
    for d in before[::-1]:
        if tuple(d.isocalendar())[:2] != (y, w):
            return pd.Timestamp(d)
    return None


def is_decision_date(sessions: pd.DatetimeIndex, t: pd.Timestamp) -> bool:
    t = pd.Timestamp(t)
    idx = pd.DatetimeIndex(sessions)
    if t not in idx:
        return False
    later = idx[idx > t]
    return len(later) == 0 or tuple(later[0].isocalendar())[:2] != tuple(t.isocalendar())[:2]


def next_session_open_utc(sessions: pd.DatetimeIndex, day) -> pd.Timestamp | None:
    """09:30 America/New_York of the first session strictly after `day` (a date), in UTC."""
    idx = pd.DatetimeIndex(sessions)
    nxt = idx[idx > pd.Timestamp(day).normalize()]
    if not len(nxt):
        return None
    hh, mm = map(int, NEXT_SESSION_OPEN.split(":"))
    return pd.Timestamp(nxt[0].date()).tz_localize("America/New_York").replace(hour=hh, minute=mm).tz_convert("UTC")


def ny_day(when) -> pd.Timestamp:
    """The America/New_York calendar date of a UTC time (a naive timestamp or a date is taken as the date itself)."""
    ts = pd.Timestamp(when)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    return ts.normalize()


def decision_on_or_after(sessions: pd.DatetimeIndex, when) -> pd.Timestamp | None:
    """The first decision date whose cutoff (16:00 America/New_York) is after `when` (a UTC time; a date means its
    start): the first week in which information available at `when` can be read."""
    idx = pd.DatetimeIndex(sessions)
    ts = pd.Timestamp(when)
    ts = ts.tz_localize("America/New_York").tz_convert("UTC") if ts.tzinfo is None else ts
    for d in idx[idx >= ny_day(ts)]:
        if is_decision_date(idx, d) and decision_cutoff(d) > ts:
            return pd.Timestamp(d)
    return None


def dev_weeks(sessions: pd.DatetimeIndex, n: int = W_DEV[2], start: str = W_DEV[0], end: str = W_DEV[1],
              salt: str = W_DEV[3]) -> list[pd.Timestamp]:
    """W_dev: `n` decision weeks in [start, end] chosen by the smallest sha256(salt + ISO date) (no selection
    freedom; design_v1 §2.5), returned in date order."""
    ds = decision_dates(sessions, start, end)
    pick = sorted(ds, key=lambda d: hashlib.sha256(f"{salt}{d.date().isoformat()}".encode()).hexdigest())[:n]
    return sorted(pick)


def subset_names(universe: Iterable[str], n: int, salt: str) -> tuple[str, ...]:
    """The `n` names of U_t with the smallest sha256(salt + symbol) (design_v1 §2.2 S50/S30), in U_t order."""
    uni = list(universe)
    keep = set(sorted(uni, key=lambda s: hashlib.sha256(f"{salt}{s}".encode()).hexdigest())[:n])
    return tuple(s for s in uni if s in keep)


def universe_asof(membership: pd.DataFrame, t: pd.Timestamp, top: int = TOP_N) -> tuple[str, ...]:
    """U_t: members of the latest reconstitution on or before d_t with rank < `top`, in rank order (the rule of
    `pipeline.layer_b.universe_by_date`)."""
    u = membership
    dates = np.array(sorted(pd.to_datetime(u["date"]).unique()))
    k = dates.searchsorted(np.datetime64(pd.Timestamp(t)), side="right") - 1
    if k < 0:
        return ()
    g = u[(pd.to_datetime(u["date"]) == pd.Timestamp(dates[k])) & (u["rank"] < top)]
    return tuple(g.sort_values("rank")["symbol"])


# ------------------------------------------------------------------------------------ 10-K Item 1 store
@dataclass(frozen=True)
class TenK:
    symbol: str
    filed: pd.Timestamp  # filing date (EDGAR filingDate; no acceptance time in the archive)
    accession: str
    path: Path

    @property
    def filing_id(self) -> str:
        return f"{self.symbol}:{self.accession}"


_TENK_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})_([0-9]{10}-[0-9]{2}-[0-9]{6})\.json$")


class TenKStore:
    """10-K Item 1 ("Business") sections, `data/raw/sec/business/<SYM>/<filed>_<accession>.json`.

    A 10-K filed on date f is usable from the open of the session after f (design_v1 §2.4), so at decision date d_t
    only filings with f < d_t are visible; `latest` returns the most recent such filing whose Item 1 extraction
    succeeded (>= TENK_MIN_CHARS characters). A failed extraction is skipped rather than shown as empty text: the
    previous 10-K's profile then stays current, which is still point in time."""

    def __init__(self, root: Path = ROOT / "data" / "raw" / "sec" / "business", min_chars: int = TENK_MIN_CHARS):
        self.root, self.min_chars = Path(root), min_chars
        self._index: dict[str, list[TenK]] = {}

    def filings(self, symbol: str) -> list[TenK]:
        if symbol not in self._index:
            d = self.root / symbol
            out = []
            if d.is_dir():
                for p in d.iterdir():
                    m = _TENK_NAME.match(p.name)
                    if m:
                        out.append(TenK(symbol, pd.Timestamp(m.group(1)), m.group(2), p))
            self._index[symbol] = sorted(out, key=lambda x: (x.filed, x.accession))
        return self._index[symbol]

    @staticmethod
    @lru_cache(maxsize=512)
    def _read(path: str) -> dict:
        return json.loads(Path(path).read_text())

    def text(self, f: TenK) -> str:
        return self._read(str(f.path)).get("business") or ""

    def usable_text(self, f: TenK) -> bool:
        return int(self._read(str(f.path)).get("n_chars") or 0) >= self.min_chars

    def latest(self, symbol: str, d: pd.Timestamp) -> TenK | None:
        """Most recent 10-K with filed < d (date) and a usable Item 1 text."""
        d = pd.Timestamp(d).normalize()
        for f in reversed(self.filings(symbol)):
            if f.filed < d and self.usable_text(f):
                return f
        return None


# ------------------------------------------------------------------------------------ PIT study membership
def _utc_ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


class StudyMembers:
    """Which study firms had been members of U (the monthly top-100) at a reconstitution on or before a date.

    Membership of a reconstitution is fixed by rankings strictly before its date, so the set at date d is public at
    d; it only grows (a firm that left U stays in it). The universe file starts on 2019-06-03: a date before that
    uses the first reconstitution's members (the earliest ranking available; logged once)."""

    def __init__(self, membership: pd.DataFrame, top: int = TOP_N):
        m = membership[membership["rank"] < top]
        first = m.groupby("symbol")["date"].min()
        self.first: dict[str, pd.Timestamp] = {s: pd.Timestamp(d).normalize() for s, d in first.items()}
        self._dates = sorted(set(self.first.values()))
        self._sets: dict[int, frozenset[str]] = {}
        self._warned = False

    def asof(self, when) -> frozenset[str]:
        """Members at `when` (a UTC time: its New York date; or a date)."""
        if not self._dates:
            return frozenset()
        d = ny_day(when)
        if d < self._dates[0]:
            if not self._warned:
                log.warning("study membership asked for %s, before the universe file starts (%s): using the first "
                            "reconstitution's members", d.date(), self._dates[0].date())
                self._warned = True
            d = self._dates[0]
        k = bisect.bisect_right(self._dates, d)
        if k not in self._sets:
            cut = self._dates[k - 1]
            self._sets[k] = frozenset(s for s, f in self.first.items() if f <= cut)
        return self._sets[k]


class MembersView:
    """`study_members` of one week: `StudyMembers.asof` with the cutoff capped at the week's cutoff (a program can
    never ask about membership after c_t)."""

    def __init__(self, members: StudyMembers, asof_utc):
        self._members, self.asof = members, _utc_ts(asof_utc)

    def __call__(self, cutoff=None) -> frozenset[str]:
        c = self.asof if cutoff is None else min(_utc_ts(cutoff), self.asof)
        return self._members.asof(c)


class PitBlinder:
    """The week's view of the study blinder: a text blinded at cutoff c masks the study firms that had been U members
    by c (`StudyMembers`) plus its focal firm (`symbol`), never a firm that joins U later (masked versus named would
    leak future membership). Tagged firms that are not yet members stay named, like any other non-member.

    Same call interface as `data.blinding.Blinder` (`__call__`, `blind_entities`, `blind_entity`, `code_name`,
    `code_names`, `mask_levels`, `blind_time`); `code_names` holds the members at the week's cutoff. A cutoff later than
    the week's is capped at it. The per-set Blinders are compiled once and shared by every week of a run."""

    def __init__(self, data: RealData, asof_utc):
        self._data = data
        self.asof = _utc_ts(asof_utc)
        self.members_asof = MembersView(data.members, self.asof)
        base = data.blinder
        self.salt = base.salt
        self.code_names = {s: base.code_names[s] for s in sorted(self.members_asof()) if s in base.code_names}

    def code_name(self, symbol: str) -> str:
        return self._data.blinder.code_names[symbol]

    def blinder_at(self, cutoff=None, symbol: str | None = None):
        """The Blinder of the firms masked in a text blinded at `cutoff` (None: the week's) about `symbol`."""
        syms = self.members_asof(cutoff)
        return self._data.blinder_for(syms | {symbol} if symbol else syms)

    def _cap(self, cutoff):
        """A cutoff after the week's is capped (also for the alias-discovery anchor and claim validity)."""
        return None if cutoff is None else min(_utc_ts(cutoff), self.asof)

    def __call__(self, text: str, symbol: str | None = None, decision_year: int | None = None, *, cutoff=None,
                 tagged: Iterable[str] = ()) -> str:
        c = self._cap(cutoff)
        return self.blinder_at(c, symbol)(text, symbol, decision_year, cutoff=c, tagged=tagged)

    def blind_entities(self, text: str, cutoff=None, tagged: Iterable[str] = ()) -> str:
        c = self._cap(cutoff)
        return self.blinder_at(c).blind_entities(text, c, tagged)

    def blind_entity(self, text: str, symbol: str | None = None) -> str:
        return self.blinder_at(None, symbol).blind_entity(text, symbol)

    @staticmethod
    def mask_levels(text: str) -> str:
        from jevtrader.data.blinding import mask_levels

        return mask_levels(text)

    @staticmethod
    def blind_time(text: str, decision_year: int) -> str:
        from jevtrader.data.blinding import blind_time

        return blind_time(text, decision_year)

    def filing_view(self) -> FilingView:
        """The blinder of 10-K text (FILING_MASKING; F-A): the members at this week's cutoff plus the focal firm."""
        return FilingView(self)

    def __getattr__(self, k: str):  # firms, shared, discovery, lives, ...: the week's Blinder
        if k.startswith("_"):
            raise AttributeError(k)
        return getattr(self.blinder_at(None), k)


class FilingView:
    """The blinder of 10-K text (FILING_MASKING, F-A): it masks the study firms that are U members at the reading
    week's cutoff (never a firm that joins U later: masked versus named would leak future membership) plus the text's
    focal firm, while alias claims, executives, levels and times follow the text's own `cutoff` (the filing's usable
    time; capped at the week's). Same call interface as `data.blinding.Blinder.__call__`."""

    def __init__(self, pit: PitBlinder):
        self._pit, self.asof = pit, pit.asof

    def members(self) -> frozenset[str]:
        return self._pit.members_asof(None)

    def __call__(self, text: str, symbol: str | None = None, decision_year: int | None = None, *, cutoff=None,
                 tagged: Iterable[str] = ()) -> str:
        c = self._pit._cap(cutoff)
        return self._pit.blinder_at(None, symbol)(text, symbol, decision_year, cutoff=c, tagged=tagged)

    def code_name(self, symbol: str) -> str:
        return self._pit.code_name(symbol)


def _file_sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_listing_sha256(d: Path) -> str:
    """Identity of a directory store: sha256 over (relative path, size, mtime) of every file (a content hash of the
    1 GB SEC archive would cost seconds per run; a re-download changes the listing, which is the conservative side)."""
    h = hashlib.sha256()
    if d.is_dir():
        for p in sorted(x for x in d.rglob("*") if x.is_file()):
            st = p.stat()
            h.update(f"{p.relative_to(d).as_posix()}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------ heavy stores
@dataclass
class RealData:
    """The archive's stores, loaded lazily and shared by every week of a run (read-only after loading)."""

    root: Path = ROOT
    symbols: tuple[str, ...] | None = None  # the study symbol set; default: every member of the universe file
    news_path: Path | None = None
    universe_path: Path | None = None
    # PIT regex alias discovery (design_v1 §5.0), rule `pit-alias-v3` (data/blinding.py). OFF by default, a logged
    # design deviation (docs/impl/day3_blind.md): the Day-1 rule masked ordinary headline words ("Joe Biden", "Powell",
    # "Tariffs", "Information") as a heavily covered firm's code name; the redesigned rule's audited precision on
    # 2019-07..2024-07 per (firm, alias) pair (0.865 after the audit relabel, in-sample; 0.659 for v2 on held-out
    # firms) is below the 0.95 gate. Per masked occurrence an independent sample gives 0.985 [0.957, 0.995]; the
    # gate's unit is the orchestrator's decision. Switching it on changes every text state and request key.
    alias_discovery: bool = False
    # the symbols whose fundamentals are loaded (default: the study set). A run needs the U members of every date it
    # ranks sizes at: its weeks, and (F-A) the first week of each 10-K it profiles, up to ~15 months earlier
    # (`symbols_for_weeks`)
    fact_symbols: tuple[str, ...] | None = None
    _pool: dict = field(default_factory=dict, init=False, repr=False, compare=False)  # member set -> Blinder

    @cached_property
    def membership(self) -> pd.DataFrame:
        u = pd.read_parquet(self.universe_path or self.root / "data" / "processed" / "universe_monthly_top100.parquet")
        u["date"] = pd.to_datetime(u["date"])
        return u

    @cached_property
    def study_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.symbols or set(self.membership["symbol"])))

    @cached_property
    def members(self) -> StudyMembers:
        return StudyMembers(self.membership)

    def symbols_for_weeks(self, weeks: Iterable[pd.Timestamp], lookback_days: int = 800) -> tuple[str, ...]:
        """Every U member at a reconstitution in [first week - lookback, last week]: the `fact_symbols` a run over
        `weeks` needs (U_t of its weeks, and U at the first week of any 10-K usable at its weeks, for F-A's size; a
        10-K is at most ~13 months old, ~25 when the latest extraction failed and the previous one is used)."""
        ws = sorted(pd.Timestamp(w) for w in weeks)
        if not ws:
            return ()
        u = self.membership
        dates = pd.to_datetime(u["date"])
        lo = ws[0] - pd.Timedelta(days=lookback_days)
        k = dates[dates <= lo].max()  # the reconstitution in force at `lo`
        keep = (dates >= (k if pd.notna(k) else dates.min())) & (dates <= ws[-1]) & (u["rank"] < TOP_N)
        return tuple(sorted(set(u.loc[keep, "symbol"])))

    @cached_property
    def panels(self):
        from jevtrader.data.prices import load_price_panels

        log.info("loading price panels")
        return load_price_panels(self.root)

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.panels.sessions

    @cached_property
    def news_table(self) -> pd.DataFrame:
        return pd.read_parquet(self.news_path or self.root / "data" / "processed" / "news_table.parquet")

    @cached_property
    def news(self):
        from jevtrader.data.news import NewsStore

        return NewsStore(self.news_table)

    @cached_property
    def news_times(self) -> np.ndarray:
        """Sorted creation times of the archive's items (UTC, int64 ns), for the week-coverage check."""
        c = pd.DatetimeIndex(pd.to_datetime(self.news_table["created_at"], utc=True)).as_unit("ns")
        return np.sort(c.asi8)

    def news_count(self, start: pd.Timestamp, end: pd.Timestamp) -> int:
        a = self.news_times
        return int(np.searchsorted(a, _utc_ts(end).value, "left") - np.searchsorted(a, _utc_ts(start).value, "left"))

    @cached_property
    def facts(self):
        from jevtrader.data.fundamentals import FundamentalsPanel

        syms = list(self.fact_symbols or self.study_symbols)
        log.info("building the fundamentals panel for %d symbols", len(syms))
        return FundamentalsPanel(self.root / "data" / "raw" / "sec" / "facts", syms, self.panels)

    @cached_property
    def macro(self):
        from jevtrader.features.macro import MacroStore

        return MacroStore(self.root / "data" / "raw" / "macro")

    @cached_property
    def earnings(self):
        from jevtrader.data.earnings import EarningsCalendar

        return EarningsCalendar(self.root / "data" / "raw" / "sec" / "submissions", list(self.study_symbols))

    @cached_property
    def filings(self) -> TenKStore:
        return TenKStore(self.root / "data" / "raw" / "sec" / "business")

    @cached_property
    def ff12(self) -> dict[str, str]:
        from jevtrader.data.universe import ff12_map

        return ff12_map(list(self.study_symbols), self.root / "data" / "raw" / "sec" / "submissions")

    @cached_property
    def sector_map(self):
        from jevtrader.features.technical import SectorMap

        return SectorMap(self.panels.adjusted)

    @cached_property
    def lives(self) -> dict[str, tuple]:
        """Trading lives (first, last raw bar), as `blinding.lives_from_bars` computes them."""
        raw = self.panels.raw.close
        return {s: (c.first_valid_index(), c.last_valid_index()) for s, c in raw.items() if c.notna().any()}

    @cached_property
    def blinder(self):
        """The study Blinder over the full study symbol set: alias tables (dated historical tickers from the resolver,
        brand aliases), pseudonyms, PIT alias discovery on the news table (items created and last edited before each
        monthly anchor), and trading lives to resolve shared aliases. Contexts never mask with it directly: they use
        `pit_blinder` (only the firms that are U members by the text's cutoff)."""
        from jevtrader.data.blinding import load_blinder
        from jevtrader.data.universe import SymbolResolver, load_membership, load_renames

        log.info("building the blinder over %d symbols", len(self.study_symbols))
        m = load_membership(self.root / "data" / "raw" / "universe" / "sp500_hist_components.csv", "2018-01-01")
        resolver = SymbolResolver(load_renames(self.root / "data" / "raw" / "universe" / "alpaca_name_changes.json", m))
        if not self.alias_discovery:
            log.warning("blinder built WITHOUT PIT alias discovery (RealData.alias_discovery=False)")
        return load_blinder(self.root / "data" / "raw" / "sec" / "submissions", self.study_symbols, resolver=resolver,
                            history=(date(2018, 1, 1), date(2026, 12, 31)), lives=self.lives,
                            **self.discovery_kwargs())

    def discovery_kwargs(self) -> dict[str, Any]:
        """The `load_blinder` arguments of PIT alias discovery (empty when it is off): every row of the news table,
        MARKET items included (exclusivity is measured against other firms' and market items), the PIT member set per
        discovery anchor (items tagged only to firms that join U later are not counted before they join, and such a
        firm gets no alias before it joins) and the trading lives (no alias for a firm outside its life)."""
        if not self.alias_discovery:
            return {}
        return {"discovery_items": self.news_table, "members": self.members.asof, "lives": self.lives}

    def blinder_for(self, symbols: Iterable[str]):
        """A Blinder masking exactly `symbols` (with the study blinder's alias tables, pseudonyms, discovery and
        lives); compiled once per symbol set."""
        from jevtrader.data.blinding import Blinder

        base = self.blinder
        key = frozenset(symbols) & frozenset(base.firms)
        b = self._pool.get(key)
        if b is None:
            b = Blinder({s: base.firms[s] for s in sorted(key)}, salt=base.salt, discovery=base.discovery,
                        lives=self.lives)
            self._pool[key] = b
        return b

    def pit_blinder(self, cutoff_utc) -> PitBlinder:
        return PitBlinder(self, cutoff_utc)

    def archive_id(self) -> str:
        """Identity of the archive a context reads (the run's data freeze): sha256 over the content of the processed
        tables and the file listings of the raw stores."""
        if "_archive_id" not in self.__dict__:
            h = hashlib.sha256()
            files = [self.universe_path or self.root / ARCHIVE_FILES[0], self.news_path or self.root / ARCHIVE_FILES[1]]
            for p in files:
                p = Path(p)
                h.update(f"{p.name}\t{_file_sha256(p) if p.exists() else 'missing'}\n".encode())
            for d in ARCHIVE_DIRS:
                h.update(f"{d}\t{_dir_listing_sha256(self.root / d)}\n".encode())
            self.__dict__["_archive_id"] = "archive-" + h.hexdigest()[:16]
        return self.__dict__["_archive_id"]

    def describe(self) -> dict[str, Any]:
        """What the context's data are (part of every unit fingerprint through `ContextBuilder.describe`): the study
        set, the blinder configuration and the archive identity. `fact_symbols` is a loading choice, not part of it."""
        from jevtrader.data.blinding import DISCOVERY_RULE, TEXT_RULES, aliases_sha256

        # the text rules' version (hosts, levels, times, brand list) changes every text state it touches, so it is part
        # of the description: a resume never mixes shards blinded under two rule versions; so do the curated alias
        # tables (manual aliases, executives: data/raw/aliases) and the 10-K masking rule
        blinder: dict[str, Any] = {"masking": MASKING_RULE, "text_rules": TEXT_RULES,
                                   "alias_discovery": self.alias_discovery, "filing_masking": FILING_MASKING,
                                   "aliases": aliases_sha256(self.root / "data" / "raw" / "sec" / "submissions")}
        if self.alias_discovery:  # the discovery rule's version
            blinder["discovery_rule"] = DISCOVERY_RULE
        return {"n_study_symbols": len(self.study_symbols),
                "study_symbols_sha256": hashlib.sha256(",".join(self.study_symbols).encode()).hexdigest()[:16],
                "blinder": blinder, "archive": self.archive_id()}


@dataclass
class AvailabilityReport:
    t: pd.Timestamp
    hard: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)  # a source that does not cover the week (an outage)
    soft: list[str] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    unavailable: set[str] = field(default_factory=set)  # the stale sources: "news", "macro", "facts"

    def to_json(self) -> dict:
        return {"t": str(pd.Timestamp(self.t).date()), "hard": self.hard, "stale": self.stale, "soft": self.soft,
                "unavailable": sorted(self.unavailable), "counts": self.counts,
                "coverage": {k: round(float(v), 4) for k, v in self.coverage.items()}}


class ContextBuilder:
    """`t -> WeekContext` over the real archive (a `GeneratorSpec.context_factory`)."""

    def __init__(self, data: RealData | None = None, *, subset: str | None = None,
                 symbols: Sequence[str] | None = None, top: int = TOP_N, strict: bool = True,
                 with_blinder: bool = True):
        """`subset`: 'S50' | 'S30' (per-week deterministic subsets of U_t) and/or `symbols` (an explicit list): the
        focus names whose stock-level requests are run. `strict` (default): a hard failure or a stale source raises
        `DataAvailabilityError`; otherwise it is logged as a warning and the stale sources are listed in
        `WeekContext.unavailable`. `with_blinder=False` skips the (slow) blinder for runs whose programs read no text
        and render no code names (never for real runs: every state needs code names)."""
        self.data = data or RealData()
        self.subset, self.symbols, self.top, self.strict = subset, (tuple(symbols) if symbols else None), top, strict
        self.with_blinder = with_blinder
        if subset is not None and subset not in SUBSETS:
            raise ValueError(f"subset must be one of {sorted(SUBSETS)}")
        self.reports: dict[pd.Timestamp, AvailabilityReport] = {}
        self._ranks: dict[tuple, dict[str, int]] = {}
        self._warned_days: set[pd.Timestamp] = set()

    def describe(self) -> dict[str, Any]:
        """Everything a context depends on besides the week (the runner puts its hash in every unit fingerprint, so
        a resume never mixes shards built for other names, another blinder or another archive)."""
        return {"builder": CONTEXT_VERSION, "subset": self.subset,
                "symbols": sorted(self.symbols) if self.symbols is not None else None, "top": self.top,
                "strict": self.strict, "with_blinder": self.with_blinder, "data": self.data.describe()}

    # ------------------------------------------------------------------ parts
    def universe(self, t: pd.Timestamp) -> tuple[str, ...]:
        return universe_asof(self.data.membership, t, self.top)

    def focus(self, t: pd.Timestamp, universe: tuple[str, ...]) -> tuple[str, ...] | None:
        names = universe
        if self.subset is not None:
            salt, n = SUBSETS[self.subset]
            names = subset_names(names, n, salt)
        if self.symbols is not None:
            want = set(self.symbols)
            names = tuple(s for s in names if s in want)
        return None if names == universe else names

    def _size_ranks(self, t: pd.Timestamp, members: tuple[str, ...]) -> dict[str, int]:
        from jevtrader.data.fundamentals import size_ranks

        key = (pd.Timestamp(t), members)
        if key not in self._ranks:
            feats = self.data.facts.features(t, list(members))
            mv = feats["market_value"] if "market_value" in feats else pd.Series(dtype=float)
            self._ranks[key] = size_ranks(mv, list(members))
        return self._ranks[key]

    def firms(self, t: pd.Timestamp, universe: tuple[str, ...]) -> dict[str, FirmInfo]:
        d = self.data
        ranks = self._size_ranks(t, tuple(universe))
        sector = d.sector_map.asof(t, list(universe))
        return {s: FirmInfo(industry=d.ff12.get(s, "Other"), size_rank=ranks[s], sector_etf=sector.get(s) or None)
                for s in universe}

    def firm_asof(self, symbol: str, day) -> FirmInfo:
        """`symbol`'s FF12 industry and size rank at decision date `day`, as a run deciding on `day` renders them:
        the rank by raw market value within U_day (the firm added at the end if it was not a member). Used by F-A for
        the focal firm of a 10-K at the first week the filing could be read, so the request does not depend on the
        week that first reads it. No sector ETF (F-A does not render one)."""
        day = pd.Timestamp(day).normalize()
        uni = self.universe(day) or self.universe(self.data.membership["date"].min())  # before the file: 1st ranking
        members = tuple(uni) if symbol in uni else (*uni, symbol)
        fs = self.data.fact_symbols
        if fs is not None and set(members) - set(fs) and day not in self._warned_days:
            self._warned_days.add(day)
            log.warning("firm_asof %s: %d members of U at that date are outside fact_symbols (their market values "
                        "are missing from the size ranks); widen RealData.symbols_for_weeks", day.date(),
                        len(set(members) - set(fs)))
        return FirmInfo(industry=self.data.ff12.get(symbol, "Other"), size_rank=self._size_ranks(day, members)[symbol])

    # ------------------------------------------------------------------ checks
    def check(self, t: pd.Timestamp, universe: tuple[str, ...] | None = None) -> AvailabilityReport:
        d, t = self.data, pd.Timestamp(t)
        rep = AvailabilityReport(t)
        sessions = d.sessions
        cutoff = decision_cutoff(t)
        if not is_decision_date(sessions, t):
            rep.hard.append(f"{t.date()} is not the last session of its ISO week (not a decision date)")
        close = d.panels.adjusted.close
        if t not in close.index or not np.isfinite(close.at[t, MARKET_SYMBOL]):
            rep.hard.append(f"no {MARKET_SYMBOL} bar on {t.date()}")
        uni = universe if universe is not None else self.universe(t)
        if not uni:
            rep.hard.append(f"no universe reconstitution on or before {t.date()}")
            return rep
        have_close = [s for s in uni if s in close and t in close.index and np.isfinite(close.at[t, s])]
        rep.coverage["close_on_d"] = len(have_close) / len(uni)
        if self.with_blinder:
            missing = [s for s in uni if s not in d.blinder.code_names]
            if missing:
                rep.hard.append(f"members without a pseudonym (cannot blind): {missing}")
            unmasked = sorted(set(uni) - d.members.asof(cutoff))
            if unmasked:
                rep.hard.append(f"members of U_t the PIT blinder would not mask: {unmasked}")

        def stale(source: str, msg: str) -> None:
            rep.stale.append(msg)
            rep.unavailable.add(source)

        # stale sources: an outage must not render as a quiet (no news) or unchanged (old macro, old filings) week
        last_news = pd.Timestamp(d.news_times[-1], tz="UTC") if len(d.news_times) else None
        if last_news is None or last_news < cutoff - NEWS_STALE:
            stale("news", f"news archive ends {last_news}, before the cutoff {cutoff}")
        prev = previous_decision(sessions, t)
        start = decision_cutoff(prev) if prev is not None else cutoff - pd.Timedelta(days=7)
        rep.counts["news_items_in_window"] = d.news_count(start, cutoff)
        if rep.counts["news_items_in_window"] == 0:
            stale("news", f"no news item created in [{start}, {cutoff})")
        snaps = [d.facts.snapshot_asof(s, t) for s in uni]
        rep.coverage["facts"] = float(np.mean([x is not None for x in snaps]))
        filed = [pd.Timestamp(x["last_filing_date"]) for x in snaps if x is not None and x.get("last_filing_date")]
        if filed and rep.coverage["facts"] >= 0.9 and (t - max(filed)).days > FACTS_STALE_DAYS:  # over a panel of U_t
            stale("facts", f"no XBRL filing by any U_t member since {max(filed).date()} "
                           f"({(t - max(filed)).days} days before {t.date()})")
        if d.fact_symbols is not None and set(uni) - set(d.fact_symbols):
            rep.soft.append(f"fact_symbols miss {len(set(uni) - set(d.fact_symbols))} U_t members (size ranks)")
        rep.coverage["form_10k"] = float(np.mean([d.filings.latest(s, t) is not None for s in uni]))
        rep.coverage["ff12"] = float(np.mean([s in d.ff12 for s in uni]))
        m = d.macro
        for s in ("VIXCLS", "DGS10", "DFF"):
            got = m.daily_asof(s, t)
            if got.empty or (t - got.index[-1]).days > FRED_STALE_DAYS:
                stale("macro", f"FRED {s} has no observation within {FRED_STALE_DAYS} days before {t.date()} minus "
                               f"its lag")
        if m.epu_asof(t).empty or (t - m.epu_asof(t).index[-1]).days > EPU_STALE_DAYS:
            stale("macro", f"EPU archive ends before {t.date()} - lag")
        if m.gpr_asof(t).empty or (t - m.gpr_asof(t).index[-1]).days > GPR_STALE_DAYS:
            stale("macro", f"GPR archive ends before {t.date()} - lag")
        if m.fed_decision_within(t) is None:
            rep.soft.append("no FOMC calendar: `Fed rate decision within the next week` renders as not available")
        for k, v in rep.coverage.items():
            if v < 0.9:
                rep.soft.append(f"coverage of {k} in U_t is {v:.0%}")
        return rep

    # ------------------------------------------------------------------ the factory
    def __call__(self, t: pd.Timestamp) -> WeekContext:
        t = pd.Timestamp(t)
        d = self.data
        uni = self.universe(t)
        rep = self.check(t, uni)
        self.reports[t] = rep
        for msg in rep.soft:
            log.warning("availability %s: %s", t.date(), msg)
        if rep.hard or rep.stale:
            if self.strict:
                raise DataAvailabilityError(f"{t.date()}: " + "; ".join(rep.hard + rep.stale))
            log.warning("availability %s (non-strict; unavailable: %s): %s", t.date(), sorted(rep.unavailable),
                        "; ".join(rep.hard + rep.stale))
        prev = previous_decision(d.sessions, t)
        cutoff = decision_cutoff(t)

        def firm_asof(symbol: str, day) -> FirmInfo:
            if pd.Timestamp(day).normalize() > t:
                raise ValueError(f"firm_asof({symbol}, {pd.Timestamp(day).date()}) is after the decision date {t.date()}")
            return self.firm_asof(symbol, day)

        return WeekContext(
            t=t, cutoff_utc=cutoff, universe=uni, prices=d.panels.adjusted, raw_prices=d.panels.raw,
            news=d.news, facts=d.facts, macro=d.macro, earnings=d.earnings,
            blinder=d.pit_blinder(cutoff) if self.with_blinder else None, firms=self.firms(t, uni),
            sessions=d.sessions, prev_cutoff_utc=None if prev is None else decision_cutoff(prev), filings=d.filings,
            focus=self.focus(t, uni), study_members=MembersView(d.members, cutoff), firm_asof=firm_asof,
            unavailable=frozenset(rep.unavailable),
        )


def load_context_factory(subset: str | None = None, symbols: Sequence[str] | None = None, root: Path = ROOT,
                         **kw: Any) -> ContextBuilder:
    return ContextBuilder(RealData(root=root), subset=subset, symbols=symbols, **kw)
