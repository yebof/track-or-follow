"""Identity-leak probes for blinding (design_v1 §5.0; audit finding "blinding leaks", 2026-09-27).

The blinder masks the names in its alias tables; a name the feed uses but the tables lack passes through untouched
("NXP Semiconductors", "Applied Mat", "eBay", "3M"). This module finds the names the feed itself uses for each firm
and measures, per symbol, how often a blinded headline still carries one of them:

- `feed_short_names`: the firm name in the archive's machine-written headline templates (broker actions "Barclays
  Maintains Overweight on <NAME>, Raises Price Target ...", options activity "Looking At <NAME>'s Recent Unusual
  Options Activity", "... Invested In <NAME> 10 Years Ago ...") of items tagged to that symbol alone, counted from two
  years before its first to two years after its last universe membership (`membership_span_mask`; a reused ticker
  such as APC, Anadarko until 2019 and an unrelated issuer from 2024, is read in its study span only);
- `probes`: per symbol, case-insensitive whole-word patterns for its frequent short names and the distinctive first
  word of its SEC name;
- `leak_report`: per symbol, the share of its blinded headlines in which a probe still matches.

It is a measurement, never a mask: the blinder does not read it (curated names go to `blinding.FEED_SHORT_NAMES` by
hand, after review). Used by `tests/test_news_blinding_leaks.py`, `scripts/blinding_leak_scan.py`, the TA-lite
smoke leak check and the red-team display-name table.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping

import pandas as pd

from jevtrader.data.blinding import _PROTECTED, COMMON_WORDS, WORD_NAMES, clean_name

_KEEP = re.compile(rf"(?i)(?<![A-Za-z0-9])(?:{_PROTECTED})(?![A-Za-z0-9])")  # "Price Target" is kept on purpose

TEMPLATES = (
    re.compile(r"^.+? (?:Maintains|Reiterates|Upgrades|Downgrades|Initiates Coverage On|Assumes|Resumes) "
               r"(?:\w+(?:[- ]\w+)? on )?(?P<n>.+?)(?:,| to | with )"),
    re.compile(r"^Looking At (?P<n>.+?)'s Recent Unusual Options Activity"),
    re.compile(r"^(?P<n>.+?) Unusual Options Activity"),
    re.compile(r"Invested In (?P<n>.+?) (?:\d+|Five|Ten|Fifteen|Twenty) Years Ago"),
    re.compile(r"^(?P<n>.+?)'s Options Frenzy"),
    re.compile(r"^(?P<n>.+?) Earnings Preview"),
    re.compile(r"^A Closer Look at (?P<n>.+?)'s Options"),
)
_NOT_A_NAME = re.compile(r"(?i)^(?:this stock|neutral|buy|sell|hold|outperform|underperform|overweight|"
                         r"underweight|equal-weight|market perform|sector perform|peer perform)$")
_RATING_PREFIX = re.compile(r"(?i)^(?:(?:buy|sell|hold|neutral|outperform|overweight|underweight|equal-weight|"
                            r"market perform|sector perform|in-line) on |analyzing |understanding )")


def _ws(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def membership_mask(rows: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """True where `rows.symbol` was a member of the universe reconstitution in force at `rows.created_at` (the last
    `universe.date` on or before the item's UTC date)."""
    u = universe.assign(date=pd.to_datetime(universe["date"]).dt.tz_localize(None).dt.normalize())
    dates = pd.DatetimeIndex(sorted(u["date"].unique()))
    members = {d: set(g) for d, g in u.groupby("date")["symbol"]}
    day = pd.to_datetime(rows["created_at"], utc=True).dt.tz_localize(None).dt.normalize()
    pos = dates.searchsorted(day.values, side="right") - 1
    return pd.Series([p >= 0 and s in members[dates[p]] for p, s in zip(pos, rows["symbol"])], index=rows.index)


def membership_span_mask(rows: pd.DataFrame, universe: pd.DataFrame, pad_days: int = 730) -> pd.Series:
    """True where `rows.created_at` lies within `pad_days` of the span from the symbol's first to its last universe
    reconstitution date (symbols never in `universe`: False)."""
    u = pd.to_datetime(universe["date"]).dt.tz_localize(None).dt.normalize()
    span = u.groupby(universe["symbol"]).agg(["min", "max"])
    pad = pd.Timedelta(days=pad_days)
    day = pd.to_datetime(rows["created_at"], utc=True).dt.tz_localize(None)
    lo = rows["symbol"].map(span["min"] - pad)
    hi = rows["symbol"].map(span["max"] + pad)
    return (day >= lo) & (day <= hi)


def feed_short_names(news: pd.DataFrame, universe: pd.DataFrame | None = None, min_count: int = 5,
                     clean: Callable[[str], str] | None = None, pad_days: int = 730
                     ) -> dict[str, list[tuple[str, int]]]:
    """symbol -> [(short name, count)] from the machine-written templates of its single-ticker items (module doc),
    most frequent first, counts >= min_count. `universe` restricts items to the symbol's padded membership span."""
    clean = clean or _ws
    w = news[news["n_tags"] == 1]
    if universe is not None:
        w = w[membership_span_mask(w, universe, pad_days)]
    w = w.drop_duplicates(["symbol", "headline"])
    out: dict[str, list[tuple[str, int]]] = {}
    for sym, g in w.groupby("symbol"):
        c: Counter = Counter()
        for h in g["headline"]:
            h = str(clean(h))
            for rx in TEMPLATES:
                m = rx.search(h)
                if m:
                    n = _RATING_PREFIX.sub("", m.group("n").strip())
                    n = re.sub(r"'s$", "", n)
                    if 2 <= len(n) <= 40 and not _NOT_A_NAME.match(n):
                        c[n] += 1
                    break
        top = [(n, k) for n, k in c.most_common() if k >= min_count]
        if top:
            out[str(sym)] = top
    return out


def sec_first_word(sec_name: str | None) -> str | None:
    """The distinctive first word of a cleaned SEC name ("NXP" of "NXP Semiconductors N.V."), or None when it is
    short, numeric, or an ordinary word ("Royal", "Applied")."""
    ws = clean_name(sec_name or "").split()
    if not ws:
        return None
    w = ws[0]
    if len(w) < 3 or not w.isalpha() or w.casefold() in COMMON_WORDS or w.casefold() in WORD_NAMES:
        return None
    return w


def probes(short_names: Mapping[str, Iterable[tuple[str, int]]], sec_names: Mapping[str, str] | None = None,
           min_count: int = 10) -> dict[str, list[re.Pattern]]:
    """symbol -> case-insensitive whole-word patterns: its short names with >= min_count uses and the distinctive
    first word of its SEC name."""
    out: dict[str, list[re.Pattern]] = {}
    for sym in set(short_names) | set(sec_names or {}):
        strings = [n for n, k in short_names.get(sym, ()) if k >= min_count]
        fw = sec_first_word((sec_names or {}).get(sym))
        if fw:
            strings.append(fw)
        pats = [re.compile(r"(?i)(?<![A-Za-z0-9])" + re.escape(x).replace(r"\ ", r"[,.]?\s+") + r"(?![A-Za-z0-9])")
                for x in dict.fromkeys(strings)]
        if pats:
            out[sym] = pats
    return out


def leak_report(blind: Callable[[str, str, pd.Timestamp], str], rows: pd.DataFrame,
                probe: Mapping[str, list[re.Pattern]], max_examples: int = 3) -> pd.DataFrame:
    """Per symbol: headlines checked, headlines whose blinded text still matches one of the symbol's probes, the
    rate, and examples. `blind(text, symbol, created_at)` is the full text path under test (D0, then blinding with
    the item's tags); `rows` has symbol, headline, created_at."""
    recs = []
    for sym, g in rows.groupby("symbol"):
        pats = probe.get(str(sym), [])
        n = leaks = 0
        ex: list[str] = []
        for h, t in zip(g["headline"], g["created_at"]):
            b = _KEEP.sub(" ", blind(h, str(sym), t))
            n += 1
            if any(p.search(b) for p in pats):
                leaks += 1
                if len(ex) < max_examples:
                    ex.append(b)
        recs.append({"symbol": sym, "n": n, "leaks": leaks, "rate": leaks / n if n else 0.0, "examples": ex})
    return pd.DataFrame(recs, columns=["symbol", "n", "leaks", "rate", "examples"]).sort_values(
        "rate", ascending=False, kind="stable").reset_index(drop=True)
