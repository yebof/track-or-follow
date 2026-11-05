"""Injection harness: apply one attack unit to a COPY of the news table (design_v1 section 13.1; RB-1..RB-3, RB-11).

The real table (`data/processed/news_table.parquet`) is never modified: `apply()` returns a new frame and
`save_attacked()` refuses every path under `data/processed/` and `data/raw/news/`.

A unit is (symbol, decision date t, family). Point in time, with the analyst's window of week w being
[c_{t_w - 1}, c_{t_w}) (c = 16:00 ET cutoff, `data.news.decision_cutoff`):
- **append / insert** (A1-A4, A2-neg, NEUTRAL; "an edited feed item"): the host is the most recent item of the
  focal symbol in week t's window that the news analyst actually reads, by the readers' own item rules
  (`programs.news.ItemSource`): tagged with <= 3 tickers, created in the window, last edited before c_t, a non-empty
  headline after D0, and not an exact D0-duplicate of an earlier item's headline (the reader keeps the earliest copy);
  most recent first, ties by the smaller id string. Its text is edited on every row of that item (all its tags), and
  `updated_at` becomes max(last edit, c_t - 1 h), still before c_t. A unit without such a host raises `NoHostItem`
  (never an edit that no reader sees).
- **new** (A5, A6, A6-neg, A7, A7-para, A8-1, A8-2; "a planted item"): new items tagged only to the focal symbol,
  created (and last edited) 3 h before the cutoff of their week; A8's first item 27 h before. Planted ids are
  negative (never a real id) and a fixed hash of the unit.

Company names. `{company}` is the name genuine items of the feed use for the firm (`display_names_v1.json`, built
and checked by `scripts/build_redteam_names.py`: every name blinds to exactly the firm's code name, so a planted item
reads like a genuine one after D0 and blinding, with no residue). A symbol without a frozen name raises
`MissingDisplayName`. The table is hashed into `attacks.templates_sha256`.

The harness imports the readers' D0 (`programs.render.d0`, sanitizer code, no wording) for the host rule; the
templates (`attacks.py`) import nothing from the bank or the programs (held-out wording).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from jevtrader.data.news import decision_cutoff
from jevtrader.redteam.attacks import FAMILIES, Rendered, render

ROOT = Path(__file__).resolve().parents[3]
PROTECTED_DIRS = (ROOT / "data/processed", ROOT / "data/raw/news")
NAMES_PATH = Path(__file__).with_name("display_names_v1.json")
MAX_TAGS = 3
EDIT_LEAD = pd.Timedelta(hours=1)
NEW_LEAD = pd.Timedelta(hours=3)


class NoHostItem(LookupError):
    """An append/insert family needs a real item of the focal symbol in week t's window that a reader reads."""


class MissingDisplayName(KeyError):
    """No frozen display name for a symbol: the unit cannot plant a genuine-looking name."""


@dataclass(frozen=True)
class AttackUnit:
    symbol: str
    t: pd.Timestamp  # decision date of the attacked week
    family: str
    later: tuple[pd.Timestamp, ...] = ()  # decision dates t+1, t+2 (A7, A7-para)
    prev_t: pd.Timestamp | None = None  # decision date t-1; window start = its cutoff (default: c_t - 7 days)

    def weeks(self) -> list[pd.Timestamp]:
        n = FAMILIES[self.family].weeks
        ws = [pd.Timestamp(self.t), *[pd.Timestamp(x) for x in self.later]]
        if len(ws) < n:
            raise ValueError(f"{self.family} plants in {n} weeks; pass the later decision dates in `later`")
        return ws[:n]

    def window(self, w: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        ws = self.weeks()
        c = decision_cutoff(ws[w])
        if w > 0:
            return decision_cutoff(ws[w - 1]), c
        return (decision_cutoff(self.prev_t) if self.prev_t is not None else c - pd.Timedelta(days=7)), c


@dataclass
class Injection:
    unit: AttackUnit
    rendered: list[Rendered]
    replace_ids: tuple[int, ...]  # real items whose rows are replaced by edited copies
    rows: pd.DataFrame  # rows added to the copy (planted items, or the edited copies)
    host_id: int | None = None
    meta: dict = field(default_factory=dict)

    @property
    def item_ids(self) -> list[int]:
        return sorted(set(self.rows.id.tolist()))


def planted_id(unit: AttackUnit, week: int, index: int) -> int:
    key = f"{unit.symbol}|{pd.Timestamp(unit.t).date()}|{unit.family}|{week}|{index}"
    return -int(hashlib.sha256(key.encode()).hexdigest()[:13], 16)


def reader_headline_key(headline) -> str:
    """The readers' duplicate key of a headline: its D0 text (programs.render.d0, headline cap); "" = dropped."""
    from jevtrader.programs.render import HEADLINE_MAX, d0

    return str(d0(headline, HEADLINE_MAX))


def host_item(news: pd.DataFrame, symbol: str, start: pd.Timestamp, cutoff: pd.Timestamp,
              max_tags: int = MAX_TAGS, headline_key=reader_headline_key) -> pd.Series | None:
    """The most recent item of `symbol` in [start, cutoff) that the readers keep (module doc): <= max_tags tickers,
    last edited before the cutoff, a non-empty D0 headline that is not a duplicate of an earlier item's."""
    g = news[(news.symbol == symbol) & (news.created_at >= start) & (news.created_at < cutoff)
             & (news.updated_at < cutoff)
             & (pd.to_numeric(news.n_tags, errors="coerce").fillna(0) <= max_tags)]
    if g.empty:
        return None
    g = g.assign(_iid=g["id"].astype(str)).sort_values(["created_at", "_iid"], kind="stable")
    seen, kept = set(), []
    for i, h in zip(g.index, g["headline"]):
        k = headline_key(h)
        if k and k not in seen:  # the earliest copy of a headline is the one the readers keep
            seen.add(k)
            kept.append(i)
    if not kept:
        return None
    k = g.loc[kept]
    latest = k["created_at"].max()
    return k[k["created_at"] == latest].sort_values("_iid", kind="stable").iloc[0].drop(labels="_iid")


def _insert_mid(summary: str, text: str) -> str:
    """Insert `text` at the word boundary nearest the middle of `summary` (mid-sentence), else append."""
    if not summary.strip():
        return text
    mid = len(summary) // 2
    spaces = [i for i, ch in enumerate(summary) if ch == " "]
    if not spaces:
        return f"{summary} {text}"
    cut = min(spaces, key=lambda i: (abs(i - mid), i))
    return f"{summary[:cut]} {text}{summary[cut:]}"


def _frame(records: list[dict], like: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(records)
    for c in like.columns:
        if c not in df:
            df[c] = pd.NA
    df = df[list(like.columns)]
    return df.astype(like.dtypes.to_dict())


def plan(news: pd.DataFrame, unit: AttackUnit, company: str | None = None, ticker: str | None = None,
         headline_key=reader_headline_key) -> Injection:
    """What the unit changes, without touching `news`. `company` defaults to the frozen display name."""
    spec = FAMILIES[unit.family]
    company = company if company is not None else display_name(unit.symbol)
    rendered = render(unit.family, unit.symbol, unit.t, company, ticker or unit.symbol)
    if spec.mode in ("append", "insert"):
        start, c = unit.window(0)
        host = host_item(news, unit.symbol, start, c, headline_key=headline_key)
        if host is None:
            raise NoHostItem(f"{unit.symbol} has no item a reader keeps in [{start}, {c}) for {unit.family}")
        r = rendered[0]
        old = host.summary if isinstance(host.summary, str) else ""
        new_summary = f"{old} {r.text}".strip() if spec.mode == "append" else _insert_mid(old, r.text)
        edited = news[news.id == host.id].copy()
        edited["summary"] = new_summary
        edited["updated_at"] = max(pd.Timestamp(host.updated_at), c - EDIT_LEAD)
        return Injection(unit, rendered, (int(host.id),), edited.reset_index(drop=True), int(host.id),
                         {"mode": spec.mode, "edit_time": edited.updated_at.iloc[0]})
    records = []
    for r in rendered:
        start, c = unit.window(r.week)
        lead = NEW_LEAD + pd.Timedelta(hours=24) * (spec.items_per_week - 1 - r.index)
        t_new = max(c - lead, start + pd.Timedelta(minutes=1 + r.index))
        rec = {"id": planted_id(unit, r.week, r.index), "symbol": unit.symbol, "created_at": t_new,
               "updated_at": t_new, "headline": r.headline, "summary": r.text, "source": "benzinga", "n_tags": 1}
        if "author" in news:
            rec["author"] = r.byline
        if "url_section" in news:
            rec["url_section"] = r.section
        records.append(rec)
    return Injection(unit, rendered, (), _frame(records, news), None, {"mode": "new"})


def apply(news: pd.DataFrame, inj: Injection) -> pd.DataFrame:
    """A new table: `news` minus the replaced items, plus the injection's rows. `news` itself is never changed."""
    keep = news[~news.id.isin(inj.replace_ids)] if inj.replace_ids else news
    out = pd.concat([keep, inj.rows], ignore_index=True)
    return out.sort_values(["symbol", "created_at", "id"], kind="stable").reset_index(drop=True)


def inject(news: pd.DataFrame, unit: AttackUnit, company: str | None = None,
           ticker: str | None = None) -> tuple[pd.DataFrame, Injection]:
    inj = plan(news, unit, company, ticker)
    return apply(news, inj), inj


def save_attacked(table: pd.DataFrame, path: Path) -> Path:
    """Write an attacked copy, never over the real news data."""
    p = Path(path).resolve()
    for d in PROTECTED_DIRS:
        if p == d.resolve() or d.resolve() in p.parents:
            raise PermissionError(f"refusing to write an attacked table under {d} (the real news data)")
    p.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(p, index=False)
    return p


def load_display_names(path: Path = NAMES_PATH) -> dict[str, str]:
    """symbol -> the name genuine feed items use for the firm (frozen table; module doc)."""
    import json

    return dict(json.loads(Path(path).read_text())["names"])


def display_name(symbol: str, names: dict[str, str] | None = None) -> str:
    names = load_display_names() if names is None else names
    if symbol not in names or not names[symbol]:
        raise MissingDisplayName(f"no frozen display name for {symbol} (scripts/build_redteam_names.py)")
    return names[symbol]
