"""Feed provenance (design_v2_delta FEED-1, SAN-5): who wrote the items, and through which site section.

Every archived item has `source` in {"benzinga", ""} and a URL on www.benzinga.com, so the feed's provenance is
described by its byline (`author`) and its URL section (`url_section`, the URL's first path segment; both carried by
`data/processed/news_table.parquet`).

Byline classes (operational rules, applied in this order to the provenance key, `data.news.provenance_key`):
1. `desk`: one of the bank's desk bylines (`code_constants.corroboration.desk_bylines`, QB-4);
2. `wire`: a wire or release byline, i.e. the key contains "newswire", "press release", "wire" or "globe";
3. `named`: a personal-name byline of 2-4 capitalized words (any script), optionally followed by " - <affiliation>";
4. `other`: everything else, split into `other_email` (an e-mail address as byline), `other_org` (one-word or
   organization bylines such as a trade publication) and `other_empty`.

Outside-reachable channels (FEED-1 "channels an outside party could plausibly reach"; our operational reading, not a
measured fact): an item is marked if its section is an opinion, trading-ideas or press-release section
(case-insensitive), or its byline is a wire byline, a contributor with an affiliation (" - "), or an e-mail address.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from jevtrader.data.news import provenance_key

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "docs/design/question_bank_v1.json"
WIRE_WORDS = ("newswire", "press release", "wire", "globe")
REACHABLE_SECTIONS = ("opinion", "trading-ideas", "press-releases", "pressreleases")
CLASSES = ("desk", "wire", "named", "other_email", "other_org", "other_empty")
_NAME = re.compile(r"^[^\W\d_][^\s\d_@]*(?: [^\W\d_][^\s\d_@]*){1,3}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I)


def desk_bylines(bank_path: Path = BANK) -> tuple[str, ...]:
    """The QB-4 desk-byline provenance keys from the question bank."""
    return tuple(json.loads(Path(bank_path).read_text())["code_constants"]["corroboration"]["desk_bylines"])


def byline_class(author: str | None, desk: tuple[str, ...]) -> str:
    key = provenance_key(author)
    if not key:
        return "other_empty"
    if key in desk:
        return "desk"
    if any(w in key for w in WIRE_WORDS):
        return "wire"
    raw = " ".join(str(author).split())
    if _EMAIL.match(raw):
        return "other_email"
    person = raw.split(" - ")[0].strip()
    if _NAME.match(person) and all(tok[:1].isupper() for tok in person.split()) and ".com" not in person.lower():
        return "named"
    return "other_org"


def outside_reachable(author: str | None, section: str | None, cls: str) -> bool:
    sec = (section or "").lower()
    return (sec in REACHABLE_SECTIONS or cls in ("wire", "other_email")
            or (cls == "named" and " - " in str(author or "")))


def classify_items(items: pd.DataFrame, desk: tuple[str, ...]) -> pd.DataFrame:
    """One row per distinct item id with `byline_class` and `outside_reachable` added."""
    it = items.drop_duplicates("id").copy()
    classes = {a: byline_class(a, desk) for a in it.author.fillna("").unique()}
    it["byline_class"] = it.author.fillna("").map(classes)
    it["outside_reachable"] = [outside_reachable(a, s, c) for a, s, c in
                               zip(it.author, it.url_section, it.byline_class)]
    return it


def provenance_mix(items: pd.DataFrame, group: pd.Series, top_sections: int = 12) -> dict:
    """Shares of byline classes, of the top URL sections and of outside-reachable items, per group value."""
    out = {}
    for g, sub in items.groupby(group, sort=True):
        n = len(sub)
        secs = sub.url_section.fillna("").str.lower().value_counts()
        out[str(g)] = {
            "n_items": int(n),
            "byline_class_share": {c: float((sub.byline_class == c).mean()) for c in CLASSES},
            "section_share": {k: float(v / n) for k, v in secs.head(top_sections).items()},
            "section_share_other": float(secs.iloc[top_sections:].sum() / n),
            "outside_reachable_share": float(sub.outside_reachable.mean()),
            "distinct_bylines": int(sub.author.fillna("").map(provenance_key).nunique()),
        }
    return out
