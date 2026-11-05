"""B10: Loughran-McDonald dictionary baseline (design_v1 section 13.3; loughran2011liability).

The feature `lm_neg` of a stock-week is the net negative-word share of its news text: (negative words - positive
words) / all words. Two producers, one word rule:

- **The W-channel reader (V-reader-LM, RB-7; the judgment-table producer).** `lm_neg_of_items` over the headlines and
  summaries of exactly the items the news analyst reads (`programs.news.ItemSource`: the window [c_{t-1}, c_t), last
  edit before c_t, <= 3 tickers, D0, the duplicate rule, the 40-item cap, blinding), i.e. the blinded text every S1
  reader sees; code names and the blinder's masks are not words (`lm_words`). The Layer-A program
  `programs.readers.LMDictionaryReader` ("LM") writes it into `features.parquet` of a generation (taint W, no model
  call), where Layer B's V-reader-LM reads it; `lm_week_features` is the same computation for one week context.
- **The classic panel (`lm_feature_panel`)**, used by `experiments/run_classic_baselines.py` for B10 under P-full: the
  raw (unblinded) text of the same window and ticker filter, without D0, duplicates or the cap.

A stock-week without items (or without a word) gets NaN / None (Layer B: uncovered; the untrusted slot treats it as
0). Point in time: the dictionary file is a 2026 release; a word counts for a decision in year Y only if its category
flag (the year it was added) is before Y, or it was removed after Y (flags are signed years).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from jevtrader.data.news import decision_cutoff

LM_CSV = Path(__file__).resolve().parents[3] / "data/raw/lm_dictionary/Loughran-McDonald_MasterDictionary.csv"
LM_FEATURE = "lm_neg"  # learn.features.CATALOG (agent lm_dictionary, channel W, kind code)
_TOKEN = re.compile(r"[A-Za-z]+")
# blinded text: code names ("Company-0A1B2C3D") and the blinder's masks ("[level]", "[date]") are not words
_NOT_WORDS = re.compile(r"Company-[0-9A-F]{8}|\[(?:level|date)\]")


def lm_words(text: str | None) -> str:
    """Blinded text with code names and masks removed, so the hex letters of a pseudonym are never counted."""
    return _NOT_WORDS.sub(" ", text or "")


class LMDictionary:
    def __init__(self, path: Path = LM_CSV):
        d = pd.read_csv(path, usecols=["Word", "Negative", "Positive"], keep_default_na=False)
        self.flags = {c: dict(zip(d.Word.str.upper(), d[c].astype(int))) for c in ("Negative", "Positive")}
        self._cache: dict[tuple[str, int], frozenset] = {}

    def words(self, category: str, year: int) -> frozenset:
        """Words in `category` for a decision in `year` (added before it, or removed after it)."""
        key = (category, year)
        if key not in self._cache:
            self._cache[key] = frozenset(w for w, f in self.flags[category].items()
                                         if (0 < f < year) or (f < 0 and -f > year))
        return self._cache[key]

    def net_negative(self, texts: list[str], year: int) -> tuple[float, int]:
        """((#negative - #positive) / #words, #words) over the texts."""
        neg, pos = self.words("Negative", year), self.words("Positive", year)
        n = nn = npos = 0
        for t in texts:
            for w in _TOKEN.findall(t or ""):
                w = w.upper()
                n += 1
                nn += w in neg
                npos += w in pos
        return ((nn - npos) / n if n else np.nan), n


def lm_feature_panel(news, decisions: list[pd.Timestamp], universe: dict[pd.Timestamp, list[str]],
                     lm: LMDictionary, max_tags: int = 3) -> pd.DataFrame:
    """Rows (t, symbol, lm_neg, n_items, n_words) for every member of U_t. `news` is a data.news.NewsStore."""
    rows = []
    prev_cut = None
    for t in sorted(pd.Timestamp(x) for x in decisions):
        c = decision_cutoff(t)
        lo = prev_cut if prev_cut is not None else c - pd.Timedelta(days=7)
        for s in universe[t]:
            w = news.window(s, lo, c, cutoff=c)
            w = w[w.n_tags <= max_tags]
            texts = (w.headline.fillna("") + " " + w.summary.fillna("")).tolist()
            val, nw = lm.net_negative(texts, t.year)
            rows.append((t, s, val, len(w), nw))
        prev_cut = c
    return pd.DataFrame(rows, columns=["t", "symbol", "lm_neg", "n_items", "n_words"])


def lm_neg_of_items(lm: LMDictionary, items: Iterable[tuple[str, str]], year: int) -> tuple[float | None, int]:
    """(lm_neg, #words) over (headline, summary) pairs of blinded items (`lm_words`); None without a word."""
    val, n = lm.net_negative([lm_words(f"{h} {s}") for h, s in items], year)
    return (None if not n else float(val)), n


def lm_week_features(ctx, lm: LMDictionary, bank=None) -> dict[str, tuple[float | None, int, int]]:
    """symbol -> (lm_neg, #items, #words) for the names a week context asks about, over exactly the items the news
    analyst reads (`programs.news.ItemSource`; module doc)."""
    from jevtrader.programs.news import ItemSource, stock_names

    src = ItemSource(ctx, bank)
    year = pd.Timestamp(ctx.t).year
    out = {}
    for sym in stock_names(ctx):
        items = src.items(sym)
        val, nw = lm_neg_of_items(lm, [(it.headline, it.summary) for it in items], year)
        out[sym] = (val, len(items), nw)
    return out
