"""Baseline text readers in the same S1 interface (design_v1 §5.8; design_v2_delta RB-5, RB-15; bank QB-3, QB-12).

Every reader uses the same backend, cache and ledger as the typed programs, sees exactly the items of the news
analyst for the stock-week (`news.ItemSource`: the same window, ticker filter, D0, duplicate rule, cap and blinding,
so D0 always runs before blinding; SAN-2), and writes the same judgment tables. The reader id is the program name, the
`stage` of its answers and requests, and its ledger tag (`<RUN>/<reader>`).

- **H** (holistic reader): one request per stock-week with at least one item; state `{note, focal_company,
  evidence: {news_items: [{headline, summary, published}]}}`, no N0, no guards; the single question `H.outlook`.
  Feature `h_outlook` = (E[outlook] - 2) / 2, which replaces every untrusted news feature in the allocator.
  H-spot and H-guard write the same quantity under their own Layer-B catalog names, `hspot_outlook` and
  `hguard_outlook` (learn.features.CATALOG; pipeline.variants.READERS), so the readers can share one generation
  without duplicate (t, symbol, feature) rows and each reader variant finds its own feature.
- **H-spot** (RB-5): H with the spotlight note (`conventions.note_text_spotlight`) and datamarked evidence text: after
  D0 and blinding every whitespace run inside headline and summary strings becomes U+02C6
  (`conventions.datamark`); keys and code-rendered fields (`published`) are not marked.
- **H-guard** (RB-15): H over the evidence that survives the typed guards: items with P(N1.instructs_reader) < .5 and
  P(N1.promotional) < .5, each reduced to its N0-kept sentences (a removed headline becomes the literal "(headline
  removed by screening)"), items with nothing kept dropped. It reads N01's blackboard rows of the same week
  (`depends_on = ("N01",)`); in a run of its own (tag HGUARD) N01's answers come from the response cache. A guard
  abstention drops the item, as in the typed path.
- **HS** (headline sentiment): one request per stock-week with one `HS.tone_k` Choice per headline (at most 20, most
  recent first); feature `hs_tone` = mean of P(good_news) - P(bad_news) over the answered headlines.
- **LM** (B10 in the W slot; RB-7): no model call. Feature `lm_neg` = the Loughran-McDonald net-negative word share of
  the stock-week's blinded headlines and summaries (`baselines.text.lm_neg_of_items`), which Layer B's V-reader-LM
  reads. Being free, it may run inside any generation (it has no dependency and reads no memory), so adding it to an
  existing generation on resume computes only its own units.

All features are untrusted (taint W). A stock-week without items (or, for H-guard, without surviving items) gets no
request and a null feature (Layer B: uncovered).

State budget. The H, H-spot and H-guard states hold the most recent items that fit in `H_STATE_TOKENS` (the L9 token
estimate of `lint.check_request`); older items are left out and counted in the log. Clean weeks are far below it (the
busiest real stock-weeks reach about 3.8k tokens), but 40 token-dense items (CJK text within the D0 caps) would pass the
24k hard budget and abort the week (audit finding 7). A request that still fails L9 leaves that stock-week uncovered
instead of stopping the reader for every stock.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from jevtrader.programs import render
from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import Evidence, ProgramOutput, Request, WeekContext, ask
from jevtrader.programs.cluster import DROPPED_HEADLINE
from jevtrader.programs.lint import LintError
from jevtrader.programs.news import ITEMS_KEY, P_KEEP, ItemSource, NewsItem, focal, stock_names
from jevtrader.s1.backends import estimate_tokens
from jevtrader.s1.types import canonical_json

log = logging.getLogger("jevtrader.programs")
HS_MAX_HEADLINES = 20
H_STATE_TOKENS = 16_000  # H-type state budget (L9 estimate); the hard request budget is 24k with the question
READERS = {"H": "H", "H-spot": "H_spot", "H-guard": "H_guard"}  # reader id -> bank state schema


def _phi_outlook(a) -> float | None:
    """(E[level] - 2) / 2 for the 5-level H.outlook Score; None when the answer abstains."""
    return None if a is None or a.probs is None else (a.expected_score - 2.0) / 2.0


def guarded_items(items: list[NewsItem], rows: list[dict]) -> list[tuple[NewsItem, dict[str, str]]]:
    """H-guard's evidence (QB-12): items with P(instructs_reader) < .5 and P(promotional) < .5 (an abstention drops
    the item), reduced to their N0-kept sentences; items with no kept sentence dropped. Most recent first."""
    by = {it.eid: it for it in items}
    out = []
    for r in rows:
        p = r["p"]
        if r["n0_quarantined"] or not r["kept_sentences"]:
            continue
        if p["instructs"] is None or p["instructs"] >= P_KEEP or p["promotional"] is None or p["promotional"] >= P_KEEP:
            continue
        it = by[r["eid"]]
        kept = set(r["kept_sentences"])
        head = it.n0_sentences[0] if 0 in kept else DROPPED_HEADLINE
        summ = " ".join(s for i, s in enumerate(it.n0_sentences) if i > 0 and i in kept)
        out.append((it, {"headline": head, "summary": summ}))
    return sorted(out, key=lambda x: (-x[0].created_at.value, x[0].item_id))


class HolisticReader:
    """H: the holistic reader (one request per stock-week, `H.outlook` over the unscreened evidence)."""

    name = "H"
    version = "H-v2"
    schema = "H"  # bank state schema
    feature = "h_outlook"  # Layer-B catalog name of the reader's feature
    taint = "W"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None):
        self.bank = bank or load_bank()
        self.spec = self.bank.q("H.outlook")

    def news_items(self, ctx: WeekContext, sym: str, items: list[NewsItem]) -> tuple[list[dict], list[Evidence]]:
        if self.name == "H-guard":
            entry = ctx.memory.work(f"{ITEMS_KEY}{sym}") if ctx.memory is not None else None
            if entry is None:
                raise RuntimeError(f"H-guard {pd.Timestamp(ctx.t).date()} {sym}: no N01 rows on the blackboard")
            chosen = guarded_items(items, entry.value)
            ev: list[Evidence] = [Evidence(f"bb:{ITEMS_KEY}{sym}", entry.available_at)]
        else:
            chosen = [(it, {"headline": it.headline, "summary": it.summary}) for it in items]
            ev = []
        rows = []
        for it, text in chosen:
            h, s = text["headline"], text["summary"]
            if self.name == "H-spot":
                h, s = render.datamark(h), render.datamark(s)
            rows.append({"headline": h, "summary": s, "published": it.published})
            ev.append(it.evidence())
        return rows, ev

    def state(self, ctx: WeekContext, sym: str, rows: list[dict], ev: list[Evidence]) -> tuple[dict, list[Evidence]]:
        """The state with the most recent items that fit in H_STATE_TOKENS (rows are most recent first; the
        blackboard evidence row of H-guard, if any, stays). Items left out are logged."""
        fc = focal(self.bank, ctx, sym)
        n_bb = len(ev) - len(rows)
        lo, hi = 0, len(rows)
        while lo < hi:  # the largest prefix that fits (the estimate grows with the prefix)
            mid = (lo + hi + 1) // 2
            st = render.envelope(self.bank, self.schema, focal_company=fc, evidence={"news_items": rows[:mid]})
            lo, hi = (mid, hi) if estimate_tokens(canonical_json(st)) <= H_STATE_TOKENS else (lo, mid - 1)
        if lo < len(rows):
            log.warning("%s %s %s: %d of %d items left out to fit the %d-token state budget", self.name,
                        pd.Timestamp(ctx.t).date(), sym, len(rows) - lo, len(rows), H_STATE_TOKENS)
        st = render.envelope(self.bank, self.schema, focal_company=fc, evidence={"news_items": rows[:lo]})
        return st, ev[: n_bb + lo]

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        src = ItemSource(ctx, self.bank)
        t = pd.Timestamp(ctx.t)

        async def one(sym: str) -> ProgramOutput:
            out = ProgramOutput(self.name, sym, t, {self.feature: None})
            rows, ev = self.news_items(ctx, sym, src.items(sym))
            if not rows:
                return out
            state, ev = self.state(ctx, sym, rows, ev)
            if not state["evidence"]["news_items"]:
                return out
            req = Request.build(self.name, self.name, sym, t, state, [("outlook", self.spec, self.spec.to_question())],
                                ev, tag=self.name)
            try:
                asked = await ask(s1, req)
            except LintError as e:  # this stock-week stays uncovered; the other stocks still run
                log.error("%s %s %s: %s", self.name, t.date(), sym, e)
                return out
            _flag_rows(asked.evidence, src.items(sym))
            out.features = {self.feature: _phi_outlook(asked.response.answers.get("outlook"))}
            out.trace, out.requests, out.evidence = asked.trace, [asked.request], asked.evidence
            return out

        return list(await asyncio.gather(*(one(s) for s in stock_names(ctx))))


class SpotlightReader(HolisticReader):
    """H-spot (RB-5): H with the spotlight note and datamarked evidence text (QB-3)."""

    name = "H-spot"
    version = "H-spot-v3"
    schema = "H_spot"
    feature = "hspot_outlook"


class GuardedReader(HolisticReader):
    """H-guard (RB-15): H over the evidence that survives the typed guards (QB-12); reads N01's rows."""

    name = "H-guard"
    version = "H-guard-v3"
    schema = "H_guard"
    feature = "hguard_outlook"
    depends_on: tuple[str, ...] = ("N01",)


def reader(reader_id: str, bank: Bank | None = None) -> HolisticReader:
    """The holistic reader for a reader id ("H", "H-spot", "H-guard")."""
    cls = {"H": HolisticReader, "H-spot": SpotlightReader, "H-guard": GuardedReader}.get(reader_id)
    if cls is None:
        raise ValueError(f"unknown holistic reader {reader_id!r}; one of {sorted(READERS)}")
    return cls(bank)


class HeadlineSentimentReader:
    """HS: one Choice per headline (design_v1 §5.8; lopez-lira style per-headline scoring mapped to {1, -1, 0})."""

    name = "HS"
    version = "HS-v2"
    taint = "W"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None):
        self.bank = bank or load_bank()
        self.spec = self.bank.q("HS.tone_k")

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        src = ItemSource(ctx, self.bank)
        t = pd.Timestamp(ctx.t)

        async def one(sym: str) -> ProgramOutput:
            out = ProgramOutput(self.name, sym, t, {"hs_tone": None})
            items = src.items(sym)[:HS_MAX_HEADLINES]
            if not items:
                return out
            state = render.envelope(self.bank, "HS", headlines=[it.headline for it in items])
            qs = []
            for k in range(len(items)):
                name, q = self.spec.expand(index=k)
                qs.append((name, self.spec, q))
            try:
                asked = await ask(s1, Request.build(self.name, self.name, sym, t, state, qs,
                                                    [it.evidence() for it in items], tag=self.name))
            except LintError as e:  # this stock-week stays uncovered; the other stocks still run
                log.error("%s %s %s: %s", self.name, t.date(), sym, e)
                return out
            _flag_rows(asked.evidence, items)
            tones = [a.probs.get("good_news", 0.0) - a.probs.get("bad_news", 0.0)
                     for a in asked.response.answers.values() if a.probs is not None]
            out.features = {"hs_tone": sum(tones) / len(tones) if tones else None}
            out.trace, out.requests, out.evidence = asked.trace, [asked.request], asked.evidence
            return out

        return list(await asyncio.gather(*(one(s) for s in stock_names(ctx))))


class LMDictionaryReader:
    """LM: the Loughran-McDonald dictionary reader in the untrusted slot (design_v1 §13.3 B10; design_v2_delta RB-7).
    Code only: the same items as every reader (`news.ItemSource`), no request, no evidence rows; feature `lm_neg`
    (None for a stock-week without items or words: uncovered)."""

    name = "LM"
    version = "LM-v1"
    feature = "lm_neg"
    taint = "W"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None, lm=None):
        self.bank = bank or load_bank()
        self._lm = lm

    @property
    def lm(self):
        if self._lm is None:
            from jevtrader.baselines.text import LMDictionary

            self._lm = LMDictionary()
        return self._lm

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        from jevtrader.baselines.text import lm_week_features

        t = pd.Timestamp(ctx.t)
        vals = lm_week_features(ctx, self.lm, self.bank)
        return [ProgramOutput(self.name, sym, t, {self.feature: vals[sym][0]}) for sym in stock_names(ctx)]


def _flag_rows(rows: list[Any], items: list[NewsItem]) -> None:
    """D0 event flags on the news evidence rows (SAN-4; evidence.parquet column `d0_flags` once carried)."""
    by = {it.eid: it for it in items}
    for r in rows:
        if r.eid in by:
            r.d0_flags = by[r.eid].flags.to_json()
