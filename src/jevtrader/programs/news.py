"""News analyst (design_v1 §5.1 as amended by design_v2_delta; bank N0-N3): N0 -> N1 (+ sentinel) -> cluster (code)
-> N2 -> write gate (D6) -> M-epi -> N3, and the news features f_* of the untrusted channel.

Two judgment programs share the analyst, so the runner can run the per-item stages of every week at once and only
the memory-dependent stages week by week (design_v1 §4.2 "Backtest scheduling"):

- `NewsScreenProgram` ("N01"; not week-ordered). For each stock of U_t: item selection and D0 (`ItemSource`), one N0
  request per item (sentence screening, D1), then N1 on the screened text in two stages of separate requests (bank
  v1.0-rc7 `code_constants["N1.requests"]`, `ask_n1`; docs/impl/dev2_n1.md): stage 1 asks about_focal and the guards
  instructs_reader and promotional (D2) each alone; stage 2, only for an item about the focal company that is not
  quarantined, asks new_development alone and the unchanged rc5 13-question request, from which code reads only the
  item attributes and the sentinel Nouls (its gate answers are logged as `N1-joint` diagnostics). (Until rc5 code read
  every answer of that one request, and the emulator's gate answers depended on the layout: DEV #1 §8.3; rc6 asked the
  attributes in a nine-question request, which over-fired `unconfirmed`: dev2_n1 audit response.) Code triages
  every item (quarantine, keep, promotional weight) and runs the sentinel (`programs/sentinel.py`). It writes typed rows only to the M-work blackboard: `N01.items.<symbol>` (per
  item: kept sentence indices, answer probabilities, triage, D0 flags, request keys, the hash of the N1 item state)
  and `N01.sentinel.<symbol>` (h, k*, kappa, q, n^prob), and the feature `h`.
- `NewsEventProgram` ("N23"; week-ordered, depends on N01). It rebuilds the screened item states from the news store
  with the same code (and checks their hashes against N01's), clusters kept items (`programs/cluster.py`), asks N2 for
  the (at most 6) most influential clusters that have prior M-epi records, writes every cluster that passes the write
  gate to M-epi, asks N3 once per stock-week with a kept cluster (its `earlier_events` show every record of the last 8
  weeks, superseded ones included), and emits f_imp, f_cont, f_rev, f_sur, f_net, f_und, f_story, f_cov, no_news and
  news_view, plus the robust-aggregation reader features f_med and f_vote (design_v2_delta RB-6; `robust_features`).
  Card evidence goes to the blackboard as `N23.card.<symbol>` (`card_evidence(gen_dir)` exports it).

Hostile text. An item whose D0 or blinding fails is left out of the stock-week (never shown unsanitized or unblinded)
and recorded as `N01.errors.<symbol>` (evidence id, exception type) and in the card's `screening.n_item_errors`; the rest
of the week runs. Bylines are counted but stored only as hashed provenance ids (`sentinel.provenance_id`).

Point-in-time rules. Items are tagged to the stock with at most 3 tickers, `created_at` in [c_{t-1}, c_t) and
`updated_at` < c_t (NewsStore.window); c_{t-1} is the previous decision's cutoff (the last session of the previous
ISO week). N2 and N3 read only M-epi records of earlier weeks (the runner's week-ordered chain), `stock_reaction`
uses closes up to d_t only.

Taint barrier (R3). Raw text reaches N0, N1 and N2 only. The N3 state holds code-rendered labels only and is checked
word by word (`render.assert_typed_only`); an injected sentence can reach N3 only through typed item attributes, and
only if it survived N0 and the N1 guards.

Fail-closed abstentions. A degraded guard answer counts as firing (quarantine; promotional weight .25); a degraded
N0 answer drops its sentence; an item whose `event_type`, `effect` or `important` answer is degraded is not kept for
events. Quarantined items (N0 kept nothing, or P(instructs_reader) >= .5) are excluded from events, coverage, the
sentinel and memory.

Registered Layer-A variants (`WeekContext.variant`): `nodef` (R-nodef, design_v1 §13.1: no N0, no guard actions (so
no guard requests: N1 stage 1 asks about_focal only), no write gate, N3 rows with the raw blinded headline, R3 lifted), `n2_off` (no N2 request; nu = 1), `mepi_off` (X2c: no N2,
empty `earlier_events`, no M-epi writes), `gate_off` (the A7 arm with the write gate off: promotional clusters are
written too; needs `MemoryStore(enforce_gate=False)`).
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jevtrader.data.news import decision_cutoff
from jevtrader.memory.store import EventRecord, write_gate
from jevtrader.programs import render
from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import Asked, Evidence, EvidenceRow, ProgramOutput, Request, WeekContext, ask
from jevtrader.programs.cluster import DROPPED_HEADLINE, Cluster, KeptItem, cluster_items, cluster_term, top_label
from jevtrader.programs.sentinel import SentinelItem, desk_bylines, provenance_id, provenance_key, sentinel
from jevtrader.s1.types import Answer, canonical_json

log = logging.getLogger("jevtrader.programs")

MAX_ITEMS = 40  # most recent first, per stock-week (design_v1 §5.1 Input)
MAX_TICKERS = 3  # items tagged to at most 3 tickers
MAX_SUMMARY_SENTENCES = 6  # N0: headline + up to 6 summary sentences
N2_MAX_CLUSTERS = 6  # N2 requests per stock-week
N2_MAX_PRIOR = 5  # prior M-epi records per N2 request
MEMORY_WEEKS = 8  # novelty and earlier_events window (design_v1 §7: TTL 8 weeks)
EARLIER_EVENTS_MAX = 12  # N3 earlier_events rows (most recent first; keeps the state inside the L9 budget)
P_KEEP = 0.5  # N0.keep, N1.keep, N1.quarantine, N1.promotional_weight thresholds (bank code_constants.thresholds)
RHO_PROMOTIONAL = 0.25
SAME_PRIOR = 0.5  # N2: a cluster that matches a prior record with P(same_as_prior) >= .5 supersedes it
MARKET = "SPY"
VOL_SESSIONS = 60  # stock_reaction / price anchor: sd of daily excess returns over 60 sessions ending at d0 - 1
VARIANT_NODEF, VARIANT_N2_OFF, VARIANT_MEPI_OFF, VARIANT_GATE_OFF = "nodef", "n2_off", "mepi_off", "gate_off"
ITEMS_KEY, SENTINEL_KEY, CARD_KEY, ERRORS_KEY = "N01.items.", "N01.sentinel.", "N23.card.", "N01.errors."
NEWS_FEATURES = ("f_imp", "f_cont", "f_rev", "f_sur", "f_net", "f_und", "f_story", "f_cov", "no_news", "news_view",
                 "f_med", "f_vote")
ROBUST_FEATURES = ("f_med", "f_vote")  # RB-6 readers R-med / R-vote (pipeline.variants.READERS)


class NewsRebuildError(RuntimeError):
    """N23 could not rebuild exactly the screened item states N01 judged (determinism broken)."""


# ------------------------------------------------------------------------------------------ items
@dataclass(frozen=True)
class NewsEvidence(Evidence):
    """News evidence with its D0 event flags (SAN-4; a JSON object of `render.D0Flags`) and the SHA-256 of the
    archived text (headline + newline + summary), so evidence.parquet identifies the exact item version judged."""

    d0_flags: str | None = None
    raw_sha256: str | None = None

    @property
    def text_sha256(self) -> str | None:
        return self.raw_sha256 if self.raw_sha256 is not None else super().text_sha256


def item_eid(item_id: Any) -> str:
    """'news:<hash>': 16 hex characters of sha256 of the feed's item id (design_v1 §12.3)."""
    return "news:" + hashlib.sha256(str(item_id).encode()).hexdigest()[:16]


_SPLIT = re.compile(r"[.!?]+[\"\u201d\u2019')\]]*(\s+)")
_ABBREV = re.compile(r"(?:\b(?:Inc|Corp|Co|Cos|Ltd|Plc|LLC|LP|Bros|Mr|Mrs|Ms|Dr|Jr|Sr|St|Mt|vs|No|Nos|approx|est|Inst"
                     r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|U\.S|U\.K|E\.U|N\.Y)\.|\b[A-Z]\.)$")


def split_sentences(text: str) -> list[str]:
    """Regex sentence split (design_v1 §5.1 N0): a break after . ! ? (and closing quotes or brackets) followed by
    whitespace and an upper-case letter, digit, opening quote or bracket; not after common abbreviations."""
    out, start = [], 0
    for m in _SPLIT.finditer(text):
        nxt = text[m.end(): m.end() + 1]
        if not nxt or not (nxt.isupper() or nxt.isdigit() or nxt in "\"\u201c'([$"):
            continue
        piece = text[start: m.start(1)]
        if _ABBREV.search(piece):
            continue
        out.append(piece.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


@dataclass(frozen=True)
class NewsItem:
    """One news item of a stock-week after D0 and blinding (the input of N0, N1 and the text readers)."""

    item_id: str
    eid: str
    symbol: str
    created_at: pd.Timestamp
    updated_at: pd.Timestamp
    prov_key: str  # SAN-5 provenance id ("key:" + hash of the normalized byline; never the byline text)
    prov_desk: bool  # the byline is one of the bank's desk bylines (the desk-collapsed sensitivity)
    source: str  # D0-sanitized, blinded source field ("not available" if empty)
    n_tags: int
    companies_tagged: tuple[str, ...]  # code names of the tagged universe firms, focal first
    headline: str  # D0-sanitized, blinded headline (never empty)
    summary: str  # D0-sanitized, blinded summary (capped at 1,200 characters)
    n0_sentences: tuple[str, ...]  # headline first, then up to 6 summary sentences
    flags: render.D0Flags  # headline + summary D0 events
    raw_sha256: str  # sha256 of the archived headline + "\n" + summary
    published: str  # relative time of created_at
    # n0_sentences before time blinding (entities blinded, levels masked), when re-blinding them with the week's year
    # gives n0_sentences back exactly; M-epi stores this form and N2 blinds it with the reading week's year
    n0_untimed: tuple[str, ...] | None = None

    @property
    def available_at(self) -> pd.Timestamp:
        """When the archived (last-edited) text was public: max(created_at, updated_at) < c_t."""
        return max(self.created_at, self.updated_at)

    def evidence(self, kept: bool = True, quarantined: bool = False) -> NewsEvidence:
        return NewsEvidence(self.eid, self.available_at, kept, quarantined, None, self.flags.to_json(), self.raw_sha256)


def week_window(ctx: WeekContext) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[c_{t-1}, c_t): c_{t-1} is the previous decision's cutoff (`ctx.prev_cutoff_utc` from the context builder;
    otherwise the last session of the previous ISO week in the session calendar or the price index, dates before
    week t only; without either, c_t - 7 days)."""
    prev_cut = getattr(ctx, "prev_cutoff_utc", None)
    if prev_cut is not None:
        return pd.Timestamp(prev_cut), ctx.cutoff_utc
    t = pd.Timestamp(ctx.t)
    start = ctx.cutoff_utc - pd.Timedelta(days=7)
    cal = getattr(ctx, "sessions", None)
    if cal is None:
        close = getattr(ctx.prices, "close", None)
        cal = None if close is None else close.index
    if cal is not None and len(cal):
        cal = pd.DatetimeIndex(cal)
        monday = (t - pd.Timedelta(days=t.weekday())).normalize()
        prev = cal[cal < monday]
        if len(prev):
            start = decision_cutoff(prev[-1])
    return start, ctx.cutoff_utc


def stock_names(ctx: WeekContext) -> tuple[str, ...]:
    """The names this run asks about (`WeekContext.stock_names`: the S50/S30 focus, else U_t)."""
    fn = getattr(ctx, "stock_names", None)
    return tuple(fn()) if callable(fn) else tuple(ctx.universe)


def _text(x) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else str(x)


def _int(x, default: int = 1) -> int:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else int(v)


def _utc(x) -> pd.Timestamp:
    ts = pd.Timestamp(x)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


class ItemSource:
    """Point-in-time item selection for one week (design_v1 §5.1 Input), shared by the news programs and the text
    readers so that every reader of a stock-week sees the same items:
    items tagged to the stock with <= 3 tickers, created in [c_{t-1}, c_t) and last edited before c_t; D0 on
    headline (300 characters) and summary (1,200); exact-duplicate headlines dropped (the earliest copy is kept);
    the 40 most recent kept; then blinding with the item's tagged firms (SAN-2: D0 runs first, `render.d0_blind`).
    An item whose sanitation or blinding fails is left out (its text cannot enter a state unsanitized or unblinded)
    and recorded in `errors[symbol]` as (evidence id, exception type): one hostile item never stops the week."""

    def __init__(self, ctx: WeekContext, bank: Bank | None = None):
        if ctx.news is None:
            raise ValueError("the week context has no news store")
        if ctx.blinder is None:
            raise ValueError("the week context has no blinder: Web text cannot enter a state unblinded")
        self.ctx, self.bank = ctx, bank or load_bank()
        self.start, self.end = week_window(ctx)
        table = ctx.news.t
        wk = table[(table["created_at"] >= self.start) & (table["created_at"] < self.end)]
        self.tags: dict[str, tuple[str, ...]] = {
            str(k): tuple(sorted(set(g) - {"MARKET"})) for k, g in wk.groupby(wk["id"].astype(str))["symbol"]}
        self._cache: dict[str, list[NewsItem]] = {}
        self.errors: dict[str, list[tuple[str, str]]] = {}
        self._desk = desk_bylines()

    def items(self, symbol: str) -> list[NewsItem]:
        if symbol not in self._cache:
            self._cache[symbol] = self._build(symbol)
        return self._cache[symbol]

    def _build(self, symbol: str) -> list[NewsItem]:
        ctx = self.ctx
        w = ctx.news.window(symbol, self.start, self.end, cutoff=ctx.cutoff_utc)
        if "n_tags" in w:
            w = w[pd.to_numeric(w["n_tags"], errors="coerce").fillna(0) <= MAX_TICKERS]
        if not len(w):
            return []
        w = w.assign(iid=w["id"].astype(str)).sort_values(["created_at", "iid"], kind="stable")
        chosen, seen, errors = [], set(), []
        for r in w.itertuples(index=False):
            try:
                head = render.d0(r.headline, render.HEADLINE_MAX)
            except Exception as e:  # noqa: BLE001 (untrusted text: one item never stops the week)
                errors.append((item_eid(r.iid), type(e).__name__))
                continue
            if not head or str(head) in seen:  # empty headline, or an exact duplicate of an earlier item's
                continue
            seen.add(str(head))
            chosen.append((r, head))
        chosen = sorted(chosen, key=lambda x: (-_utc(x[0].created_at).value, x[0].iid))[:MAX_ITEMS]
        out = []
        for r, head in chosen:
            try:
                out.append(self._item(r, head, symbol))
            except Exception as e:  # noqa: BLE001 (a blinder or D0 failure on hostile text drops that item only)
                errors.append((item_eid(r.iid), type(e).__name__))
        if errors:
            log.warning("news items of %s %s left out after a sanitation or blinding error: %s", symbol,
                        pd.Timestamp(ctx.t).date(), errors)
            self.errors[symbol] = errors
        return out

    def edited(self, symbol: str, item_id: Any, summary: str) -> NewsItem:
        """The item `item_id` of `symbol`'s week as the readers would read it with its summary replaced by `summary`
        (D0 and blinding as for every item): the adaptive attacker's candidate edit of a host item (redteam.adaptive).
        Raises KeyError if the item is not in the week's window."""
        w = self.ctx.news.window(symbol, self.start, self.end, cutoff=self.ctx.cutoff_utc)
        w = w.assign(iid=w["id"].astype(str))
        rows = [r for r in w.itertuples(index=False) if r.iid == str(item_id)]
        if not rows:
            raise KeyError(f"item {item_id} of {symbol} is not in the week's window")
        r = rows[0]._replace(summary=summary)
        return self._item(r, render.d0(r.headline, render.HEADLINE_MAX), symbol)

    def _item(self, r, head: render.Clean, symbol: str) -> NewsItem:
        ctx = self.ctx
        year = pd.Timestamp(ctx.t).year
        tags = self.tags.get(r.iid, (symbol,))
        kw = {"symbol": symbol, "cutoff": ctx.cutoff_utc, "tagged": tags}
        summ = render.d0(getattr(r, "summary", ""), render.SUMMARY_MAX)
        # blinding = time blinding of the entity-blinded, level-masked text (the Blinder's own order), so the untimed
        # form is kept for M-epi, which re-blinds times with the reading week's year (audit finding 11)
        timer = getattr(ctx.blinder, "blind_time", None)
        if timer is not None:
            h_u = render.blind_clean(head, ctx.blinder, decision_year=None, **kw)
            s_u = render.blind_clean(summ, ctx.blinder, decision_year=None, **kw) if summ else ""
            h, s = timer(h_u, year), timer(s_u, year) if s_u else ""
        else:
            h_u = s_u = None
            h = render.blind_clean(head, ctx.blinder, decision_year=year, **kw)
            s = render.blind_clean(summ, ctx.blinder, decision_year=year, **kw) if summ else ""
        src_raw = _text(getattr(r, "source", ""))
        src = render.blind_clean(render.d0(src_raw, 60), ctx.blinder, decision_year=year, **kw) if src_raw else ""
        codes = [ctx.blinder.code_name(symbol)] + sorted(
            ctx.blinder.code_name(x) for x in tags if x != symbol and x in ctx.blinder.code_names)
        created, updated = _utc(r.created_at), _utc(r.updated_at)
        raw = f"{_text(r.headline)}\n{_text(getattr(r, 'summary', ''))}"
        sents = (h, *split_sentences(s)[:MAX_SUMMARY_SENTENCES])
        untimed = None
        if h_u is not None:
            u = (h_u, *split_sentences(s_u)[:MAX_SUMMARY_SENTENCES])
            if len(u) == len(sents) and all(timer(x, year) == y for x, y in zip(u, sents)):
                untimed = u
        author = getattr(r, "author", None)
        return NewsItem(
            item_id=r.iid, eid=item_eid(r.iid), symbol=symbol, created_at=created, updated_at=updated,
            prov_key=provenance_id(author), prov_desk=provenance_key(author) in self._desk,
            source=src or render.NOT_AVAILABLE, n_tags=_int(getattr(r, "n_tags", 1)),
            companies_tagged=tuple(codes), headline=h, summary=s, n0_sentences=sents, flags=head.flags + summ.flags,
            raw_sha256=hashlib.sha256(raw.encode()).hexdigest(),
            published=render.relative_time(created, ctx.cutoff_utc), n0_untimed=untimed)


def focal(bank: Bank, ctx: WeekContext, symbol: str) -> dict[str, str]:
    firm = ctx.firms[symbol]
    return render.focal_company(bank, ctx.blinder.code_name(symbol), firm.industry, firm.size_rank)


def _p(a: Answer | None) -> float | None:
    return None if a is None or a.probs is None else float(a.probs["yes"])


def _probs(a: Answer | None) -> dict[str, float] | None:
    return None if a is None or a.probs is None else {k: float(v) for k, v in a.probs.items()}


def _set_rows(rows: Sequence[EvidenceRow], kept: bool, quarantined: bool, flags: render.D0Flags) -> None:
    """Final item status on its evidence rows (known only after screening), and its D0 flags (SAN-4)."""
    for r in rows:
        r.kept, r.quarantined = kept, quarantined
        r.d0_flags = flags.to_json()  # evidence.parquet column `d0_flags` once the runner carries it


# ------------------------------------------------------------------------------------------ N0
def n0_request(bank: Bank, item: NewsItem, t) -> Request:
    """N0: one request per item, a Noul pair (reports_i, judges_i) per sentence, sentence by sentence. Requests are
    labelled by their stage (program N0 ... N3 in requests.parquet, ledger tag `<RUN>/N0`), so the per-program cost
    rows (OPS-1) and the L9 state budgets apply per stage; the unit (N01/N23) labels the features and shards."""
    specs = bank.program("N0")
    state = render.envelope(bank, "N0", sentences=list(item.n0_sentences))
    qs = []
    for i in range(len(item.n0_sentences)):
        for tmpl in ("reports_i", "judges_i"):
            name, q = specs[tmpl].expand(index=i)
            qs.append((name, specs[tmpl], q))
    return Request.build("N0", "N0", item.symbol, t, state, qs, [item.evidence()], tag="N0")


def n0_keep(answers: Mapping[str, Answer], n: int) -> list[bool]:
    """D1: keep sentence i iff P(reports_i) >= .5 and P(judges_i) < .5; an abstention drops the sentence."""
    keep = []
    for i in range(n):
        pr, pj = _p(answers.get(f"reports_{i}")), _p(answers.get(f"judges_{i}"))
        keep.append(pr is not None and pj is not None and pr >= P_KEEP and pj < P_KEEP)
    return keep


# ------------------------------------------------------------------------------------------ N1
def n1_item_state(item: NewsItem, keep: Sequence[bool]) -> dict[str, Any]:
    """N1's `item`: the headline if N0 kept it (else the dropped-headline literal) and the kept summary sentences."""
    return {
        "headline": item.n0_sentences[0] if keep[0] else DROPPED_HEADLINE,
        "summary": " ".join(s for s, k in zip(item.n0_sentences[1:], keep[1:]) if k),
        "source": item.source,
        "published": item.published,
        "companies_tagged": list(item.companies_tagged),
    }


def item_hash(n1_item: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(n1_item)).encode()).hexdigest()


def n1_request(bank: Bank, item: NewsItem, fc: Mapping[str, str], n1_item: Mapping[str, Any], t) -> Request:
    """Every N1 question in ONE request: the rc5 layout. **Not the production N1 since bank v1.0-rc6** (`ask_n1` is):
    DEV #1 and docs/impl/dev2_n1.md show the emulator answers the gates differently inside this joint request. Kept
    only for callers that have not migrated to `ask_n1` and for diagnostics that measure the joint layout."""
    specs = bank.program("N1")
    return n1_group_request(bank, item, fc, n1_item, t, tuple(specs))


# N1 request plan (bank code_constants["N1.requests"], v1.0-rc6/rc7; docs/impl/dev2_n1.md): the gates are asked alone,
# in two stages; rc7 reads the attributes from the unchanged rc5 13-question request. A bank without the block (<= rc5)
# asks the one joint request.
N1_PLAN_KEY = "N1.requests"
N1_LAYOUT = "question_first"  # the only layout the programs send today (emu-v1's); see N1Plan
N1_JOINT_LABEL = "N1-joint"  # answers a request asks but code does not read (rc7: the joint request's gate answers)


@dataclass(frozen=True)
class N1Group:
    """One N1 request: `ask` its questions (bank order), `use` the answers code reads. They differ only for a group
    given as {"ask": [...], "use": [...]} (rc7's attribute request asks all 13 questions and reads the nine
    attributes); its other answers are logged under N1_JOINT_LABEL (the joint-versus-single layout, FID V1, for free)
    and never read by code."""

    ask: tuple[str, ...]
    use: tuple[str, ...]

    @property
    def name(self) -> str:
        """The group's name in an item row's request keys: the question of a single-question request, "attributes"
        for the attribute request, "joint" for the rc5 single request (it reads every question)."""
        return n1_group_name(self.use)


@dataclass(frozen=True)
class N1Plan:
    """The N1 request groups of a bank: `stage1` for every item that reaches N1, `stage2` only for items that pass
    `stage2_gate` (P(about_focal) >= .5 and P(instructs_reader) < .5; R-nodef: P(about_focal) >= .5)."""

    stage1: tuple[N1Group, ...]
    stage2: tuple[N1Group, ...]
    layout: str = N1_LAYOUT

    def groups(self) -> tuple[N1Group, ...]:
        return self.stage1 + self.stage2


def _n1_group(g, order: list[str]) -> N1Group:
    if isinstance(g, Mapping):
        ask, use = list(g.get("ask") or ()), list(g.get("use") or ())
    else:
        ask = use = list(g)
    return N1Group(tuple(sorted(ask, key=order.index)), tuple(sorted(use, key=order.index)))


def n1_plan(bank: Bank) -> N1Plan:
    """The bank's N1 request plan, checked: the answers read (`use`) partition the N1 questions exactly, each group
    reads only questions it asks, asks every question at most once, and renders in bank order, and stage 1 reads the
    questions the stage-2 gate reads. Raises ValueError on a malformed plan, or on a layout the programs cannot send
    (a state-first layout needs a per-request layout in the backend; dev2_n1 S-2)."""
    order = list(bank.program("N1"))
    spec = bank.code_constants.get(N1_PLAN_KEY)
    if spec is None:
        return N1Plan((N1Group(tuple(order), tuple(order)),), ())
    try:
        s1 = tuple(_n1_group(g, order) for g in spec["stage1"])
        s2 = tuple(_n1_group(g, order) for g in spec.get("stage2", ()))
    except ValueError as e:  # a question that is not an N1 question (order.index)
        raise ValueError(f"bank {N1_PLAN_KEY}: {e}") from None
    used = [q for g in s1 + s2 for q in g.use]
    if sorted(used) != sorted(order) or len(used) != len(set(used)):
        raise ValueError(f"bank {N1_PLAN_KEY}: the groups must partition the N1 questions exactly, got {used}")
    for g in s1 + s2:
        if not g.use or not set(g.use) <= set(g.ask) or len(set(g.ask)) != len(g.ask):
            raise ValueError(f"bank {N1_PLAN_KEY}: group {list(g.ask)} must ask each question once and read only "
                             f"questions it asks (reads {list(g.use)})")
    if s2 and not {"about_focal", "instructs_reader"} <= {q for g in s1 for q in g.use}:
        raise ValueError(f"bank {N1_PLAN_KEY}: the stage-2 gate reads about_focal and instructs_reader from stage 1")
    layout = spec.get("layout", N1_LAYOUT)
    if layout != N1_LAYOUT:
        raise ValueError(f"bank {N1_PLAN_KEY}: layout {layout!r} is not supported by the programs (only {N1_LAYOUT!r})")
    return N1Plan(s1, s2, layout)


def n1_group_name(names: Sequence[str]) -> str:
    """The name of an N1 request group (by the answers it reads) in an item row's request keys: the question for a
    single-question request, "attributes" for the attribute request, "joint" for the rc5 single request."""
    return names[0] if len(names) == 1 else ("attributes" if "about_focal" not in names else "joint")


def n1_group_request(bank: Bank, item: NewsItem, fc: Mapping[str, str], n1_item: Mapping[str, Any], t,
                     names: Sequence[str]) -> Request:
    """One N1 request: the item state and the named questions, in bank order (program and stage N1, tag N1)."""
    specs = bank.program("N1")
    order = list(specs)
    state = render.envelope(bank, "N1", focal_company=dict(fc), item=dict(n1_item))
    qs = [(n, specs[n], specs[n].to_question()) for n in sorted(names, key=order.index)]
    return Request.build("N1", "N1", item.symbol, t, state, qs, [item.evidence()], tag="N1")


def stage2_gate(answers: Mapping[str, Answer], nodef: bool = False) -> bool:
    """Bank `N1.requests.stage2_gate`: P(about_focal) >= .5 and P(instructs_reader) < .5 (an abstaining guard counts
    as firing, as in `triage`); under R-nodef (no guard actions) P(about_focal) >= .5 alone."""
    a = _p(answers.get("about_focal"))
    if a is None or a < P_KEEP:
        return False
    if nodef:
        return True
    g = _p(answers.get("instructs_reader"))
    return g is not None and g < P_KEEP


@dataclass
class N1Asked:
    answers: dict[str, Answer]  # every N1 answer code reads (stage-2 questions only if the gate passed)
    asked: list[Asked]  # the requests, stage 1 first, each stage in plan order
    keys: dict[str, str]  # "N1.<group name>" -> request key


def _relabel_unread(a: Asked, group: N1Group) -> Asked:
    """A group's answers that code does not read become diagnostic rows (program N1_JOINT_LABEL, qid 'N1.<name>'), so
    answers.parquet never holds two decision rows for one (item, question); as the CF probes' relabel."""
    if set(group.ask) == set(group.use):
        return a
    trace = [r if r.qid in group.use else dataclasses.replace(r, program=N1_JOINT_LABEL, qid=f"N1.{r.qid}")
             for r in a.trace]
    return dataclasses.replace(a, trace=trace)


async def ask_n1(s1, bank: Bank, item: NewsItem, fc: Mapping[str, str], n1_item: Mapping[str, Any], t,
                 nodef: bool = False) -> N1Asked:
    """N1 for one item as the bank's plan says (v1.0-rc7): stage 1 (about_focal, instructs_reader, promotional, each
    alone) at once; then, if `stage2_gate` passes, stage 2 (new_development alone; the rc5 13-question request, read
    for the attribute questions only) at once. R-nodef takes no guard action, so its stage 1 asks only the group that
    reads about_focal. Every request shares the item state. Returns the answers code reads, the requests (unread
    answers relabelled, `_relabel_unread`) and their keys."""
    plan = n1_plan(bank)
    stage1 = tuple(g for g in plan.stage1 if "about_focal" in g.use) if nodef else plan.stage1
    out = N1Asked({}, [], {})

    async def run(groups: Sequence[N1Group]):
        res = await asyncio.gather(*(ask(s1, n1_group_request(bank, item, fc, n1_item, t, g.ask)) for g in groups))
        for g, a in zip(groups, res):
            out.answers.update({q: v for q, v in a.response.answers.items() if q in g.use})
            out.asked.append(_relabel_unread(a, g))
            out.keys[f"N1.{g.name}"] = a.request.request_key

    await run(stage1)
    if plan.stage2 and stage2_gate(out.answers, nodef):
        await run(plan.stage2)
    return out


@dataclass(frozen=True)
class Triage:
    about: bool
    kept: bool
    quarantined: bool
    promotional: bool
    rho: float
    reason: str  # why an item is not kept for events ("" if kept)


def triage(answers: Mapping[str, Answer], n0_quarantined: bool, nodef: bool = False) -> Triage:
    """N1 triage (design_v1 §5.1; D2): quarantine if P(instructs_reader) >= .5; keep for events if
    P(about_focal) >= .5 and P(new_development) >= .5; rho = .25 if P(promotional) >= .5. Guard abstentions count as
    firing. `nodef`: no guard actions (R-nodef)."""
    a, n = _p(answers.get("about_focal")), _p(answers.get("new_development"))
    g, pi = _p(answers.get("instructs_reader")), _p(answers.get("promotional"))
    about = a is not None and a >= P_KEEP
    if nodef:
        quarantined, promotional = False, False
    else:
        quarantined = n0_quarantined or g is None or g >= P_KEEP
        promotional = pi is None or pi >= P_KEEP
    attrs = all(answers.get(k) is not None and answers[k].probs is not None
                for k in ("event_type", "effect", "important"))
    if quarantined:
        reason = "quarantined"
    elif not about:
        reason = "not_about_focal"
    elif n is None or n < P_KEEP:
        reason = "no_new_development"
    elif not attrs:
        reason = "attributes_abstained"
    else:
        reason = ""
    return Triage(about, reason == "", quarantined, promotional, RHO_PROMOTIONAL if promotional else 1.0, reason)


def item_row(item: NewsItem, keep: Sequence[bool], n0_quarantined: bool, n1_item: Mapping | None,
             answers: Mapping[str, Answer], tri: Triage, keys: Mapping[str, str | None]) -> dict[str, Any]:
    """The blackboard row of one item (typed values only; the text is rebuilt from the news store)."""
    return {
        "eid": item.eid,
        "created_at": item.created_at.isoformat(),
        "available_at": item.available_at.isoformat(),
        "prov_key": item.prov_key,  # the hashed provenance id: no byline text in M-work (audit finding 10)
        "prov_desk": item.prov_desk,
        "n_sentences": len(item.n0_sentences),
        "kept_sentences": [i for i, k in enumerate(keep) if k],
        "n0_quarantined": n0_quarantined,
        "n1_sha256": None if n1_item is None else item_hash(n1_item),
        "request_keys": dict(keys),
        "d0": item.flags.as_dict(),
        "p": {k: _p(answers.get(q)) for k, q in (
            ("about", "about_focal"), ("new", "new_development"), ("instructs", "instructs_reader"),
            ("promotional", "promotional"), ("important", "important"), ("scheduled", "scheduled"),
            ("unconfirmed", "unconfirmed"), ("serious", "serious_problem"), ("unresolved", "problem_unresolved"))},
        "event_type": _probs(answers.get("event_type")),
        "effect": _probs(answers.get("effect")),
        "vs_expectations": _probs(answers.get("vs_expectations")),
        "problem_kind": _probs(answers.get("problem_kind")),
        "status": {"about": tri.about, "kept": tri.kept, "quarantined": tri.quarantined,
                   "promotional": tri.promotional, "rho": tri.rho, "reason": tri.reason},
    }


def sentinel_item(row: Mapping[str, Any]) -> SentinelItem:
    st, p = row["status"], row["p"]
    return SentinelItem(row["eid"], pd.Timestamp(row["created_at"]), row["prov_key"], st["about"], st["kept"],
                        st["quarantined"], st["promotional"], p["serious"], row["problem_kind"], p["unresolved"],
                        row.get("prov_desk"))


class NewsScreenProgram:
    """N0 + N1 + sentinel for every stock of the week ("N01"; design_v1 §5.1-5.2). Not week-ordered."""

    name = "N01"
    # v4: N1 in two stages of separate requests (bank v1.0-rc6 N1.requests); v5: the attributes from the rc5 joint
    # request (bank v1.0-rc7), its unread gate answers relabelled N1-joint; older shards are redone
    version = "N01-v5"
    taint = "W"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None):
        self.bank = bank or load_bank()

    async def _item(self, ctx: WeekContext, s1, fc, item: NewsItem, nodef: bool):
        t = pd.Timestamp(ctx.t)
        trace, reqs, evid = [], [], []
        keys: dict[str, str | None] = {"N0": None}
        if nodef:
            keep = [True] * len(item.n0_sentences)
        else:
            asked0 = await ask(s1, n0_request(self.bank, item, t))
            keep = n0_keep(asked0.response.answers, len(item.n0_sentences))
            trace += asked0.trace
            reqs.append(asked0.request)
            evid += asked0.evidence
            keys["N0"] = asked0.request.request_key
        n0_quarantined = not any(keep)
        answers: dict[str, Answer] = {}
        n1_item = None
        if not n0_quarantined:
            n1_item = n1_item_state(item, keep)
            res1 = await ask_n1(s1, self.bank, item, fc, n1_item, t, nodef)
            answers = res1.answers
            for asked1 in res1.asked:
                trace += asked1.trace
                reqs.append(asked1.request)
                evid += asked1.evidence
            keys.update(res1.keys)
        tri = triage(answers, n0_quarantined, nodef)
        _set_rows(evid, tri.kept, tri.quarantined, item.flags)
        row = item_row(item, keep, n0_quarantined, n1_item, answers, tri, keys)
        return row, trace, reqs, evid

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        src = ItemSource(ctx, self.bank)
        nodef = VARIANT_NODEF in ctx.variant

        async def one(sym: str) -> ProgramOutput:
            items = src.items(sym)
            fc = focal(self.bank, ctx, sym) if items else None
            res = await asyncio.gather(*(self._item(ctx, s1, fc, it, nodef) for it in items))
            out = ProgramOutput(self.name, sym, pd.Timestamp(ctx.t))
            rows = []
            for row, trace, reqs, evid in res:
                rows.append(row)
                out.trace += trace
                out.requests += reqs
                out.evidence += evid
            sres = sentinel([sentinel_item(r) for r in rows])
            # news_count = ln(1 + #items of the stock-week the analyst reads): the raw count of the V-codeonly
            # contrast (design_v1 §6.6, C3), computed in code before any S1 answer
            out.features = {"h": sres.h, "news_count": math.log1p(len(items))}
            out.blackboard = {f"{ITEMS_KEY}{sym}": rows, f"{SENTINEL_KEY}{sym}": sres.as_dict()}
            if sym in src.errors:  # items left out after a sanitation or blinding error (ids and error types only)
                out.blackboard[f"{ERRORS_KEY}{sym}"] = [{"eid": e, "error": t} for e, t in src.errors[sym]]
            return out

        return list(await asyncio.gather(*(one(s) for s in stock_names(ctx))))


# ------------------------------------------------------------------------------------------ N23 helpers
def kept_items(rows: Sequence[Mapping[str, Any]], items: Mapping[str, NewsItem]) -> list[KeptItem]:
    """KeptItems of the rows N01 kept for events, with the screened text rebuilt from `items`."""
    out = []
    for r in rows:
        if not r["status"]["kept"]:
            continue
        it = items[r["eid"]]
        keep = [i in set(r["kept_sentences"]) for i in range(len(it.n0_sentences))]
        n1 = n1_item_state(it, keep)
        # M-epi keeps the kept text before time blinding when it is known exactly (`n0_untimed`), so a later week's
        # N2 renders its times relative to that week, not to the week it was written (audit finding 11)
        mem_src = it.n0_untimed if it.n0_untimed is not None else it.n0_sentences
        kept_text = [s for s, k in zip(mem_src, keep) if k]
        mem_head = kept_text[0][: render.HEADLINE_MAX] if kept_text else DROPPED_HEADLINE
        out.append(KeptItem(
            eid=r["eid"], created_at=pd.Timestamp(r["created_at"]), available_at=pd.Timestamp(r["available_at"]),
            headline=n1["headline"], type_probs=r["event_type"], effect_probs=r["effect"],
            p_important=r["p"]["important"], rho=r["status"]["rho"], p_unconfirmed=r["p"]["unconfirmed"],
            p_scheduled=r["p"]["scheduled"], vs_probs=r["vs_expectations"], memory_headline=mem_head, n1_item=n1,
            request_keys=tuple(k for k in r["request_keys"].values() if k)))
    return out


def vs_label(vs_probs: Mapping[str, float] | None) -> str:
    return render.option_name("not_stated") if not vs_probs else render.option_name(top_label(vs_probs))


def event_row(bank: Bank, type_top: str, effect_probs, m: float, u: float, nu: float, width: int,
              vs_probs) -> dict[str, Any]:
    """One N3 `this_week_events` / `earlier_events` row: code-rendered labels only (bank §B N3)."""
    return {
        "event": bank.option_text("N1.event_type", type_top),
        "effect": render.effect_label(Answer("score", dict(effect_probs) if effect_probs else None),
                                      bank.q("N1.effect").levels, bank),
        "important": render.prob_bin(m, bank),
        "confirmed": render.prob_bin(None if u is None else 1.0 - u, bank),
        "new": render.prob_bin(nu, bank),
        "outlets": int(width),
        "vs_expectations": vs_label(vs_probs),
    }


def session_of(ts_utc, sessions: pd.DatetimeIndex) -> pd.Timestamp | None:
    """C2's rule: the session of `ts` if it is before 16:00 ET on a session day, otherwise the next session."""
    et = pd.Timestamp(ts_utc).tz_convert("America/New_York")
    day = pd.Timestamp(et.date())
    if et.hour < 16 and day in sessions:
        return day
    later = sessions[sessions > day]
    return later[0] if len(later) else None


def excess_move(prices, symbol: str, since_utc, d_t, market: str = MARKET,
                vol_sessions: int = VOL_SESSIONS) -> tuple[float, float, int] | None:
    """(r_ex, sigma_d, n) for N3's `stock_reaction` and the BG-3 price anchor: r_ex = the stock's cumulative return
    minus the market's from the close of d0 - 1 to the close of d_t (adjusted closes), d0 = the session of `since`
    (C2's rule), sigma_d = sd of the stock's daily excess returns over the `vol_sessions` sessions ending at d0 - 1,
    n = sessions from d0 to d_t inclusive. Only closes up to d_t are read. None if any input is missing."""
    close = getattr(prices, "close", None)
    if close is None or symbol not in close or market not in close:
        return None
    sess = close.index[close.index <= pd.Timestamp(d_t)]
    if not len(sess):
        return None
    d0 = session_of(since_utc, sess)
    if d0 is None:
        return None
    i0 = sess.get_loc(d0)
    if i0 < 1:
        return None
    s, m = close[symbol].reindex(sess), close[market].reindex(sess)
    base, end = sess[i0 - 1], sess[-1]
    if not np.isfinite([s[base], s[end], m[base], m[end]]).all():
        return None
    r_ex = float(s[end] / s[base] - m[end] / m[base])
    ex = (s.pct_change(fill_method=None) - m.pct_change(fill_method=None)).iloc[max(1, i0 - vol_sessions): i0].dropna()
    if len(ex) < 20:
        return None
    return r_ex, float(ex.std(ddof=1)), int(len(sess) - i0)


def stock_reaction(ctx: WeekContext, symbol: str, since_utc) -> str:
    mv = excess_move(ctx.prices, symbol, since_utc, pd.Timestamp(ctx.t)) if ctx.prices is not None else None
    return render.NOT_AVAILABLE if mv is None else render.reaction(*mv)


def _sign(x: float | None) -> float:
    return 0.0 if x is None or x == 0 else math.copysign(1.0, x)


def robust_terms(clusters: Sequence[Cluster], novelty: Sequence[float]) -> list[float]:
    """rho_e nu_e m_e d_e per kept cluster: the base term of f_cont and f_rev, before the unconfirmed discount, the
    width bonus and the caps of f_imp (design_v1 §5.1)."""
    return [float(c.rho * nu * c.m * c.d) for c, nu in zip(clusters, novelty)]


def robust_features(clusters: Sequence[Cluster], novelty: Sequence[float]) -> dict[str, float | None]:
    """Robust-aggregation typed readers (design_v2_delta RB-6; the isolate-then-aggregate principle of
    xiang2026certifiably, p.4-7). Over the stock-week's kept clusters, with b_e = rho_e nu_e m_e d_e (`robust_terms`):

    - f_med = the median of the b_e (the mean of the two middle values for an even count): one planted cluster moves
      it at most to a neighbouring order statistic, while it moves the sum f_imp by its whole (capped) term;
    - f_vote = the sign majority: +1 if more clusters have b_e > 0 than b_e < 0, -1 if fewer, 0 on a tie; a cluster
      with b_e = 0 (e.g. nu = 0, a repeat N2 matched with certainty) abstains. An attacker needs more planted clusters
      than the benign majority's margin to flip it (the majority-vote certificate, xiang2026certifiably p.7).

    Both are None without a kept cluster (Layer B: uncovered), like every news feature except f_cov and no_news. The
    variants R-med / R-vote read them with f_cov and no_news in place of the nine news features, under the same band
    and cap (pipeline.variants.READERS); they cost no request."""
    if not clusters:
        return {"f_med": None, "f_vote": None}
    b = robust_terms(clusters, novelty)
    n_pos, n_neg = sum(x > 0 for x in b), sum(x < 0 for x in b)
    return {"f_med": float(np.median(b)), "f_vote": float(np.sign(n_pos - n_neg))}


def news_features(clusters: Sequence[Cluster], novelty: Sequence[float], n3: Mapping[str, Answer] | None,
                  coverage: int, cont_types: Sequence[str], rev_types: Sequence[str]) -> dict[str, float | None]:
    """The news features of design_v1 §5.1 for one stock-week, plus RB-6's f_med and f_vote (`robust_features`).
    Features other than f_cov and no_news are None when the stock-week has no kept cluster (Layer B: uncovered)."""
    f: dict[str, float | None] = {k: None for k in NEWS_FEATURES}
    f["f_cov"] = math.log1p(coverage)
    f["no_news"] = 0.0 if clusters else 1.0
    if not clusters:
        return f
    f.update(robust_features(clusters, novelty))
    base = robust_terms(clusters, novelty)
    f["f_imp"] = max(-3.0, min(3.0, sum(cluster_term(c, nu) for c, nu in zip(clusters, novelty))))
    f["f_cont"] = float(sum(b * c.p_types(cont_types) for b, c in zip(base, clusters)))
    f["f_rev"] = float(sum(b * c.p_types(rev_types) for b, c in zip(base, clusters)))
    f["f_sur"] = float(sum(c.rho * c.m * c.sigma for c in clusters))
    n3 = n3 or {}
    net, reac = n3.get("net_effect"), n3.get("reaction_fit")
    if net is not None and net.probs is not None:
        f["f_net"] = (net.expected_score - 2.0) / 2.0
        if reac is not None and reac.probs is not None:
            p = reac.probs
            f["f_und"] = _sign(f["f_net"]) * (p.get("smaller_than_news", 0.0) + p.get("opposite_to_news", 0.0)
                                              - p.get("larger_than_news", 0.0))
    story = n3.get("continuing_story")
    if story is not None and story.probs is not None:
        f["f_story"] = _sign(f["f_imp"]) * float(story.probs["yes"])
    view = n3.get("analyst_view")
    if view is not None and view.probs is not None:
        f["news_view"] = (view.expected_score - 2.0) / 2.0
    return f


def influence(c: Cluster) -> float:
    """|rho_e m_e d_e (1 - 0.5 u_e)(1 + 0.25 ln w_e)|: the size of a cluster's f_imp term before novelty and the cap."""
    return abs(c.rho * c.m * c.d * (1.0 - 0.5 * c.u) * (1.0 + 0.25 * math.log(c.width)))


def reblind_time(ctx: WeekContext, text: str) -> str:
    """An M-epi headline with its times blinded relative to the reading week (records store them unblinded)."""
    timer = getattr(ctx.blinder, "blind_time", None)
    return text if timer is None else timer(text, pd.Timestamp(ctx.t).year)


def event_id(symbol: str, t, members: Sequence[str]) -> str:
    key = f"{symbol}|{pd.Timestamp(t).date()}|{','.join(sorted(members))}"
    return "ev:" + hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class Novelty:
    nu: float = 1.0
    max_same: float | None = None  # max P(same_as_prior) over non-degraded answers
    matched: str | None = None  # the prior record superseded (P(same) >= .5)
    prior_ids: tuple[str, ...] = ()
    request_key: str | None = None
    checked: bool = False  # N2 asked (False: no prior record, cap reached, or N2 off)


class NewsEventProgram:
    """Cluster -> N2 -> write gate -> N3 -> news features ("N23"; design_v1 §5.1). Week-ordered: week t reads the
    M-epi records the chain wrote in earlier weeks; depends on N01 of the same week (its blackboard rows)."""

    name = "N23"
    version = "N23-v3"  # v3: + f_med, f_vote (RB-6); old shards are refused on resume, cached answers are reused
    taint = "W"
    depends_on: tuple[str, ...] = ("N01",)
    week_ordered = True
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None):
        self.bank = bank or load_bank()
        cc = self.bank.code_constants
        self.cont_types = tuple(cc["news.continuation_types"])
        self.rev_types = tuple(cc["news.reversal_types"])

    # ------------------------------------------------------------------ N2
    async def _novelty(self, ctx: WeekContext, s1, sym: str, fc, c: Cluster, prior,
                       rep_ev: Evidence) -> tuple[Novelty, Any]:
        spec = self.bank.q("N2.same_as_prior_j")
        qs = []
        for j, r in enumerate(prior):
            kind = self.bank.option_text("N1.event_type", r.top_type) if r.top_type else render.NOT_AVAILABLE
            pe = {"when": render.relative_time(r.first_seen, ctx.cutoff_utc), "kind": kind,
                  "headline": reblind_time(ctx, r.headline_blinded)}
            name, q = spec.expand(index=j, fields={"prior_event": pe})
            qs.append((name, spec, q))
        rep = c.representative
        state = render.envelope(self.bank, "N2", focal_company=dict(fc), item=dict(rep.n1_item))
        ev = [rep_ev, *(Evidence(f"mem:{r.event_id}", r.available_at) for r in prior)]
        asked = await ask(s1, Request.build("N2", "N2", sym, ctx.t, state, qs, ev, tag="N2"))
        ps = [_p(asked.response.answers.get(f"same_as_prior_{j}")) for j in range(len(prior))]
        valid = [(p, r) for p, r in zip(ps, prior) if p is not None]
        nov = Novelty(prior_ids=tuple(r.event_id for r in prior), request_key=asked.request.request_key, checked=True)
        if valid:
            best, rec = max(valid, key=lambda x: (x[0], x[1].first_seen.value, x[1].event_id))
            nov.max_same, nov.nu = best, 1.0 - best
            nov.matched = rec.event_id if best >= SAME_PRIOR else None
        return nov, asked

    # ------------------------------------------------------------------ one stock-week
    async def _one(self, ctx: WeekContext, s1, src: ItemSource, sym: str, desc: Mapping[str, Any]) -> ProgramOutput:
        t = pd.Timestamp(ctx.t)
        variant = ctx.variant
        out = ProgramOutput(self.name, sym, t)
        entry = ctx.memory.work(f"{ITEMS_KEY}{sym}") if ctx.memory is not None else None
        if entry is None:
            raise NewsRebuildError(f"N23 {t.date()} {sym}: no N01 rows on the blackboard (run N01 in the same run)")
        rows = entry.value
        items = {it.eid: it for it in src.items(sym)}
        for r in rows:  # the screened text must be exactly what N1 judged
            it = items.get(r["eid"])
            if it is None:
                raise NewsRebuildError(f"N23 {t.date()} {sym}: item {r['eid']} of N01 is not in the week's items")
            if r["n1_sha256"] is not None:
                keep = [i in set(r["kept_sentences"]) for i in range(len(it.n0_sentences))]
                if item_hash(n1_item_state(it, keep)) != r["n1_sha256"]:
                    raise NewsRebuildError(f"N23 {t.date()} {sym}: item {r['eid']} rebuilds to another N1 state")
        coverage = sum(1 for r in rows if r["status"]["about"] and not r["status"]["quarantined"])
        clusters = cluster_items(kept_items(rows, items))
        bb_ev = Evidence(f"bb:{ITEMS_KEY}{sym}", entry.available_at)
        card: dict[str, Any] = {"code_name": ctx.blinder.code_name(sym), "screening": screening(rows),
                                "items": [item_card(r) for r in rows], "clusters": [], "synthesis": None}
        errs = ctx.memory.work(f"{ERRORS_KEY}{sym}")
        card["screening"]["n_item_errors"] = 0 if errs is None else len(errs.value)
        sent = ctx.memory.work(f"{SENTINEL_KEY}{sym}")
        if sent is not None:
            card["sentinel"] = sent.value
        if not clusters:
            out.features = news_features([], [], None, coverage, self.cont_types, self.rev_types)
            out.blackboard = {f"{CARD_KEY}{sym}": card}
            return out
        fc = focal(self.bank, ctx, sym)
        # ---------------------------------------------------------- N2 (novelty against M-epi)
        nov = [Novelty() for _ in clusters]
        use_memory = VARIANT_MEPI_OFF not in variant
        ask_n2 = use_memory and VARIANT_N2_OFF not in variant
        if ask_n2:
            priors = [ctx.memory.events(sym, window_weeks=MEMORY_WEEKS, prefer_type=c.top_type, limit=N2_MAX_PRIOR,
                                        current_only=True) for c in clusters]
            # the clusters N2 checks are the most influential ones (|f_imp term| at nu = 1), so neutral decoys never
            # push an influential repeated plant past the cap (audit finding 5); ties keep the cluster order
            todo = sorted((k for k, p in enumerate(priors) if p),
                          key=lambda k: (-influence(clusters[k]), k))[:N2_MAX_CLUSTERS]
            res = await asyncio.gather(*(
                self._novelty(ctx, s1, sym, fc, clusters[k], priors[k],
                              items[clusters[k].representative.eid].evidence()) for k in todo))
            for k, (nv, asked) in zip(todo, res):
                nov[k] = nv
                out.trace += asked.trace
                out.requests.append(asked.request)
                out.evidence += asked.evidence
        novelty = [n.nu for n in nov]
        # ---------------------------------------------------------- write gate (D6) -> M-epi
        writes = []
        if use_memory:
            for c, nv in zip(clusters, nov):
                gate = write_gate(kept=True, quarantined=False, rho=c.rho)
                if c.rho != 1.0 and VARIANT_GATE_OFF not in variant:
                    continue
                keys = c.request_keys + ([nv.request_key] if nv.request_key else [])
                writes.append(EventRecord(
                    event_id=event_id(sym, t, [m.eid for m in c.members]), symbol=sym, first_seen=c.first_seen,
                    available_at=c.available_at, item_ids=tuple(m.eid for m in c.members), width=c.width,
                    type_probs=c.type_probs, effect_probs=c.effect_probs, p_important=c.m, p_unconfirmed=c.u,
                    p_scheduled=c.p_scheduled, surprise=c.sigma, novelty=nv.nu,
                    headline_blinded=c.representative.memory_headline, supersedes=nv.matched,
                    provenance={"bank_hash": desc.get("bank_hash"), "backend": desc.get("backend"),
                                "model": desc.get("model_pinned"), "prompt_version": desc.get("prompt_version"),
                                "request_keys": keys, "gate": gate, "variant": sorted(variant)},
                    vs_probs=c.vs_probs))
        out.memory_writes = writes
        written = {w.event_id for w in writes}
        # ---------------------------------------------------------- N3 (typed-only synthesis)
        nodef = VARIANT_NODEF in variant
        this_week = []
        for c, nu in zip(clusters, novelty):
            row = event_row(self.bank, c.top_type, c.effect_probs, c.m, c.u, nu, c.width, c.vs_probs)
            if nodef:  # R-nodef: N3 with the raw (blinded, unscreened) headline
                row["headline"] = items[c.representative.eid].headline
            this_week.append(row)
        # every M-epi record of the last 8 weeks, superseded ones included (design_v1 §5.1, §7): a matched later
        # record must not hide the earlier one from the synthesis (audit finding 4); most recent first, capped
        earlier_recs = ctx.memory.events(sym, window_weeks=MEMORY_WEEKS)[:EARLIER_EVENTS_MAX] if use_memory else []
        earlier = [{"when": render.relative_time(r.first_seen, ctx.cutoff_utc),
                    **event_row(self.bank, r.top_type, r.effect_probs, r.p_important,
                                1.0 if r.p_unconfirmed is None else r.p_unconfirmed,
                                1.0 if r.novelty is None else r.novelty, r.width, r.vs_probs)}
                   for r in earlier_recs if r.top_type]
        largest = min(clusters, key=lambda c: (-c.width, -c.m, c.first_seen, c.members[0].eid))
        state = render.envelope(self.bank, "N3", typed_only=False if nodef else None, focal_company=dict(fc),
                                this_week_events=this_week, earlier_events=earlier,
                                stock_reaction=stock_reaction(ctx, sym, largest.first_seen))
        specs = self.bank.program("N3")
        ev = [bb_ev, Evidence(f"px:{sym}", ctx.cutoff_utc), Evidence(f"px:{MARKET}", ctx.cutoff_utc)]
        ev += [items[m.eid].evidence() for c in clusters for m in c.members]
        ev += [Evidence(f"mem:{r.event_id}", r.available_at) for r in earlier_recs]
        asked = await ask(s1, Request.build("N3", "N3", sym, t, state,
                                            [(s.name, s, s.to_question()) for s in specs.values()], ev, tag="N3"))
        for r in asked.evidence:
            if r.eid.startswith("news:"):
                r.d0_flags = items[r.eid].flags.to_json()
        out.trace += asked.trace
        out.requests.append(asked.request)
        out.evidence += asked.evidence
        n3 = asked.response.answers
        out.features = news_features(clusters, novelty, n3, coverage, self.cont_types, self.rev_types)
        card["clusters"] = [cluster_card(self.bank, c, nv, event_id(sym, t, [m.eid for m in c.members]), written)
                            for c, nv in zip(clusters, nov)]
        card["synthesis"] = synthesis_card(self.bank, n3)
        card["provenance"] = {"N3": asked.request.request_key, "N2": [n.request_key for n in nov if n.request_key]}
        out.blackboard = {f"{CARD_KEY}{sym}": card}
        return out

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        src = ItemSource(ctx, self.bank)
        desc = s1.describe() if hasattr(s1, "describe") else {}
        return list(await asyncio.gather(*(self._one(ctx, s1, src, s, desc) for s in stock_names(ctx))))


# ------------------------------------------------------------------------------------------ cards
def screening(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The card's screening block (decide/cards.render step 1) plus the D0 events (SAN-4: items per non-zero flag)."""
    instr = [r["p"]["instructs"] for r in rows if r["p"]["instructs"] is not None]
    d0 = {}
    for r in rows:
        for k, v in r["d0"].items():
            if v:
                d0[k] = d0.get(k, 0) + 1
    return {
        "n_items": len(rows),
        "n_about": sum(1 for r in rows if r["status"]["about"]),
        "n_quarantined": sum(1 for r in rows if r["status"]["quarantined"]),
        "n_n0_quarantined": sum(1 for r in rows if r["n0_quarantined"]),
        "n_promotional": sum(1 for r in rows if r["status"]["promotional"] and not r["status"]["quarantined"]),
        "n_kept": sum(1 for r in rows if r["status"]["kept"]),
        "max_instructs_reader": max(instr) if instr else 0.0,
        "n_sentences_removed": sum(r["n_sentences"] - len(r["kept_sentences"]) for r in rows),
        "d0_items": d0,
    }


def item_card(r: Mapping[str, Any]) -> dict[str, Any]:
    return {"eid": r["eid"], "kept": r["status"]["kept"], "quarantined": r["status"]["quarantined"],
            "rho": r["status"]["rho"], "reason": r["status"]["reason"], "n_sentences": r["n_sentences"],
            "n_kept_sentences": len(r["kept_sentences"]), "headline_removed": 0 not in r["kept_sentences"],
            "prov_key": r["prov_key"], "d0": {k: v for k, v in r["d0"].items() if v}}


def cluster_card(bank: Bank, c: Cluster, nv: Novelty, eid: str, written: set[str]) -> dict[str, Any]:
    tp = c.type_probs
    eff = c.effect_probs
    top_eff = top_label(eff)
    return {
        "event_id": eid, "event_type": c.top_type, "p_type": float(tp[c.top_type]),
        "effect_level": render.level_name(bank.q("N1.effect").levels[int(top_eff)]), "effect_E": c.expected_effect,
        "p_important": c.m, "p_unconfirmed": c.u, "surprise": c.sigma, "width": c.width, "rho": c.rho,
        "novelty": nv.nu, "max_same_p": 0.0 if nv.max_same is None else nv.max_same, "novelty_checked": nv.checked,
        "matched_records": list(nv.prior_ids), "supersedes": nv.matched, "written": eid in written,
        "items": [m.eid for m in c.members],
    }


def synthesis_card(bank: Bank, n3: Mapping[str, Answer]) -> dict[str, Any]:
    net, reac, story = n3.get("net_effect"), n3.get("reaction_fit"), n3.get("continuing_story")
    out: dict[str, Any] = {}
    if net is not None and net.probs is not None:
        out["net_effect"] = render.level_name(bank.q("N3.net_effect").levels[int(net.top)])
        out["net_effect_E"] = net.expected_score
    if reac is not None and reac.probs is not None:
        out["reaction"] = render.option_name(reac.top)
        out["p_reaction"] = float(reac.probs[reac.top])
    if story is not None and story.probs is not None:
        out["p_continuing_story"] = float(story.probs["yes"])
    return out


def card_evidence(gen_dir: Path | str) -> dict[tuple[str, str], dict[str, Any]]:
    """Card evidence per (decision date, symbol) from a generation's blackboard (`memory/work.jsonl`), for
    `decide.cards.build_cards(evidence=...)` (keyed by symbol within a week)."""
    f = Path(gen_dir) / "memory" / "work.jsonl"
    out: dict[tuple[str, str], dict[str, Any]] = {}
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)["record"]
            if rec["key"].startswith(CARD_KEY):
                out[(str(pd.Timestamp(rec["t"]).date()), rec["key"][len(CARD_KEY):])] = rec["value"]
    return out

