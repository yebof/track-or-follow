"""Fundamentals analyst (design_v1 §5.4): F-A reads the 10-K business section into the M-sem profile; F-B reads the
trusted fundamentals and valuation buckets.

**F-A** (`FilingProfileProgram`, taint S, writes M-sem). Once per 10-K, usable from the session after `filed`:
1. Retrieval (code). The Item 1 text (`programs.context.TenKStore`) is split into paragraphs: extraction line breaks
   inside sentences are rejoined, trademark fragments ("®") reattached, short heading lines prefixed to the paragraph
   they head, and paragraphs longer than MAX_PARAGRAPH_CHARS cut at sentence ends. For each exposure group of the
   bank (`groups.FA`: customers, trade, regulation, macro, technology) the paragraphs are scored by the number of
   hits of that group's lexicon (`GROUP_LEXICON`, fixed a priori, version LEXICON_VERSION); the top 8 with at least
   one hit are kept (ties by position) while the excerpt stays within 2,000 tokens, then put back in document order.
2. Each kept paragraph is D0-sanitized and blinded against the study firms that are U members at the week that reads
   the filing, plus the focal firm (v4: `programs.context.FILING_MASKING`), with their executives, levels, times
   relative to the filing year, and alias claims and PIT alias discovery as of the usable-from time. (v2 masked the
   members at the filing date, so a firm joining U before the reading week stayed named, DEV #1 P0-4; v3 masked every
   study firm, so masked-versus-named revealed which competitors would join U later, DEV #2 audit.) A paragraph whose
   blinded text still holds an absolute time is refused (`REFUSED_KEY` in M-work) and retrieval is redone without it.
3. One request per group with that group's Nouls (21 in total); the shared criteria map "only hints" and "not said"
   to no. A group whose retrieval keeps no paragraph makes no request: its flags abstain (None), because a state
   without excerpts is the content-free state (VC-3) and its answer would be the model's prior, not the filing's.
   The focal firm is rendered as at the filing's first readable week (`decision_on_or_after(usable_from)`, through
   `ctx.firm_asof`), so the state, and hence the request key, of a 10-K depends on the week that first reads it
   only through the member set of step 2 (DEV, PRE and MAIN runs share its cache entries unless a firm named in an
   excerpt joined U in between).
4. The profile p_{i,f} = P(yes) per flag (None if degraded or not retrieved) is written to M-sem as a
   `ProfileRecord` that supersedes the firm's previous profile; its `excerpt_ids` name the retrieved paragraphs per
   group (a group without one is an empty group, `empty_groups`). F-A is week-ordered (it reads its own earlier
   profiles): a firm is profiled in the first week in which its latest usable 10-K has no profile yet, and never
   again.

**F-B** (`FundamentalsProgram`, taint T). Every week for every stock with a point-in-time snapshot (filed < d_t);
the state changes only when a filing arrives or a bucket moves, so the response cache absorbs the repeats and Layer B
gets a row every week (WS-B request). State `{note, focal_company, fundamentals, valuation}` from
`data.fundamentals.FundamentalsPanel.features` (V-B1 fixes: market value = raw close x the latest plausible share
count filed before d_t, split-adjusted; earnings yield = TTM net income / market value; quarters by YTD
differencing, V-M14):
- `fundamentals`: revenue growth (latest quarter y/y, TTM y/y), gross, operating and net margin, debt/assets,
  cash/assets, 1-year share-count change, and the operating and free-cash-flow margins (the YTD-differencing tests
  pass: `CASHFLOW_READY`), each '<value> (<percentile bucket in the universe>)', buckets over U_t;
- `valuation`: earnings yield and sales yield as '<bucket in the universe>; <bucket in its industry>' (FF12 peers
  within U_t); no dollar amounts, no ratio values.
F-B's S1 features (x_growth, x_prof, x_strain, x_val) are answer-level phi maps computed by Layer B.
"""

from __future__ import annotations

import asyncio
import logging
import re

import numpy as np
import pandas as pd

from jevtrader.data.blinding import absolute_time
from jevtrader.data.fundamentals import CASHFLOW_READY
from jevtrader.features.technical import _pct_bucket as universe_bucket
from jevtrader.memory.store import ProfileRecord
from jevtrader.programs import render
from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import Evidence, ProgramOutput, Request, WeekContext, ask
from jevtrader.programs.context import decision_on_or_after, next_session_open_utc
from jevtrader.s1.backends import estimate_tokens
from jevtrader.s1.types import canonical_json

log = logging.getLogger("jevtrader.programs")

# ====================================================================================== F-A: 10-K profile
LEXICON_VERSION = "fa-lex-v1"
MAX_PARAGRAPHS = 8
EXCERPT_TOKEN_CAP = 2000
MAX_PARAGRAPH_CHARS = 1500
MIN_PARAGRAPH_CHARS = 80
PARAGRAPH_MAX_SANITIZED = 4000  # D0 length cap for one paragraph (after the MAX_PARAGRAPH_CHARS cut; never binds)
REFUSED_KEY = "FA.refused."  # M-work: paragraphs refused because an absolute time survived blinding (v4)

# Exposure-group lexicons (fixed a priori from the wording of the group's Nouls; matched case-insensitively on word
# boundaries, except the upper-case abbreviations in GROUP_ABBREVIATIONS, matched case-sensitively).
GROUP_LEXICON: dict[str, tuple[str, ...]] = {
    "customers": (
        r"government(?:s|al)?", r"federal", r"state and local", r"public sector", r"department of", r"agenc(?:y|ies)",
        r"medicare", r"medicaid", r"contracts?", r"subscriptions?", r"subscribers?", r"recurring", r"renewals?",
        r"multi-year", r"long-term (?:contracts?|agreements?)", r"fees?", r"licens(?:e|es|ing)", r"consumers?",
        r"individuals?", r"households?", r"retail", r"advertis(?:ing|ers?|ement)", r"customers?", r"clients?",
    ),
    "trade": (
        r"international", r"foreign", r"outside (?:of )?the united states", r"overseas", r"countries", r"global(?:ly)?",
        r"china", r"chinese", r"hong kong", r"taiwan", r"asia(?:n)?", r"exports?", r"export controls?", r"imports?",
        r"imported", r"tariffs?", r"trade (?:restrictions?|polic(?:y|ies)|barriers?|disputes?)", r"sanctions?",
        r"entity list", r"suppliers?", r"supply chains?", r"sourc(?:e|ed|ing) (?:from|of)", r"contract manufactur\w*",
        r"third-party manufactur\w*", r"manufacturing facilit\w*", r"currenc(?:y|ies)",
    ),
    "regulation": (
        r"regulat(?:ion|ions|ory|ed|es|ors?)", r"public (?:utility|service) commissions?", r"rate cases?",
        r"rate[- ]regulated", r"approv(?:e|es|ed|al) (?:of )?(?:our )?(?:rates|prices)", r"antitrust",
        r"competition (?:law|authorit\w*)", r"european commission", r"federal trade commission", r"department of justice",
        r"investigations?", r"medicare", r"medicaid", r"reimbursement", r"payers?", r"pricing pressure",
        r"capital requirements?", r"basel", r"federal reserve", r"fdic", r"comptroller", r"finra", r"insurance regulators?",
        r"environmental", r"emissions?", r"greenhouse", r"climate", r"clean air", r"privacy", r"data protection",
        r"personal (?:data|information)", r"content moderation", r"licens(?:e|ed|ing) requirements?",
    ),
    "macro": (
        r"interest rates?", r"deposits?", r"loans?", r"lending", r"mortgages?", r"financing", r"credit", r"energy",
        r"fuel", r"electricity", r"natural gas", r"crude", r"oil", r"raw materials?", r"commodit(?:y|ies)", r"metals?",
        r"copper", r"steel", r"aluminum", r"crops?", r"grain", r"economic (?:conditions|downturns?|cycles?|activity)",
        r"recessions?", r"cyclical", r"seasonal(?:ity)?", r"consumer spending", r"discretionary", r"defen[cs]e",
        r"military", r"armed forces", r"weapons?", r"department of defense",
    ),
    "technology": (
        r"artificial intelligence", r"machine learning", r"generative", r"large language models?", r"data cent(?:er|re)s?",
        r"accelerated computing", r"semiconductors?", r"chips?", r"wafers?", r"foundr(?:y|ies)", r"fabs?",
        r"integrated circuits?", r"lithography", r"gpus?", r"processors?", r"memory (?:chips?|products?)",
    ),
}
GROUP_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "customers": ("DoD",), "trade": ("PRC",), "regulation": ("FDA", "FTC", "DOJ", "EPA", "FERC", "GDPR", "CCPA", "SEC",
                                                             "OCC", "CFPB"),
    "macro": ("DoD",), "technology": ("AI", "GenAI", "EUV"),
}
_PATTERNS = {
    g: (re.compile(r"(?i)(?<![A-Za-z])(?:" + "|".join(terms) + r")(?![A-Za-z])"),
        re.compile(r"(?<![A-Za-z])(?:" + "|".join(GROUP_ABBREVIATIONS.get(g, ())) + r")(?![A-Za-z])")
        if GROUP_ABBREVIATIONS.get(g) else None)
    for g, terms in GROUP_LEXICON.items()
}
_TM = re.compile(r"^[®™℠©]+$|^\(?(?:SM|TM|R)\)?$")
_SENT_END = re.compile(r"[.!?:;]['\")\]]?$")
_FOOTER = re.compile(r"(?i)\|\s*\d{4}\s*form\s*10-k\s*\|?\s*\d*\s*$|^\s*\d{1,3}\s*$|^table of contents$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")


def _is_heading(line: str) -> bool:
    return len(line) <= 80 and not _SENT_END.search(line)


def _join(head: str, body: str) -> str:
    """Join a line to what precedes it: no space before punctuation, a space before a continuation."""
    if not head:
        return body
    if body[0] in ",;:).%":
        return head + body
    return head + " " + body


def paragraphs(text: str) -> list[str]:
    """Paragraphs of an extracted Item 1 text (see the module doc for the rules)."""
    blocks: list[str] = []
    cur = ""
    heads: list[str] = []  # heading lines waiting for the paragraph they head

    def flush() -> None:
        nonlocal cur
        if cur.strip():
            head = " ".join(heads)
            if head and cur[0] not in ",;:).%" and not cur[0].islower():
                head += "."
            blocks.append(_join(head, cur).strip())
            heads.clear()
        cur = ""

    for raw_line in (text or "").split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if cur and _SENT_END.search(cur):
                flush()
            continue
        if _FOOTER.search(line):
            continue
        if _TM.match(line):  # a trademark fragment of the previous line
            if cur:
                cur += line
            elif heads:
                heads[-1] += line
            continue
        if line[0].islower() or line[0] in ",;:).%":  # a line broken inside a sentence
            if cur and not _SENT_END.search(cur):
                cur = _join(cur, line)
                continue
            if not cur and heads:
                cur = _join(heads.pop(), line)
                continue
        if cur and not _SENT_END.search(cur) and _is_heading(cur):
            heads.append(cur)  # the previous line was a heading
            cur = line
            continue
        if cur and _SENT_END.search(cur):
            flush()
        if not cur and _is_heading(line):
            heads.append(line)
            continue
        cur = _join(cur, line)
    flush()
    out: list[str] = []
    for b in blocks:
        b = re.sub(r"\s+([®™℠])", r"\1", b)
        if len(b) <= MAX_PARAGRAPH_CHARS:
            out.append(b)
            continue
        piece = ""
        for s in _SENT_SPLIT.split(b):  # cut long paragraphs at sentence ends
            if piece and len(piece) + 1 + len(s) > MAX_PARAGRAPH_CHARS:
                out.append(piece)
                piece = ""
            piece = (piece + " " + s).strip()
            while len(piece) > MAX_PARAGRAPH_CHARS:
                out.append(piece[:MAX_PARAGRAPH_CHARS])
                piece = piece[MAX_PARAGRAPH_CHARS:]
        if piece:
            out.append(piece)
    return [p for p in out if len(p) >= MIN_PARAGRAPH_CHARS]


def group_hits(paragraph: str, group: str) -> int:
    rx, abbr = _PATTERNS[group]
    return len(rx.findall(paragraph)) + (len(abbr.findall(paragraph)) if abbr is not None else 0)


def retrieve(paras: list[str], group: str, max_paragraphs: int = MAX_PARAGRAPHS,
             token_cap: int = EXCERPT_TOKEN_CAP) -> list[int]:
    """Indices of the kept paragraphs for `group`, in document order: the top `max_paragraphs` by lexicon hits
    (at least one; ties by position) that fit together within `token_cap` tokens (a paragraph that does not fit is
    skipped and the next one is tried)."""
    scored = sorted(((group_hits(p, group), i) for i, p in enumerate(paras)), key=lambda x: (-x[0], x[1]))
    kept, used = [], 0
    for hits, i in scored:
        if hits <= 0 or len(kept) >= max_paragraphs:
            break
        n = estimate_tokens(paras[i])
        if used + n > token_cap:
            continue
        kept.append(i)
        used += n
    return sorted(kept)


def empty_groups(rec: ProfileRecord, groups) -> list[str]:
    """The exposure groups of a profile whose retrieval kept no paragraph (their flags abstain)."""
    have = {x.split(":", 1)[0] for x in rec.excerpt_ids}
    return [g for g in groups if g not in have]


class FilingProfileProgram:
    name = "FA"
    # v2: no request for a group without excerpts (flags abstain); focal firm at the filing's first readable week;
    # v3: excerpts blinded against the full study set; v4 (DEV #2 audit): against the U members at the reading week
    # (FILING_MASKING), and a paragraph that cannot be blinded is refused and retrieval redone without it
    version = f"FA-v4+{LEXICON_VERSION}"
    taint = "S"
    depends_on: tuple[str, ...] = ()
    week_ordered = True  # reads its own earlier profiles: one profile per 10-K
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None):
        self.bank = bank or load_bank()
        self.groups: dict[str, list[str]] = self.bank.groups["FA"]
        self.specs = self.bank.program("FA")

    def excerpts(self, ctx: WeekContext, symbol: str, filing,
                 refused: dict[int, str] | None = None) -> dict[str, list[tuple[int, str]]]:
        """group -> [(paragraph index, sanitized and blinded text)] for one 10-K: a function of the filing and of the
        U members at the reading week (FILING_MASKING), so its request keys change only when a firm named in the
        excerpt joins U between two reading weeks.

        A paragraph whose blinded text still holds an absolute time (`blinding.absolute_time`, the envelope's test) is
        refused, logged, recorded in `refused` ({paragraph index: the offending string}) and retrieval is redone
        without it: one unblindable paragraph must never abort the run (DEV #2 audit: DE's FY2023 10-K)."""
        usable = next_session_open_utc(ctx.sessions if ctx.sessions is not None else ctx.prices.close.index,
                                       filing.filed)
        paras = paragraphs(ctx.filings.text(filing))
        year = int(pd.Timestamp(filing.filed).year)
        cache: dict[int, str] = {}
        bad: dict[int, str] = {} if refused is None else refused
        # the members at the reading week plus the focal firm (a PitBlinder's filing view); a plain Blinder (synthetic
        # worlds) masks all its firms
        view = getattr(ctx.blinder, "filing_view", None)
        blinder = view() if callable(view) else ctx.blinder

        def blinded(i: int) -> str:
            if i not in cache:  # D0 (hidden markup, invisibles, confusables) before blinding, as for Web text
                clean = render.d0(paras[i], PARAGRAPH_MAX_SANITIZED)
                cache[i] = render.blind_clean(clean, blinder, symbol=symbol, decision_year=year, cutoff=usable)
                hit = absolute_time(cache[i])
                if hit is not None:
                    bad[i] = hit
                    log.warning("FA %s %s: paragraph %d refused, absolute time %r survives blinding", symbol,
                                filing.accession, i, hit)
            return cache[i]

        while True:  # `bad` only grows, so this ends
            live = [("" if i in bad else p) for i, p in enumerate(paras)]  # "" has no lexicon hit: never retrieved
            out = {g: [(i, blinded(i)) for i in retrieve(live, g)] for g in self.groups}
            if not any(i in bad for rows in out.values() for i, _ in rows):
                return out

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        if ctx.filings is None:
            raise ValueError("FA needs ctx.filings (a programs.context.TenKStore)")
        sessions = ctx.sessions if ctx.sessions is not None else ctx.prices.close.index
        t = pd.Timestamp(ctx.t)

        async def one(sym: str) -> ProgramOutput | None:
            f = ctx.filings.latest(sym, t)
            if f is None:
                return None
            cur = ctx.memory.profile(sym) if ctx.memory is not None else None
            if cur is not None and cur.accession == f.accession:
                return None  # this 10-K is already profiled
            usable = next_session_open_utc(sessions, f.filed)
            if usable is None or usable >= ctx.cutoff_utc:
                return None
            firm = self.focal_firm(ctx, sym, usable)
            focal = render.focal_company(self.bank, ctx.blinder.code_name(sym), firm.industry, firm.size_rank)
            refused: dict[int, str] = {}
            ex = self.excerpts(ctx, sym, f, refused)
            reqs = []
            for g, flags in self.groups.items():
                if not ex[g]:
                    continue  # no retrieved paragraph: no request, the group's flags abstain (content-free state)
                state = render.envelope(self.bank, "FA", focal_company=focal, business_excerpts=[x for _, x in ex[g]])
                items = [(q, self.specs[q], self.specs[q].to_question()) for q in flags]
                ev = [Evidence(f"fund:10k:{f.accession}:p{i}", usable, text=x) for i, x in ex[g]]
                reqs.append(Request.build("FA", f"FA:{g}", sym, t, state, items, ev))
            asked = await asyncio.gather(*(ask(s1, r) for r in reqs))
            trace = [row for a in asked for row in a.trace]
            flags = {q: None for qs in self.groups.values() for q in qs}
            flags.update({row.qid: (None if row.probs is None else float(row.probs["yes"])) for row in trace})
            if len(reqs) < len(self.groups):
                log.info("FA %s %s: %d of %d groups without excerpts (flags abstain)", sym, f.accession,
                         len(self.groups) - len(reqs), len(self.groups))
            supersedes = cur.record_id if cur is not None and cur.filed < pd.Timestamp(f.filed).tz_localize("UTC") \
                else None
            rec = ProfileRecord(
                symbol=sym, accession=f.accession, filed=pd.Timestamp(f.filed), usable_from=usable, flags=flags,
                excerpt_ids=tuple(f"{g}:p{i}" for g in self.groups for i, _ in ex[g]), bank_hash=self.bank.sha256,
                supersedes=supersedes,
            )
            # refused paragraphs (index and the absolute time that survived blinding, no text) go to M-work
            bb = {f"{REFUSED_KEY}{sym}": [{"accession": f.accession, "paragraph": i, "hit": h}
                                          for i, h in sorted(refused.items())]} if refused else {}
            return ProgramOutput("FA", sym, t, {}, trace, [a.request for a in asked],
                                 [e for a in asked for e in a.evidence], blackboard=bb, memory_writes=[rec])

        outs = await asyncio.gather(*(one(s) for s in ctx.stock_names()))
        return [o for o in outs if o is not None]

    @staticmethod
    def focal_firm(ctx: WeekContext, symbol: str, usable: pd.Timestamp):
        """The focal firm's industry and size as at the first week the 10-K could be read (`ctx.firm_asof`), so the
        state does not depend on the week that first reads the filing; `ctx.firms` without that hook."""
        if ctx.firm_asof is None:
            return ctx.firms[symbol]
        sessions = ctx.sessions if ctx.sessions is not None else ctx.prices.close.index
        ref = decision_on_or_after(sessions, usable)
        if ref is None or ref > pd.Timestamp(ctx.t):
            raise ValueError(f"FA {symbol}: 10-K usable {usable} has no readable week on or before {ctx.t}")
        return ctx.firm_asof(symbol, ref)


# ====================================================================================== F-B: filing read
FUNDAMENTAL_FIELDS: tuple[tuple[str, str, bool], ...] = (
    # (state key, valuation() field, signed)
    ("revenue growth, latest quarter vs a year ago", "rev_growth_q_yoy", True),
    ("revenue growth, trailing 12 months", "rev_growth_ttm", True),
    ("gross margin", "gross_margin", False),
    ("operating margin", "operating_margin", False),
    ("net margin", "net_margin", False),
    ("debt to assets", "debt_to_assets", False),
    ("cash to assets", "cash_to_assets", False),
    ("change in share count over 1 year", "share_change_1y", True),
) + ((("operating cash flow margin", "operating_cf_margin", False), ("free cash flow margin", "fcf_margin", False))
     if CASHFLOW_READY else ())
VALUATION_FIELDS: tuple[tuple[str, str], ...] = (("earnings yield", "earnings_yield"), ("sales yield", "sales_yield"))
MIN_INDUSTRY_PEERS = 4  # an industry bucket needs this many U_t members with a value (the firm included)


def industry_bucket(p: float | None) -> str:
    if p is None or not np.isfinite(p):
        return "unknown within its industry"
    if p >= 0.9:
        return "top 10% of its industry"
    if p >= 0.7:
        return "upper 30% of its industry"
    if p > 0.3:
        return "middle of its industry"
    if p > 0.1:
        return "lower 30% of its industry"
    return "bottom 10% of its industry"


def _fmt(x: float, signed: bool) -> str:
    if x is None or not np.isfinite(x):
        return render.NOT_AVAILABLE
    return f"{x:+.1%}" if signed else f"{x:.1%}"


class FundamentalsProgram:
    name = "FB"
    version = "FB-v1"
    taint = "T"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None):
        self.bank = bank or load_bank()
        self.specs = self.bank.program("FB")

    def states(self, ctx: WeekContext, names: list[str] | None = None) -> dict[str, tuple[dict, list[Evidence]]]:
        """symbol -> (F-B state, evidence); names without a point-in-time snapshot get none (abstain)."""
        t = pd.Timestamp(ctx.t)
        uni = list(ctx.universe)
        feats = ctx.facts.features(t, uni)
        if feats.empty:
            return {}
        pct = feats.apply(pd.to_numeric, errors="coerce").rank(pct=True)  # percentile within U_t (finite values)
        industry = pd.Series({s: ctx.firms[s].industry for s in feats.index if s in ctx.firms})
        sessions = ctx.sessions if ctx.sessions is not None else ctx.prices.close.index
        out = {}
        for sym in (names if names is not None else ctx.stock_names()):
            snap = ctx.facts.snapshot_asof(sym, t)
            if snap is None or sym not in feats.index:
                continue
            row = feats.loc[sym]
            fundamentals = {}
            for key, col, signed in FUNDAMENTAL_FIELDS:
                v = row.get(col, np.nan)
                v = float(v) if v is not None and np.isfinite(v) else np.nan
                fundamentals[key] = render.NOT_AVAILABLE if not np.isfinite(v) else \
                    f"{_fmt(v, signed)} ({universe_bucket(pct.at[sym, col])})"
            peers = industry[industry == industry.get(sym)].index
            valuation = {}
            for key, col in VALUATION_FIELDS:
                v = row.get(col, np.nan)
                if v is None or not np.isfinite(v):
                    valuation[key] = render.NOT_AVAILABLE
                    continue
                pv = pd.to_numeric(feats.loc[peers, col], errors="coerce").dropna()
                ind = industry_bucket(float(pv.rank(pct=True).get(sym))) if len(pv) >= MIN_INDUSTRY_PEERS else \
                    "too few industry peers to compare"
                valuation[key] = f"{universe_bucket(pct.at[sym, col])}; {ind}"
            firm = ctx.firms[sym]
            state = render.envelope(
                self.bank, "FB",
                focal_company=render.focal_company(self.bank, ctx.blinder.code_name(sym), firm.industry, firm.size_rank),
                fundamentals=fundamentals, valuation=valuation,
            )
            filed = snap.get("last_filing_date")
            ev = [Evidence(f"px:{sym}", ctx.cutoff_utc, text=canonical_json(valuation))]
            if filed is not None:  # always with its true availability: a late filing fails the runner's PIT check
                avail = next_session_open_utc(sessions, filed)
                if avail is None:  # no session after the filing in the calendar: at least the next day's open
                    avail = (pd.Timestamp(filed).normalize() + pd.Timedelta(days=1, hours=9, minutes=30)
                             ).tz_localize("America/New_York").tz_convert("UTC")
                ev.insert(0, Evidence(f"fund:{sym}:{filed}", avail, text=canonical_json(fundamentals)))
            out[sym] = (state, ev)
        return out

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        if "facts" in ctx.unavailable:  # stale filings would render an outage as unchanged fundamentals: abstain
            log.warning("FB %s: the facts store does not cover the week; FB abstains", pd.Timestamp(ctx.t).date())
            return []
        items = [(q.name, q, q.to_question()) for q in self.specs.values()]

        async def one(sym: str, state: dict, ev: list[Evidence]) -> ProgramOutput:
            asked = await ask(s1, Request.build("FB", "FB", sym, ctx.t, state, items, ev))
            return ProgramOutput("FB", sym, pd.Timestamp(ctx.t), {}, asked.trace, [asked.request], asked.evidence)

        return list(await asyncio.gather(*(one(s, st, ev) for s, (st, ev) in self.states(ctx).items())))
