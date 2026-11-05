"""Framework types for typed judgment programs (design_v1 §5, §12.3; formalism of jev_native §2).

A program builds point-in-time states in code, sends one request per state with all its questions at once
(`ask`), and post-processes the typed answers in code into features, blackboard values and memory writes.
Everything a program emits is a `ProgramOutput`; the runner turns those into the judgment tables
(answers / requests / evidence / features parquet), the only interface between Layer A and Layer B.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import pandas as pd

from jevtrader.s1.types import Answer, Choice, JSONContent, Noul, Question, S1Response, Score, state_sha256

if TYPE_CHECKING:
    from jevtrader.data.blinding import Blinder
    from jevtrader.memory.store import MemoryView

log = logging.getLogger("jevtrader.programs")
_PLACEHOLDER = "<filled by code"
_BRACKET = re.compile(r"\[(i|j|k)\]")
_SUFFIX = re.compile(r"_(i|j|k)$")


# ------------------------------------------------------------------------------------ bank entries
@dataclass(frozen=True)
class QSpec:
    """One question template of the bank (source of truth: docs/design/question_bank_v1.json).

    Templates whose name ends in `_i`/`_j`/`_k` expand per sentence, prior record, industry or headline;
    placeholders ("<filled by code ...>" instruction fields, "<...>" option labels) are filled by `expand`.
    """

    program: str
    name: str
    kind: Literal["noul", "choice", "score"]
    instructions: JSONContent
    true: JSONContent | None = None
    false: JSONContent | None = None
    options: dict[str, JSONContent | None] | None = None
    levels: tuple[JSONContent, ...] | None = None
    judgment: str = "descriptive"
    taint: str = "raw_text"
    decision: str = ""
    channel: str = ""
    role: str = ""

    @property
    def qid(self) -> str:
        return f"{self.program}.{self.name}"

    @property
    def suffix(self) -> str | None:
        m = _SUFFIX.search(self.name)
        return m.group(1) if m else None

    def placeholders(self) -> dict[str, list[str]]:
        """Unfilled instruction fields and option labels."""
        fields = [k for k, v in self.instructions.items() if isinstance(v, str) and v.startswith(_PLACEHOLDER)] \
            if isinstance(self.instructions, dict) else []
        opts = [k for k in (self.options or {}) if k.startswith("<") and k.endswith(">")]
        return {"fields": fields, "options": opts}

    def expand(
        self,
        index: int | str | None = None,
        fields: Mapping[str, JSONContent] | None = None,
        option_labels: Mapping[str, str] | None = None,
    ) -> tuple[str, Question]:
        """Instantiate the template as an s1 question. Returns (question name, question).

        `index` names the instance (`reports_i` -> `reports_3`; a non-template gets `_<index>` appended, e.g.
        one V question per claim) and fills `[i]`/`[j]`/`[k]` in the instructions (`sentences[3]`).
        `fields` fills "<filled by code ...>" instruction fields; `option_labels` renames "<...>" options.
        Every placeholder must be filled: an unfilled one would reach the model as literal text.
        """
        suffix = self.suffix
        if suffix is not None and index is None:
            raise ValueError(f"{self.qid} is a template; give an index")
        name = self.name if index is None else (_SUFFIX.sub(f"_{index}", self.name) if suffix else f"{self.name}_{index}")
        ins = copy.deepcopy(self.instructions)
        if index is not None and suffix is not None:
            if isinstance(ins, str):
                ins = _BRACKET.sub(f"[{index}]", ins)
            elif isinstance(ins, dict) and isinstance(ins.get("question"), str):
                ins["question"] = _BRACKET.sub(f"[{index}]", ins["question"])
        ph = self.placeholders()
        fields = dict(fields or {})
        unknown = set(fields) - set(ph["fields"])
        if unknown:
            raise ValueError(f"{self.qid}: no placeholder field(s) {sorted(unknown)}")
        missing = set(ph["fields"]) - set(fields)
        if missing:
            raise ValueError(f"{self.qid}: unfilled placeholder field(s) {sorted(missing)}")
        for k, v in fields.items():
            ins[k] = copy.deepcopy(v)
        if self.kind == "noul":
            q: Question = Noul(ins, copy.deepcopy(self.true), copy.deepcopy(self.false))
        elif self.kind == "choice":
            ren = dict(option_labels or {})
            if set(ren) != set(ph["options"]):
                raise ValueError(f"{self.qid}: option placeholders {ph['options']} need labels, got {sorted(ren)}")
            opts = {ren.get(k, k): copy.deepcopy(v) for k, v in (self.options or {}).items()}
            if len(opts) != len(self.options or {}):
                raise ValueError(f"{self.qid}: option labels collide")
            q = Choice(ins, opts)
        else:
            q = Score(ins, tuple(copy.deepcopy(self.levels or ())))
        if _PLACEHOLDER in json.dumps([q.instructions, getattr(q, "options", None)], ensure_ascii=False):
            raise ValueError(f"{self.qid}: placeholder left after filling")
        return name, q

    def to_question(self, **kw) -> Question:
        return self.expand(**kw)[1]


# ------------------------------------------------------------------------------------ week context
@dataclass(frozen=True)
class FirmInfo:
    """Per-firm facts the state envelope needs (§5.0), computed point-in-time by the context builder."""

    industry: str  # FF12 code (bank ff12_industries key)
    size_rank: int  # 1 = largest raw market value within U_t
    sector_etf: str | None = None  # trailing-correlation sector ETF (m6), for relative strength


@dataclass(frozen=True)
class WeekContext:
    """Everything point-in-time at decision cutoff c_t (design_v1 §4.2 step 1, §12.3).

    Programs read only through this object. Stores are the existing data classes; each program is
    responsible for querying them with `t`/`cutoff_utc` (property tests perturb data after the cutoff).
    """

    t: pd.Timestamp
    cutoff_utc: pd.Timestamp
    universe: tuple[str, ...]
    prices: Any = None  # features.technical.PanelPrices (adjusted)
    raw_prices: Any = None  # PanelPrices (raw, for levels; B1)
    news: Any = None  # data.news.NewsStore
    facts: Any = None  # data.fundamentals fact store
    macro: Any = None  # features.macro.MacroStore
    earnings: Any = None  # data.earnings.EarningsCalendar
    blinder: Blinder | None = None
    memory: MemoryView | None = None
    variant: frozenset[str] = frozenset()
    firms: Mapping[str, FirmInfo] = field(default_factory=dict)
    # Day-2 extensions (all optional, so existing callers keep working):
    # the exchange calendar (public in advance; programs read prices only up to t, probes also read later dates);
    sessions: Any = None  # pd.DatetimeIndex
    # c_{t-1}, the previous weekly decision's cutoff: news windows are [c_{t-1}, c_t) (§5.1, §5.5)
    prev_cutoff_utc: pd.Timestamp | None = None
    # 10-K Item 1 sections (programs.context.TenKStore): usable from the session after `filed` (§2.4, §5.4)
    filings: Any = None
    # the names whose stock-level requests this run makes (S50/S30 subsets); None = all of `universe`. Cross-
    # sectional buckets are always over the full U_t, so a subset run asks exactly a full run's requests
    focus: tuple[str, ...] | None = None
    # study_members(c): the study firms that had been U members by the date of cutoff c (capped at c_t); a text's
    # firms are masked only if they are members by its cutoff, and P1's broad-item pool keeps an item only if it is
    # tagged MARKET or to such a member (programs.context.MembersView). None = no restriction (synthetic worlds)
    study_members: Any = None
    # firm_asof(symbol, day <= t): FirmInfo as a run deciding on the earlier `day` renders it (F-A's focal firm at
    # the first week its 10-K was usable). None = use `firms` (synthetic worlds)
    firm_asof: Any = None
    # sources that do not cover the week ("news", "macro", "facts"; a non-strict context builder only): the programs
    # that read them abstain instead of rendering an outage as a quiet or unchanged week
    unavailable: frozenset[str] = frozenset()

    def stock_names(self) -> tuple[str, ...]:
        """The names a stock-level program asks about this week: `focus` if the run selects a subset, else U_t."""
        return tuple(self.universe) if self.focus is None else tuple(s for s in self.focus if s in set(self.universe))


# ------------------------------------------------------------------------------------ requests & rows
@dataclass(frozen=True)
class Evidence:
    """One input a request's state was built from. `eid`: 'news:<hash>' | 'fund:<field>' | 'mem:<id>' |
    'bb:<key>' | 'px:<symbol>'."""

    eid: str
    available_at_utc: pd.Timestamp
    kept: bool = True
    quarantined: bool = False
    text: str | None = None
    d0_flags: str | None = None  # D0 event flags of a Web text (SAN-4; JSON of render.D0Flags), else None

    @property
    def text_sha256(self) -> str | None:
        return None if self.text is None else hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class Request:
    program: str
    stage: str
    entity: str  # symbol or 'MARKET'
    t: pd.Timestamp
    state: JSONContent
    questions: dict[str, Question]  # insertion order = render order (bank order)
    qids: dict[str, tuple[str, str, str]]  # question name -> (bank program, qid, template name)
    evidence: tuple[Evidence, ...] = ()
    tag: str = ""

    @classmethod
    def build(
        cls,
        program: str,
        stage: str,
        entity: str,
        t: pd.Timestamp,
        state: JSONContent,
        items: Iterable[tuple[str, QSpec, Question]],
        evidence: Iterable[Evidence] = (),
        tag: str | None = None,
        relabel: str | None = None,
    ) -> Request:
        """`relabel`: write the answers under this program label with qid '<bank program>.<name>' instead of the
        bank key. Diagnostic programs (CF content-free states) use it, so their rows can never be read as a
        decision program's answers (Layer B keys trusted features on (program, qid))."""
        qs, qids = {}, {}
        for name, spec, q in items:
            if name in qs:
                raise ValueError(f"duplicate question name {name!r}")
            qs[name] = q
            qids[name] = (spec.program, name, spec.name) if relabel is None else (relabel, f"{spec.program}.{name}",
                                                                                   spec.name)
        return cls(program, stage, entity, pd.Timestamp(t), state, qs, qids, tuple(evidence), tag or program)


@dataclass
class TraceRow:
    """One typed answer: a row of answers.parquet. (`program`, `qid`) is a bank key: the bank program of the
    question ("T", "N1", "M") and its expanded name ("trend", "reports_3", "industry_tailwind_BusEq");
    `template` is the bank template name ("reports_i")."""

    t: pd.Timestamp
    symbol: str
    program: str
    stage: str
    qid: str
    template: str
    kind: str
    probs: dict[str, float] | None
    request_key: str


@dataclass
class RequestRow:
    """One request: a row of requests.parquet (design_v1 §12.3)."""

    request_key: str
    t: pd.Timestamp
    symbol: str
    program: str
    stage: str
    backend: str
    model_pinned: str
    response_model: str
    prompt_version: str
    bank_hash: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    usd: float
    latency_s: float
    cached: bool
    degraded: int  # number of answers with probs=None
    attempts: int
    usage_estimated: bool
    state_sha256: str
    tag: str
    n_questions: int
    data_freeze: str = ""
    lint_warnings: int = 0  # L9 soft-budget warnings (state above the program's state budget)
    backend_meta: str = ""  # backend-specific JSON (typed-local: wall time, model token count, threads); "" if none


@dataclass
class EvidenceRow:
    request_key: str
    eid: str
    available_at_utc: pd.Timestamp
    kept: bool
    quarantined: bool
    text_sha256: str | None
    d0_flags: str | None = None  # SAN-4: D0 events of the item (hidden markup, invisibles, confusables, ...)


@dataclass
class ProgramOutput:
    """What one program run emits for one entity and week."""

    program: str
    entity: str
    t: pd.Timestamp
    # code aggregates Layer B cannot derive from single answers (news f_imp, sentinel h, policy s_pol, ...);
    # answer-level phi features are computed once, in Layer B (learn.features.features_from_answers). None = abstain.
    features: dict[str, float | None] = field(default_factory=dict)
    trace: list[TraceRow] = field(default_factory=list)
    requests: list[RequestRow] = field(default_factory=list)
    evidence: list[EvidenceRow] = field(default_factory=list)
    blackboard: dict[str, Any] = field(default_factory=dict)  # M-work entries "<program>.<key>" -> JSON value
    memory_writes: list[Any] = field(default_factory=list)  # memory.store EventRecord / ProfileRecord


@runtime_checkable
class JudgmentProgram(Protocol):
    """A typed judgment program (design_v1 §12.3).

    `depends_on`: programs whose same-week outputs (blackboard) this one reads; `week_ordered`: the run
    for week t needs this program's memory writes of week t-1 (N2/N3 via M-epi); `memory_reads` (optional,
    default ()): programs whose M-epi/M-sem records of earlier weeks this one reads (e.g. policy exposure reads
    FA's M-sem profiles); the runner waits for all their earlier-week units and shows only their records;
    `taint`: trust channel of everything it emits ('T', 'S' or 'W', §3.2)."""

    name: str
    version: str
    taint: str
    depends_on: tuple[str, ...]
    week_ordered: bool

    async def run(self, ctx: WeekContext, s1: Any) -> list[ProgramOutput]: ...


_WARNED: set[str] = set()


@dataclass
class Asked:
    response: S1Response
    trace: list[TraceRow]
    request: RequestRow
    evidence: list[EvidenceRow]


async def ask(s1: Any, req: Request) -> Asked:
    """Send one request through `S1Backend.ask` (cache, ledger, rate limit) and turn it into table rows.

    The L9 budget (state + longest question <= 24k tokens, design_v1 §5.0) is enforced before sending; a state
    above the program's soft budget is logged (first time per program) and counted in `lint_warnings`."""
    from jevtrader.programs.lint import check_request

    warns = check_request(req.program, req.state, req.questions)
    for w in warns:
        (log.warning if req.program not in _WARNED else log.debug)("%s", w)
        _WARNED.add(req.program)
    resp = await s1.ask(req.state, req.questions, tag=req.tag)
    trace = [
        TraceRow(req.t, req.entity, req.qids[n][0], req.stage, req.qids[n][1], req.qids[n][2], a.kind,
                 None if a.probs is None else dict(a.probs), resp.request_key)
        for n, a in resp.answers.items()
    ]
    row = RequestRow(
        request_key=resp.request_key, t=req.t, symbol=req.entity, program=req.program, stage=req.stage,
        backend=resp.backend, model_pinned=resp.model, response_model=resp.response_model,
        prompt_version=resp.prompt_version, bank_hash=getattr(s1, "bank_hash", ""), input_tokens=resp.input_tokens,
        cached_tokens=resp.cached_tokens, output_tokens=resp.output_tokens, usd=resp.cost_usd, latency_s=resp.latency_s,
        cached=resp.cached, degraded=sum(a.probs is None for a in resp.answers.values()), attempts=resp.attempts,
        usage_estimated=resp.usage_estimated, state_sha256=state_sha256(req.state), tag=req.tag,
        n_questions=len(req.questions), lint_warnings=len(warns),
        backend_meta=json.dumps(resp.meta, sort_keys=True, default=str) if resp.meta else "",
    )
    ev = [EvidenceRow(resp.request_key, e.eid, pd.Timestamp(e.available_at_utc), e.kept, e.quarantined, e.text_sha256,
                      getattr(e, "d0_flags", None)) for e in req.evidence]
    return Asked(resp, trace, row, ev)


# ------------------------------------------------------------------------------------ feature map phi
def phi_noul(a: Answer, clip: float = 4.0) -> float | None:
    """x = clip(logit p, -clip, clip) (§5.0); None when the answer abstains."""
    if a.probs is None:
        return None
    p = a.probs["yes"]
    if p <= 0.0:
        return -clip
    if p >= 1.0:
        return clip
    return max(-clip, min(clip, math.log(p / (1.0 - p))))


def phi_score(a: Answer) -> float | None:
    """x = (E[level] - L/2) / (L/2) in [-1, 1] for levels 0..L; an ordinal feature, never a magnitude."""
    if a.probs is None:
        return None
    half = (len(a.probs) - 1) / 2.0
    return (a.expected_score - half) / half


def phi_ordered(a: Answer, value_map: Mapping[str, float]) -> float | None:
    """x = sum_l v(l) p(l) for an ordered Choice with a code-defined value map."""
    if a.probs is None:
        return None
    return float(sum(value_map[k] * p for k, p in a.probs.items()))
