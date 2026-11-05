"""Policy analyst (design_v1 §5.5): P1 reads pre-filtered policy items, P2 reads the week's typed theme rows, and the
stock exposure is code.

**Pre-filter (code).** Candidates of week t are news items tagged MARKET (SPY/QQQ/DIA/IWM) or with more than 3
tickers, created in [c_{t-1}, c_t) and last edited before c_t, whose headline + summary matches `POLICY_LEXICON`
(case-insensitive; the constant of `experiments/plan/plan_v1.py`, pinned by a test). At most 60 per week: the 60 with
the smallest sha256(item id), a deterministic, uniform sample over the week. Point in time, the pool keeps a broad
item only if it is tagged MARKET or to a study firm that had been a U member by the cutoff (`ctx.study_members`): the
archive holds a broad item only because one of its tickers is a study symbol, and a study symbol that joins U later
would otherwise select 2021 items by 2024 membership. A week whose news archive is unavailable (non-strict context)
makes no request and writes no `P1.week` (PX and P2 abstain).

**P1** (`PolicyItemProgram`, taint W; one request per item; 18 questions in bank order). State `{note (market note),
item: {headline, summary, published}}`, D0-sanitized, then blinded (all universe firms, with the item's tickers as
`tagged`; levels; times relative to the decision year). Weekly aggregates (code), with item weight
omega_k = P(is_policy) * E_k[s(stage)] * 1[P(instructs_reader) < .5] and s = code_constants."P1.stage.weight":
    I_theta = sum_k omega_k P_k(theme)                                                    (intensity)
    D_theta = sum_k omega_k P_k(theme) (E_k[profit_conditions] - 2) / 2 / max(I_theta, 1e-6)
    U_t     = sum_k omega_k (P_k(raises_uncertainty) - P_k(reduces_uncertainty))
    E_theta,g = sum_k omega_k P_k(theme) (E_k[affects_industry_g] - 1)
Active themes: I >= 1.5, at most the top 4 (code_constants thresholds "P1.active_theme"). `not_policy` is not a
theme. An item with a degraded gate answer (is_policy, theme, stage, instructs_reader) is left out of the week
(counted); a degraded effect answer drops only that term. The aggregates go to the M-work blackboard (`P1.week`,
W-tainted); P1 emits no per-stock features.

**PX** (`PolicyExposureProgram`, code only, taint W = W themes x S flags). Per stock:
X_{i,theta} = min(1, sum_f Mat[theta, f] p_{i,f}) with the M-sem profile p (F-A; a missing profile or a degraded
flag counts 0, the X2d convention) and the fixed theme -> flag matrix `THEME_FLAGS` (design_v1 §5.5 table);
s_pol_i = sum_{theta active} (I_theta / sum_active I) D_theta X_{i,theta}; s_ind_i = sum_theta E_{theta, g(i)}.
Both go to features.parquet under their catalog names; a week without P1 aggregates abstains (null).

**P2** (`PolicyWeekProgram`, taint W; one request per week; typed-only state, rule R3). This week's active-theme rows
`{theme, news_volume, mostly_at_stage, effect_on_profits, uncertainty}`, the rows of the 4 prior weeks with
`weeks_before`, and the EPU and GPR 30-day averages as 5-year percentile buckets. `news_volume` is the tercile of
I_theta against its own past 52 weekly values; the prior weeks' and the history's aggregates are recomputed by the P1
code for those weeks, each rendered at its own cutoff, so their requests are exactly P1's own (the response cache
and single-flight make them free in a contiguous run; a run over scattered weeks pays for them, see the report).
Row words: effect_on_profits harsher if D <= -1/4, easier if D >= 1/4, else mixed; uncertainty rising if
U_theta >= 1/5, falling if <= -1/5, else stable, with U_theta the theme's weighted mean of P(raises) - P(reduces);
mostly_at_stage = the stage with the largest omega-weighted probability mass (without the stage weight).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jevtrader.data.news import decision_cutoff
from jevtrader.programs import render
from jevtrader.programs.bank_v1 import Bank, load_bank
from jevtrader.programs.base import Evidence, ProgramOutput, Request, RequestRow, WeekContext, ask
from jevtrader.programs.context import previous_decision
from jevtrader.programs.macro import program_items
from jevtrader.s1.types import canonical_json

log = logging.getLogger("jevtrader.programs")
MARKET = "MARKET"
# design_v1 §5.5: matched case-insensitively on headline + summary (verbatim from experiments/plan/plan_v1.py)
POLICY_LEXICON = (r"\b(?:tariffs?|fed|fomc|federal reserve|powell|congress|senate|house|white house|president|"
                  r"administration|regulators?|sec|ftc|doj|antitrust|sanctions?|export controls?|elections?|shutdown|"
                  r"debt ceiling|war|missiles?|ceasefire|subsid(?:y|ies)|tax(?:es)?|treasury|trade deal|opec|"
                  r"rate cuts?|rate hikes?|inflation)\b")
POLICY_CAP = 60
MIN_TAGS_BROAD = 4  # "> 3 tickers"
ACTIVE_I, ACTIVE_TOP = 1.5, 4
HISTORY_WEEKS, PRIOR_WEEKS = 52, 4
EFFECT_CUT, UNCERTAINTY_CUT = 0.25, 0.2
NOT_THEME = "not_policy"

# design_v1 §5.5 theme -> M-sem profile flag weights
THEME_FLAGS: dict[str, dict[str, float]] = {
    "trade_and_tariffs": {"imported_inputs": 1.0, "china_link": 0.5, "foreign_sales": 0.5},
    "sanctions_and_export_controls": {"export_controls": 1.0, "china_link": 0.7, "semiconductor_chain": 0.7},
    "tech_and_antitrust": {"antitrust_scrutiny": 1.0, "data_privacy_rules": 0.7, "advertising_revenue": 0.3,
                           "ai_central": 0.3},
    "financial_regulation": {"bank_style_regulation": 1.0},
    "healthcare_and_drug_pricing": {"health_payer_prices": 1.0, "gov_customer": 0.3},
    "energy_and_environment": {"environmental_rules": 1.0, "energy_input_costs": 0.5, "commodity_revenue": 0.5},
    "monetary_policy": {"rate_sensitive": 1.0, "cyclical_demand": 0.3},
    "fiscal_and_tax": {"gov_customer": 0.7, "defense_customer": 0.5, "consumer_facing": 0.3},
    "government_operations": {"gov_customer": 1.0, "defense_customer": 0.5},
    "geopolitical_conflict": {"defense_customer": 1.0, "commodity_revenue": 0.5, "foreign_sales": 0.5},
    "labor_and_immigration": {"consumer_facing": 0.3, "cyclical_demand": 0.3},
}


# ------------------------------------------------------------------------------------ pre-filter
_BROAD: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}  # id(table) -> (table, broad items); the reference keeps
# the id from being reused by another table while the entry exists


def _broad_items(table: pd.DataFrame) -> pd.DataFrame:
    """One row per broad item (MARKET-tagged or > 3 tickers) that matches the lexicon, with its universe tags and
    whether it is MARKET-tagged; cached per news table (the table is read-only)."""
    key = id(table)
    if key not in _BROAD or _BROAD[key][0] is not table:
        market_ids = set(table.loc[table["symbol"] == MARKET, "id"])
        broad_ids = market_ids | set(table.loc[table["n_tags"] >= MIN_TAGS_BROAD, "id"])
        rows = table[table["id"].isin(broad_ids)]
        tags = rows[rows["symbol"] != MARKET].groupby("id")["symbol"].agg(lambda s: tuple(sorted(set(s))))
        items = rows.drop_duplicates("id").set_index("id")
        items = items.assign(tags=tags.reindex(items.index).apply(lambda x: x if isinstance(x, tuple) else ()),
                             market=items.index.isin(market_ids))
        text = (items["headline"].fillna("") + " " + items["summary"].fillna("")).str.lower()
        items = items[text.str.contains(POLICY_LEXICON, regex=True)]
        items = items.assign(created_at=pd.to_datetime(items["created_at"], utc=True),
                             updated_at=pd.to_datetime(items["updated_at"], utc=True))
        _BROAD[key] = (table, items.sort_values("created_at", kind="stable"))
    return _BROAD[key][1]


def item_rank(item_id) -> str:
    return hashlib.sha256(str(item_id).encode()).hexdigest()


def _members_at(members, cutoff) -> frozenset[str]:
    return frozenset(members(cutoff) if callable(members) else members)


def pit_pool(items: pd.DataFrame, members, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Broad items that are MARKET-tagged or tagged to a study firm that had been a U member by `cutoff`
    (`members`: a callable cutoff -> set, or a set; None keeps every item)."""
    if members is None:
        return items
    ms = _members_at(members, cutoff)
    keep = items["market"] | items["tags"].map(lambda tags: bool(ms.intersection(tags)))
    return items[keep.astype(bool)]


def policy_candidates(table: pd.DataFrame, start: pd.Timestamp | None, cutoff: pd.Timestamp,
                      cap: int = POLICY_CAP, members=None) -> pd.DataFrame:
    """The week's P1 items: broad lexicon matches created in [start, cutoff) and last edited before the cutoff, in
    the point-in-time pool (`pit_pool`), at most `cap` by the smallest sha256(id), returned in that order."""
    b = _broad_items(table)
    lo = b["created_at"].searchsorted(start, side="left") if start is not None else 0
    hi = b["created_at"].searchsorted(cutoff, side="left")
    w = b.iloc[lo:hi]
    w = pit_pool(w[w["updated_at"] < cutoff], members, cutoff)
    order = sorted(w.index, key=item_rank)[:cap]
    return w.loc[order]


AUDIT_N, AUDIT_SALT = 300, "P1-audit-www27"


def recall_audit_items(table: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, n: int = AUDIT_N,
                       salt: str = AUDIT_SALT, members=None) -> pd.DataFrame:
    """The recall audit of the pre-filter (design_v1 §5.5): `n` broad items (MARKET-tagged or > 3 tickers) created
    in [start, end) that the lexicon does NOT match, chosen by the smallest sha256(salt + id); P1.is_policy on them
    estimates the false-omission rate. `members` (a callable cutoff -> set) restricts to the point-in-time pool, each
    item judged at its creation time (`pit_pool`). Diagnostic only."""
    market_ids = set(table.loc[table["symbol"] == MARKET, "id"])
    broad_ids = market_ids | set(table.loc[table["n_tags"] >= MIN_TAGS_BROAD, "id"])
    rows = table[table["id"].isin(broad_ids)]
    tags = rows[rows["symbol"] != MARKET].groupby("id")["symbol"].agg(lambda s: tuple(sorted(set(s))))
    rows = rows.drop_duplicates("id").set_index("id")
    rows = rows.assign(tags=tags.reindex(rows.index).apply(lambda x: x if isinstance(x, tuple) else ()),
                       market=rows.index.isin(market_ids))
    created = pd.to_datetime(rows["created_at"], utc=True)
    updated = pd.to_datetime(rows["updated_at"], utc=True)
    rows = rows[(created >= start) & (created < end) & (updated < end)]
    if members is not None:
        rows = rows[[bool(m) or bool(_members_at(members, c).intersection(tg))
                     for m, c, tg in zip(rows["market"], pd.to_datetime(rows["created_at"], utc=True), rows["tags"])]]
    text = (rows["headline"].fillna("") + " " + rows["summary"].fillna("")).str.lower()
    rows = rows[~text.str.contains(POLICY_LEXICON, regex=True)]
    order = sorted(rows.index, key=lambda i: hashlib.sha256(f"{salt}{i}".encode()).hexdigest())[:n]
    return rows.loc[order]


# ------------------------------------------------------------------------------------ aggregation
@dataclass
class PolicyWeek:
    """P1's weekly aggregates (JSON on the blackboard as `P1.week`)."""

    n_candidates: int = 0
    n_used: int = 0
    n_degraded: int = 0
    intensity: dict[str, float] = field(default_factory=dict)  # I_theta
    direction: dict[str, float] = field(default_factory=dict)  # D_theta
    uncertainty_theme: dict[str, float] = field(default_factory=dict)  # U_theta (weighted mean)
    stage: dict[str, dict[str, float]] = field(default_factory=dict)  # theme -> stage mass
    industry: dict[str, dict[str, float]] = field(default_factory=dict)  # E_{theta, g}
    uncertainty: float = 0.0  # U_t
    active: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"n_candidates": self.n_candidates, "n_used": self.n_used, "n_degraded": self.n_degraded,
                "intensity": self.intensity, "direction": self.direction,
                "uncertainty_theme": self.uncertainty_theme, "stage": self.stage, "industry": self.industry,
                "uncertainty": self.uncertainty, "active": self.active}

    @classmethod
    def from_json(cls, d: Mapping) -> PolicyWeek:
        return cls(**{k: d[k] for k in cls().to_json() if k in d})


def aggregate(bank: Bank, answers: list[dict[str, dict | None]], n_candidates: int | None = None) -> PolicyWeek:
    """The weekly aggregates from each item's answers (name -> probs or None)."""
    stage_w = bank.code_constants["P1.stage.weight"]
    themes = [th for th in bank.q("P1.theme").options if th != NOT_THEME]
    codes = list(bank.ff12)
    wk = PolicyWeek(n_candidates=len(answers) if n_candidates is None else n_candidates)
    I = dict.fromkeys(themes, 0.0)
    Dn, Un, U = dict.fromkeys(themes, 0.0), dict.fromkeys(themes, 0.0), 0.0
    St = {th: {} for th in themes}
    E = {th: dict.fromkeys(codes, 0.0) for th in themes}
    for a in answers:
        gate = [a.get(k) for k in ("is_policy", "theme", "stage", "instructs_reader")]
        if any(g is None for g in gate):
            wk.n_degraded += 1
            continue
        pol, th_p, st_p, instr = gate
        if instr["yes"] >= 0.5:
            continue  # guard: an item that tries to steer the reader carries no weight
        base = float(pol["yes"])
        w = base * sum(float(stage_w[k]) * float(v) for k, v in st_p.items())
        wk.n_used += 1
        eff = a.get("profit_conditions")
        e_eff = None if eff is None else sum(int(k) * float(v) for k, v in eff.items())
        unc = a.get("uncertainty")
        du = None if unc is None else float(unc["raises_uncertainty"]) - float(unc["reduces_uncertainty"])
        if du is not None:
            U += w * du
        for th in themes:
            m = w * float(th_p.get(th, 0.0))
            I[th] += m
            if e_eff is not None:
                Dn[th] += m * (e_eff - 2.0) / 2.0
            if du is not None:
                Un[th] += m * du
            for k, v in st_p.items():  # stage mass without the stage weight
                St[th][k] = St[th].get(k, 0.0) + base * float(th_p.get(th, 0.0)) * float(v)
            for g in codes:
                ind = a.get(f"affects_industry_{g}")
                if ind is not None:
                    E[th][g] += m * (sum(int(k) * float(v) for k, v in ind.items()) - 1.0)
    wk.intensity = {th: float(I[th]) for th in themes}
    wk.direction = {th: float(Dn[th] / max(I[th], 1e-6)) for th in themes}
    wk.uncertainty_theme = {th: float(Un[th] / max(I[th], 1e-6)) for th in themes}
    wk.stage = {th: {k: float(v) for k, v in St[th].items()} for th in themes}
    wk.industry = {th: {g: float(x) for g, x in E[th].items()} for th in themes}
    wk.uncertainty = float(U)
    ranked = sorted((th for th in themes if I[th] >= ACTIVE_I), key=lambda th: (-I[th], th))
    wk.active = ranked[:ACTIVE_TOP]
    return wk


# ------------------------------------------------------------------------------------ P1
class PolicyItemProgram:
    name = "P1"
    version = "P1-v1"
    taint = "W"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None, cap: int = POLICY_CAP):
        self.bank = bank or load_bank()
        self.cap = cap
        self.items = program_items(self.bank, "P1")

    def requests(self, t: pd.Timestamp, cutoff: pd.Timestamp, prev_cutoff: pd.Timestamp | None, news, blinder,
                 unit_t: pd.Timestamp | None = None, stage: str = "P1", program: str = "P1", members=None
                 ) -> tuple[list[Request], int]:
        """The week's P1 requests (one per item), rendered relative to `cutoff`; also the candidate count.
        `unit_t`/`stage`/`program` label requests another program makes with P1's states and questions (P2's
        history): the request key depends only on the state and questions, so they are P1's own requests.
        `members`: `ctx.study_members` (the point-in-time pool, `pit_pool`)."""
        table = news.t if hasattr(news, "t") else news
        if prev_cutoff is None:  # first week of the calendar: a one-week window
            log.warning("P1 %s: no previous decision; using the 7 days before the cutoff", pd.Timestamp(t).date())
            prev_cutoff = cutoff - pd.Timedelta(days=7)
        cand = policy_candidates(table, prev_cutoff, cutoff, self.cap, members=members)
        year = int(pd.Timestamp(t).year)
        out = []
        for item_id, it in cand.iterrows():
            tags = tuple(it["tags"])
            kw = dict(decision_year=year, cutoff=cutoff, tagged=tags)
            h0, s0 = render.d0(it["headline"], render.HEADLINE_MAX), render.d0(it["summary"], render.SUMMARY_MAX)
            head = render.blind_clean(h0, blinder, **kw)  # D0 before blinding (SAN-2)
            summ = render.blind_clean(s0, blinder, **kw) if str(s0) else ""
            item = {"headline": head, "summary": summ, "published": render.relative_time(it["created_at"], cutoff)}
            state = render.envelope(self.bank, "P1", item=item)
            avail = max(pd.Timestamp(it["created_at"]), pd.Timestamp(it["updated_at"]))
            ev = [Evidence(f"news:{item_rank(item_id)[:16]}", avail, text=canonical_json(item),
                           d0_flags=(h0.flags + s0.flags).to_json())]
            out.append(Request.build(program, stage, MARKET, pd.Timestamp(unit_t if unit_t is not None else t), state,
                                     self.items, ev, tag=program))
        return out, len(cand)

    async def week(self, t, cutoff, prev_cutoff, news, blinder, s1, unit_t=None, stage="P1", program="P1",
                   members=None):
        reqs, n = self.requests(t, cutoff, prev_cutoff, news, blinder, unit_t=unit_t, stage=stage, program=program,
                                members=members)
        asked = await asyncio.gather(*(ask(s1, r) for r in reqs))
        answers = [{row.qid: row.probs for row in a.trace} for a in asked]
        return asked, aggregate(self.bank, answers, n)

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        if "news" in ctx.unavailable:
            log.warning("P1 %s: the news archive does not cover the week; P1 abstains (no P1.week)",
                        pd.Timestamp(ctx.t).date())
            return []
        asked, wk = await self.week(ctx.t, ctx.cutoff_utc, ctx.prev_cutoff_utc, ctx.news, ctx.blinder, s1,
                                    members=ctx.study_members)
        t = pd.Timestamp(ctx.t)
        outs = [ProgramOutput("P1", MARKET, t, {}, a.trace, [a.request], a.evidence) for a in asked]
        outs.append(ProgramOutput("P1", MARKET, t, blackboard={"P1.week": wk.to_json()}))
        return outs


# ------------------------------------------------------------------------------------ exposure (code)
def exposure(flags: Mapping[str, float | None] | None) -> dict[str, float]:
    """X_{i,theta} = min(1, sum_f Mat[theta, f] p_f); a missing profile or a degraded flag counts 0."""
    flags = flags or {}
    return {th: min(1.0, sum(w * float(flags.get(f) or 0.0) for f, w in m.items())) for th, m in THEME_FLAGS.items()}


def stock_policy_features(wk: PolicyWeek, x: Mapping[str, float], industry: str | None) -> dict[str, float]:
    tot = sum(wk.intensity[th] for th in wk.active)
    s_pol = sum(wk.intensity[th] / tot * wk.direction[th] * x.get(th, 0.0) for th in wk.active) if tot > 0 else 0.0
    s_ind = sum(e.get(industry, 0.0) for e in wk.industry.values()) if industry else 0.0
    return {"s_pol": float(s_pol), "s_ind": float(s_ind)}


class PolicyExposureProgram:
    name = "PX"
    version = "PX-v1"
    taint = "W"
    depends_on = ("P1", "FA")  # this week's P1 aggregates; FA's same-week profiles
    week_ordered = False
    memory_reads = ("FA",)  # FA's earlier-week profiles (M-sem)

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        t = pd.Timestamp(ctx.t)
        entry = ctx.memory.work("P1.week") if ctx.memory is not None else None
        outs = []
        for sym in ctx.stock_names():
            if entry is None:
                outs.append(ProgramOutput("PX", sym, t, {"s_pol": None, "s_ind": None}))
                continue
            wk = PolicyWeek.from_json(entry.value)
            prof = ctx.memory.profile(sym)
            x = exposure(prof.flags if prof is not None else None)
            ev = [Evidence("bb:P1.week", entry.available_at, text=canonical_json(wk.active))]
            if prof is not None:
                ev.append(Evidence(f"mem:{prof.record_id}", prof.available_at, text=canonical_json(prof.flags)))
            feats = stock_policy_features(wk, x, ctx.firms[sym].industry if sym in ctx.firms else None)
            o = ProgramOutput("PX", sym, t, feats)
            o.evidence = [_evidence_row(sym, t, e) for e in ev]
            outs.append(o)
        return outs


def _evidence_row(sym: str, t: pd.Timestamp, e: Evidence):
    """Evidence of a code-only program is keyed by a synthetic request key (no S1 request was made)."""
    from jevtrader.programs.base import EvidenceRow

    key = "code:" + hashlib.sha256(f"PX|{sym}|{pd.Timestamp(t).date()}".encode()).hexdigest()[:32]
    return EvidenceRow(key, e.eid, pd.Timestamp(e.available_at_utc), e.kept, e.quarantined, e.text_sha256)


# ------------------------------------------------------------------------------------ P2
def volume_word(i_now: float, history: list[float]) -> str:
    """Tercile of this week's intensity against the past weeks' (light / moderate / heavy)."""
    h = np.asarray(history, dtype=float)
    if not len(h):
        return "moderate"
    q1, q2 = np.quantile(h, [1 / 3, 2 / 3])
    return "light" if i_now <= q1 else "heavy" if i_now > q2 else "moderate"


def theme_rows(bank: Bank, wk: PolicyWeek, history: list[PolicyWeek]) -> list[dict[str, str]]:
    rows = []
    for th in wk.active:
        st = wk.stage.get(th) or {}
        top_stage = max(st, key=lambda k: (st[k], k)) if st else "not_policy"
        d, u = wk.direction[th], wk.uncertainty_theme[th]
        rows.append({
            "theme": bank.option_text("P1.theme", th),
            "news_volume": volume_word(wk.intensity[th], [h.intensity.get(th, 0.0) for h in history]),
            "mostly_at_stage": render.option_name(top_stage),
            "effect_on_profits": "harsher" if d <= -EFFECT_CUT else "easier" if d >= EFFECT_CUT else "mixed",
            "uncertainty": "rising" if u >= UNCERTAINTY_CUT else "falling" if u <= -UNCERTAINTY_CUT else "stable",
        })
    return rows


class PolicyWeekProgram:
    name = "P2"
    version = "P2-v1"
    taint = "W"
    depends_on = ("P1",)
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, bank: Bank | None = None, history_weeks: int = HISTORY_WEEKS, prior_weeks: int = PRIOR_WEEKS):
        self.bank = bank or load_bank()
        self.history_weeks, self.prior_weeks = history_weeks, prior_weeks
        self.p1 = PolicyItemProgram(self.bank)
        self.specs = self.bank.program("P2")

    def past_weeks(self, ctx: WeekContext, n: int) -> list[pd.Timestamp]:
        """The n decision dates before t, most recent first (from the exchange calendar)."""
        sessions = ctx.sessions if ctx.sessions is not None else ctx.prices.close.index
        out, d = [], pd.Timestamp(ctx.t)
        for _ in range(n):
            d = previous_decision(sessions, d)
            if d is None:
                break
            out.append(d)
        return out

    async def history(self, ctx: WeekContext, s1) -> tuple[list[PolicyWeek], list[RequestRow], list[Evidence]]:
        """P1 aggregates of the past `history_weeks + prior_weeks` weeks (most recent first: the prior weeks' rows
        need their own 52-week history), each computed by P1's code at its own cutoff; the requests are logged under
        P2's unit with stage 'P1-history:<date>'."""
        sessions = ctx.sessions if ctx.sessions is not None else ctx.prices.close.index
        weeks = self.past_weeks(ctx, self.history_weeks + self.prior_weeks)

        async def one(s: pd.Timestamp):
            prev = previous_decision(sessions, s)
            asked, wk = await self.p1.week(s, decision_cutoff(s), None if prev is None else decision_cutoff(prev),
                                           ctx.news, ctx.blinder, s1, unit_t=ctx.t, stage=f"P1-history:{s.date()}",
                                           program="P2", members=ctx.study_members)
            return asked, wk, Evidence(f"hist:P1.week:{s.date()}", decision_cutoff(s), text=canonical_json(wk.active))

        res = await asyncio.gather(*(one(s) for s in weeks))
        return [r[1] for r in res], [a.request for r in res for a in r[0]], [r[2] for r in res]

    def state(self, ctx: WeekContext, wk: PolicyWeek, hist: list[PolicyWeek]) -> dict:
        this = theme_rows(self.bank, wk, hist[: self.history_weeks])
        prior = []
        for k, h in enumerate(hist[: self.prior_weeks], 1):
            prior += [{**row, "weeks_before": k} for row in theme_rows(self.bank, h, hist[k:k + self.history_weeks])]
        snap = ctx.macro.snapshot(ctx.t)

        def idx(name: str) -> str:
            rec = snap.get(name)
            p = rec.get("pctile_5y") if isinstance(rec, dict) else None
            return render.NOT_AVAILABLE if p is None or not np.isfinite(p) else \
                f"{render.hist_bucket(p)} compared with the past 5 years"

        indices = {"economic policy uncertainty (30-day average)": idx("economic policy uncertainty (EPU, 30d avg)"),
                   "geopolitical risk (30-day average)": idx("geopolitical risk (GPR, 30d avg)")}
        return render.envelope(self.bank, "P2", policy_themes_this_week=this, policy_themes_prior_weeks=prior,
                               uncertainty_indices=indices)

    async def run(self, ctx: WeekContext, s1) -> list[ProgramOutput]:
        t = pd.Timestamp(ctx.t)
        entry = ctx.memory.work("P1.week") if ctx.memory is not None else None
        if entry is None:
            log.warning("P2 %s: no P1.week on the blackboard; P2 abstains", t.date())
            return []
        wk = PolicyWeek.from_json(entry.value)
        hist, hist_rows, hist_ev = await self.history(ctx, s1)
        state = self.state(ctx, wk, hist)
        items = [(q.name, q, q.to_question()) for q in self.specs.values()]
        ev = [Evidence("bb:P1.week", entry.available_at, text=canonical_json(wk.active)), *hist_ev,
              Evidence("macro:EPU_GPR", t.tz_localize("UTC"), text=canonical_json(state["uncertainty_indices"]))]
        asked = await ask(s1, Request.build("P2", "P2", MARKET, t, state, items, ev))
        ans = {r.qid: r.probs for r in asked.trace}
        bb = {"P2.policy_regime": {"probs": ans.get("policy_regime")}, "P2.policy_risk": {"probs": ans.get("policy_risk")}}
        return [ProgramOutput("P2", MARKET, t, {}, asked.trace, [asked.request, *hist_rows], asked.evidence, bb)]
