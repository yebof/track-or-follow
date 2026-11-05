#!/usr/bin/env python3
"""Write paper/generated/numbers.tex: one \\newcommand per number the paper states (outline_v1 §5).

No number is typed by hand into paper/sections/*.tex (CLAUDE.md rule 2; scripts/check_numbers.py). Every number
comes from this file, in four groups:

  1. Design constants, read now from configs/*.yaml (K, J, the cap multiplier, costs, windows, model pins).
  2. Protocol constants of the pre-registration (unit counts, query budget, MDEs): placeholders until the frozen
     pre-registration or its plan output is wired in (the planned source is named per macro).
  3. Result numbers (outline §5 names; positioning_v2 X-ids): placeholders until results/ holds the named output.
  4. Numbers quoted from the literature: each value is checked against literature/summaries/<key>.md (the value must
     occur there next to a context phrase); a failed check leaves a placeholder and makes the script exit 1.
  5. Result sentences whose wording switches to the pre-registered fallback (positioning_v2 §2.7) when a
     confirmatory test is null; placeholders until the test outcome exists.

A placeholder prints as a bold "[name]" via \\NumPlaceholder, so an unfilled number is visible in the PDF and
`scripts/check_numbers.py --final` can refuse it. Units are written in the text ("\\hOneASRTyped\\%",
"\\costBps{} bps"); write "\\nWeeks{}" when a space follows.

To fill a number, give its entry a getter (see `json_value`, `yaml_value`) that reads the file named in `source`.

Usage: python experiments/report/make_numbers.py [--check] [--list]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "generated" / "numbers.tex"
SUMMARIES = ROOT / "literature" / "summaries"
CONFIGS = ROOT / "configs"
PLACEHOLDER = "NumPlaceholder"


# ---------------------------------------------------------------------------------------------------------------------
# Readers and formatters
# ---------------------------------------------------------------------------------------------------------------------


def yaml_value(rel: str, dotted: str):
    """Value at a dotted key path in a YAML file under the repo, or None if the file or key is missing."""
    path = ROOT / rel
    if not path.is_file():
        return None
    node = yaml.safe_load(path.read_text(encoding="utf-8"))
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def json_value(rel: str, dotted: str):
    path = ROOT / rel
    if not path.is_file():
        return None
    node = json.loads(path.read_text(encoding="utf-8"))
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def fmt_num(x, decimals: int | None = None) -> str:
    """Plain number: integers without decimals, thousands with a comma, negatives with a math minus."""
    if isinstance(x, float) and decimals is None and float(x).is_integer():
        x = int(x)
    if isinstance(x, int) or decimals is None:
        s = f"{abs(x):,}"  # shortest repr for floats (0.8, 0.04)
    else:
        s = f"{abs(x):,.{decimals}f}"
    s = s.replace(",", "{,}")
    return f"\\ensuremath{{-{s}}}" if x < 0 else s


def fmt_pct(x: float, decimals: int = 1) -> str:
    """A share in [0, 1] as a percentage number (the text writes the % sign)."""
    return fmt_num(round(100.0 * x, decimals), decimals) if decimals else fmt_num(round(100.0 * x))


def fmt_pct_int(x: float) -> str:
    return fmt_pct(x, 0)


def fmt_ci(lo: float, hi: float, decimals: int = 3) -> str:
    return f"[{fmt_num(lo, decimals)}, {fmt_num(hi, decimals)}]"


def fmt_list(xs) -> str:
    xs = [fmt_num(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def fmt_codes(xs) -> str:
    """Identifiers kept verbatim (e.g. SEC item codes "1.03"), joined as "a, b and c"."""
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def tt(s: str) -> str:
    return "\\texttt{" + s.replace("_", "\\_") + "}"


# ---------------------------------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Num:
    name: str
    desc: str
    source: str
    ref: str = ""                                   # outline/positioning id (X1, Table 2 caption, ...)
    get: Callable[[], str | None] | None = None     # returns the LaTeX value, or None -> placeholder


@dataclass
class Lit:
    name: str
    key: str
    page: str
    value: str            # LaTeX value printed in the paper
    verify: str           # string that must occur in the summary ...
    context: str          # ... within CONTEXT_WINDOW characters of this regex
    desc: str


@dataclass
class Clause:
    """An extra clause of a result sentence, stated only if its own output says so (`cond` returns True); inserted
    before the sentence's final period. A clause whose output is missing or False is dropped (audit 2026-09-28: no
    generated sentence may claim what no script checks)."""
    text: str
    cond: Callable[[], bool | None]
    source: str


@dataclass
class Sentence:
    name: str
    desc: str
    source: str
    confirmed: str
    fallback: str
    outcome: Callable[[], bool | None] | None = None  # True = confirmatory test rejected H0; None = not yet run
    ref: str = ""
    confirmed_clauses: list = field(default_factory=list)  # Clause, appended to `confirmed` when their cond holds
    fallback_clauses: list = field(default_factory=list)  # Clause, appended to `fallback` when their cond holds
    # DEV-9 (docs/design/prereg_deviations.md): descriptive wording used instead of `fallback` when the registered
    # one-sided test did not reject AND the estimate's 95% CI excludes 0 on the side OPPOSITE to the registered claim
    # (`reversal` returns True). It states the estimate and its CI only (no new p-value); fallback clauses still apply.
    reversed: str = ""
    reversal: Callable[[], bool | None] | None = None

    def is_reversed(self, outcome: bool) -> bool:
        if outcome or not self.reversed or self.reversal is None:
            return False
        try:
            return self.reversal() is True
        except Exception:  # noqa: BLE001 -- a malformed source never switches the wording
            return False

    def compose(self, outcome: bool) -> str:
        base, clauses = (self.confirmed, self.confirmed_clauses) if outcome else (self.fallback, self.fallback_clauses)
        if self.is_reversed(outcome):
            base = self.reversed
        extra = []
        for c in clauses:
            try:
                ok = c.cond()
            except Exception:  # noqa: BLE001 -- a malformed source never adds a claim
                ok = None
            if ok is True:
                extra.append(c.text)
        if not extra:
            return base
        stem = base[:-1] if base.endswith(".") else base
        return stem + "".join(extra) + ("." if base.endswith(".") else "")


@dataclass
class Group:
    title: str
    items: list = field(default_factory=list)


SYS, LRN, BCK = "configs/system_v1.yaml", "configs/learning_v1.yaml", "configs/backends.yaml"
PREREG = "configs/prereg_v1.yaml (pending; plan values in results/plan/plan_v1.json)"


NEWS = "src/jevtrader/programs/news.py"
BANK = "docs/design/question_bank_v1.json"


def lazy_const(rel: str, name: str, scale: float = 1.0, decimals: int | None = None) -> Callable[[], str | None]:
    """code_const, resolved at call time (code_const is defined further down)."""
    return lambda: code_const(rel, name, scale, decimals)()


def bank_count(program: str, *keys: str) -> Callable[[], str | None]:
    """Number of questions of a frozen-bank program, or of a question's options or levels."""
    def get():
        path = ROOT / BANK
        if not path.is_file():
            return None
        v = json.loads(path.read_text(encoding="utf-8")).get("programs", {}).get(program)
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        return None if v is None else fmt_num(len(v))
    return get


def cfg(rel: str, key: str, fmt: Callable = fmt_num) -> Callable[[], str | None]:
    def get():
        v = yaml_value(rel, key)
        return None if v is None else fmt(v)
    return get


def cfg_item(rel: str, key: str, index: int) -> Callable[[], str | None]:
    """One element of a list-valued config entry (e.g. a window's start date)."""
    def get():
        v = yaml_value(rel, key)
        return str(v[index]) if isinstance(v, list) and len(v) > index else None
    return get


def test_case_count(rel: str, *names: str) -> Callable[[], str | None]:
    """Product of integer module constants of a test file, read from its source (e.g. `N_CHUNKS, PER_CHUNK = 8, 250`
    or one assignment per name), so a case count in the paper follows the test that runs it."""
    def get():
        path = ROOT / rel
        if not path.is_file():
            return None
        src = path.read_text(encoding="utf-8")
        vals: dict[str, int] = {}
        for m in re.finditer(r"^([A-Z_][A-Z0-9_]*(?:\s*,\s*[A-Z_][A-Z0-9_]*)*)\s*=\s*([\d_]+(?:\s*,\s*[\d_]+)*)\s*(?:#.*)?$",
                             src, re.MULTILINE):
            lhs = [s.strip() for s in m.group(1).split(",")]
            rhs = [int(s.strip().replace("_", "")) for s in m.group(2).split(",")]
            if len(lhs) == len(rhs):
                vals.update(zip(lhs, rhs))
        if not all(n in vals for n in names):
            return None
        total = 1
        for n in names:
            total *= vals[n]
        return fmt_num(total)
    return get


def model_name(rel: str, key: str) -> Callable[[], str | None]:
    """A pinned model id in typewriter type, with a line-break point after every hyphen (\\texttt does not hyphenate,
    so a long id such as gpt-4.1-mini-2025-04-14 would otherwise overrun the column)."""
    def get():
        v = yaml_value(rel, key)
        if not isinstance(v, str):
            return None
        name = v.split("@", 1)[0].rsplit("/", 1)[-1]  # drop the HF organisation and the pinned revision
        return tt(name).replace("-", "-\\allowbreak{}")
    return get


def reach_at_operating_point(part: str) -> Callable[[], str | None]:
    """Corollary 3.3 of the paper evaluated at the registered K and J, with equal reference targets that sum to one
    and no anchor (a design illustration, not a result):
      (ii)  displacement budget D = V/2 + (3/2) F with V = 1 and F = J/K (the non-protected share of the reference
            book), i.e. 1/2 + 3J/(2K);
      (iii) invested floor = (1/2) * (K - J)/K of the reference book (protected names halved at most).
    Both follow from configs/system_v1.yaml selection.K and selection.J only."""
    def get():
        k, j = yaml_value(SYS, "selection.K"), yaml_value(SYS, "selection.J")
        if not isinstance(k, (int, float)) or not isinstance(j, (int, float)) or not 0 <= j <= k:
            return None
        if part == "disp":
            return fmt_num(0.5 + 1.5 * j / k, 2)
        if part == "invested":
            return fmt_pct_int(0.5 * (k - j) / k)
        raise ValueError(part)
    return get


def pilot_stock_weeks(rel: str) -> Callable[[], str | None]:
    """Distinct (date, symbol) pairs of the declared pilot look at W_clean (design_v1 §13.6: 'the pilot (about 600
    stock-weeks, unblinded headlines, hindsight names)'); needs pandas with a parquet engine."""
    def get():
        path = ROOT / rel
        if not path.is_file():
            return None
        try:
            import pandas as pd  # noqa: PLC0415 -- optional dependency, only for this count
        except ImportError:
            return None
        d = pd.read_parquet(path, columns=["date", "symbol"])
        return fmt_num(int(d.drop_duplicates().shape[0]))
    return get


def count_files(pattern: str) -> Callable[[], str | None]:
    """Number of files matching a glob under the repo (e.g. one deployment.json per deployment-check decision)."""
    def get():
        n = len(list(ROOT.glob(pattern)))
        return fmt_num(n) if n else None
    return get


GROUPS: list[Group] = [
    Group("Design constants (read from configs/)", [
        Num("bookSize", "book size K", f"{SYS} selection.K", get=cfg(SYS, "selection.K")),
        Num("bandHalfWidth", "band half-width J", f"{SYS} selection.J", get=cfg(SYS, "selection.J")),
        Num("incumbentBuffer", "rank buffer for held names", f"{SYS} selection.incumbent_buffer",
            get=cfg(SYS, "selection.incumbent_buffer")),
        Num("capMultiplier", "text-cap multiplier mu (kappa_t = mu * s^T_t)", f"{LRN} text_cap.multiplier",
            get=cfg(LRN, "text_cap.multiplier")),
        Num("capFloor", "floor of the trusted cross-sectional sd s^T_t in the cap", f"{LRN} text_cap.floor",
            get=cfg(LRN, "text_cap.floor")),
        Num("reachDispOp", "Corollary (ii) at the operating point, equal reference targets summing to one, no anchor: "
            "1/2 + 3J/(2K) (fraction of NAV)", f"{SYS} selection.K, selection.J (derived)",
            get=reach_at_operating_point("disp")),
        Num("reachInvestedPct", "Corollary (iii) at the operating point, equal reference targets, no anchor: "
            "(K-J)/(2K) of the reference book (%)", f"{SYS} selection.K, selection.J (derived)",
            get=reach_at_operating_point("invested")),
        Num("nameCapPct", "per-name weight cap (%)", f"{SYS} risk.name_cap", get=cfg(SYS, "risk.name_cap", fmt_pct_int)),
        Num("industryCapPct", "per-industry weight cap (%)", f"{SYS} risk.industry_cap",
            get=cfg(SYS, "risk.industry_cap", fmt_pct_int)),
        Num("turnoverCapPct", "one-way turnover cap per rebalance (%)", f"{SYS} risk.turnover_cap",
            get=cfg(SYS, "risk.turnover_cap", fmt_pct_int)),
        Num("earningsMult", "earnings-window multiplier", f"{SYS} risk.earnings_mult", get=cfg(SYS, "risk.earnings_mult")),
        Num("earningsSessions", "earnings window (sessions ahead)", f"{SYS} risk.earnings_sessions",
            get=cfg(SYS, "risk.earnings_sessions")),
        Num("sentinelThreshold", "sentinel probability threshold", f"{SYS} risk.sentinel_high",
            get=cfg(SYS, "risk.sentinel_high")),
        Num("corroborationKeys", "distinct provenance keys needed for an exclusion in anchor weeks",
            f"{SYS} risk.corroboration", get=cfg(SYS, "risk.corroboration")),
        Num("anchorPriceSd", "price anchor threshold (sd of idiosyncratic return)", f"{SYS} anchors.price_k",
            get=cfg(SYS, "anchors.price_k")),
        Num("anchorFilingDays", "filing anchor window (calendar days)", f"{SYS} anchors.filing_window_days",
            get=cfg(SYS, "anchors.filing_window_days")),
        Num("jevReleaseProse", "Jev (jev-1.13.0) release date, prose (docs/jev archive)",
            "src/jevtrader/s1/cloudflare_jev.py RELEASE_DATE",
            get=lambda: (lambda d: f"{d.day}~{d.strftime('%B %Y')}")(__import__("datetime").date.fromisoformat(
                str(__import__("jevtrader.s1.cloudflare_jev", fromlist=["RELEASE_DATE"]).RELEASE_DATE)))),
        Num("universeLookback", "universe: sessions of median dollar volume before each reconstitution",
            "src/jevtrader/data/universe.py build_universe(lookback_days)",
            get=lambda: (lambda inspect, U: fmt_num(inspect.signature(U.build_universe).parameters["lookback_days"].default))(
                __import__("inspect"), __import__("jevtrader.data.universe", fromlist=["build_universe"]))),
        Num("maxTickers", "news: an item is read for a firm only if it tags at most this many tickers",
            "src/jevtrader/programs/news.py MAX_TICKERS",
            get=lambda: fmt_num(__import__("jevtrader.programs.news", fromlist=["MAX_TICKERS"]).MAX_TICKERS)),
        Num("arBetaWindow", "AR criterion: sessions in the trailing point-in-time beta (abnormal_returns default)",
            "src/jevtrader/learn/labels.py abnormal_returns(beta_window)",
            get=lambda: (lambda inspect, L: fmt_num(inspect.signature(L.abnormal_returns).parameters["beta_window"].default))(
                __import__("inspect"), __import__("jevtrader.learn.labels", fromlist=["abnormal_returns"]))),
        Num("anchorSdWindow", "price anchor: sessions in the daily excess-return sd window", f"{SYS} anchors.sd_window_sessions",
            get=cfg(SYS, "anchors.sd_window_sessions")),
        Num("anchorFilingItems", "8-K items that fire the filing anchor (SEC item codes)", f"{SYS} anchors.filing_items",
            get=cfg(SYS, "anchors.filing_items", fmt_codes)),
        Num("costBps", "transaction cost per side (bps)", f"{SYS} costs.bps_per_side", get=cfg(SYS, "costs.bps_per_side")),
        Num("costSensitivityBps", "cost sensitivity levels (bps per side)", f"{SYS} costs.sensitivity_bps",
            get=cfg(SYS, "costs.sensitivity_bps", fmt_list)),
        Num("wCleanStart", "first decision of W_clean", f"{SYS} windows.W_clean[0]",
            get=cfg_item(SYS, "windows.W_clean", 0)),
        Num("wCleanEnd", "last decision of W_clean", f"{SYS} windows.W_clean[1]",
            get=cfg_item(SYS, "windows.W_clean", 1)),
        Num("wReleaseStart", "first decision of W_release", f"{SYS} windows.W_release[0]",
            get=cfg_item(SYS, "windows.W_release", 0)),
        Num("cutoffTime", "information cutoff c_t (ET)", f"{SYS} clock.cutoff_local", get=cfg(SYS, "clock.cutoff_local", str)),
        Num("subsetLarge", "names in the S50 subset", f"{SYS} subsets.S50.n", get=cfg(SYS, "subsets.S50.n")),
        Num("subsetSmall", "names in the S30 subset", f"{SYS} subsets.S30.n", get=cfg(SYS, "subsets.S30.n")),
        Num("halfLifeWeeks", "allocator decay half-life (weeks)", f"{LRN} allocator.half_life_weeks",
            get=cfg(LRN, "allocator.half_life_weeks")),
        Num("priorSd", "prior sd of allocator slopes", f"{LRN} allocator.prior_sd_beta",
            get=cfg(LRN, "allocator.prior_sd_beta")),
        Num("logitClip", "clip of typed-judgment logits", f"{LRN} feature_map.noul_logit_clip",
            get=cfg(LRN, "feature_map.noul_logit_clip")),
        Num("stdWeeks", "feature standardization: trailing weeks", f"{LRN} standardization.trailing_weeks",
            get=cfg(LRN, "standardization.trailing_weeks")),
        Num("nTrustedFeatures", "trusted features of the learner", f"{LRN} features.trusted (count)",
            get=lambda: (lambda v: None if v is None else fmt_num(len(v)))(yaml_value(LRN, "features.trusted"))),
        Num("nUntrustedFeatures", "untrusted (text) features of the learner", f"{LRN} features.untrusted (count)",
            get=lambda: (lambda v: None if v is None else fmt_num(len(v)))(yaml_value(LRN, "features.untrusted"))),
        Num("newsMaxItems", "news: items read per stock-week, most recent first", f"{NEWS} MAX_ITEMS",
            get=lazy_const(NEWS, "MAX_ITEMS")),
        Num("nSummarySentences", "news screening: summary sentences after the headline", f"{NEWS} MAX_SUMMARY_SENTENCES",
            get=lazy_const(NEWS, "MAX_SUMMARY_SENTENCES")),
        Num("memoryWeeks", "event store: weeks of prior records for novelty and synthesis", f"{NEWS} MEMORY_WEEKS",
            get=lazy_const(NEWS, "MEMORY_WEEKS")),
        Num("nPriorRecords", "news novelty: prior records per request", f"{NEWS} N2_MAX_PRIOR",
            get=lazy_const(NEWS, "N2_MAX_PRIOR")),
        Num("promoWeight", "news: weight of an item the promotion guard flags", f"{NEWS} RHO_PROMOTIONAL",
            get=lazy_const(NEWS, "RHO_PROMOTIONAL", 1.0, 2)),
        Num("clusterJaccard", "news clustering: Jaccard threshold on blinded headlines",
            "src/jevtrader/programs/cluster.py JACCARD", get=lazy_const("src/jevtrader/programs/cluster.py", "JACCARD", 1.0, 2)),
        Num("policyCap", "policy items read per week", "src/jevtrader/programs/policy.py POLICY_CAP",
            get=lazy_const("src/jevtrader/programs/policy.py", "POLICY_CAP")),
        Num("nQuestionsNewsItem", "questions of the per-item news stage", f"{BANK} programs.N1 (count)",
            get=bank_count("N1")),
        Num("nEventTypes", "options of the event-type question", f"{BANK} programs.N1.event_type.options (count)",
            get=bank_count("N1", "event_type", "options")),
        Num("nEffectLevels", "levels of the effect question", f"{BANK} programs.N1.effect.levels (count)",
            get=bank_count("N1", "effect", "levels")),
        Num("nProfileFlags", "exposure flags of the profile agent", f"{BANK} groups.FA (count)",
            get=lambda: (lambda g: None if not g else fmt_num(sum(len(v) for v in g.values())))(
                json.loads((ROOT / BANK).read_text(encoding="utf-8")).get("groups", {}).get("FA"))),
        Num("emuModel", "pinned LLM behind the emulator and VERB", f"{BCK} backends.llm-logprob.model",
            get=model_name(BCK, "backends.llm-logprob.model")),
        Num("jevModel", "pinned Jev model id (runs by configuration)", f"{BCK} backends.jev.model",
            get=model_name(BCK, "backends.jev.model")),
        Num("ngModel", "open non-generative typed decision model (NG)", f"{BCK} backends.typed-local.model",
            get=model_name(BCK, "backends.typed-local.model")),
    ]),
    Group("Protocol constants of the pre-registration (placeholders until frozen)", [
        Num("universeSize", "names in the point-in-time universe (U100)", "configs (universe top_n; now a code default)"),
        Num("nPromotionUnits", "promotion stock-weeks in the H1 test", PREREG, "design §13.1"),
        Num("nBoundCheckUnits", "bound-check units", PREREG),
        Num("nDemotionUnits", "demotion units (plus an anchor stratum)", PREREG),
        Num("nAdaptiveItems", "items in the adaptive arm", PREREG),
        Num("maxAdaptiveQueries", "black-box query budget per item", PREREG),
        Num("nPairedAttacks", "paired attacks scored in H1 (realized frame, DEV-6/DEV-9; registered frame: "
            "nPairedAttacksRegistered)", PREREG),
        Num("nPairedAttacksRegistered", "paired attacks registered for H1 (units x families; prereg §3.1)", PREREG),
        Num("nPropertyCases", "randomized cases per property test (at least)",
            "tests/test_band_guard_bound.py N_CHUNKS x PER_CHUNK (test_proposition_randomized)",
            get=test_case_count("tests/test_band_guard_bound.py", "N_CHUNKS", "PER_CHUNK")),
        Num("hOneMDELow", "H1 MDE at the realized pairs, low end (pi_d = .2; registered formula; DEV-6/DEV-9)",
            PREREG + " power.C1_MDE_n2800_pid0.2"),
        Num("hOneMDEHigh", "H1 MDE at the realized pairs, high end (pi_d = .4; registered formula; DEV-6/DEV-9)",
            PREREG + " power.C1_MDE_n2800_pid0.4"),
        Num("hOneMDELowRegistered", "H1 MDE at the registered 1,400 pairs, low end (pi_d = .2; prereg §3.1)", PREREG),
        Num("hOneMDEHighRegistered", "H1 MDE at the registered 1,400 pairs, high end (pi_d = .4; prereg §3.1)", PREREG),
        Num("hTwoMDELow", "H2 MDE, low end", PREREG + " power.C2_MDE_between_sd0.0"),
        Num("hTwoMDEHigh", "H2 MDE, high end", PREREG + " power.C2_MDE_between_sd0.05"),
        Num("hThreeMDELow", "H3 MDE, low end (IC)", PREREG + " power.C3_MDE_rho0.8"),
        Num("hThreeMDEHigh", "H3 MDE, high end (IC)", PREREG + " power.C3_MDE_rho0.5"),
        Num("irMDE", "portfolio information-ratio MDE", PREREG + " power.IR_MDE_approx", "design §13.6"),
        Num("nwLag", "Newey-West lag (weeks)", PREREG),
        Num("ciLevel", "confidence level of reported intervals (%)", PREREG),
        Num("familyAlpha", "family-wise level of the Holm correction over H1-H3", PREREG, "design §13.1"),
        Num("bufferWeeks", "weekly decisions in W_buffer (between the emulator LLM's cutoff and W_clean)",
            "pre-registration; results/plan/plan_v1.json windows_decisions", "design §2.5"),
        Num("nWeeksRelease", "weekly decisions in W_release",
            "pre-registration; results/plan/plan_v1.json windows_decisions", "design §2.5"),
        Num("priorLookStockWeeks", "stock-weeks of the declared pilot look at W_clean (unblinded headlines)",
            "results/pilot/news_signal_pilot.parquet (distinct date x symbol); design_v1 §13.6",
            get=pilot_stock_weeks("results/pilot/news_signal_pilot.parquet")),
    ]),
    Group("Results: abstract and introduction (positioning_v2 §6 placeholders)", [
        Num("nWeeks", "weekly decisions in W_clean", "pre-registration; results/plan/plan_v1.json windows_decisions", "X1"),
        Num("nImpl", "implementations of the interface with H2 rows", "results/c2/*/summary.json", "NB"),
        Num("hOneASRHolistic", "portfolio-level ASR, holistic typed reader (%)", "results/c1/summary.json", "X5"),
        Num("hOneASRTyped", "portfolio-level ASR, typed programs (%)", "results/c1/summary.json", "X6"),
        Num("hTwoSpearmanEmu", "H2 weekly Spearman, emu", "results/c2/emu/summary.json", "X4"),
        Num("hThreeDeltaIC", "H3 rank-IC increment over code-only", "results/c3/summary.json", "X3"),
        Num("costPerRebalance", "$ per weekly rebalance (system)", "results/judgments/<gen_id>/requests.parquet (OPS-1)", "X8"),
        Num("nBaselines", "number of baselines", "final baseline list (results/eval)", "X9"),
        Num("localizationRate", "successful attacks localized to the planted item (%)", "results/faith/localization.json", "X11"),
        Num("deployViolations", "pre-trade violations in the deployment check", "results/live/<t>/deployment.json", "X7"),
        Num("nDeployDecisions", "weekly decisions in the deployment check (one deployment.json each)",
            "results/live/*/deployment.json (count)", get=count_files("results/live/*/deployment.json")),
        Num("nBoundCases", "bound-check cases (implementation conformance)", "results/bound/ (BG-5)", "outline §1 ¶5"),
        Num("nBoundViolations", "bound violations (implementation conformance)", "results/bound/ (BG-5)", "outline §1 ¶5"),
    ]),
    Group("Results: setting (§2)", [
        Num("feedDeskShare", "desk bylines on W_clean (%)", "feed provenance output (FEED-1)", "outline §2 ¶2"),
        Num("feedContribShare", "named contributors on W_clean (%)", "feed provenance output (FEED-1)", "outline §2 ¶2"),
        Num("feedWireShare", "wire or release bylines on W_clean (%)", "feed provenance output (FEED-1)", "outline §2 ¶2"),
        Num("feedOtherShare", "other bylines on W_clean (%)", "feed provenance output (FEED-1)"),
        Num("flagInstructPct", "clean W_clean items that the guard question flags as addressing automated readers "
            "(P(instructs_reader) >= .5; %; flag rate, not a verified label)", "results/feed/guard_flags.json (FEED-1 + MAIN N1)",
            "draft-0 review www M5"),
        Num("flagPromoPct", "clean W_clean items that the guard question flags as promotional (P(promotional) >= .5; %)",
            "results/feed/guard_flags.json (FEED-1 + MAIN N1)", "draft-0 review www M5"),
    ]),
    Group("Results: interface conformance (§4, App. G)", [
        Num("confJointAgree", "emu: argmax agreement of joint vs single-question answers (%)", "results/fid/emu/summary.json (IF-3 V1)"),
        Num("confReaskFloor", "emu: argmax labels changed by re-asking an identical request (%)", "results/fid/emu/summary.json (IF-3 V2)"),
        Num("confOrderShift", "emu: argmax labels changed by a cyclic permutation of option content (%)",
            "results/fid/emu/summary.json (LA-3)"),
        Num("confValidity", "emu: valid (non-degraded) answers (%)", "results/fid/emu/summary.json (IF-3 V4)"),
    ]),
    Group("Results: operating profile (§5.1, Table 1)", [
        Num("costPerStockWeek", "$ per stock-week (x 10^-3)", "requests.parquet (OPS-1)"),
        Num("latencyPNinetyFive", "p95 latency per request (s)", "requests.parquet (OPS-1)"),
        Num("mainWallClock", "MAIN wall-clock (h)", "MANIFEST.json (OPS-1)"),
        Num("reevalIdentical", "fresh re-evaluation runs with identical output hashes", "results/ops/replay_check.json (OPS-3)"),
        Num("reevalRuns", "fresh re-evaluation runs", "results/ops/replay_check.json (OPS-3)"),
        Num("scalingRequests", "requests per week per additional 10 names", "results/ops/scaling.json (OPS-2)"),
        Num("scalingCost", "$ per week per additional 10 names", "results/ops/scaling.json (OPS-2)"),
        Num("deployShortfall", "implementation shortfall in the deployment check (bps)", "results/live/<t>/deployment.json"),
        Num("deployLatency", "cutoff-to-order latency (h)", "results/live/<t>/deployment.json"),
        Num("deployCost", "$ per rebalance in the deployment check", "results/live/<t>/deployment.json"),
    ]),
    Group("Results: bound, tightness and price (§5.2, Fig. 2)", [
        Num("maxEntrantRank", "largest trusted rank among successful entries (vs K + J)", "results/bound/ (BG-6)"),
        Num("maxLogitRatio", "largest |delta logit| / 2 kappa_t", "results/bound/ (BG-6)"),
        Num("maxEntrantWeightRatio", "largest entrant weight / trusted target", "results/bound/ (BG-6)"),
        # Rank-IC is computed on L, which does not depend on J (pipeline/layer_b.py::prediction_metrics), so the band's
        # price is measured on the signal book's clean selection return (draft-0 review www M9; paper §4).
        Num("cleanCostAnchor", "clean change in the system book's mean weekly excess return from anchor-gated "
            "exclusion (bps)", "results/bound/clean_costs.json (BG-3c)", "Fig. 2 caption"),
        Num("cleanCostSizing", "clean change in the system book's mean weekly excess return from reference-anchored "
            "sizing (bps)", "results/bound/clean_costs.json (V-size-v1)", "Fig. 2 caption"),
        Num("valueText", "value of the text: paired mean weekly selection return of the signal book, J = inf minus "
            "J = 0 (bps per week)", "results/eval/frontier.json (V-J0, V-Jinf replays)", "www M9"),
        Num("valueTextCI", "two-sided Newey-West(4) 95% CI of valueText (formatted [lo, hi]; frontier.py paired_bps -> "
            "stats.nw_test; not a week-block bootstrap; audit A6)", "results/eval/frontier.json", "www M9"),
        Num("priceBand", "price of the band at the operating point: selection return at J = inf minus J (bps per week)",
            "results/eval/frontier.json", "www M9"),
        Num("priceSharePct", "priceBand as a share of valueText (%)", "results/eval/frontier.json", "www M9"),
        Num("asrOperating", "in-band ASR at the operating point, question programs (%)", "results/redteam/rts/<reader>/<variant>/asr.json"),
        Num("asrNoBand", "ASR without the band, question programs (%)", "results/redteam/rts/<reader>/<variant>/asr.json"),
        Num("budgetUseTyped", "largest share of the Corollary's displacement budget D_t used by an attack, question "
            "programs (%)", "results/bound/budget_use.json", "www M1"),
        Num("budgetUseHolistic", "largest share of D_t used by an attack, holistic reader (%)",
            "results/bound/budget_use.json", "www M1"),
        Num("budgetUseMax", "largest share of D_t used by an attack over all readers (%)", "results/bound/budget_use.json",
            "www M1"),
        Num("anchorRateProtected", "protected stock-weeks on W_clean with a trusted anchor A = 1 (%)",
            "results/bound/anchor_rates.json", "www M4"),
        Num("asrAnchorStratum", "exclusion ASR in the anchor stratum of protected holdings (%)",
            "results/redteam/rts/<reader>/demotion/asr.json (RB-2)", "www M4"),
        Num("asrReachable", "in-band ASR with attacks under reachable bylines only, question programs (%)",
            "results/redteam/rts/<reader>/<variant>/asr_reachable.json", "www M5"),
    ]),
    Group("Results: typed interface, validity and signal (§5.3-§5.4)", [
        Num("hOneDelta", "H1 paired ASR difference (typed - holistic)", "results/c1/summary.json", "Table 2 caption"),
        Num("hOneCI", "H1 95% CI (formatted [lo, hi])", "results/c1/summary.json", "Table 2 caption"),
        Num("hOnePholm", "H1 Holm-adjusted p", "results/c1/summary.json", "Table 2 caption"),
        Num("hOneAdaptiveTyped", "adaptive ASR at the query budget, typed (%)", "results/redteam/rta/*/asr_budget.json"),
        Num("hOneAdaptiveHolistic", "adaptive ASR at the query budget, holistic (%)", "results/redteam/rta/*/asr_budget.json"),
        Num("hTwoCIEmu", "H2 95% CI, emu", "results/c2/emu/summary.json"),
        Num("hTwoPholm", "H2 Holm-adjusted p", "results/c2/emu/summary.json"),
        Num("hTwoPlaceboEmu", "H2 placebo Spearman (AR[-5,-1]), emu", "results/c2/emu/summary.json"),
        Num("hTwoSpearmanVerb", "H2 per implementation, VERB: POOLED Spearman over its 2,000 units with a week-block CI "
            "(descriptive; not the weekly mean of hTwoSpearmanEmu; audit A6)", "results/c2/verb/summary.json"),
        Num("hTwoSpearmanNG", "H2 per implementation, NG: POOLED Spearman over its 500 units with a week-block CI "
            "(descriptive; not the weekly mean of hTwoSpearmanEmu; audit A6)", "results/c2/ng/summary.json"),
        Num("hTwoSpearmanEmuPooled", "H2 per implementation, emu: POOLED Spearman on the full frame (the statistic "
            "of hTwoSpearmanVerb/NG; descriptive; DEV-9)", "results/c2/emu/summary.json"),
        Num("hTwoSpearmanPreOpen", "H2 weekly Spearman, emu, items created before 09:30 ET only (control)",
            "results/c2/emu/summary.json", "www M8"),
        Num("hTwoSpearmanRelease", "H2 weekly Spearman, emu, on W_release (sensitivity)", "results/c2/emu/summary.json",
            "www M10"),
        Num("hThreeDeltaICRelease", "H3 rank-IC increment on W_release (sensitivity)", "results/c3/summary.json", "www M10"),
        Num("icTextPart", "H3 split: IC(L) - IC(L^T), the untrusted text (descriptive)", "results/c3/summary.json", "www M9"),
        Num("icTrustedTypedPart", "H3 split: IC(L^T) - IC(code-only), the trusted typed judgments (descriptive)",
            "results/c3/summary.json", "www M9"),
        Num("hThreeCI", "H3 95% CI", "results/c3/summary.json"),
        Num("hThreePholm", "H3 Holm-adjusted p", "results/c3/summary.json"),
        Num("hThreePlaceboDeltaIC", "rank-IC increment over the coverage placebo", "results/c3/coverage_placebo.json"),
        Num("poolOracleShare", "share of the oracle gap captured by the learned pool (%)", "results/ma/pools.json"),
        Num("poolLogLossDiff", "PD2: weekly log-loss difference, V-EW-log minus the learned pool (negative = the "
            "learned pool has the higher loss; DEV-9, audit A5)", "results/ma/pools.json"),
        Num("poolLogLossDiffCI", "PD2: NW(4) 95% CI of poolLogLossDiff (formatted [lo, hi]; DEV-9, audit A5)",
            "results/ma/pools.json"),
    ]),
    Group("Results: traceability and portfolio (§5.5-§5.6)", [
        Num("ledgerResidual", "largest ledger sum residual", "cards.jsonl (T1 exactness)", "Fig. 3 caption"),
        Num("nTrials", "registered trials in the DSR", "results/trials.jsonl", "Table 4 caption"),
        Num("dsr", "deflated Sharpe ratio", "results/trials.jsonl (PE-8)", "Table 4 caption"),
        Num("dsrNhat", "deflated Sharpe ratio at the correlation-implied number of trials N-hat",
            "results/trials.jsonl (PE-8)", "Table 4 caption"),
        Num("randomPct", "Random-20 percentile of the system book", "results/baselines/random20.json", "Table 4 caption"),
    ]),
    Group("Statements (outline §8)", [
        Num("preregDigest", "SHA-256 of the frozen pre-registration", "configs/prereg_v1.yaml (after the freeze)",
            "reproducibility statement"),
        Num("preregTimestamp", "OpenTimestamps attestation of preregDigest: Bitcoin block height (made before W_clean is "
            "unlocked; no account)", "results/prereg/opentimestamps.json (block_height)", "www M10",
            get=lambda: (lambda v: None if v is None else fmt_num(int(v)))(
                json_value("results/prereg/opentimestamps.json", "block_height"))),
    ]),
]

CONTEXT_WINDOW = 400
LIT: list[Lit] = [
    # §1 ¶1 (positioning_v2 §1.3 item 1-2)
    Lit("litNestaasBefore", "nestaas2025adversarial", "p.6", "34.0", "34.0%", r"recommendation rate", "fake product's recommendation rate before (Bing Copilot, %)"),
    Lit("litNestaasAfter", "nestaas2025adversarial", "p.6", "59.4", "59.4%", r"recommendation rate", "... after preference manipulation (%)"),
    Lit("litGeoGain", "aggarwal2024geo", "p.6", "41", "41%", r"Position-Adjusted Word Count", "best GEO gain on position-adjusted word count (%)"),
    Lit("litNaJailbreak", "na2026poisoning", "p.4", "98.9", "98.9%", r"attackable days", "LLM risk-manager jailbreak success (% of attackable days)"),
    Lit("litRizvaniHomoglyph", "rizvani2026adversarial", "p.8", "3.67", "3.67 pp", r"homoglyph", "mean cumulative-return loss, homoglyph edits, FinBERT (pp)"),
    Lit("litRizvaniHidden", "rizvani2026adversarial", "p.8", "3.18", "3.18 pp", r"hidden text", "mean cumulative-return loss, hidden-text edits, FinBERT (pp)"),
    # always-negative attacker, Table 1: Delta-Sharpe is -5.501, -5.970, -6.167, -6.060 at p_poison 30/45/60/75%, so every
    # drop from 30% on exceeds 5.5 (the printed value is the smallest drop, rounded down; the text says "more than")
    Lit("litSuaPoisonShare", "sua2026contagion", "p.8", "30", "30%", r"p_poison", "attacker share of the social posts read, from which on every Sharpe drop exceeds litSuaSharpeMin (%)"),
    Lit("litSuaSharpeMin", "sua2026contagion", "p.8", "5.5", "−5.501", r"Sharpe", "smallest annualized Sharpe drop at an attacker share of 30% or more, rounded down"),
    # §1 ¶2
    Lit("litXuFlip", "xu2026jevout", "p.6", "61.4", "61.4%", r"initially correct", "initially correct jev-1.13.0 decisions redirected (%)"),
    # The paper calls the adaptive budget "the budget of a published attack"; generate() asserts maxAdaptiveQueries == this.
    Lit("litXuBudget", "xu2026jevout", "p.6", "64", "64 accepted", r"Budget", "accepted target evaluations per decision (budget)"),
    # §3 / §6 ¶3 defenses
    Lit("litCordonAsrBase", "yu2026cordon", "p.4", "27.5", "27.5%", r"Vanilla RAG", "mean ASR, vanilla RAG (%)"),
    Lit("litCordonAsr", "yu2026cordon", "p.4", "2.1", "2.1%", r"Cordon-MAS", "mean ASR, Cordon-MAS (%)"),
    Lit("litCordonCollusion", "yu2026cordon", "p.6-7", "70.3", "70.3%", r"collusion", "colluding documents bypass the audit (%)"),
    Lit("litShkolnikovConfigs", "shkolnikov2026composable", "p.3", "8{,}729", "8729", r"configurations", "configurations checked by enumeration"),
    Lit("litShkolnikovDefBase", "shkolnikov2026composable", "p.8", "27", "27%", r"defended rate", "held-out injection defended rate, base (%)"),
    Lit("litShkolnikovDef", "shkolnikov2026composable", "p.8", "94", "94%", r"defended rate", "held-out injection defended rate, cascade (%)"),
    Lit("litShkolnikovQrel", "shkolnikov2026composable", "p.8-9", "0.96", "0.96", r"Qrel", "clean-quality ratio Qrel"),
    # §4 / §5 evaluation
    Lit("litPathadeMDE", "pathade2026attack", "p.1", "18.2", "18.2", r"minimum detectable|minimum difference", "ASR MDE on a 100-instance benchmark (points)"),
    Lit("litPathadeInstances", "pathade2026attack", "p.1", "100", "100-instance", r"minimum", "benchmark size behind that MDE"),
    Lit("litPortbenchShare", "zhao2026portbench", "p.1", "32.5", "32.5% of 120", r"equal weight", "LLM evaluations beating equal weight (%)"),
    Lit("litPortbenchN", "zhao2026portbench", "p.1", "120", "32.5% of 120", r"equal weight", "evaluations in that count"),
    Lit("litLopezOvernight", "lopezlira2026chatgpt", "p.51", "93.28", "93.28%", r"overnight", "long-short hit rate, initial reaction, overnight (%)"),
    Lit("litLopezIntraday", "lopezlira2026chatgpt", "p.51", "88.78", "88.78%", r"intraday", "long-short hit rate, initial reaction, intraday (%)"),
    Lit("litKirtacBrier", "kirtac2026financial", "p.26", "0.151", "0.151", r"Brier", "LLaMA-3 Brier score"),
    Lit("litKirtacECE", "kirtac2026financial", "p.26", "0.032", "0.032", r"ECE", "LLaMA-3 ECE"),
    Lit("litGuoAUROC", "guo2026just", "p.2", "0.962", "0.962", r"AUROC", "per-type median AUROC, one typed injection question"),
    Lit("litGuoECE", "guo2026just", "p.3", "0.168", "0.168", r"ECE", "median ECE of those probabilities"),
    Lit("litKongShare", "kong2026evaluating", "p.2", "26.8", "26.8%", r"look-ahead", "studies acknowledging look-ahead bias (%)"),
    Lit("litXiaLookahead", "xia2026agentic", "p.39", "0", "N=0 primary studies", r"Look-ahead", "primary studies foregrounding look-ahead"),
    Lit("litXiaCosts", "xia2026agentic", "p.41", "1", "transaction-cost models: 1/19", r"transaction-cost", "studies with explicit cost models"),
    Lit("litXiaStudies", "xia2026agentic", "p.41", "19", "1/19", r"transaction-cost", "primary studies reviewed"),
    Lit("litFonsecaMiss", "fonseca2026lookahead", "p.25", "54.5", "54.5%", r"differential", "planted leaks missed by a two-run differential detector (%)"),
    # §6 ¶2 typed decision models
    Lit("litRafeSlope", "rafe2026calibrated", "p.25", "1.63", "slope 1.63", r"not calibrated as delivered", "Jev calibration slope as delivered"),
    Lit("litRafeZ", "rafe2026calibrated", "p.25", "\\ensuremath{-5.3}", "z −5.3", r"Spiegelhalter", "Jev Spiegelhalter z as delivered"),
    Lit("litLingProjects", "ling2026jev", "p.1", "2,170", "2,170", r"GitHub", "public GitHub projects using Jev within about a week"),
    Lit("litChengJevBrier", "cheng2026thisthat", "p.7", "0.133", "Brier 0.133", r"Jev", "Jev Brier score on a recorded cohort"),
    Lit("litChengCohort", "cheng2026thisthat", "p.7", "68", "68 questions", r"recorded", "questions in that cohort"),
    # K1 comparison rows
    Lit("litTradingAgentsCalls", "xiao2024tradingagents", "p.12", "11", "11 LLM calls", r"tool calls", "LLM calls per prediction"),
    Lit("litTradingAgentsTools", "xiao2024tradingagents", "p.12", "20+", "20+ tool calls", r"LLM calls", "tool calls per prediction"),
    Lit("litDeepFundSignals", "li2025time", "p.5", "96", "96% validity", r"signals", "valid analyst signals (%)"),
    Lit("litDeepFundDecisions", "li2025time", "p.5", "98", "98% validity", r"decisions", "valid trading decisions (%)"),
    Lit("litForesightBeaten", "luo2026foresight", "p.1", "31", "31 of 36", r"outperform", "agent runs beaten by equal weight and inverse volatility"),
    Lit("litForesightRuns", "luo2026foresight", "p.1", "36", "31 of 36", r"outperform", "agent runs"),
    Lit("litClqtNoise", "qu2026clqt", "p.27", "0.42", "±0.42", r"noise", "run-to-run Sharpe noise band (plus or minus)"),
    Lit("litClqtGapBacktest", "qu2026clqt", "p.34", "0.30", "+0.30 in backtest", r"Coherence gap", "stated-vs-executed gap, backtest"),
    Lit("litClqtGapLive", "qu2026clqt", "p.34", "0.23", "+0.23 live", r"Coherence gap", "stated-vs-executed gap, live"),
]

SENTENCES: list[Sentence] = [
    Sentence(
        "hOneSentence", "H1 result sentence", "results/c1/summary.json (Holm decision)",
        confirmed=(r"Per-item guarded decomposition lowers portfolio-level attack success from \hOneASRHolistic\% to "
                   r"\hOneASRTyped\% relative to the holistic reader under the same band and cap "
                   r"(Holm-adjusted $p = \hOnePholm$)."),
        # registered wording (prereg §3.1): the null sentence; the bound and localization clauses are added only when
        # their own outputs hold (bound.json via confirmatory.json claims.BOUND; localization.json claim)
        fallback=(r"At the tested attacker and budget, per-item guarded decomposition and the holistic reader did "
                  r"not differ detectably (MDE $\approx$ \hOneMDELow--\hOneMDEHigh)."),
        # DEV-9 (audit A1): the registration has no sentence for an estimate whose CI excludes 0 on the side opposite
        # to the claim; the null sentence would then be false. Descriptive: the registered test's result (its own
        # Holm-adjusted p, no new p-value), then the two ASRs and the paired estimate with its 95% CI.
        reversed=(r"At the tested attacker and budget, the pre-registered one-sided test that per-item guarded "
                  r"decomposition lowers attack success relative to the holistic reader was not supported "
                  r"(Holm-adjusted $p = \hOnePholm$); the estimate points the other way, with attack success of "
                  r"\hOneASRTyped\% for per-item guarded decomposition and \hOneASRHolistic\% for the holistic reader "
                  r"(difference \hOneDelta{}, \ciLevel\% CI \hOneCI{})."),
        ref="prereg §3.1; positioning_v2 §2.7 (reader name per draft-0 editor review §4); DEV-9"),
    Sentence(
        "hTwoSentence", "H2 result sentence", "results/c2/emu/summary.json (Holm decision)",
        # registered wording (prereg §3.2); the SEC clause is added only if h2.json's sec_claim holds
        confirmed=(r"Typed item judgments track announcement-window abnormal returns (weekly Spearman "
                   r"\hTwoSpearmanEmu{}, Holm-adjusted $p = \hTwoPholm$)."),
        fallback=(r"Validity was not detected at MDE $\approx$ \hTwoMDELow--\hTwoMDEHigh; SEC concept agreement is "
                  r"reported per implementation."),
        ref="positioning_v2 §2.7"),
    Sentence(
        "hThreeSentence", "H3 result sentence", "results/c3/summary.json (Holm decision)",
        confirmed=(r"Typed judgments add \hThreeDeltaIC{} rank-IC over an otherwise identical code-only system "
                   r"(Holm-adjusted $p = \hThreePholm$)."),
        # The old clause "the frontier shows what the band costs when the text adds little" is dropped: when the
        # text's value is not distinguishable from zero the frontier does not identify the price (draft-0 www M9).
        fallback=r"No increment was detected at MDE $\approx$ \hThreeMDELow--\hThreeMDEHigh{} IC.",
        ref="positioning_v2 §2.7, amended per draft-0 www M9"),
    Sentence(
        "boundSentence", "reach and price sentence (§1, §7)",
        "results/bound/budget_use.json; results/eval/frontier.json (does the valueText CI exclude zero?)",
        confirmed=(r"Successful attacks used at most \budgetUseTyped\% of the displacement budget of "
                   r"Corollary~\ref{cor:reach} with our question programs and \budgetUseHolistic\% with the holistic "
                   r"reader, and the band gave up \priceSharePct\% of the text's clean value of \valueText{} bps per "
                   r"week \valueTextCI{}."),
        fallback=(r"Successful attacks used at most \budgetUseTyped\% of the displacement budget of "
                  r"Corollary~\ref{cor:reach} with our question programs and \budgetUseHolistic\% with the holistic "
                  r"reader; the text's clean value, \valueText{} bps per week \valueTextCI{}, was not distinguishable "
                  r"from zero, so the frontier does not identify the band's price."),
        ref="draft-0 www B2, M1, M9"),
]


def _same_outcome(name: str) -> Callable[[], bool | None]:
    """An abstract clause follows the Holm decision of the full result sentence it shortens (never wired twice)."""
    def get():
        s = next(x for x in SENTENCES if x.name == name)
        return s.outcome() if s.outcome else None
    return get


# Clauses of the abstract's last sentence (positioning_v2 §6). A fallback is never longer than its confirmed clause, so
# the abstract stays within 200 words whatever the outcomes.
SENTENCES += [
    Sentence(
        "hOneAbstract", "H1 clause of the abstract",
        "results/analysis/confirmatory.json claims.H1 via hOneSentence (and h1.json confirmatory.ci for DEV-9)",
        confirmed=(r"per-item guarded decomposition lowers attack success from \hOneASRHolistic\% to \hOneASRTyped\% "
                   r"versus a holistic reader"),
        fallback=r"per-item guarded decomposition and a holistic reader do not differ detectably in attack success",
        reversed=(r"per-item guarded decomposition does not lower attack success versus a holistic reader "
                  r"(\hOneASRTyped\% vs.\ \hOneASRHolistic\%)"),
        outcome=_same_outcome("hOneSentence"), ref="positioning_v2 §6, §2.7; DEV-9"),
    Sentence(
        "boundAbstract", "reach and price clause of the abstract and conclusion",
        "results/analysis/frontier.json value_of_text.ci_excludes_zero via boundSentence",
        confirmed=(r"attacks use at most \budgetUseMax\% of the bound's reach, and the band gives up \priceSharePct\% of "
                   r"the text's value"),
        fallback=(r"attacks use at most \budgetUseMax\% of the bound's reach, and the text's value is not "
                  r"distinguishable from zero"),
        outcome=_same_outcome("boundSentence"), ref="draft-0 www B2, M9"),
    Sentence(
        "hTwoAbstract", "H2 clause of the abstract", "results/analysis/confirmatory.json claims.H2 via hTwoSentence",
        confirmed=r"item judgments track announcement-window abnormal returns (weekly Spearman \hTwoSpearmanEmu{})",
        fallback=r"item judgments do not detectably track announcement-window abnormal returns",
        outcome=_same_outcome("hTwoSentence"), ref="positioning_v2 §6, §2.7"),
    Sentence(
        "hThreeAbstract", "H3 clause of the abstract", "results/analysis/confirmatory.json claims.H3 via hThreeSentence",
        confirmed=r"the judgments add \hThreeDeltaIC{} rank-IC over code-only features",
        fallback=r"no rank-IC increment over code-only features is detected",
        outcome=_same_outcome("hThreeSentence"), ref="positioning_v2 §6, §2.7"),
    # DEV-9 (audit A5): PD2's claim-register wording (prereg §4.2) as a generated sentence. The registered retirement
    # text ("no detectable difference from equal-weight pooling") is false when the CI lies entirely below 0, so that
    # case gets a descriptive wording (estimate and CI only).
    Sentence(
        "poolSentence", "PD2 composition sentence (prereg §4.2 claim register)",
        "results/analysis/pd2.json claim_composition_adds",
        confirmed=(r"The learned pool has lower log loss than the equal-weight log pool (weekly difference, "
                   r"equal-weight minus learned, \poolLogLossDiff{}, \ciLevel\% CI \poolLogLossDiffCI{})."),
        fallback=(r"The learned pool shows no detectable difference from equal-weight pooling (weekly log-loss "
                  r"difference, equal-weight minus learned, \poolLogLossDiff{}, \ciLevel\% CI \poolLogLossDiffCI{})."),
        reversed=(r"The learned pool is detectably worse than the equal-weight log pool in log loss (weekly "
                  r"difference, equal-weight minus learned, \poolLogLossDiff{}, \ciLevel\% CI \poolLogLossDiffCI{})."),
        ref="prereg §4.2 PD2; DEV-9"),
]


# ---------------------------------------------------------------------------------------------------------------------
# Wiring to the pre-registered analysis outputs (experiments/analysis/*, prereg §14.3 P-8)
# ---------------------------------------------------------------------------------------------------------------------
# A result macro is filled only from a file written by its analysis entry point for the REGISTERED window (W_clean)
# with the lock open (the file's `meta` block says so); a development run (results/analysis/<window>/...) or a file
# written while the lock was closed never fills a macro. Protocol constants come from the pre-registration's own
# power output (results/prereg/power_prereg.json) and from the analysis code's module constants. Rounding of the
# registered MDEs is half-up on the decimal value, so the paper prints what the pre-registration text quotes.

ANALYSIS = "results/analysis"
POWER = "results/prereg/power_prereg.json"
PLAN = "results/plan/plan_v1.json"


def _walk(node, path):
    for part in path:
        if isinstance(node, list) and isinstance(part, int):
            node = node[part] if -len(node) <= part < len(node) else None
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def registered_result(name: str) -> dict | None:
    """results/analysis/<name>.json if it is a registered-window run made with the lock open, else None."""
    path = ROOT / ANALYSIS / f"{name}.json"
    if not path.is_file():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    meta = d.get("meta", {})
    if not meta.get("registered_window") or not (meta.get("lock") or {}).get("open"):
        return None
    return d


def half_up(x, decimals: int) -> str:
    from decimal import ROUND_HALF_UP, Decimal

    q = Decimal(str(x)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    return fmt_num(float(q), decimals)


def _finite(v) -> bool:
    import math

    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def res(name: str, *path, decimals: int = 3, scale: float = 1.0, integer: bool = False) -> Callable[[], str | None]:
    def get():
        v = _walk(registered_result(name), path)
        if not _finite(v):
            return None
        return fmt_num(int(round(v * scale))) if integer else half_up(v * scale, decimals)
    return get


def res_ci(name: str, *path, decimals: int = 3, scale: float = 1.0) -> Callable[[], str | None]:
    def get():
        v = _walk(registered_result(name), path)
        if not (isinstance(v, list) and len(v) == 2 and all(_finite(x) for x in v)):
            return None
        return f"[{half_up(v[0] * scale, decimals)}, {half_up(v[1] * scale, decimals)}]"
    return get


def fmt_p(p: float) -> str:
    """A p-value: three decimals, or m x 10^e (two significant digits) below 0.001, so that a p-value never prints
    as 0.000 (DEV-9, audit A4)."""
    if p >= 0.001:
        return half_up(p, 3)
    m, e = f"{p:.1e}".split("e")
    return f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"


def holm_p(label: str) -> Callable[[], str | None]:
    def get():
        fam = _walk(registered_result("confirmatory"), ("family",)) or []
        r = next((x for x in fam if x.get("hypothesis") == label), None)
        return None if r is None or not _finite(r.get("p_holm")) else fmt_p(r["p_holm"])
    return get


def h1_realized_mde(pid: float) -> Callable[[], str | None]:
    """DEV-6/DEV-9 (audit A2): H1's MDE by the REGISTERED formula (experiments/prereg/power_prereg.py h1_block,
    z_holm * sqrt(2 pi_d / n)) at the realized number of scored pairs (h1.json confirmatory.n_pairs)."""
    def get():
        n = _walk(registered_result("h1"), ("confirmatory", "n_pairs"))
        if not _finite(n) or n <= 0:
            return None
        import importlib.util

        spec = importlib.util.spec_from_file_location("power_prereg", ROOT / "experiments/prereg/power_prereg.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return half_up(mod.h1_block(int(n))[f"mde_pid{pid}"], 3)
    return get


def ci_opposite(name: str, ci_path: tuple, claim_side: str) -> Callable[[], bool | None]:
    """DEV-9: True iff the registered output's 95% CI lies entirely on the side OPPOSITE to the registered claim
    (claim_side "below": the claim is estimate < 0; "above": estimate > 0)."""
    def get():
        ci = _walk(registered_result(name), ci_path)
        if not (isinstance(ci, list) and len(ci) == 2 and all(_finite(x) for x in ci)):
            return None
        return bool(ci[0] > 0) if claim_side == "below" else bool(ci[1] < 0)
    return get


def power(*path, decimals: int = 3, integer: bool = False) -> Callable[[], str | None]:
    def get():
        p = ROOT / POWER
        if not p.is_file():
            return None
        v = _walk(json.loads(p.read_text(encoding="utf-8")), path)
        if not _finite(v):
            return None
        return fmt_num(int(v)) if integer else half_up(v, decimals)
    return get


def plan_window(prefix: str) -> Callable[[], str | None]:
    def get():
        p = ROOT / PLAN
        if not p.is_file():
            return None
        w = json.loads(p.read_text(encoding="utf-8")).get("windows_decisions", {})
        v = next((n for k, n in w.items() if k.split("(")[0] == prefix), None)
        return None if v is None else fmt_num(int(v))
    return get


def code_const(rel: str, name: str, scale: float = 1.0, decimals: int | None = None) -> Callable[[], str | None]:
    """A numeric module constant (`NAME = value`) of a code file, so the paper follows the analysis code."""
    def get():
        path = ROOT / rel
        if not path.is_file():
            return None
        m = re.search(rf"^{name}\s*=\s*([0-9_.]+)", path.read_text(encoding="utf-8"), re.MULTILINE)
        if not m:
            return None
        v = float(m.group(1).replace("_", "")) * scale
        return fmt_num(int(round(v))) if decimals is None else fmt_num(round(v, decimals), decimals)
    return get


def n_impl() -> str | None:
    impl = _walk(registered_result("h2"), ("per_implementation",)) or {}
    n = sum(1 for v in impl.values() if isinstance(v, dict) and v.get("status") == "ok")
    return fmt_num(n) if n else None


def sci(name: str, *path) -> Callable[[], str | None]:
    """A small positive number as m x 10^e (ledger residuals)."""
    def get():
        d = None
        p = ROOT / ANALYSIS / f"{name}.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
        if not d or not d.get("registered"):
            return None
        v = _walk(d, path)
        if not _finite(v):
            return None
        if v == 0:
            return "0"
        m, e = f"{v:.0e}".split("e")
        return f"\\ensuremath{{{int(m)}\\times 10^{{{int(e)}}}}}"
    return get


A = "experiments/analysis"
WIRING: dict[str, tuple[str, Callable[[], str | None]]] = {
    # protocol constants of the pre-registration
    "universeSize": ("src/jevtrader/pipeline/inputs.py UNIVERSE_TOP", code_const("src/jevtrader/pipeline/inputs.py",
                                                                                 "UNIVERSE_TOP")),
    "nPromotionUnits": (f"{POWER} H1.registered.units", power("H1", "registered", "units", integer=True)),
    # DEV-9 (audit A2): the realized H1 frame; the registered values keep their own macros
    "nPairedAttacks": (f"{ANALYSIS}/h1.json confirmatory.n_pairs (realized; DEV-6)",
                       res("h1", "confirmatory", "n_pairs", integer=True)),
    "nPairedAttacksRegistered": (f"{POWER} H1.registered.pairs", power("H1", "registered", "pairs", integer=True)),
    "hOneMDELow": ("power_prereg.h1_block(h1.json confirmatory.n_pairs)[mde_pid0.2] (realized; DEV-6)",
                   h1_realized_mde(0.2)),
    "hOneMDEHigh": ("power_prereg.h1_block(h1.json confirmatory.n_pairs)[mde_pid0.4] (realized; DEV-6)",
                    h1_realized_mde(0.4)),
    "hOneMDELowRegistered": (f"{POWER} H1.registered.mde_pid0.2", power("H1", "registered", "mde_pid0.2")),
    "hOneMDEHighRegistered": (f"{POWER} H1.registered.mde_pid0.4", power("H1", "registered", "mde_pid0.4")),
    "hTwoMDELow": (f"{POWER} H2.registered_W_clean.mde_between_sd0.0_units",
                   power("H2", "registered_W_clean", "mde_between_sd0.0_units")),
    "hTwoMDEHigh": (f"{POWER} H2.registered_W_clean.mde_between_sd0.05_focal_d0",
                    power("H2", "registered_W_clean", "mde_between_sd0.05_focal_d0")),
    "hThreeMDELow": (f"{POWER} H3.registered_W_clean.sigma_0.12.mde_rho0.8",
                     power("H3", "registered_W_clean", "sigma_0.12", "mde_rho0.8")),
    "hThreeMDEHigh": (f"{POWER} H3.registered_W_clean.sigma_0.2418.mde_rho0.8",
                      power("H3", "registered_W_clean", "sigma_0.2418", "mde_rho0.8")),
    "irMDE": (f"{POWER} descriptive.portfolio_IR_mde", power("descriptive", "portfolio_IR_mde", decimals=2)),
    "nwLag": (f"{A}/stats.py NW_LAGS", code_const(f"{A}/stats.py", "NW_LAGS")),
    "ciLevel": (f"{A}/stats.py LEVEL", code_const(f"{A}/stats.py", "LEVEL", 100.0)),
    "familyAlpha": (f"{A}/confirmatory.py ALPHA", code_const(f"{A}/confirmatory.py", "ALPHA", 1.0, 2)),
    "bufferWeeks": (f"{PLAN} windows_decisions", plan_window("W_buffer")),
    "nWeeksRelease": (f"{PLAN} windows_decisions", plan_window("W_release")),
    "nWeeks": (f"{POWER} windows.W_clean_decisions", power("windows", "W_clean_decisions", integer=True)),
    # results (registered-window files only)
    "nImpl": (f"{ANALYSIS}/h2.json per_implementation (status ok)", n_impl),
    "hOneASRHolistic": (f"{ANALYSIS}/h1.json confirmatory.asr_b", res("h1", "confirmatory", "asr_b", decimals=1,
                                                                      scale=100)),
    "hOneASRTyped": (f"{ANALYSIS}/h1.json confirmatory.asr_a", res("h1", "confirmatory", "asr_a", decimals=1,
                                                                   scale=100)),
    "hOneDelta": (f"{ANALYSIS}/h1.json confirmatory.estimate", res("h1", "confirmatory", "estimate")),
    "hOneCI": (f"{ANALYSIS}/h1.json confirmatory.ci", res_ci("h1", "confirmatory", "ci")),
    "hOnePholm": (f"{ANALYSIS}/confirmatory.json family[H1].p_holm", holm_p("H1")),
    "hTwoSpearmanEmu": (f"{ANALYSIS}/h2.json confirmatory.mean", res("h2", "confirmatory", "mean")),
    "hTwoCIEmu": (f"{ANALYSIS}/h2.json confirmatory.ci", res_ci("h2", "confirmatory", "ci")),
    "hTwoPholm": (f"{ANALYSIS}/confirmatory.json family[H2].p_holm", holm_p("H2")),
    "hTwoPlaceboEmu": (f"{ANALYSIS}/h2.json placebo_ar_m5_m1.mean", res("h2", "placebo_ar_m5_m1", "mean")),
    "hTwoSpearmanVerb": (f"{ANALYSIS}/h2.json per_implementation.VERB.spearman_ar.rho",
                         res("h2", "per_implementation", "VERB", "spearman_ar", "rho")),
    "hTwoSpearmanNG": (f"{ANALYSIS}/h2.json per_implementation.NG.spearman_ar.rho",
                       res("h2", "per_implementation", "NG", "spearman_ar", "rho")),
    "hTwoSpearmanEmuPooled": (f"{ANALYSIS}/h2.json per_implementation.emu.spearman_ar_pooled.rho",
                              res("h2", "per_implementation", "emu", "spearman_ar_pooled", "rho")),
    "hTwoSpearmanPreOpen": (f"{ANALYSIS}/h2.json controls.pre_open_only.mean",
                            res("h2", "controls", "pre_open_only", "mean")),
    "hTwoSpearmanRelease": (f"{ANALYSIS}/h2.json sensitivities.W_release.mean",
                            res("h2", "sensitivities", "W_release", "mean")),
    "hThreeDeltaIC": (f"{ANALYSIS}/h3.json confirmatory.mean", res("h3", "confirmatory", "mean")),
    "hThreeCI": (f"{ANALYSIS}/h3.json confirmatory.ci", res_ci("h3", "confirmatory", "ci")),
    "hThreePholm": (f"{ANALYSIS}/confirmatory.json family[H3].p_holm", holm_p("H3")),
    "hThreeDeltaICRelease": (f"{ANALYSIS}/h3.json sensitivities.W_release.mean",
                             res("h3", "sensitivities", "W_release", "mean")),
    "icTextPart": (f"{ANALYSIS}/h3.json split.text_part_IC_L_minus_IC_LT.mean",
                   res("h3", "split", "text_part_IC_L_minus_IC_LT", "mean")),
    "icTrustedTypedPart": (f"{ANALYSIS}/h3.json split.trusted_typed_part_IC_LT_minus_IC_codeonly.mean",
                           res("h3", "split", "trusted_typed_part_IC_LT_minus_IC_codeonly", "mean")),
    "hThreePlaceboDeltaIC": (f"{ANALYSIS}/h3.json coverage_placebo.delta_ic_over_placebo.mean",
                             res("h3", "coverage_placebo", "delta_ic_over_placebo", "mean")),
    "poolOracleShare": (f"{ANALYSIS}/pd2.json oracle_gap_share.primary.share",
                        res("pd2", "oracle_gap_share", "primary", "share", decimals=0, scale=100)),
    "poolLogLossDiff": (f"{ANALYSIS}/pd2.json paired.primary_vs_V-EW-log.delta_log_loss_b_minus_a.mean",
                        res("pd2", "paired", "primary_vs_V-EW-log", "delta_log_loss_b_minus_a", "mean", decimals=5)),
    "poolLogLossDiffCI": (f"{ANALYSIS}/pd2.json paired.primary_vs_V-EW-log.delta_log_loss_b_minus_a.ci",
                          res_ci("pd2", "paired", "primary_vs_V-EW-log", "delta_log_loss_b_minus_a", "ci",
                                 decimals=5)),
    "costPerStockWeek": (f"{ANALYSIS}/pd0.json typed.ledger.usd_per_stock_week (x 10^-3)",
                         res("pd0", "typed", "ledger", "usd_per_stock_week", decimals=2, scale=1e3)),
    "costPerRebalance": (f"{ANALYSIS}/pd0.json typed.ledger.usd_per_week",
                         res("pd0", "typed", "ledger", "usd_per_week", decimals=2)),
    "latencyPNinetyFive": (f"{ANALYSIS}/pd0.json typed.total.latency_p95_s",
                           res("pd0", "typed", "total", "latency_p95_s", decimals=2)),
    "mainWallClock": (f"{ANALYSIS}/pd0.json typed.ledger.wall_clock_h",
                      res("pd0", "typed", "ledger", "wall_clock_h", decimals=1)),
    "reevalIdentical": (f"{ANALYSIS}/pd0.json replay_determinism.identical",
                        res("pd0", "replay_determinism", "identical", integer=True)),
    "reevalRuns": (f"{ANALYSIS}/pd0.json replay_determinism.runs", res("pd0", "replay_determinism", "runs",
                                                                       integer=True)),
    "scalingRequests": (f"{ANALYSIS}/pd0.json scaling.requests_per_week_per_10_names",
                        res("pd0", "scaling", "requests_per_week_per_10_names", integer=True)),
    "scalingCost": (f"{ANALYSIS}/pd0.json scaling.usd_per_week_per_10_names",
                    res("pd0", "scaling", "usd_per_week_per_10_names", decimals=3)),
    "maxEntrantRank": (f"{ANALYSIS}/h1.json tightness.P1.max_R_T_entry",
                       res("h1", "tightness", "P1", "max_R_T_entry", integer=True)),
    "maxLogitRatio": (f"{ANALYSIS}/h1.json tightness.P4.max_dL_over_2kappa",
                      res("h1", "tightness", "P4", "max_dL_over_2kappa", decimals=2)),
    "maxEntrantWeightRatio": (f"{ANALYSIS}/h1.json tightness.P3.max_w_over_bound",
                              res("h1", "tightness", "P3", "max_w_over_bound", decimals=2)),
    "cleanCostAnchor": (f"{ANALYSIS}/frontier.json clean_costs.anchor_rule.mean_bps",
                        res("frontier", "clean_costs", "anchor_rule", "mean_bps", decimals=1)),
    "cleanCostSizing": (f"{ANALYSIS}/frontier.json clean_costs.reference_sizing.mean_bps",
                        res("frontier", "clean_costs", "reference_sizing", "mean_bps", decimals=1)),
    "valueText": (f"{ANALYSIS}/frontier.json value_of_text.mean_bps",
                  res("frontier", "value_of_text", "mean_bps", decimals=1)),
    "valueTextCI": (f"{ANALYSIS}/frontier.json value_of_text.ci", res_ci("frontier", "value_of_text", "ci",
                                                                         decimals=1)),
    "priceBand": (f"{ANALYSIS}/frontier.json price_of_band.mean_bps",
                  res("frontier", "price_of_band", "mean_bps", decimals=1)),
    "priceSharePct": (f"{ANALYSIS}/frontier.json price_of_band.share_of_value",
                      res("frontier", "price_of_band", "share_of_value", decimals=0, scale=100)),
    "asrOperating": (f"{ANALYSIS}/frontier.json readers.typed.J.10.asr.asr",
                     res("frontier", "readers", "typed", "J", "10", "asr", "asr", decimals=1, scale=100)),
    "asrNoBand": (f"{ANALYSIS}/frontier.json readers.typed.J.inf.asr.asr",
                  res("frontier", "readers", "typed", "J", "inf", "asr", "asr", decimals=1, scale=100)),
    "ledgerResidual": (f"{ANALYSIS}/fig3_card.json ledger_residual (registered selection)",
                       sci("fig3_card", "ledger_residual")),
    "nTrials": (f"{ANALYSIS}/pd1.json dsr.n_trials", res("pd1", "dsr", "n_trials", integer=True)),
    "dsr": (f"{ANALYSIS}/pd1.json dsr.by_book.sized.raw_M.dsr",
            res("pd1", "dsr", "by_book", "sized", "raw_M", "dsr", decimals=2)),
    "dsrNhat": (f"{ANALYSIS}/pd1.json dsr.by_book.sized.n_hat.dsr",
                res("pd1", "dsr", "by_book", "sized", "n_hat", "dsr", decimals=2)),
    "nBoundCases": (f"{ANALYSIS}/bound.json bg5.n_cases", res("bound", "bg5", "n_cases", integer=True)),
    "nBoundViolations": (f"{ANALYSIS}/bound.json bg5.n_violations",
                         res("bound", "bg5", "n_violations", integer=True)),
    "localizationRate": (f"{ANALYSIS}/localization.json by_reader.typed.rate",
                         res("localization", "by_reader", "typed", "rate", decimals=0, scale=100)),
    "randomPct": (f"{ANALYSIS}/pd1.json random20.percentile_of_book.sized",
                  res("pd1", "random20", "percentile_of_book", "sized", decimals=0, scale=100)),
}


def claim(label: str) -> Callable[[], bool | None]:
    """The Holm decision of confirmatory.json (registered run only)."""
    def get():
        v = _walk(registered_result("confirmatory"), ("claims", label))
        return v if isinstance(v, bool) else None
    return get


def bound_outcome() -> bool | None:
    v = _walk(registered_result("frontier"), ("value_of_text", "ci_excludes_zero"))
    return v if isinstance(v, bool) else None


def flag(name: str, *path) -> Callable[[], bool | None]:
    """A boolean of a registered, lock-open analysis output (None when the file or the key is missing)."""
    def get():
        v = _walk(registered_result(name), path)
        return v if isinstance(v, bool) else None
    return get


CLAUSE_WIRING = {
    "hOneSentence": ("fallback", [
        Clause(r"; the influence bound capped the damage for every reader", claim("BOUND"),
               f"{ANALYSIS}/confirmatory.json claims.BOUND (bound.json: BG-5 0 violations and the property tests)"),
        Clause(r", and only per-item readers localized the planted item",
               flag("localization", "claim", "per_item_readers_localize"),
               f"{ANALYSIS}/localization.json claim.per_item_readers_localize")]),
    "hTwoSentence": ("confirmed", [
        Clause(r" and agree with SEC item codes on each implementation",
               flag("h2", "sec_claim", "each_implementation_auc_ci_above_0.5"),
               f"{ANALYSIS}/h2.json sec_claim (every implementation's 8-K AUC CIs above 0.5)")]),
}


SENTENCE_WIRING = {"poolSentence": (f"{ANALYSIS}/pd2.json claim_composition_adds", flag("pd2", "claim_composition_adds")),
                   "hOneSentence": (f"{ANALYSIS}/confirmatory.json claims.H1", claim("H1")),
                   "hTwoSentence": (f"{ANALYSIS}/confirmatory.json claims.H2", claim("H2")),
                   "hThreeSentence": (f"{ANALYSIS}/confirmatory.json claims.H3", claim("H3")),
                   "boundSentence": (f"{ANALYSIS}/frontier.json value_of_text.ci_excludes_zero", bound_outcome)}


# DEV-9: when the reversed (descriptive) wording replaces a fallback (see Sentence.reversed)
REVERSAL_WIRING = {
    "hOneSentence": (f"{ANALYSIS}/h1.json confirmatory.ci lies above 0 (claim: typed - holistic < 0)",
                     ci_opposite("h1", ("confirmatory", "ci"), "below")),
    "hOneAbstract": (f"{ANALYSIS}/h1.json confirmatory.ci lies above 0 (as hOneSentence)",
                     ci_opposite("h1", ("confirmatory", "ci"), "below")),
    "poolSentence": (f"{ANALYSIS}/pd2.json paired.primary_vs_V-EW-log.delta_log_loss_b_minus_a.ci lies below 0 "
                     "(claim: above 0)",
                     ci_opposite("pd2", ("paired", "primary_vs_V-EW-log", "delta_log_loss_b_minus_a", "ci"), "above")),
}


def _wire() -> None:
    known = {n.name: n for g in GROUPS for n in g.items}
    for name, (source, getter) in WIRING.items():
        n = known[name]  # a KeyError here means the wiring names a macro that does not exist
        if n.get is None:
            n.get, n.source = getter, source
    sents = {s.name: s for s in SENTENCES}
    for name, (source, outcome) in SENTENCE_WIRING.items():
        s = sents[name]
        if s.outcome is None:
            s.outcome, s.source = outcome, source
    for name, (_source, rev) in REVERSAL_WIRING.items():
        s = sents[name]
        if s.reversal is None:
            s.reversal = rev
    for name, (which, clauses) in CLAUSE_WIRING.items():
        s = sents[name]
        target = s.confirmed_clauses if which == "confirmed" else s.fallback_clauses
        if not target:
            target.extend(clauses)


_wire()


# ---------------------------------------------------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------------------------------------------------


def verify_lit(item: Lit) -> str | None:
    """None if the value is supported by the summary; otherwise the reason."""
    path = SUMMARIES / f"{item.key}.md"
    if not path.is_file():
        return f"no summary {path.relative_to(ROOT)}"
    text = path.read_text(encoding="utf-8")
    ctx = re.compile(item.context, re.IGNORECASE)
    hits = [m.start() for m in re.finditer(re.escape(item.verify), text)]
    if not hits:
        return f"'{item.verify}' does not occur in the summary"
    for i in hits:
        if ctx.search(text[max(0, i - CONTEXT_WINDOW) : i + len(item.verify) + CONTEXT_WINDOW]):
            return None
    return f"'{item.verify}' occurs, but not within {CONTEXT_WINDOW} characters of /{item.context}/"


def placeholder(name: str) -> str:
    return f"\\{PLACEHOLDER}{{{name}}}"


def generate() -> tuple[str, dict, list[str]]:
    lines = [
        "% paper/generated/numbers.tex -- GENERATED by experiments/report/make_numbers.py. Do not edit by hand.",
        "% Every number in paper/sections/*.tex is one of these macros (outline_v1 §5; scripts/check_numbers.py).",
        "% Units are written in the text (\\hOneASRTyped\\%); write \\nWeeks{} when a space follows.",
        "% A placeholder prints as a bold [name] until its source exists.",
        f"\\providecommand{{\\{PLACEHOLDER}}}[1]{{\\textbf{{[#1]}}}}",
    ]
    stats = {"filled": 0, "placeholder": 0, "lit_ok": 0, "lit_failed": 0}
    problems: list[str] = []
    names: set[str] = set()

    def emit(name: str, value: str | None, comment: str) -> None:
        if name in names:
            raise SystemExit(f"make_numbers: duplicate macro name {name}")
        if re.search(r"[^A-Za-z]", name):
            raise SystemExit(f"make_numbers: macro name must be letters only: {name}")
        names.add(name)
        lines.append(f"% {comment}")
        if value is None:
            stats["placeholder"] += 1
            value = placeholder(name)
        else:
            stats["filled"] += 1
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for group in GROUPS:
        lines += ["", f"% ===== {group.title} ====="]
        for n in group.items:
            try:
                value = n.get() if n.get else None
            except Exception as exc:  # noqa: BLE001 -- a malformed source must never print a wrong number
                problems.append(f"{n.name}: getter failed ({exc}); left as a placeholder")
                value = None
            ref = f" [{n.ref}]" if n.ref else ""
            emit(n.name, value, f"{n.desc}{ref} | source: {n.source}")

    lines += ["", "% ===== Numbers quoted from the literature (checked against literature/summaries/<key>.md) ====="]
    for item in LIT:
        why = verify_lit(item)
        if why:
            problems.append(f"{item.name} ({item.key}, {item.page}): {why}")
            stats["lit_failed"] += 1
            emit(item.name, None, f"{item.desc} | {item.key}, {item.page} | NOT VERIFIED: {why}")
        else:
            stats["lit_ok"] += 1
            emit(item.name, item.value, f"{item.desc} | [{item.key}, {item.page}] | verified: \"{item.verify}\"")

    # Consistency: the text calls the adaptive budget "the budget of a published attack" (xu2026jevout, p.6).
    budget = next((n for g in GROUPS for n in g.items if n.name == "maxAdaptiveQueries"), None)
    xu = next((x for x in LIT if x.name == "litXuBudget"), None)
    if budget is not None and budget.get is not None and xu is not None:
        try:
            val = budget.get()
        except Exception:  # noqa: BLE001
            val = None
        if val is not None and val != xu.value:
            problems.append(f"maxAdaptiveQueries = {val} differs from the published budget litXuBudget = {xu.value}; "
                            "the paper's phrase 'the budget of a published attack' would be false")

    lines += ["", "% ===== Result sentences (pre-registered fallback wording when a test is null) ====="]
    for s in SENTENCES:
        outcome = s.outcome() if s.outcome else None
        value = None if outcome is None else s.compose(outcome)
        lines.append(f"% if confirmed: {s.confirmed}")
        for c in s.confirmed_clauses:
            lines.append(f"%   + only if {c.source}: {c.text}")
        lines.append(f"% if null ({s.ref}): {s.fallback}")
        if s.reversed:
            lines.append(f"% if null and the CI lies on the side opposite to the claim (DEV-9): {s.reversed}")
        for c in s.fallback_clauses:
            lines.append(f"%   + only if {c.source}: {c.text}")
        emit(s.name, value, f"{s.desc} | source: {s.source}")
    return "\n".join(lines) + "\n", stats, problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="do not write; exit 1 if numbers.tex is stale")
    ap.add_argument("--list", action="store_true", help="print every macro with its status and source (no write)")
    args = ap.parse_args(argv)

    text, stats, problems = generate()
    if args.list:
        for m in re.finditer(r"^% (.*)\n\\newcommand\{\\(\w+)\}\{(.*)\}$", text, re.MULTILINE):
            status = "PLACEHOLDER" if PLACEHOLDER in m.group(3) else m.group(3)
            print(f"\\{m.group(2):<24} {status:<34.34} {m.group(1)[:110]}")
        for p in problems:
            print(f"  ERROR {p}", file=sys.stderr)
        return 1 if problems else 0
    for p in problems:
        print(f"  ERROR {p}", file=sys.stderr)
    summary = (f"{stats['filled']} filled, {stats['placeholder']} placeholders "
               f"(literature: {stats['lit_ok']} verified, {stats['lit_failed']} failed)")
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.is_file() else ""
        if current != text:
            print("make_numbers: paper/generated/numbers.tex is stale (rerun the script)", file=sys.stderr)
            return 1
        print(f"make_numbers: OK, up to date; {summary}")
        return 1 if problems else 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"make_numbers: wrote {OUT.relative_to(ROOT)}: {summary}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
