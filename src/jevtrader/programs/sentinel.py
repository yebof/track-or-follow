"""Sentinel inside N1 (design_v1 §5.2, amended by design_v2_delta SAN-5 and BG-3; bank QB-4): no extra requests.

Per stock-week, from the N1 answers of the week's items:
- h_i = max P(`serious_problem`) over the items about the focal firm (P(about_focal) >= .5) that are not quarantined.
- k* = the item attaining h_i (ties: earliest `created_at`, then evidence id). Its `problem_kind` top label and
  q_i = P(`problem_unresolved`) are read only when the absolute Noul fires (P(serious_problem) >= .5, the bank's
  `sentinel.escalate`): the gate-then-categorize pattern (failure mode 8 of the Jev model-jaggedness page). k*'s label
  is kept for cards (`k_star_problem_kind`).
- kappa_i (`problem_kind`, the kind the risk officer's exclusion rule reads) is the top label of the most serious
  corroborating item (below) when there is one, else k*'s. So an item that can never corroborate (promotional, not
  kept) cannot choose the kind of a corroborated problem: with two honest keys of a non-exclusion kind, one planted
  promotional item of an exclusion kind no longer turns the halving into an exclusion (audit of 2026-09-27).
- Corroboration n^prob_i = the number of distinct **provenance keys** among kept, non-promotional, non-quarantined
  items with P(serious_problem) >= .8 (`sentinel.high`). A provenance key is the item's author byline after NFKC,
  lower-casing and whitespace collapse (bank `code_constants.corroboration.unit`; one definition,
  `data.news.provenance_key`, shared with the FEED-1 table). Two items under one byline therefore never count twice,
  however many copies an attacker posts; every item without a byline has the one key "" (`NO_BYLINE`). The
  sensitivity `n_prob_collapsed` collapses the bank's desk bylines into one key; `n_prob_items` counts items (the
  design_v1 item rule, for the BG-3c replay). Keys leave the news program only as `provenance_id`s ("key:" + 16 hex
  of the key's SHA-256) with a desk flag: a byline is untrusted text and never reaches M-work, cards or risk.parquet
  (the id is reproducible from the archive's `author` for audits).

The rows feed the risk officer (WS-B `pipeline/inputs.load_risk_table` and `decide/risk.py`: h, n_prob,
n_prob_collapsed, n_prob_items, problem_kind). They are written to the M-work blackboard as `N01.sentinel.<symbol>`
and exported by `write_risk_table(gen_dir)` to `risk.parquet`; h also goes to features.parquet (catalog name `h`,
untrusted channel). k* and its `created_at` are kept for cards and audits (the price anchor of BG-3 deliberately
does not use them: features/anchors.py takes every session of the week).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from jevtrader.data.news import provenance_key as _news_provenance_key
from jevtrader.programs.bank_v1 import load_bank

ESCALATE = 0.5  # code_constants.thresholds["sentinel.escalate"]: the Noul "fires"; kappa and q are read
HIGH = 0.8  # code_constants.thresholds["sentinel.high"]: single-cluster demotion and corroboration threshold
ABOUT = 0.5  # N1.keep / sentinel: P(about_focal) >= .5
NO_BYLINE = ""  # the one key of every item without an author (data.news.provenance_key; never several sources)
DESK_KEY = "(desk bylines)"  # the sensitivity's single key for all desk bylines (QB-4)
BLACKBOARD_PREFIX = "N01.sentinel."
RISK_COLUMNS = ["t", "symbol", "h", "n_prob", "n_prob_collapsed", "n_prob_items", "problem_kind", "p_problem_kind",
                "k_star_problem_kind", "q_unresolved", "k_star", "k_star_created_at", "n_about", "prov_keys"]


def desk_bylines() -> frozenset[str]:
    return frozenset(load_bank().code_constants["corroboration"]["desk_bylines"])


def provenance_key(author) -> str:
    """Provenance key of an item (SAN-5; bank QB-4): its author byline after NFKC, lower-casing and whitespace
    collapse (`data.news.provenance_key`, the one definition). A missing or empty byline is `NO_BYLINE`."""
    return _news_provenance_key(author)


def provenance_id(author) -> str:
    """The stored form of a provenance key: "key:" + 16 hex characters of its SHA-256. Distinct keys stay distinct
    (collisions are negligible at feed scale), and no byline text is stored."""
    return "key:" + hashlib.sha256(provenance_key(author).encode()).hexdigest()[:16]


def collapse_desk(key: str, desk: Iterable[str] | None = None) -> str:
    """The sensitivity key: every desk byline becomes one key (QB-4 `sensitivity`)."""
    return DESK_KEY if key in (desk_bylines() if desk is None else frozenset(desk)) else key


@dataclass(frozen=True)
class SentinelItem:
    """What the sentinel needs from one item's N1 answers and the code-side triage (None = abstained)."""

    eid: str
    created_at: pd.Timestamp
    prov_key: str
    about: bool  # P(about_focal) >= .5
    kept: bool  # kept for events: about, P(new_development) >= .5, not quarantined, attributes answered
    quarantined: bool  # N0 kept nothing, or P(instructs_reader) >= .5 (or abstained)
    promotional: bool  # P(promotional) >= .5 (or abstained): rho = .25
    p_serious: float | None
    kind_probs: dict[str, float] | None = None
    p_unresolved: float | None = None
    desk: bool | None = None  # the key is a desk byline (None: look `prov_key` up in the desk list)


@dataclass
class SentinelResult:
    h: float | None = None  # None: no about-focal, non-quarantined item with an answer (Layer B: abstain)
    k_star: str | None = None
    k_star_created_at: str | None = None  # ISO UTC; d0 of the price anchor (BG-3) is its session
    problem_kind: str | None = None  # kappa_i: the most serious corroborating item's kind, else k*'s (if escalated)
    p_problem_kind: float | None = None
    k_star_problem_kind: str | None = None  # k*'s top kind, only when P(serious_problem) of k* >= ESCALATE (cards)
    q_unresolved: float | None = None
    n_prob: int = 0  # distinct provenance keys among corroborating items
    n_prob_collapsed: int = 0  # the same with desk bylines collapsed (sensitivity)
    n_prob_items: int = 0  # corroborating items (not keys): the design_v1 item rule, for the BG-3c replay
    n_about: int = 0
    prov_keys: list[str] = field(default_factory=list)  # the distinct corroborating keys (sorted)

    def as_dict(self) -> dict:
        return asdict(self)


def sentinel(items: Sequence[SentinelItem], desk: Iterable[str] | None = None) -> SentinelResult:
    """The stock-week's sentinel values from its items (design_v1 §5.2 with SAN-5)."""
    desk = desk_bylines() if desk is None else frozenset(desk)
    about = [it for it in items if it.about and not it.quarantined]
    res = SentinelResult(n_about=len(about))
    scored = [it for it in about if it.p_serious is not None]
    def top(it: SentinelItem) -> tuple[str | None, float | None]:
        if not it.kind_probs:
            return None, None
        lab = max(it.kind_probs, key=lambda x: (it.kind_probs[x], x))
        return lab, float(it.kind_probs[lab])

    def rank(it: SentinelItem):
        return (-it.p_serious, pd.Timestamp(it.created_at), it.eid)

    if scored:
        k = min(scored, key=rank)
        res.h, res.k_star = float(k.p_serious), k.eid
        res.k_star_created_at = pd.Timestamp(k.created_at).isoformat()
        if k.p_serious >= ESCALATE:
            res.k_star_problem_kind, res.p_problem_kind = top(k)
            res.problem_kind = res.k_star_problem_kind
            res.q_unresolved = None if k.p_unresolved is None else float(k.p_unresolved)
    corr = [it for it in items if it.kept and not it.quarantined and not it.promotional
            and it.p_serious is not None and it.p_serious >= HIGH]
    if corr:  # kappa of a corroborated problem comes from the corroborating items only
        res.problem_kind, res.p_problem_kind = top(min(corr, key=rank))
    keys = sorted({it.prov_key for it in corr})
    res.n_prob_items = len(corr)
    res.prov_keys = keys
    res.n_prob = len(keys)
    res.n_prob_collapsed = len({DESK_KEY if (it.prov_key in desk if it.desk is None else it.desk) else it.prov_key
                                for it in corr})
    return res


def risk_table(gen_dir: Path | str) -> pd.DataFrame:
    """The risk-officer inputs of a Layer-A generation (`t, symbol, h, n_prob, problem_kind, ...`), read from the
    sentinel's blackboard entries in `<gen_dir>/memory/work.jsonl` (one row per stock-week the N01 program ran).
    WS-B merges `sessions_to_earnings` (WS-C `firm_info.parquet`) and, for the anchor rule, k*'s time."""
    f = Path(gen_dir) / "memory" / "work.jsonl"
    rows = []
    if f.exists():
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)["record"]
            if rec["key"].startswith(BLACKBOARD_PREFIX):
                v = dict(rec["value"])
                v["prov_keys"] = json.dumps(v.get("prov_keys", []))
                t = pd.Timestamp(rec["t"])  # the store keeps the decision date as UTC midnight
                rows.append({"t": t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t,
                             "symbol": rec["key"][len(BLACKBOARD_PREFIX):], **v})
    df = pd.DataFrame(rows)
    for c in RISK_COLUMNS:
        if c not in df:
            df[c] = pd.Series(dtype=object)
    return df[RISK_COLUMNS].sort_values(["t", "symbol"], kind="stable").reset_index(drop=True)


def write_risk_table(gen_dir: Path | str) -> Path:
    out = Path(gen_dir) / "risk.parquet"
    risk_table(gen_dir).to_parquet(out, index=False)
    return out
