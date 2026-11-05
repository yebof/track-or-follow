"""Event clustering in code (design_v1 §5.1 "Cluster"): no model call.

Kept items of one stock-week with the same top `event_type` are merged when the Jaccard similarity of their blinded
headline content tokens is >= 0.30 (bank `code_constants.thresholds["cluster.jaccard"]`); merging is transitive
(single linkage within a type). Each cluster e carries:
- width w_e = the number of member items ("outlets");
- d_e = (E_e[effect] - 2) / 2 in [-1, 1];
- m_e = P(important), u_e = P(unconfirmed), sigma_e = P(above) - P(below), p_scheduled, the type distribution tau_e
  and the effect distribution, each a P(important)-weighted mean over members (so m_e = sum m_k^2 / sum m_k; equal
  weights if every member has P(important) = 0);
- rho_e = min_k rho_k (a promotional member makes the whole cluster promotional).
An attribute a member abstained on is averaged over the other members; if no member answered, u_e = 1 (the full
0.5 discount), sigma_e = 0 and p_scheduled = None.

The headline tokens are the N1-state headline's (screened and blinded): code names, stop words and one-letter tokens
are dropped, and a headline removed by screening has no tokens (it never merges on the literal placeholder).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

JACCARD = 0.30  # code_constants.thresholds["cluster.jaccard"]
TERM_CAP = 1.5  # per-cluster |term| cap of f_imp (design_v1 §3.3 D5)
IMPACT_CAP = 3.0  # f_imp in [-3, 3]
DROPPED_HEADLINE = "(headline removed by screening)"  # conventions.masks.dropped_headline
_CODE_NAME = re.compile(r"Company-[0-9A-F]{8}")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = frozenset("""
a an the and or but nor of for to in on at by with from as into onto over under about after before than then is are
was were be been being has have had do does did will would shall should can could may might must its it this that
these those their there they them he she his her we our you your who whom which what when where why how not no yes
up down out off per via vs amid says said say reports reported report new
""".split())  # noqa: SIM905 (a word list reads better as text)


def content_tokens(headline: str) -> frozenset[str]:
    """Content tokens of a blinded headline: lower-case words and numbers without code names, stop words or
    one-character tokens. The dropped-headline literal has none."""
    if not headline or headline.strip() == DROPPED_HEADLINE:
        return frozenset()
    text = _CODE_NAME.sub(" ", headline).lower()
    return frozenset(w for w in _TOKEN.findall(text) if len(w) > 1 and w not in STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """|a & b| / |a | b|; 0 when either side is empty (no evidence of similarity)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def top_label(probs: Mapping[str, float]) -> str:
    """Argmax label; ties broken by the label (deterministic)."""
    return max(probs, key=lambda k: (probs[k], k))


@dataclass(frozen=True)
class KeptItem:
    """One kept, non-quarantined item with its N1 attributes (None = the member abstained on that attribute)."""

    eid: str
    created_at: pd.Timestamp
    available_at: pd.Timestamp  # max(created_at, updated_at): when the judged text was public
    headline: str  # N1-state headline: screened and blinded (or the dropped-headline literal)
    type_probs: dict[str, float]
    effect_probs: dict[str, float]  # Score levels "0".."4"
    p_important: float
    rho: float  # 1, or .25 if promotional
    p_unconfirmed: float | None = None
    p_scheduled: float | None = None
    vs_probs: dict[str, float] | None = None  # N1.vs_expectations
    memory_headline: str = ""  # screened text stored in M-epi (kept headline, else first kept sentence), not yet
    # time-blinded when that form is known exactly (NewsItem.n0_untimed)
    n1_item: dict[str, Any] = field(default_factory=dict)  # the N1 state's `item` (N2 reuses it verbatim)
    request_keys: tuple[str, ...] = ()

    @property
    def top_type(self) -> str:
        return top_label(self.type_probs)

    @property
    def tokens(self) -> frozenset[str]:
        return content_tokens(self.headline)

    @property
    def expected_effect(self) -> float:
        return sum(int(k) * p for k, p in self.effect_probs.items())


def _wmean(values: Sequence[float | None], weights: Sequence[float]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights) if v is not None]
    if not pairs:
        return None
    sw = sum(w for _, w in pairs)
    if sw <= 1e-12:
        return sum(v for v, _ in pairs) / len(pairs)
    return sum(v * w for v, w in pairs) / sw


def _wdist(dists: Sequence[Mapping[str, float] | None], weights: Sequence[float]) -> dict[str, float] | None:
    pairs = [(d, w) for d, w in zip(dists, weights) if d]
    if not pairs:
        return None
    sw = sum(w for _, w in pairs)
    eq = sw <= 1e-12
    labels: list[str] = []
    for d, _ in pairs:
        labels += [k for k in d if k not in labels]
    out = {k: sum(d.get(k, 0.0) * (1.0 if eq else w) for d, w in pairs) / (len(pairs) if eq else sw) for k in labels}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()} if s > 0 else out


@dataclass(frozen=True)
class Cluster:
    """One event cluster (design_v1 §5.1). Members are sorted by (created_at, eid)."""

    members: tuple[KeptItem, ...]

    @property
    def weights(self) -> list[float]:
        return [m.p_important for m in self.members]

    @property
    def width(self) -> int:
        return len(self.members)

    @property
    def first_seen(self) -> pd.Timestamp:
        return min(m.created_at for m in self.members)

    @property
    def available_at(self) -> pd.Timestamp:
        return max(m.available_at for m in self.members)

    @property
    def type_probs(self) -> dict[str, float]:
        return _wdist([m.type_probs for m in self.members], self.weights)

    @property
    def top_type(self) -> str:
        return self.members[0].top_type  # every member has the same top type by construction

    @property
    def effect_probs(self) -> dict[str, float]:
        return _wdist([m.effect_probs for m in self.members], self.weights)

    @property
    def expected_effect(self) -> float:
        return _wmean([m.expected_effect for m in self.members], self.weights)

    @property
    def d(self) -> float:
        return (self.expected_effect - 2.0) / 2.0

    @property
    def m(self) -> float:
        return _wmean([m.p_important for m in self.members], self.weights)

    @property
    def u(self) -> float:
        v = _wmean([m.p_unconfirmed for m in self.members], self.weights)
        return 1.0 if v is None else v

    @property
    def sigma(self) -> float:
        vals = [None if m.vs_probs is None else m.vs_probs.get("above", 0.0) - m.vs_probs.get("below", 0.0)
                for m in self.members]
        v = _wmean(vals, self.weights)
        return 0.0 if v is None else v

    @property
    def vs_probs(self) -> dict[str, float] | None:
        return _wdist([m.vs_probs for m in self.members], self.weights)

    @property
    def p_scheduled(self) -> float | None:
        return _wmean([m.p_scheduled for m in self.members], self.weights)

    @property
    def rho(self) -> float:
        return min(m.rho for m in self.members)

    @property
    def representative(self) -> KeptItem:
        """The most important member (N2's `item`): max P(important), then earliest, then evidence id."""
        return min(self.members, key=lambda k: (-k.p_important, k.created_at, k.eid))

    @property
    def request_keys(self) -> list[str]:
        return sorted({k for m in self.members for k in m.request_keys})

    def p_types(self, types: Iterable[str]) -> float:
        tp = self.type_probs
        return float(sum(tp.get(k, 0.0) for k in types))


def cluster_items(items: Sequence[KeptItem], threshold: float = JACCARD) -> list[Cluster]:
    """Single-linkage clusters of kept items within each top event type, Jaccard >= threshold on headline content
    tokens. Deterministic: the result does not depend on the input order. Clusters are returned most important
    first (m_e, then width, then first_seen, then the first member's evidence id)."""
    items = sorted(items, key=lambda k: (pd.Timestamp(k.created_at), k.eid))
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    toks = [k.tokens for k in items]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i].top_type == items[j].top_type and jaccard(toks[i], toks[j]) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[KeptItem]] = {}
    for i, k in enumerate(items):
        groups.setdefault(find(i), []).append(k)
    clusters = [Cluster(tuple(g)) for g in groups.values()]
    return sorted(clusters, key=lambda c: (-c.m, -c.width, c.first_seen, c.members[0].eid))


def cluster_term(c: Cluster, novelty: float) -> float:
    """rho_e nu_e m_e d_e (1 - 0.5 u_e)(1 + 0.25 ln w_e), capped at +-1.5 (design_v1 §5.1, §3.3 D5)."""
    x = c.rho * novelty * c.m * c.d * (1.0 - 0.5 * c.u) * (1.0 + 0.25 * math.log(c.width))
    return max(-TERM_CAP, min(TERM_CAP, x))
