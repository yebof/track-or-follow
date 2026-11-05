"""Risk rules and books (design_v1 §9.1-9.2 as amended by design_v2_delta BG-2 and BG-3).

D-sel-EW (signal book): 1/K on each selected name, fully invested, no exclusions, no turnover control.

D-sized (system book), reference-anchored sizing (BG-2, `size_reference_anchored`, the default):
  1. reference selection S^ref = select(R := R^T, K, J, incumbents = held names, vetoed = {}): forced names, then
     incumbents with R^T <= K + buffer and R^T <= K + J, then fill by R^T;
  2. u_i = (1/sigma_i) / sum_{j in trusted top-K} (1/sigma_j); trusted target tau_i = min(0.10, u_i) m^T_i
     (m^T: trusted multipliers, i.e. the earnings rule);
  3. reference targets v^ref = tau on S^ref, industries above 30% scaled to 30%, then everything by 1/sum if > 1;
  4. turnover speed phi = min(1, 0.40 / TO), TO = 1/2 sum |v^ref - w_cur| over S^ref and the held names;
  5. actual selection S (combined rank, incumbent buffer, vetoed = X, the anchored exclusions);
  6. N = S & S^ref keep v = v^ref m^W (m^W: min of the text multipliers, in {1/2, 1}); the displaced names' reference
     weight F = sum_{D = S^ref - S} v^ref is shared by the entrants E = S - S^ref, each capped at its trusted target:
     v_e = min(tau_e, F/|E|) m^W_e; entrants fill only the industry capacity the non-entrants leave;
  7. turnover on both paths with the same phi: z = w_cur + phi (v - w_cur), z^ref likewise; X exits fully, names that
     left U_t get 0 on both paths;
  8. post-turnover caps with the REFERENCE industry factors f_g = min(1, 0.30 / sum_g min(0.10, z^ref)):
     w_i = f_g min(0.10, z_i) for non-entrants; entrants get min(0.10, z_e), scaled into the capacity left in g;
  9. cash = 1 - sum w.
Holdings enter both paths as w_cur / max(1, sum w_cur) (`sizing_holdings`): the engine and `drift_weights` pay fill
costs out of cash, so after a fully invested fill the drifted book sums to 1 + cost (cash < 0), and a convex
combination with such a book would give sum w > 1 (P3 and the engine's cash-only check need sum w_cur <= 1). The
scale is a function of the holdings only, so it is the same on both paths (P0).
Every scale factor (phi, f_g, the 1/sum of step 3) comes from the text-free reference path, so text can only multiply
a reference target by <= 1 or pass a displaced name's reference weight to an in-band entrant capped at the entrant's
trusted target (Proposition P2/P3 of design_v2_delta BG-1; tests/test_reference_anchored_sizing.py,
tests/test_band_guard_bound.py). The Day-1 sizing stays available as `size_v1` (registered variant V-size-v1).

Risk rules (all multipliers combine by min; BG-3 anchor-gated exclusion):
  earnings        forecast release within 2 sessions                                    x0.75  (T)
  sentinel single h >= .8 and n^prob <= 1                                                x0.5   (W)
  sentinel excl.  n^prob >= 2 provenance keys, kind in exclusion kinds, anchor A = 1    0      (W, gated by T anchor)
  sentinel unanch the same without an anchor                                            x0.5   (W)
  sentinel other  n^prob >= 2, other kind                                               x0.5   (W)
  S2              s2_mult = 0 (Layer A's exclude conditions), >= 2 keys cited, A = 1    0      (W, gated by T anchor)
                  any other S2 reduction                                                x0.5   (W)
n^prob is the number of distinct provenance keys (author bylines, SAN-5) among kept, non-promotional items with
P(serious_problem) >= .8 (`key_column`; `n_prob_collapsed` for the desk-collapsed sensitivity). The anchor A is a
trusted input (features.anchors) passed separately from the risk table, so no untrusted column can switch it on.
Text multipliers take values in {1, 1/2, 0} only. Rule ids (RK-1) are stable and appear on cards.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jevtrader.decide.select import Selection, select

EXCLUSION_KINDS = ("accounting_or_disclosure", "government_enforcement", "financial_distress")
# cause -> channel; used on cards and to separate trusted from text-driven multipliers
MULTIPLIER_CHANNEL = {"earnings": "T", "sentinel_single": "W", "sentinel_corroborated": "W",
                      "sentinel_unanchored": "W", "sentinel_other_kind": "W", "s2": "W"}
TRUSTED_CAUSES = tuple(c for c, ch in MULTIPLIER_CHANNEL.items() if ch == "T")
TEXT_CAUSES = tuple(c for c, ch in MULTIPLIER_CHANNEL.items() if ch == "W")
# stable rule ids (design_v2_delta RK-1): every multiplier, cap, turnover step and exit on cards and in the logs
RULE_IDS = {
    "earnings": "RK-EARN", "sentinel_single": "RK-SENT-SINGLE", "sentinel_corroborated": "RK-SENT-EXCL",
    "sentinel_unanchored": "RK-SENT-UNANCHORED", "sentinel_other_kind": "RK-SENT-OTHER", "s2": "RK-S2",
    "name_cap": "RK-CAP-NAME", "industry_cap": "RK-CAP-IND", "scale_down": "RK-BUDGET", "turnover": "RK-TURNOVER",
    "post_turnover_cap": "RK-CAP-POST", "forced_exit": "RK-EXIT", "reference_caps": "RK-REF-CAPS",
    "displaced": "RK-DISPLACED", "entrant_share": "RK-ENTRANT", "entrant_industry_cap": "RK-CAP-IND-ENTRANT",
}
EXCLUSION_RULES = ("anchored", "bylines", "items")  # BG-3 rule; the two others are BG-3c clean-cost replays only
SENTINEL_INPUT_COLUMNS = ("h", "n_prob", "n_prob_collapsed", "n_prob_items", "problem_kind")
SIZING_MODES = ("reference", "v1")


@dataclass(frozen=True)
class RiskConfig:
    name_cap: float = 0.10
    industry_cap: float = 0.30
    turnover_cap: float | None = 0.40  # None disables the partial adjustment
    earnings_mult: float = 0.75
    earnings_sessions: int = 2
    sentinel_high: float = 0.8
    sentinel_single_mult: float = 0.5
    corroboration: int = 2
    exclusion_kinds: tuple[str, ...] = EXCLUSION_KINDS
    other_kind_mult: float = 0.5
    incumbent_buffer: int = 10
    sizing: str = "reference"  # BG-2; "v1" = registered variant V-size-v1
    exclusion_rule: str = "anchored"  # BG-3; "bylines" / "items" = BG-3c replays (no anchor gate)
    key_column: str = "n_prob"  # distinct provenance keys; "n_prob_collapsed" = desk bylines collapsed (SAN-5)
    unanchored_mult: float = 0.5  # an exclusion without a trusted anchor becomes x0.5
    s2_reduce_mult: float = 0.5

    @property
    def key_col(self) -> str:
        """The risk-table column the corroboration count comes from: `n_prob_items` under the design_v1 item rule
        (BG-3c), else `key_column`. A table with sentinel columns but without this one is an input error."""
        return "n_prob_items" if self.exclusion_rule == "items" else self.key_column

    def __post_init__(self):
        if self.sizing not in SIZING_MODES:
            raise ValueError(f"sizing must be one of {SIZING_MODES}")
        if self.exclusion_rule not in EXCLUSION_RULES:
            raise ValueError(f"exclusion_rule must be one of {EXCLUSION_RULES}")


DEFAULT_RISK = RiskConfig()


def _col(inputs: pd.DataFrame, name: str, default=np.nan) -> pd.Series:
    return (pd.to_numeric(inputs[name], errors="coerce") if name in inputs
            else pd.Series(default, index=inputs.index, dtype=float))


def risk_multipliers(inputs: pd.DataFrame, cfg: RiskConfig = DEFAULT_RISK,
                     anchors: pd.Series | Mapping | None = None) -> pd.DataFrame:
    """Multipliers per cause from per-symbol inputs (index = symbols). Optional columns: `sessions_to_earnings`
    (trusted, past releases only), `h`, `n_prob` (distinct provenance keys), `n_prob_collapsed`, `n_prob_items`
    (design_v1 item count, BG-3c only), `problem_kind`, `s2_mult` (1, 1/2 or 0: Layer A's S2 action, 0 = the exclude
    conditions hold) and `s2_keys` (distinct provenance keys cited by the verified claims). `anchors` is the trusted
    anchor A_{i,t} (features.anchors); a name without one has none. Missing inputs give 1, except that a table with
    sentinel columns must carry the rule's key column (`cfg.key_col`; ValueError otherwise). Returns one column per
    cause of MULTIPLIER_CHANNEL."""
    idx = inputs.index
    out = pd.DataFrame(1.0, index=idx, columns=list(MULTIPLIER_CHANNEL))
    s = _col(inputs, "sessions_to_earnings")
    out.loc[(s <= cfg.earnings_sessions).fillna(False).to_numpy(), "earnings"] = cfg.earnings_mult
    A = (pd.Series(anchors, dtype=float) if anchors is not None else pd.Series(dtype=float)).reindex(idx)
    anchored = (A.fillna(0.0) > 0).to_numpy()
    h = _col(inputs, "h")
    key = cfg.key_col
    if key not in inputs and any(c in inputs for c in SENTINEL_INPUT_COLUMNS):
        # never a silent substitute: a BG-3c row computed from another key column (or from 0 keys) would be reported
        # under the wrong rule
        raise ValueError(f"sentinel inputs without the key column {key!r} of exclusion rule "
                         f"{cfg.exclusion_rule!r} (key_column {cfg.key_column!r})")
    n = _col(inputs, key, 0.0).fillna(0.0)
    kind = inputs["problem_kind"] if "problem_kind" in inputs else pd.Series(None, index=idx, dtype=object)
    corr = (n >= cfg.corroboration).to_numpy()
    excl_kind = kind.isin(cfg.exclusion_kinds).to_numpy()
    gate = anchored if cfg.exclusion_rule == "anchored" else np.ones(len(idx), dtype=bool)
    out.loc[((h >= cfg.sentinel_high).fillna(False).to_numpy()) & ~corr, "sentinel_single"] = cfg.sentinel_single_mult
    out.loc[corr & excl_kind & gate, "sentinel_corroborated"] = 0.0
    out.loc[corr & excl_kind & ~gate, "sentinel_unanchored"] = cfg.unanchored_mult
    out.loc[corr & ~excl_kind, "sentinel_other_kind"] = cfg.other_kind_mult
    s2 = _col(inputs, "s2_mult", 1.0).fillna(1.0).clip(0.0, 1.0).to_numpy()  # reduce-only by construction
    keys = _col(inputs, "s2_keys", 0.0).fillna(0.0).to_numpy()
    if cfg.exclusion_rule == "items":
        s2_excl = s2 <= 0.0  # design_v1 §8: Layer A's exclude conditions (>= 2 distinct items) are enough
    else:
        s2_excl = (s2 <= 0.0) & (keys >= cfg.corroboration) & gate
    out["s2"] = np.where(s2_excl, 0.0, np.where(s2 < 1.0, cfg.s2_reduce_mult, 1.0))
    return out


def vetoed_names(mult: pd.DataFrame) -> set[str]:
    """Names excluded by a zero multiplier (anchored sentinel exclusion or anchored System-2 exclusion)."""
    return set(mult.index[(mult <= 0.0).any(axis=1)])


def combined_multiplier(mult: pd.DataFrame | None, names: Iterable[str],
                        causes: Sequence[str] | None = None) -> tuple[pd.Series, pd.Series]:
    """(m_i = min over causes, binding cause); multipliers combine by min, never by product. `causes` restricts
    the min to a subset (the trusted or the text causes)."""
    names = list(names)
    if mult is None or mult.empty:
        return pd.Series(1.0, index=names), pd.Series("", index=names)
    cols = [c for c in (causes if causes is not None else mult.columns) if c in mult]
    if not cols:
        return pd.Series(1.0, index=names), pd.Series("", index=names)
    arr = mult[cols].to_numpy(dtype=float)
    pos = {s: k for k, s in enumerate(mult.index)}
    m = np.ones((len(names), len(cols)))
    for i, s in enumerate(names):
        k = pos.get(s)
        if k is not None:
            m[i] = arr[k]
    m = np.where(np.isnan(m), 1.0, m)
    val = m.min(axis=1) if len(cols) else np.ones(len(names))
    arg = m.argmin(axis=1)  # first column attaining the min, as DataFrame.idxmin
    cause = np.where(val < 1.0, np.array(cols, dtype=object)[arg], "")
    return pd.Series(val, index=names), pd.Series(cause, index=names, dtype=object)


def equal_weights(names: Sequence[str]) -> pd.Series:
    names = list(names)
    return pd.Series(1.0 / len(names), index=names) if names else pd.Series(dtype=float)


def trusted_weights(sigma: pd.Series, trusted_topK: Sequence[str], universe: Sequence[str]) -> pd.Series:
    """u_i = (1/sigma_i) / sum over the trusted top-K of 1/sigma_j, for every i in U_t. A missing or non-positive
    sigma takes the cross-sectional median over U_t (a trusted, same-week statistic)."""
    s = pd.to_numeric(sigma.reindex(list(universe)), errors="coerce").astype(float)
    s = s.where(s > 0)
    med = s.median()
    s = s.fillna(med if np.isfinite(med) else 1.0)
    inv = 1.0 / s
    return inv / inv.reindex(list(trusted_topK)).sum()


@dataclass
class Weights:
    trusted: pd.Series  # u_i
    target: pd.Series  # after caps and multipliers, before turnover (v for reference-anchored sizing)
    final: pd.Series  # after turnover, post-turnover caps and forced exits
    ledger: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    multipliers: dict[str, dict[str, float]] = field(default_factory=dict)  # every cause < 1 (only the min binds)
    turnover: float = 0.0  # realized one-way turnover 1/2 sum |final - w_cur|
    phi: float = 1.0
    cash: float = 0.0
    reference: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # reference targets (v^ref / w_ref)
    turnover_ref: float = 0.0  # one-way turnover of the reference targets, 1/2 sum |ref - w_cur| (sets phi)
    sizing: str = "v1"
    # reference-anchored sizing only (BG-2)
    tau: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # trusted targets tau_i
    final_ref: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))  # w^ref (post-turnover reference)
    entrants: tuple[str, ...] = ()
    displaced: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    reference_selection: tuple[str, ...] = ()
    F: float = 0.0  # displaced reference weight shared by the entrants
    industry_factor: dict[str, float] = field(default_factory=dict)  # f_g from the reference path
    log: list[dict] = field(default_factory=list)  # per-name sizing rows (sizing.jsonl)
    holding_scale: float = 1.0  # max(1, sum w_cur): the holdings entered sizing as w_cur / holding_scale


def sizing_holdings(w_cur: pd.Series | Mapping | None) -> tuple[dict[str, float], float]:
    """(held names -> weight entering sizing, scale). Nonzero holdings as given, divided by max(1, sum w_cur) so the
    book sizing starts from never exceeds a full book. The sum is exact (math.fsum), hence independent of the order
    of the names; the scale is a function of the holdings only (a trusted object, the same on both paths)."""
    cur = {} if w_cur is None else {s: float(x) for s, x in dict(w_cur).items() if abs(float(x)) > 0}
    total = math.fsum(cur.values())
    if total > 1.0:
        return {s: x / total for s, x in cur.items()}, total
    return cur, 1.0


def _groups(names: Sequence[str], industry: pd.Series) -> pd.Series:
    """FF12 group per name; a name with no industry is its own group (pooling unknowns would cap unrelated names)."""
    g = industry.reindex(list(names))
    return g.where(g.notna(), pd.Series(["?" + str(s) for s in names], index=list(names)))


def _industry_scale(w: pd.Series, industry: pd.Series, cap: float) -> pd.Series:
    """Per-name factor that scales every FF12 industry above `cap` down to it."""
    g = _groups(list(w.index), industry)
    tot = w.groupby(g).transform("sum")
    return (cap / tot).where(tot > cap, 1.0).clip(upper=1.0)


def one_way(a: pd.Series, b: pd.Series) -> float:
    """One-way traded notional 1/2 sum |a - b| (buys = sells for a fully invested swap; §9.1 step 4)."""
    return 0.5 * float(a.sub(b, fill_value=0.0).abs().sum())


def _capped(base: pd.Series, m: pd.Series, ind: pd.Series, caps: RiskConfig) -> list[pd.Series]:
    """§9.1 step 3 as a chain: name cap, multiplier, industry cap, scale-down (each output feeds the next)."""
    a = np.minimum(base, caps.name_cap)
    b = a * m
    c = b * _industry_scale(b, ind, caps.industry_cap)
    tot = c.sum()
    return [a, b, c, c / tot if tot > 1.0 else c]


def reference_book(sel: Selection, u: pd.Series, multipliers: pd.DataFrame | None, caps: RiskConfig,
                   w_cur: pd.Series, industry: pd.Series) -> pd.Series:
    """V-size-v1's neutral-text book: L^W = 0 (so R = R^T), no text vetoes, trusted multipliers only, the same
    incumbent buffer, trusted weights `u` and caps. A function of trusted inputs and the holding only; sets phi."""
    held = list(w_cur.index[w_cur > 0])
    fr = pd.DataFrame({"R_T": sel.rank_T, "R": sel.rank_T})
    ref = select(fr, sel.K, sel.J, incumbents=held, incumbent_buffer=caps.incumbent_buffer)
    names = list(ref.names)
    u = u.reindex(names).fillna(0.0)
    m_T, _ = combined_multiplier(multipliers, names, TRUSTED_CAUSES)
    w = _capped(u, m_T, industry, caps)[-1]
    return w[w > 0]


def size_v1(
    sel: Selection, sigma: pd.Series, trusted_topK: Sequence[str], multipliers: pd.DataFrame | None,
    caps: RiskConfig = DEFAULT_RISK, w_cur: pd.Series | None = None, industry: pd.Series | Mapping | None = None,
    universe: Sequence[str] | None = None,
) -> Weights:
    """Day-1 D-sized weights (design_v1 §9.1 steps 2-4 with the audit fixes): kept as registered variant
    V-size-v1. Text-driven entries can lower other names through the industry cap and the 1/sum scale-down, which
    Proposition P2 does not cover; reference-anchored sizing closes that channel."""
    uni = list(universe) if universe is not None else list(sel.rank_T.index)
    ind = pd.Series(industry if industry is not None else {}, dtype=object)
    held_w, scale = sizing_holdings(w_cur)
    cur = pd.Series(held_w, dtype=float)
    names = list(dict.fromkeys([*sel.names, *cur.index]))
    u_all = trusted_weights(sigma, trusted_topK, uni)
    selected = pd.Series([s in set(sel.names) for s in names], index=names)
    base = u_all.reindex(names).fillna(0.0).where(selected, 0.0)
    lines: dict[str, list[tuple[str, float]]] = {s: [("trusted_weight" if selected[s] else "not_selected", float(base[s]))]
                                                  for s in names}

    def step(prev: pd.Series, new: pd.Series, label) -> pd.Series:
        d = (new - prev).to_numpy(dtype=float)
        for k in np.flatnonzero(d != 0.0):
            lines[names[k]].append((label(names[k]) if callable(label) else label, float(d[k])))
        return new

    m, cause = combined_multiplier(multipliers, names)
    chain = _capped(base, m, ind, caps)
    a = step(base, chain[0], "name_cap")
    b = step(a, chain[1], lambda s: f"mult:{cause[s]}")
    c = step(b, chain[2], "industry_cap")
    d = step(c, chain[3], "scale_down")
    target = d.copy()
    wc = cur.reindex(names).fillna(0.0)
    ref = reference_book(sel, u_all, multipliers, caps, cur, ind)
    to_ref = one_way(ref, cur)
    phi = 1.0 if caps.turnover_cap is None or to_ref <= 0 else min(1.0, caps.turnover_cap / to_ref)
    e = step(d, wc + phi * (d - wc), "turnover")
    e2 = np.minimum(e, caps.name_cap)
    e2 = e2 * _industry_scale(e2, ind, caps.industry_cap)
    e2 = step(e, e2, "post_turnover_cap")
    vetoed = set(sel.vetoed) | (vetoed_names(multipliers) if multipliers is not None else set())
    out = vetoed | {s for s in names if s not in set(uni)}
    f = step(e2, e2.where(~pd.Series([s in out for s in names], index=names), 0.0), "forced_exit")
    final = f[f > 0]
    applied = {}
    if multipliers is not None and not multipliers.empty:
        mm = multipliers.reindex(names).fillna(1.0)
        applied = {s: {c: float(v) for c, v in mm.loc[s].items() if v < 1.0} for s in names}
    return Weights(
        trusted=u_all, target=target[target > 0], final=final, ledger=lines, multipliers=applied,
        turnover=one_way(f, wc), phi=phi, cash=float(1.0 - final.sum()), reference=ref, turnover_ref=to_ref,
        sizing="v1", holding_scale=scale,
    )


def _group_key(industry: Mapping):
    def grp(s: str) -> str:
        g = industry.get(s)
        return "?" + str(s) if g is None or (isinstance(g, float) and np.isnan(g)) else str(g)
    return grp


def size_reference_anchored(
    sel: Selection, sigma: pd.Series, trusted_topK: Sequence[str], multipliers: pd.DataFrame | None,
    caps: RiskConfig = DEFAULT_RISK, w_cur: pd.Series | None = None, industry: pd.Series | Mapping | None = None,
    universe: Sequence[str] | None = None,
) -> Weights:
    """Reference-anchored D-sized weights (design_v2_delta BG-2; module doc, steps 1-9). `sel` is the actual
    selection (with the anchored exclusions as `vetoed`); the reference selection is recomputed from `sel.rank_T`,
    `sel.K`, `sel.J` and the held names only.

    Every reference-path sum (the step-3 industry and budget totals, TO, the step-8 industry factors) runs in a fixed
    order over the text-free names only (S^ref in its trusted order, S^ref | held sorted), so phi, v^ref, f_g and w^ref
    are bit-identical under any attack (P0); a sum over a text-dependent name list would move them in the last bit."""
    uni = list(dict.fromkeys(universe if universe is not None else sel.rank_T.index))
    in_uni = set(uni)
    grp = _group_key(dict(industry) if industry is not None else {})
    cur, scale = sizing_holdings(w_cur)  # w_cur / max(1, sum w_cur), the same on both paths
    held = list(cur)
    icap, ncap = caps.industry_cap, caps.name_cap
    # 1. reference selection (text-free: R = R^T, no exclusions)
    ref_sel = select(pd.DataFrame({"R_T": sel.rank_T, "R": sel.rank_T}), sel.K, sel.J, incumbents=held,
                     incumbent_buffer=caps.incumbent_buffer)
    S_ref = [s for s in ref_sel.names if s in in_uni]
    Rset = set(S_ref)
    # 2. trusted targets
    u_all = trusted_weights(sigma, trusted_topK, uni)
    m_T, _ = combined_multiplier(multipliers, uni, TRUSTED_CAUSES)
    tau = np.minimum(u_all, ncap) * m_T
    tau_d = tau.to_dict()
    # 3. reference targets (industry scale, then budget), summed in S^ref order
    vr = {s: float(tau_d[s]) for s in S_ref}
    gs: dict[str, float] = {}
    for s in S_ref:
        gs[grp(s)] = gs.get(grp(s), 0.0) + vr[s]
    for s in S_ref:
        if gs[grp(s)] > icap:
            vr[s] = vr[s] * (icap / gs[grp(s)])
    tot = 0.0
    for s in S_ref:
        tot += vr[s]
    if tot > 1.0:
        vr = {s: x / tot for s, x in vr.items()}
    # 5. actual selection, exclusions
    excluded = (set(sel.vetoed) | (vetoed_names(multipliers) if multipliers is not None else set())) & in_uni
    S = [s for s in sel.names if s in in_uni and s not in excluded]
    Sset = set(S)
    N = [s for s in S if s in Rset]
    E = [s for s in S if s not in Rset]
    Eset = set(E)
    D = [s for s in S_ref if s not in Sset]
    ref_names = sorted(Rset | set(held))
    names = sorted(Rset | Sset | set(held))
    # 4. turnover speed from the reference path
    to_ref = 0.0
    for s in ref_names:
        to_ref += abs(vr.get(s, 0.0) - cur.get(s, 0.0))
    to_ref *= 0.5
    phi = 1.0 if caps.turnover_cap is None or to_ref <= 0 else min(1.0, caps.turnover_cap / to_ref)
    # 6. actual targets
    mW, cW = combined_multiplier(multipliers, names, TEXT_CAUSES)
    m_W = {s: (x if x > 0 else 1.0) for s, x in mW.items()}  # zero multipliers are exclusions, handled through X
    c_W = cW.to_dict()
    v = {s: 0.0 for s in names}
    lines: dict[str, list[tuple[str, float]]] = {
        s: [("trusted_target", float(tau_d.get(s, 0.0)))] if (s in Rset or s in Sset) else [("not_selected", 0.0)]
        for s in names}
    for s in S_ref:
        if vr[s] != tau_d[s]:
            lines[s].append(("reference_caps", vr[s] - float(tau_d[s])))
    for s in N:
        v[s] = vr[s] * m_W[s]
        if v[s] != vr[s]:
            lines[s].append((f"mult:{c_W[s]}", v[s] - vr[s]))
    for s in D:
        if vr[s] != 0.0:
            lines[s].append(("displaced", -vr[s]))
    F = 0.0
    for s in D:
        F += vr[s]
    if E:
        share = F / len(E)
        for s in E:
            t_ = float(tau_d.get(s, 0.0))
            pre = min(t_, share)
            if pre != t_:
                lines[s].append(("entrant_share", pre - t_))
            v[s] = pre * m_W[s]
            if v[s] != pre:
                lines[s].append((f"mult:{c_W[s]}", v[s] - pre))
        nsum: dict[str, float] = {}
        for s in N:
            nsum[grp(s)] = nsum.get(grp(s), 0.0) + v[s]
        esum: dict[str, float] = {}
        for s in E:
            esum[grp(s)] = esum.get(grp(s), 0.0) + v[s]
        for g, sE in esum.items():
            cap_g = icap - nsum.get(g, 0.0)
            if sE > cap_g and sE > 0.0:  # sE = 0 with cap_g an ulp below 0 (N's sum rounds above 0.30): nothing to scale
                f = max(cap_g, 0.0) / sE
                for s in E:
                    if grp(s) == g:
                        new_ = v[s] * f
                        if new_ != v[s]:
                            lines[s].append(("entrant_industry_cap", new_ - v[s]))
                        v[s] = new_
    # 7. turnover, the same phi on both paths
    z, z_ref = {}, {}
    for s in names:
        wc = cur.get(s, 0.0)
        zp = wc + phi * (v[s] - wc)
        out = (s in excluded) or (s not in in_uni)
        z[s] = 0.0 if out else zp
        z_ref[s] = (wc + phi * (vr.get(s, 0.0) - wc)) if s in in_uni else 0.0
        if zp - v[s] != 0.0:
            lines[s].append(("turnover", zp - v[s]))
        if out and zp != 0.0:
            lines[s].append(("forced_exit", -zp))
    # 8. post-turnover caps with the reference industry factors (summed over the text-free names only)
    rs: dict[str, float] = {}
    for s in ref_names:
        rs[grp(s)] = rs.get(grp(s), 0.0) + min(z_ref[s], ncap)
    f_g = {g: (min(1.0, icap / x) if x > 0 else 1.0) for g, x in rs.items()}
    w, w_ref, fg = {}, {}, {}
    for s in names:
        fg[s] = f_g.get(grp(s), 1.0)  # an entrant's group without reference mass: factor 1
        w_ref[s] = fg[s] * min(z_ref[s], ncap)
        w[s] = min(z[s], ncap) if s in Eset else fg[s] * min(z[s], ncap)
    if E:
        non_e: dict[str, float] = {}
        for s in names:
            if s not in Eset:
                non_e[grp(s)] = non_e.get(grp(s), 0.0) + w[s]
        e_tot: dict[str, float] = {}
        for s in E:
            e_tot[grp(s)] = e_tot.get(grp(s), 0.0) + w[s]
        for g, sE in e_tot.items():
            room = icap - non_e.get(g, 0.0)
            if sE > room and sE > 0.0:  # as in step 6: zero-weight entrants need no scaling (and 0/0 would raise)
                f = max(room, 0.0) / sE
                for s in E:
                    if grp(s) == g:
                        w[s] = w[s] * f
    for s in names:
        if w[s] - z[s] != 0.0:
            lines[s].append(("post_turnover_cap", w[s] - z[s]))
    final = pd.Series({s: x for s, x in w.items() if x > 0}, dtype=float)
    applied = {}
    if multipliers is not None and not multipliers.empty:
        mm = multipliers.reindex(names).fillna(1.0)
        cols = list(mm.columns)
        arr = mm.to_numpy(dtype=float)
        applied = {s: {cols[k]: float(arr[i, k]) for k in np.flatnonzero(arr[i] < 1.0)} for i, s in enumerate(names)}
    Dset = set(D)
    log = [{
        "symbol": s, "in_ref": s in Rset, "entrant": s in Eset, "displaced": s in Dset, "excluded": s in excluded,
        "held": s in cur, "tau": float(tau_d.get(s, 0.0)), "v_ref": float(vr.get(s, 0.0)), "m_W": float(m_W[s]),
        "v": float(v[s]), "F": F, "n_entrants": len(E), "phi": phi, "f_g": float(fg[s]), "z": float(z[s]),
        "z_ref": float(z_ref[s]), "w": float(w[s]), "w_ref": float(w_ref[s]), "w_cur": float(cur.get(s, 0.0)),
        "holding_scale": scale,
    } for s in names]
    cur_s = pd.Series(cur, dtype=float)
    return Weights(
        trusted=u_all, target=pd.Series({s: x for s, x in v.items() if x > 0}, dtype=float), final=final,
        ledger=lines, multipliers=applied, turnover=one_way(final, cur_s), phi=phi, cash=float(1.0 - final.sum()),
        reference=pd.Series({s: vr[s] for s in S_ref if vr[s] > 0}, dtype=float), turnover_ref=to_ref,
        sizing="reference", tau=tau, final_ref=pd.Series({s: x for s, x in w_ref.items() if x > 0}, dtype=float),
        entrants=tuple(E), displaced=tuple(D), excluded=tuple(sorted(excluded)), reference_selection=tuple(S_ref), F=F,
        industry_factor={str(k): float(x) for k, x in f_g.items()}, log=log, holding_scale=scale,
    )


def size(
    sel: Selection, sigma: pd.Series, trusted_topK: Sequence[str], multipliers: pd.DataFrame | None,
    caps: RiskConfig = DEFAULT_RISK, w_cur: pd.Series | None = None, industry: pd.Series | Mapping | None = None,
    universe: Sequence[str] | None = None,
) -> Weights:
    """D-sized weights (design §12.3 interface): reference-anchored sizing (BG-2) unless `caps.sizing == "v1"`."""
    fn = size_v1 if caps.sizing == "v1" else size_reference_anchored
    return fn(sel, sigma, trusted_topK, multipliers, caps, w_cur, industry, universe)
