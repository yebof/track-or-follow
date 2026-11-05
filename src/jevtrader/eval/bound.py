"""Bound checks (design_v2_delta BG-5), tightness witnesses (BG-6) and the clean cost of the exclusion rules (BG-3c).

All three replay decisions through `pipeline.layer_b.decide_week`, the one decision function of the pipeline, from
the week states a run logged (`DecisionLog.states`: trusted L^T, kappa, sigma, anchors, holdings; the untrusted
L-bar^W and risk rows). Nothing here calls a model: the attacker's best untrusted inputs are written in directly.

BG-5 worst-case reader, per learned week t:
- entry: for every name i with R^T > K + J, set L-bar^W_i = +kappa_t and L-bar^W_j = -kappa_t for all j != i; i must
  not be in either book, and its D-sized weight must not exceed (1 - phi) w_cur,i (a held name outside the band is
  only sold down; P1);
- protection: every protected name (R^T <= K - J) without a trusted anchor gets h = 1, 10 provenance keys, kind
  `accounting_or_disclosure` and an S2 exclusion with 10 verified keys, plus L-bar^W = -kappa for it and +kappa for
  all others; it must stay in both books with v >= 1/2 v^ref and w >= 1/2 w^ref (P2).
Both replays also check P3's budget, sum w <= 1 (`sized_budget`). The number of checks and every violation are
reported; the pass criterion is 0 violations and a trusted set of T-channel features only.

Per-reader checks: attacked units (reader, t, symbol, the attacked name's untrusted logit and optional risk
overrides) are replayed the same way; units with R^T > K + J must have ASR exactly 0.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from jevtrader.decide.risk import RiskConfig, risk_multipliers, size
from jevtrader.decide.select import select
from jevtrader.learn.allocator import trusted_not_T
from jevtrader.pipeline.layer_b import DecisionConfig, DecisionLog, WeekState, decide_week

WORST_KIND = "accounting_or_disclosure"
WORST_RISK = {"h": 1.0, "n_prob": 10.0, "n_prob_collapsed": 10.0, "n_prob_items": 10.0, "problem_kind": WORST_KIND,
              "s2_mult": 0.0, "s2_keys": 10.0}
TOL = 1e-12


def _ranks(state: WeekState) -> pd.Series:
    from jevtrader.decide.select import rank_desc

    return rank_desc(state.L_T.reindex(list(state.universe)))


def _decide(cfg: DecisionConfig, state: WeekState, L_W_bar: pd.Series, risk: pd.DataFrame | None = None):
    uni = list(state.universe)
    L = state.L_T.reindex(uni) + L_W_bar.reindex(uni).fillna(0.0)
    return decide_week(cfg, state.L_T, L, state.w_cur, state.sigma, state.risk if risk is None else risk,
                       state.anchors, state.industry, uni)


def _over_budget(sized) -> bool:
    """P3's budget part: sum w <= 1 (cash-only; the engine refuses more)."""
    return float(sized.final.sum()) > 1.0 + TOL


def worst_case_entry(cfg: DecisionConfig, state: WeekState) -> tuple[int, list[dict]]:
    uni = list(state.universe)
    rT = _ranks(state)
    k = state.kappa
    out = []
    n = 0
    for i in rT.index[rT > cfg.K + cfg.J]:
        lw = pd.Series(-k, index=uni)
        lw[i] = k
        wd = _decide(cfg, state, lw)
        n += 1
        w_i = float(wd.sized.final.get(i, 0.0))
        cap = (1.0 - wd.sized.phi) * float(state.w_cur.get(i, 0.0))
        bad = [m for m, b in (("sel_ew", i in wd.sel_ew.names), ("sized_selection", i in wd.sel_sz.names),
                              ("sized_weight", w_i > cap + TOL), ("sized_budget", _over_budget(wd.sized))) if b]
        if bad:
            out.append({"t": str(state.t.date()), "symbol": i, "R_T": int(rT[i]), "failed": bad, "w": w_i, "cap": cap})
    return n, out


def worst_case_protection(cfg: DecisionConfig, state: WeekState) -> tuple[int, list[dict]]:
    uni = list(state.universe)
    rT = _ranks(state)
    A = state.anchors.reindex(uni).fillna(0.0)
    prot = [s for s in rT.index[rT <= cfg.K - cfg.J] if A[s] <= 0]
    if not prot:
        return 0, []
    risk = state.risk.reindex(uni).copy()
    for c, v in WORST_RISK.items():
        if c not in risk:
            risk[c] = np.nan if c != "problem_kind" else None
        risk.loc[prot, c] = v
    lw = pd.Series(state.kappa, index=uni)
    lw[prot] = -state.kappa
    wd = _decide(cfg, state, lw, risk)
    sz = wd.sized
    out = []
    for s in prot:
        bad = ["sized_budget"] if _over_budget(sz) else []
        if s not in wd.sel_ew.names:
            bad.append("sel_ew")
        if s not in wd.sel_sz.names:
            bad.append("sized_selection")
        if sz.sizing == "reference":
            row = next((r for r in sz.log if r["symbol"] == s), None)
            if row is None or row["v"] < 0.5 * row["v_ref"] - TOL:
                bad.append("target_half")
            if row is None or row["w"] < 0.5 * row["w_ref"] - TOL:
                bad.append("weight_half")
        if bad:
            out.append({"t": str(state.t.date()), "symbol": s, "R_T": int(rT[s]), "failed": bad})
    return len(prot), out


def bound_check(log: DecisionLog, cfg: DecisionConfig) -> dict:
    """BG-5 worst-case reader over every learned week of a run. The replays attack only the untrusted side, so they
    certify the bound only if the trusted side reads T-channel features alone: a trusted set with a W-channel feature
    (possible only without a trust split, as V-joint) is reported in `trusted_not_T` and never passes."""
    ne = npr = 0
    ve, vp = [], []
    for st in log.states:
        n, v = worst_case_entry(cfg, st)
        ne, ve = ne + n, ve + v
        n, v = worst_case_protection(cfg, st)
        npr, vp = npr + n, vp + v
    not_T = trusted_not_T(cfg.allocator.trusted)
    return {"n_weeks": len(log.states), "n_entry_checks": ne, "entry_violations": ve, "n_protection_checks": npr,
            "protection_violations": vp, "trusted_all_T": not not_T, "trusted_not_T": not_T,
            "passed": not ve and not vp and not not_T, "K": cfg.K, "J": cfg.J,
            "sizing": cfg.risk.sizing, "exclusion_rule": cfg.risk.exclusion_rule}


# ------------------------------------------------------------------------------------ per-unit replays
RISK_OVERRIDE_COLUMNS = ("h", "n_prob", "n_prob_collapsed", "n_prob_items", "problem_kind", "s2_mult", "s2_keys")


def attacked_week_logits(cfg: DecisionConfig, raw: pd.DataFrame, log: DecisionLog, t,
                         overrides: Mapping[tuple[str, str], float | None]) -> pd.DataFrame:
    """The week's untrusted logits after an attack, for replaying a C1 unit (RT-S replay scoring): the raw untrusted
    features of decision t (`raw`, long `t, symbol, feature, value[, covered]` as LayerBInputs.raw) with
    `overrides` {(symbol, feature): attacked raw value, None = abstain} applied, standardized exactly as the run did
    (covered median of the attacked week; the logged trailing scale when the history had >= min_history weeks, else
    the attacked week's own sd, floor and clip as configured; rank-Gaussian for code features), times the week's
    gamma-hat, capped at the week's kappa. Trusted inputs are untouched (P0). Returns U_t x [L_W, L_W_bar]."""
    from jevtrader.learn.features import CATALOG, rank_gaussian, wide_week

    t = pd.Timestamp(t)
    st = next(x for x in log.states if x.t == t)
    snap = next(x for x in log.snapshots if x["t"] == t.date().isoformat())
    scales = next(x for x in log.scales if x["t"] == t.date().isoformat())
    feats = list(cfg.allocator.untrusted)
    uni = list(st.universe)
    val, cov = wide_week(raw[pd.to_datetime(raw["t"]) == t], uni, feats)
    for (sym, f), v in overrides.items():
        if f in val.columns and sym in val.index:
            ok = v is not None and np.isfinite(float(v))
            val.loc[sym, f] = float(v) if ok else np.nan
            cov.loc[sym, f] = ok
    x = pd.DataFrame(0.0, index=uni, columns=feats)
    for f in feats:
        if CATALOG[f].kind == "code":
            x[f] = rank_gaussian(val[f].where(cov[f]))
            continue
        c = cov[f].to_numpy()
        v = val[f].to_numpy()[c]
        if not len(v):
            continue
        dm = v - float(np.median(v))
        lg = scales.get(f, {})
        if lg.get("history_weeks", 0) >= cfg.std_min_history:
            scale = float(lg["scale"])
        else:
            sd = float(np.std(dm, ddof=1)) if len(dm) > 1 else 0.0
            scale = max(sd if np.isfinite(sd) else 0.0, cfg.std_floor)
        col = np.zeros(len(c))
        col[c] = np.clip(dm / scale, -cfg.std_clip, cfg.std_clip)
        x[f] = col
    gamma = pd.Series(snap["gamma"], dtype=float).reindex(feats).fillna(0.0)
    LW = (x * gamma).sum(axis=1)
    k = st.kappa
    return pd.DataFrame({"L_W": LW, "L_W_bar": np.sign(LW) * np.minimum(np.abs(LW), k)})


def replay_units(log: DecisionLog, cfg: DecisionConfig, units: pd.DataFrame, labels=None,
                 week_logits: Mapping[int, pd.Series] | None = None) -> pd.DataFrame:
    """Replay attacked units. `units`: reader, t, symbol and optionally `L_W_att` (the attacked name's untrusted
    logit before the cap; the other names keep their clean L-bar^W) and risk overrides for the attacked name
    (RISK_OVERRIDE_COLUMNS; NaN = unchanged). Returns one row per unit with the clean and attacked membership of both
    books, the weights and reference quantities the tightness witnesses need, and |dL| / (2 kappa). With `labels`
    (a LabelStore; evaluation only), each row also carries RB-4's signed harm of both books over the holding week
    (attacked - clean, bps of NAV, planted part and rest; eval.metrics.signed_harm). `week_logits` maps a unit's
    position in `units` to the full attacked L-bar^W of its week (attacked_week_logits), used instead of L_W_att."""
    from jevtrader.eval.metrics import signed_harm

    states = {st.t: st for st in log.states}
    rets = (labels.table.set_index(["t", "symbol"])["r"] if labels is not None else None)
    rows = []
    for pos, u in enumerate(units.itertuples(index=False)):
        t = pd.Timestamp(u.t)
        st = states.get(t)
        if st is None:
            continue
        s = u.symbol
        uni = list(st.universe)
        rT = _ranks(st)
        clean = _decide(cfg, st, st.L_W_bar)
        lw = st.L_W_bar.reindex(uni).copy()
        latt = getattr(u, "L_W_att", np.nan)
        if week_logits is not None and pos in week_logits:
            lw = week_logits[pos].reindex(uni).fillna(0.0).astype(float)
        elif latt is not None and np.isfinite(latt):
            lw[s] = float(np.sign(latt) * min(abs(latt), st.kappa))
        risk = st.risk.reindex(uni).copy()
        for c in RISK_OVERRIDE_COLUMNS:
            val = getattr(u, c, None)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                if c not in risk:
                    risk[c] = np.nan if c != "problem_kind" else None
                risk.loc[s, c] = val
        att = _decide(cfg, st, lw, risk)
        log_row = next((r for r in att.sized.log if r["symbol"] == s), {}) if att.sized.sizing == "reference" else {}
        tau = float(att.sized.tau.get(s, np.nan)) if len(att.sized.tau) else float("nan")
        phi, wcur = att.sized.phi, float(st.w_cur.get(s, 0.0))
        w_att = float(att.sized.final.get(s, 0.0))
        rows.append({
            "reader": getattr(u, "reader", ""), "t": t, "symbol": s, "R_T": int(rT[s]), "kappa": st.kappa,
            "protected": bool(rT[s] <= cfg.K - cfg.J), "in_band": bool(rT[s] <= cfg.K + cfg.J),
            "anchor": bool(st.anchors.get(s, 0.0) > 0), "held": wcur > 0,
            "in_ew_clean": s in clean.sel_ew.names, "in_ew_att": s in att.sel_ew.names,
            "in_sized_clean": s in clean.sized.final.index, "in_sized_att": s in att.sized.final.index,
            "entered_ew": (s in att.sel_ew.names) and (s not in clean.sel_ew.names),
            "entered_sized": (s in att.sized.final.index) and (s not in clean.sized.final.index),
            "excluded": s in set(att.sized.excluded), "w_clean": float(clean.sized.final.get(s, 0.0)), "w_att": w_att,
            "w_ref": float(log_row.get("w_ref", np.nan)), "v": float(log_row.get("v", np.nan)),
            "v_ref": float(log_row.get("v_ref", np.nan)), "tau": tau, "phi": phi, "w_cur": wcur,
            "entrant": bool(log_row.get("entrant", False)),
            "dL_over_2kappa": abs(float(lw[s] - st.L_W_bar.get(s, 0.0))) / (2 * st.kappa) if st.kappa > 0 else float("nan"),
        })
        if rets is not None:
            r_t = rets.xs(t, level="t") if t in rets.index.get_level_values("t") else pd.Series(dtype=float)
            for book, wc_, wa_ in (("sel_ew", clean.w_ew, att.w_ew), ("sized", clean.sized.final, att.sized.final)):
                h = signed_harm(wc_, wa_, r_t, s)
                rows[-1].update({f"harm_{book}_bps": h["d_ret_bps"], f"harm_{book}_planted_bps": h["planted_bps"],
                                 f"harm_{book}_rest_bps": h["rest_bps"]})
    return pd.DataFrame(rows)


def per_reader_bound_check(results: pd.DataFrame, cfg: DecisionConfig) -> dict:
    """BG-5 per-reader check on replayed units: among units with R^T > K + J the entry ASR (either book) must be 0."""
    out = {}
    for reader, g in results.groupby("reader"):
        o = g[g["R_T"] > cfg.K + cfg.J]
        succ = o["entered_ew"] | o["entered_sized"]
        out[str(reader)] = {"n_units_outside_band": int(len(o)), "asr": float(succ.mean()) if len(o) else float("nan"),
                            "violations": o.loc[succ, ["t", "symbol", "R_T"]].astype(str).to_dict("records")}
    return out


# ------------------------------------------------------------------------------------ BG-6 tightness
def tightness(results: pd.DataFrame, cfg: DecisionConfig) -> dict:
    """BG-6 from replayed attacked units (all readers). P1: the largest R^T among successful entries vs K + J.
    P2: among protected held units in non-anchor weeks, the share whose target is halved (v <= 1/2 v^ref), the
    smallest w / w^ref (bound: >= 1/2) and the number of exclusions (must be 0). P3: max over entrants of v_e / tau_e and of w_e / ((1 - phi) w_cur + phi tau_e).
    P4: max |dL| / (2 kappa). A part no real unit attains is reported as not attained (use the constructed witness)."""
    r = results
    succ = r[r["entered_ew"] | r["entered_sized"]]
    p1 = int(succ["R_T"].max()) if len(succ) else None
    ph = r[r["protected"] & r["held"] & ~r["anchor"]]
    # halving is measured on the target (v = 1/2 v^ref); traded weights move toward it at the reference turnover speed
    halved = (ph["v"] <= 0.5 * ph["v_ref"] + TOL) & (ph["v_ref"] > 0)
    w_ratio = (ph["w_att"] / ph["w_ref"]).where(ph["w_ref"] > 0)
    ent = r[r["entrant"] & (r["tau"] > 0)]
    v_tau = float((ent["v"] / ent["tau"]).max()) if len(ent) else None
    bound = (1 - ent["phi"]) * ent["w_cur"] + ent["phi"] * ent["tau"]
    w_b = float((ent["w_att"] / bound).max()) if len(ent) else None
    p4 = float(r["dL_over_2kappa"].max()) if len(r) else None
    return {
        "P1": {"max_R_T_entry": p1, "K_plus_J": cfg.K + cfg.J, "attained": p1 == cfg.K + cfg.J},
        "P2": {"n_protected_held_nonanchor": int(len(ph)), "share_halved": float(halved.mean()) if len(ph) else None,
               "min_w_over_w_ref": float(w_ratio.min()) if w_ratio.notna().any() else None,
               "n_exclusions": int(ph["excluded"].sum()), "attained": bool(halved.any())},
        "P3": {"max_v_over_tau": v_tau, "max_w_over_bound": w_b,
               "attained": v_tau is not None and v_tau >= 1 - 1e-9},
        "P4": {"max_dL_over_2kappa": p4, "attained": p4 is not None and p4 >= 1 - 1e-9},
    }


def constructed_witnesses() -> dict:
    """Hand-built cases, run through the real selection, risk and sizing code, that attain each part of the bound
    exactly (ratio 1, or the factor 1/2 of P2). Reported when no real unit attains a part (BG-6)."""
    from jevtrader.learn.allocator import Allocator, AllocatorConfig, CalibSnapshot
    from jevtrader.learn.labels import LABEL_COLUMNS, LabelStore

    K, J = 4, 2
    names = [f"W{k}" for k in range(10)]
    LT = pd.Series(np.linspace(1.0, -1.0, 10), index=names)
    kappa = 2.0 * float(LT.std(ddof=1))
    # P1: the name at R^T = K + J enters when its text logit is +kappa and every other name's is -kappa
    s1 = names[K + J - 1]
    lw = pd.Series(-kappa, index=names)
    lw[s1] = kappa
    sel = select(pd.DataFrame({"L_T": LT, "L": LT + lw}), K, J)
    p1 = {"symbol_R_T": K + J, "entered": s1 in sel.names, "ratio": (K + J) / (K + J) if s1 in sel.names else None}
    # P2: one planted serious-problem item on a protected name, no anchor: exactly half of the reference weight
    sigma = pd.Series(0.2, index=names)
    caps = RiskConfig(turnover_cap=None)
    mult = risk_multipliers(pd.DataFrame({"h": [1.0], "n_prob": [1.0]}, index=[names[0]]).reindex(names), caps)
    sel2 = select(pd.DataFrame({"L_T": LT, "L": LT}), K, J)
    w2 = size(sel2, sigma, sel2.trusted_topK, mult, caps, None, None, names)
    r2 = next(r for r in w2.log if r["symbol"] == names[0])
    p2 = {"v_over_v_ref": r2["v"] / r2["v_ref"], "w_over_w_ref": r2["w"] / r2["w_ref"]}
    # P3: an entrant whose share of the displaced weight F / |E| is at least its trusted target: v_e = tau_e
    lw3 = pd.Series(0.0, index=names)
    lw3[names[K]] = 10.0  # rank K + 1 (in band) displaces rank K
    sel3 = select(pd.DataFrame({"L_T": LT, "L": LT + lw3}), K, J)
    sig3 = sigma.copy()
    sig3[names[K]] = 0.25  # the entrant's trusted target is below the displaced reference weight
    w3 = size(sel3, sig3, sel3.trusted_topK, None, caps, None, None, names)
    r3 = next(r for r in w3.log if r["symbol"] == names[K])
    bound3 = (1 - w3.phi) * r3["w_cur"] + w3.phi * r3["tau"]
    p3 = {"entrant": r3["entrant"], "v_over_tau": r3["v"] / r3["tau"], "w_over_bound": r3["w"] / bound3}
    # P4: clean and attacked untrusted logits at opposite caps
    cfg = AllocatorConfig(trusted=("mom_12_1",), untrusted=("f_imp",))
    snap = CalibSnapshot("w", pd.Timestamp("2025-01-03", tz="UTC"), cfg.trusted, cfg.untrusted, np.array([1.0]),
                         np.array([1.0]), np.eye(1), np.eye(1), 1, 1, 1.0)
    al = Allocator(cfg, LabelStore(pd.DataFrame(columns=LABEL_COLUMNS)))
    x = pd.DataFrame({"mom_12_1": LT.to_numpy(), "f_imp": 0.0}, index=names)
    clean = al.score(x.assign(f_imp=np.where(x.index == names[5], -100.0, 0.0)), snap)
    att = al.score(x.assign(f_imp=np.where(x.index == names[5], 100.0, 0.0)), snap)
    p4 = {"ratio": abs(att.frame.at[names[5], "L"] - clean.frame.at[names[5], "L"]) / (2 * clean.kappa)}
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4}


# ------------------------------------------------------------------------------------ BG-3c clean cost
def exclusion_rule_report(runs: Mapping[str, object], labels) -> dict:
    """BG-3c: under each exclusion rule (EvalResults keyed by rule name, e.g. 'anchored', 'bylines', 'items',
    'anchored_collapsed'), the number of exclusions (a zero multiplier) and halvings (a text multiplier of 1/2) over
    the learned weeks, the T3 tail rate (label z) of excluded, halved and unflagged names, and D-sized's Sharpe, IR and
    Calmar with their differences from the first rule. The prediction IC does not depend on the exclusion rule (the
    rule changes weights, not L), so the IC difference is 0 by construction."""
    lab = labels.table[["t", "symbol", "z"]]
    out, base = {}, None
    for name, res in runs.items():
        m = res.log.multipliers
        if m is None or m.empty:
            out[name] = {"n_exclusions": 0, "n_halvings": 0}
            continue
        text = [c for c in ("sentinel_single", "sentinel_corroborated", "sentinel_unanchored", "sentinel_other_kind", "s2")
                if c in m]
        mn = m[text].min(axis=1)
        flag = pd.Series(np.where(mn <= 0, "excluded", np.where(mn < 1, "halved", "unflagged")), index=m.index)
        j = m[["t", "symbol"]].assign(flag=flag).merge(lab, on=["t", "symbol"], how="left")
        tail = j.groupby("flag")["z"].mean().to_dict()
        pf = res.metrics.get("portfolio_sized", {})
        row = {"n_exclusions": int((flag == "excluded").sum()), "n_halvings": int((flag == "halved").sum()),
               "t3_tail_rate": {k: float(v) for k, v in tail.items()},
               "sharpe": pf.get("sharpe"), "information_ratio": pf.get("information_ratio"), "calmar": pf.get("calmar")}
        if base is None:
            base = row
        for k in ("sharpe", "information_ratio", "calmar"):
            a, b = row.get(k), base.get(k)
            row[f"d_{k}"] = (a - b) if isinstance(a, float) and isinstance(b, float) else None
        row["d_ic"] = 0.0
        out[name] = row
    return out


def reference_weights_hold(sized, w_cur: pd.Series) -> dict:
    """Helper for tests and reports: the P3 inequalities of one reference-anchored sizing result."""
    tau = sized.tau
    lg = pd.DataFrame(sized.log).set_index("symbol") if sized.log else pd.DataFrame()
    wc = w_cur.reindex(lg.index).fillna(0.0) if len(lg) else pd.Series(dtype=float)
    return {
        "v_le_tau": bool((lg["v"] <= lg["tau"] + TOL).all()) if len(lg) else True,
        "w_le_bound": bool((lg["w"] <= (1 - sized.phi) * wc + sized.phi * lg["tau"] + TOL).all()) if len(lg) else True,
        "w_le_name_cap": bool((sized.final <= 0.10 + TOL).all()),
        "sum_v_le_sum_vref": bool(sized.target.sum() <= sized.reference.sum() + TOL),
        "sum_vref_le_1": bool(sized.reference.sum() <= 1 + TOL),
        "sum_w_le_1": bool(sized.final.sum() <= 1 + TOL),
        "tau_finite": bool(len(tau) == 0 or np.isfinite(tau.to_numpy(dtype=float)).all()),
    }
