"""Proposition P0-P5 (design_v2_delta BG-1) under randomized attacks: BG-4 property tests, the BG-5 worst-case-reader
check and the BG-6 witnesses. (P5, poisoned history, is tests/test_bound_history.py.)

`test_proposition_randomized` draws 2,000 seeded cases (8 chunks x 250). Each case draws: untrusted features with
arbitrary finite values (up to 1e6) and a poisoned gamma; sentinel inputs (h in [0, 1], 0..12 provenance keys, any
kind) and S2 stances with 0..12 cited keys; trusted inputs (sigma incl. missing ones, the earnings rule m^T, L^T hence
R^T, industries incl. unknown ones, anchors); holdings w_cur (sum <= 1, some > 0.10 after drift, some outside U_t);
K and J from the registered grid. `test_proposition_overfull_holdings` adds 500 cases with the holdings the pipeline
produces after a fully invested fill (the engine pays the fill cost from cash): sum w_cur = 1 + cost, half of them with
a fully invested reference book (equal sigma, one industry per name, no earnings rule). The clean run has neutral text.
For both books and every name it checks:
  P0 Theta identical: R^T, L^T, kappa, the trusted top-K, u, tau, S^ref, v^ref, phi, TO, f_g, w^ref;
  P1 every selected or held-and-kept name has R^T <= K + J; a name outside the band is only sold down;
  P2 protected names (R^T <= K - J) are selected unless an anchored exclusion fires (A = 1 and >= 2 keys); with
     A = 0 nothing is excluded; without exclusion v >= v^ref / 2 and w >= w^ref / 2;
  P3 v <= tau; w <= (1 - phi) w_cur + phi tau and <= 0.10; sum v <= sum v^ref <= 1; sum w <= 1; industries <= 0.30;
  P4 |L - L^T| <= kappa (hence |L(A) - L(clean)| <= 2 kappa).
Tolerance 1e-12 throughout."""

import math

import numpy as np
import pandas as pd
import pytest

from jevtrader.decide.risk import RiskConfig, risk_multipliers, size, size_v1, trusted_weights, vetoed_names
from jevtrader.decide.select import select
from jevtrader.eval import bound as B
from jevtrader.learn.allocator import Allocator, AllocatorConfig, CalibSnapshot
from jevtrader.learn.features import TRUSTED_V1, UNTRUSTED_V1
from jevtrader.learn.labels import LABEL_COLUMNS, LabelStore
from jevtrader.pipeline.layer_b import DecisionConfig, LayerBInputs, run_decisions
from jevtrader.pipeline.synthetic import SYN_TRUSTED, SYN_UNTRUSTED, synthetic_inputs

EMPTY = LabelStore(pd.DataFrame(columns=LABEL_COLUMNS))
CFG = AllocatorConfig(trusted=TRUSTED_V1, untrusted=UNTRUSTED_V1)
ALLOC = Allocator(CFG, EMPTY)
EXCL = ("accounting_or_disclosure", "government_enforcement", "financial_distress")
KINDS = (*EXCL, "product_safety", "litigation")
J_GRID = (0.0, 5.0, 10.0, 20.0, math.inf)
TOL = 1e-12
N_CHUNKS, PER_CHUNK = 8, 250  # 2,000 cases


def _snap(beta, gamma):
    return CalibSnapshot("s", pd.Timestamp("2025-01-03", tz="UTC"), CFG.trusted, CFG.untrusted, beta, gamma,
                         np.eye(len(beta)), np.eye(len(gamma)), 1, 1, 1.0)


def _draw(rng):
    n = int(rng.integers(12, 70))
    K = int(rng.integers(3, max(4, min(25, n - 2))))
    J = float(rng.choice(J_GRID))
    names = [f"S{i:03d}" for i in range(n)]
    xt = pd.DataFrame(rng.normal(size=(n, len(TRUSTED_V1))), index=names, columns=TRUSTED_V1)
    beta = rng.normal(0, 0.1, len(TRUSTED_V1))
    clean_w = pd.DataFrame(rng.normal(size=(n, len(UNTRUSTED_V1))), index=names, columns=UNTRUSTED_V1)
    att_w = clean_w.copy()
    hit = rng.uniform(size=n) < rng.uniform(0.05, 1.0)
    att_w.loc[hit] = rng.normal(0, 10 ** rng.uniform(0, 6), (int(hit.sum()), len(UNTRUSTED_V1)))
    g_clean = rng.normal(0, 0.1, len(UNTRUSTED_V1))
    g_att = g_clean + rng.normal(0, 10 ** rng.uniform(-2, 2), len(UNTRUSTED_V1))  # poisoned history
    sigma = pd.Series(rng.uniform(0.002, 0.15, n), index=names)
    sigma[rng.uniform(size=n) < 0.05] = np.nan
    earn = rng.integers(0, 9, n)
    industry = pd.Series(rng.choice(["BusEq", "Hlth", "Money", "Shops", None], n, p=[.3, .25, .2, .15, .1]), index=names)
    anchors = pd.Series((rng.uniform(size=n) < rng.uniform(0, 0.4)).astype(float), index=names)
    held = list(rng.choice(names, int(rng.integers(0, min(n, K + 8))), replace=False))
    held += [f"GONE{k}" for k in range(int(rng.integers(0, 3)))]
    w_cur = pd.Series(rng.dirichlet(np.ones(len(held)) * rng.uniform(0.2, 2.0)) * rng.uniform(0.2, 1.0),
                      index=held) if held else pd.Series(dtype=float)
    risk_att = pd.DataFrame({
        "sessions_to_earnings": earn, "h": rng.uniform(0, 1, n), "n_prob": rng.integers(0, 13, n),
        "n_prob_collapsed": rng.integers(0, 13, n), "problem_kind": rng.choice(KINDS, n),
        "s2_mult": rng.choice([1.0, 0.5, 0.0], n, p=[0.6, 0.2, 0.2]), "s2_keys": rng.integers(0, 13, n),
    }, index=names)
    return dict(n=n, K=K, J=J, names=names, xt=xt, beta=beta, clean_w=clean_w, att_w=att_w, g_clean=g_clean,
                g_att=g_att, sigma=sigma, earn=earn, industry=industry, anchors=anchors, w_cur=w_cur, risk=risk_att)


def _run(c, attacked: bool, caps=RiskConfig()):
    x = pd.concat([c["xt"], c["att_w"] if attacked else c["clean_w"]], axis=1)
    sc = ALLOC.score(x, _snap(c["beta"], c["g_att"] if attacked else c["g_clean"]))
    risk = c["risk"] if attacked else pd.DataFrame({"sessions_to_earnings": c["earn"]}, index=c["names"])
    mult = risk_multipliers(risk, caps, anchors=c["anchors"])
    sel_ew = select(sc, c["K"], c["J"])
    held = list(c["w_cur"].index[c["w_cur"] > 0])
    sel_sz = select(sc, c["K"], c["J"], incumbents=held, vetoed=vetoed_names(mult), incumbent_buffer=caps.incumbent_buffer)
    w = size(sel_sz, c["sigma"], sel_sz.trusted_topK, mult, caps, c["w_cur"], c["industry"], c["names"])
    return sc, mult, sel_ew, sel_sz, w


def _check_case(c):
    K, J, names = c["K"], c["J"], c["names"]
    sc_c, _, sel_c, selsz_c, w_c = _run(c, False)
    sc, mult, sel_ew, sel_sz, w = _run(c, True)
    fr = sc.frame
    rT = fr["R_T"]
    # P0: every trusted object is bit-identical
    assert fr["R_T"].equals(sc_c.frame["R_T"]) and fr["L_T"].equals(sc_c.frame["L_T"]) and sc.kappa == sc_c.kappa
    assert sel_sz.trusted_topK == selsz_c.trusted_topK == sel_c.trusted_topK
    assert w.trusted.equals(w_c.trusted) and w.tau.equals(w_c.tau)
    assert w.reference_selection == w_c.reference_selection and w.reference.equals(w_c.reference)
    assert w.phi == w_c.phi and w.turnover_ref == w_c.turnover_ref and w.industry_factor == w_c.industry_factor
    assert w.final_ref.equals(w_c.final_ref)
    # P4
    assert np.all(np.abs(fr["L"] - fr["L_T"]) <= sc.kappa * (1 + TOL) + TOL)
    assert np.all(np.abs(fr["L"] - sc_c.frame["L"]) <= 2 * sc.kappa * (1 + TOL) + TOL)
    # P1 (both books, incl. incumbents)
    assert all(rT[s] <= K + J for s in sel_ew.names) and all(rT[s] <= K + J for s in sel_sz.names)
    wc_ = c["w_cur"]
    for s, x in w.final.items():
        if rT.get(s, np.inf) > K + J:  # held outside the band (or outside U_t): only sold down
            assert x <= (1 - w.phi) * float(wc_.get(s, 0.0)) + TOL
    assert all(float(wc_.get(s, 0.0)) > 0 or rT[s] <= K + J for s in w.final.index)
    # P2
    forced = set(rT.index[rT <= K - J])
    assert forced <= set(sel_ew.names)  # D-sel-EW has no exclusions
    A, r = c["anchors"], c["risk"]
    for s in forced:
        if s in w.excluded:
            assert A[s] == 1.0
            sent = r.at[s, "n_prob"] >= 2 and r.at[s, "problem_kind"] in EXCL
            s2 = r.at[s, "s2_mult"] == 0.0 and r.at[s, "s2_keys"] >= 2
            assert sent or s2
            continue
        assert s in sel_sz.names
        row = next(x for x in w.log if x["symbol"] == s)
        assert row["v"] >= 0.5 * row["v_ref"] - TOL and row["w"] >= 0.5 * row["w_ref"] - TOL
    assert all(A[s] == 1.0 for s in w.excluded)
    # P3
    lg = pd.DataFrame(w.log).set_index("symbol")
    wcur = wc_.reindex(lg.index).fillna(0.0)
    assert (lg["v"] <= lg["tau"] + TOL).all()
    assert (lg["w"] <= (1 - w.phi) * wcur + w.phi * lg["tau"] + TOL).all()
    assert (w.final <= 0.10 + TOL).all()
    assert w.target.sum() <= w.reference.sum() + TOL and w.reference.sum() <= 1 + TOL
    assert w.final.sum() <= 1 + TOL
    ind = c["industry"].reindex(w.final.index)
    grp = ind.where(ind.notna(), pd.Series(["?" + s for s in w.final.index], index=w.final.index))
    assert (w.final.groupby(grp).sum() <= 0.30 + TOL).all()
    for s, lines in w.ledger.items():
        assert abs(sum(v for _, v in lines) - w.final.get(s, 0.0)) <= 1e-12


@pytest.mark.parametrize("chunk", range(N_CHUNKS))
def test_proposition_randomized(chunk):
    for k in range(PER_CHUNK):
        seed = chunk * PER_CHUNK + k
        _check_case(_draw(np.random.default_rng(seed)))


def _overfull(rng):
    """A BG-4 case whose holdings sum to 1 + cost (0.05% to 0.3%: 15 bps per side on a traded notional up to 2), the
    book `drift_weights` and the engine hold after a fully invested fill. In half of the cases the holdings are a
    drifted full book of K names (1/K <= 0.10, so the name cap does not trim the excess) and the reference book is
    fully invested too (equal sigma, one industry per name, no earnings rule): with phi < 1 the excess would reach the
    final weights. The other half keep the generator's draws with random holdings scaled to 1 + cost."""
    c = _draw(rng)
    n, names = c["n"], c["names"]
    if rng.uniform() < 0.5:
        c["K"] = int(rng.integers(10, max(11, min(25, n - 2))))
        c["sigma"] = pd.Series(0.02, index=names)
        c["industry"] = pd.Series([f"G{k}" for k in range(n)], index=names)
        c["earn"] = np.full(n, 9)
        c["risk"] = c["risk"].assign(sessions_to_earnings=c["earn"])
        if rng.uniform() < 0.5:  # neutral sentinel inputs too: the attack moves the selection only
            c["risk"] = pd.DataFrame({"sessions_to_earnings": c["earn"]}, index=names)
        held = list(rng.choice(names, c["K"], replace=False))
        base = np.exp(rng.normal(0.0, 0.01, len(held)))
    else:
        held = list(rng.choice(names, int(rng.integers(1, min(n, c["K"] + 8))), replace=False))
        base = rng.dirichlet(np.ones(len(held)))
    c["w_cur"] = pd.Series(base / base.sum() * (1.0 + rng.uniform(0.0005, 0.003)), index=held)
    return c


@pytest.mark.parametrize("chunk", range(2))
def test_proposition_overfull_holdings(chunk):
    """Audit regression (Day 2): holdings above a full book used to pass through the turnover step into final weights
    with sum > 1 (negative cash), which the engine refuses. Both sizings now start from w_cur / max(1, sum w_cur)."""
    n_over = 0
    for k in range(PER_CHUNK):
        c = _overfull(np.random.default_rng(100_000 + chunk * PER_CHUNK + k))
        _check_case(c)
        _, mult, _, sel_sz, w = _run(c, True)
        assert w.holding_scale == math.fsum(c["w_cur"]) > 1.0
        v1 = size_v1(sel_sz, c["sigma"], sel_sz.trusted_topK, mult, RiskConfig(), c["w_cur"], c["industry"], c["names"])
        assert v1.final.sum() <= 1 + TOL and v1.holding_scale == w.holding_scale
        n_over += w.phi < 1.0 and w.reference.sum() >= 1 - 1e-9
    assert n_over > 30  # the binding case (fully invested reference, phi < 1) is exercised (about 68 per chunk)


def test_zero_weight_entrants_in_a_full_industry_do_not_divide_by_zero():
    """Found by the overfull generator (seed 100_245): an industry whose non-entrant weights sum to 0.30 + 1 ulp
    (room = -5.6e-17) and whose only entrant has weight 0 made the entrant capacity step compute 0 / 0. Zero-weight
    entrants are now left alone; the case satisfies P0-P4."""
    c = _overfull(np.random.default_rng(100_245))
    _check_case(c)
    w = _run(c, False)[-1]
    assert "S002" in w.entrants and w.final.get("S002", 0.0) == 0.0


@pytest.mark.parametrize("sizer", [size_v1, size])
def test_text_swaps_of_other_names_cannot_slow_a_sell_down(sizer):
    """WS-B audit regression (Day 1): a text-driven swap of ten in-band names used to raise TO, lower phi and keep an
    unrelated, forced, overweight name above its trusted weight. phi comes from the text-free reference path."""
    n, K, J = 40, 20, 10
    names = [f"S{i:02d}" for i in range(n)]
    LT = pd.Series(np.linspace(2, -2, n), index=names)
    sigma = pd.Series(0.03, index=names)
    sigma["S05"] = 0.09
    att_LW = pd.Series(0.0, index=names)
    att_LW[[f"S{i}" for i in range(20, 30)]] = 5.0
    sel_c = select(pd.DataFrame({"L_T": LT, "L": LT}), K, J)
    sel_a = select(pd.DataFrame({"L_T": LT, "L": LT + att_LW}), K, J)
    assert "S05" in sel_c.forced and "S05" in sel_a.forced
    w_cur = pd.Series(0.91 / 19, index=[s for s in sel_c.names if s != "S05"])
    w_cur["S05"] = 0.09
    wc = sizer(sel_c, sigma, sel_c.trusted_topK, None, RiskConfig(), w_cur, None, names)
    wa = sizer(sel_a, sigma, sel_a.trusted_topK, None, RiskConfig(), w_cur, None, names)
    tau = min(0.10, trusted_weights(sigma, sel_c.trusted_topK, names)["S05"])
    assert wa.phi == wc.phi
    assert wa.final["S05"] == pytest.approx(wc.final["S05"], abs=1e-15)
    assert wa.final["S05"] <= (1 - wa.phi) * 0.09 + wa.phi * tau + 1e-15


def test_poisoning_every_week_of_text_never_moves_the_trusted_side():
    inp = synthetic_inputs(n_stocks=30, n_days=330, seed=13)
    cfg = DecisionConfig(allocator=AllocatorConfig(trusted=SYN_TRUSTED, untrusted=SYN_UNTRUSTED), K=8, J=4)
    dates = sorted(inp.universe)
    clean = run_decisions(cfg, inp, dates)
    rng = np.random.default_rng(1)
    raw = inp.raw.copy()
    w = raw["feature"].isin(SYN_UNTRUSTED)
    raw.loc[w, "value"] = rng.normal(0, 50, w.sum())
    raw.loc[w, "covered"] = rng.uniform(size=w.sum()) < 0.9
    risk = inp.risk.copy()
    risk["n_prob"], risk["problem_kind"] = 1, "financial_distress"  # single-key "serious problems" everywhere
    att = run_decisions(cfg, LayerBInputs(**{**inp.__dict__, "raw": raw, "risk": risk}), dates)
    for a, b in zip(clean.snapshots, att.snapshots):
        assert a["beta"] == b["beta"] and a["kappa"] == b["kappa"]
    pa, pb = clean.predictions, att.predictions
    assert np.array_equal(pa["R_T"], pb["R_T"]) and np.array_equal(pa["L_T"], pb["L_T"])
    assert np.array_equal(pa["anchor"], pb["anchor"])
    assert not np.array_equal(pa["L"], pb["L"])
    K, J = cfg.K, cfg.J
    for t in dates:
        g = pb[pb["t"] == t].set_index("symbol")
        if g["cold"].iloc[0]:
            continue
        for book in ("sel_ew", "sized"):
            chosen = set(att.targets[book][t])
            new = chosen - set(att.targets[book].get(dates[dates.index(t) - 1], {}))
            assert all(g.at[s, "R_T"] <= K + J for s in new)  # entry only from inside the band
        assert set(g.index[g["R_T"] <= K - J]) <= set(att.targets["sel_ew"][t])
        assert set(g.index[g["R_T"] <= K - J]) <= set(att.targets["sized"][t])  # no anchor-free exclusion


@pytest.fixture(scope="module")
def run():
    inp = synthetic_inputs(n_stocks=24, n_days=330, seed=4)
    cfg = DecisionConfig(allocator=AllocatorConfig(trusted=SYN_TRUSTED, untrusted=SYN_UNTRUSTED), K=6, J=3)
    return inp, cfg, run_decisions(cfg, inp, sorted(inp.universe)[:30])


def test_worst_case_reader_has_no_violation(run):
    """BG-5 on a synthetic panel: every outside-band name pushed to +kappa with all others at -kappa never enters;
    every unanchored protected name under the worst sentinel/S2 inputs stays with >= half its reference weight."""
    inp, cfg, log = run
    res = B.bound_check(log, cfg)
    assert res["passed"] and res["n_entry_checks"] > 100 and res["n_protection_checks"] > 20


def test_worst_case_reader_catches_an_unanchored_exclusion_rule(run):
    """Non-vacuity: under the BG-3c byline rule (exclusion without a trusted anchor) the protection check fails."""
    import dataclasses

    inp, cfg, log = run
    bad = dataclasses.replace(cfg, risk=dataclasses.replace(cfg.risk, exclusion_rule="bylines"))
    res = B.bound_check(log, bad)
    assert not res["passed"] and res["protection_violations"]
    assert all("sized_selection" in v["failed"] for v in res["protection_violations"])


def test_per_reader_checks_and_tightness(run):
    inp, cfg, log = run
    from jevtrader.decide.select import rank_desc

    units = []
    for st in log.states[::4]:
        rT = rank_desc(st.L_T)
        for s in rT.index:
            units.append({"reader": "typed", "t": st.t, "symbol": s, "L_W_att": 1e3})
            if rT[s] <= cfg.K - cfg.J:
                units.append({"reader": "A8-1", "t": st.t, "symbol": s, "h": 1.0, "n_prob": 1,
                              "problem_kind": "financial_distress"})
    res = B.replay_units(log, cfg, pd.DataFrame(units), labels=inp.labels)
    assert {"harm_sel_ew_bps", "harm_sized_bps", "harm_sized_planted_bps"} <= set(res.columns)  # RB-4 per unit
    out = res[res["R_T"] > cfg.K + cfg.J]
    assert (out["harm_sel_ew_bps"] == 0).all()  # nothing entered, so no signed harm outside the band
    chk = B.per_reader_bound_check(res, cfg)
    assert chk["typed"]["n_units_outside_band"] > 0 and chk["typed"]["asr"] == 0.0
    tt = B.tightness(res, cfg)
    assert tt["P1"]["max_R_T_entry"] <= cfg.K + cfg.J
    assert tt["P2"]["n_exclusions"] == 0 and (tt["P2"]["min_w_over_w_ref"] is None or tt["P2"]["min_w_over_w_ref"] >= 0.5 - TOL)
    assert tt["P3"]["max_v_over_tau"] <= 1 + TOL and tt["P3"]["max_w_over_bound"] <= 1 + 1e-9
    assert tt["P4"]["max_dL_over_2kappa"] <= 1 + TOL


def test_constructed_witnesses_attain_every_part():
    w = B.constructed_witnesses()
    assert w["P1"]["entered"] and w["P1"]["ratio"] == 1.0
    assert w["P2"]["v_over_v_ref"] == pytest.approx(0.5, abs=TOL) and w["P2"]["w_over_w_ref"] == pytest.approx(0.5, abs=TOL)
    assert w["P3"]["entrant"] and w["P3"]["v_over_tau"] == pytest.approx(1.0, abs=TOL)
    assert w["P3"]["w_over_bound"] == pytest.approx(1.0, abs=1e-12)
    assert w["P4"]["ratio"] == pytest.approx(1.0, abs=TOL)


def test_attacked_week_logits_reproduce_the_run_and_move_only_the_text_side(run):
    """With no override the replay primitive reproduces the logged capped untrusted logits; an attacked raw feature
    moves L-bar^W within +-kappa and never the trusted side; replaying it gives the same membership as the run's own
    decision function."""
    inp, cfg, log = run
    st = log.states[-1]
    base = B.attacked_week_logits(cfg, inp.raw, log, st.t, {})
    assert np.allclose(base["L_W_bar"].reindex(st.L_W_bar.index), st.L_W_bar, atol=1e-12)
    s = st.L_W_bar.index[-1]
    att = B.attacked_week_logits(cfg, inp.raw, log, st.t, {(s, "f_imp"): 1e6, (s, "h"): None})
    assert (att["L_W_bar"].abs() <= st.kappa * (1 + TOL)).all() and att.at[s, "L_W"] != base.at[s, "L_W"]
    res = B.replay_units(log, cfg, pd.DataFrame([{"reader": "typed", "t": st.t, "symbol": s}]),
                         week_logits={0: att["L_W_bar"]})
    assert len(res) == 1 and res["dL_over_2kappa"].iloc[0] <= 1 + TOL


def test_worst_case_reader_checks_the_budget(run, monkeypatch):
    """BG-5 also checks P3's budget (sum w <= 1) in both replays. A week whose holdings are a drifted full book after
    a fully invested fill (10 names at 1.003 / 10), all outside the band, with a fully invested reference (K = 10,
    equal sigma, one industry per name, no risk rows): phi < 1, and no replay exceeds 1. With sizing from the raw
    holdings (the pre-fix behaviour) the same replays report `sized_budget` (non-vacuity)."""
    import dataclasses

    import jevtrader.decide.risk as R
    from jevtrader.decide.select import rank_desc

    _, cfg, log = run
    cfg = dataclasses.replace(cfg, K=10, J=3.0)
    st = log.states[-1]
    uni = list(st.universe)
    bottom = list(rank_desc(st.L_T.reindex(uni)).sort_values().index[-10:])  # R^T > K + J: sold down only
    state = dataclasses.replace(st, sigma=pd.Series(0.02, index=uni), w_cur=pd.Series(1.003 / 10, index=bottom),
                                industry=pd.Series({s: f"G{k}" for k, s in enumerate(uni)}), risk=pd.DataFrame(index=uni))
    wd = B._decide(cfg, state, pd.Series(0.0, index=uni))
    assert wd.sized.phi < 1.0 and wd.sized.reference.sum() == pytest.approx(1.0, abs=1e-12)
    assert wd.sized.holding_scale == pytest.approx(1.003) and wd.sized.final.sum() <= 1 + TOL
    n, v = B.worst_case_entry(cfg, state)
    assert n > 0 and not v
    orig = R.sizing_holdings
    monkeypatch.setattr(R, "sizing_holdings", lambda w: ({s: x * orig(w)[1] for s, x in orig(w)[0].items()}, 1.0))
    assert B._decide(cfg, state, pd.Series(0.0, index=uni)).sized.final.sum() > 1 + TOL
    _, v_raw = B.worst_case_entry(cfg, state)
    assert v_raw and all("sized_budget" in x["failed"] for x in v_raw)
