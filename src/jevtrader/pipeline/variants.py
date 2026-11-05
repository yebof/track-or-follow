"""Registered Layer-B variants (design §6.6; design_v2_delta MA-12, BG-2 V-size-v1, BG-3c). Every variant is a $0
replay whose return series is appended to the trial registry (results/trials.jsonl), which feeds the deflated Sharpe.

Two kinds:
- configuration variants: a DecisionConfig change, replayed by `layer_b.evaluate` (both books, cards, metrics);
- composition variants (fusion/pools.py): scores built from the as-of agent heads that a base run logged, scored
  after the fact and turned into a D-sel-EW book (no band guard: a composition variant has no trust split).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from jevtrader.backtest.engine import run_backtest
from jevtrader.eval import metrics as M
from jevtrader.fusion import pools as P
from jevtrader.learn.features import CATALOG, TRUSTED_V1, UNTRUSTED_V1
from jevtrader.pipeline.layer_b import (
    LEARNING_YAML,
    ROOT,
    SYSTEM_YAML,
    DecisionConfig,
    EvalResult,
    LayerBInputs,
    evaluate,
)
from jevtrader.pipeline.registry import _plain, config_hash, register_trial

TRUSTED_CODE = tuple(f for f in TRUSTED_V1 if CATALOG[f].kind == "code")


def _alloc(**kw) -> Callable[[DecisionConfig], DecisionConfig]:
    return lambda c: dataclasses.replace(c, allocator=dataclasses.replace(c.allocator, **kw))


def _risk(**kw) -> Callable[[DecisionConfig], DecisionConfig]:
    return lambda c: dataclasses.replace(c, risk=dataclasses.replace(c.risk, **kw))


CONFIG_VARIANTS: dict[str, Callable[[DecisionConfig], DecisionConfig]] = {
    "V-codeonly": _alloc(trusted=TRUSTED_CODE, untrusted=("news_count",)),  # C3 contrast (design §6.6)
    "V-joint": _alloc(trusted=TRUSTED_V1 + UNTRUSTED_V1, untrusted=(), cap_mult=math.inf),  # no split, no cap
    "V-nocap": _alloc(cap_mult=math.inf),
    "V-kappa1": _alloc(cap_mult=1.0),
    "V-kappa4": _alloc(cap_mult=4.0),
    "V-J0": lambda c: dataclasses.replace(c, J=0.0),
    "V-J5": lambda c: dataclasses.replace(c, J=5.0),
    "V-J20": lambda c: dataclasses.replace(c, J=20.0),
    "V-Jinf": lambda c: dataclasses.replace(c, J=math.inf),
    "V-nodecay": _alloc(half_life_weeks=None),
    "V-size-v1": _risk(sizing="v1"),  # BG-2: the Day-1 sizing, its clean cost is part of the price (Fig. 2)
    "V-excl-items": _risk(exclusion_rule="items"),  # BG-3c: design_v1 item rule, no anchor
    "V-excl-bylines": _risk(exclusion_rule="bylines"),  # BG-3c: byline keys, no anchor
    "V-excl-collapsed": _risk(key_column="n_prob_collapsed"),  # BG-3c / SAN-5 sensitivity: desk bylines collapsed
    "V-static": lambda c: dataclasses.replace(c, freeze_after_weeks=26),  # design §6.6 (validity m4)
    # V-outlook (design §6.6, X4): the per-agent forecast-type views as the only S1 features
    "V-outlook": _alloc(trusted=TRUSTED_CODE + ("tech_view", "fund_view"), untrusted=("news_view",)),
}


# Reader variants (design §6.3, §13.1 C1; delta RB-5, RB-6, RB-15): a baseline reader is dropped into the untrusted
# slot in place of the nine news features, with its own gamma-hat from the same fit and the same cap kappa_t; the
# sentinel and policy features stay. (Which features a reader replaces is fixed with the C1 readers in the
# pre-registration; this is the reading of "replaces all W-channel news features", design §5.8.)
NEWS_FEATURES = tuple(f for f in UNTRUSTED_V1 if CATALOG[f].agent == "news")
READERS = {"H": ("h_outlook",), "H-guard": ("hguard_outlook",), "H-spot": ("hspot_outlook",), "HS": ("hs_tone",),
           "LM": ("lm_neg",), "R-med": ("f_med", "f_cov", "no_news"), "R-vote": ("f_vote", "f_cov", "no_news")}


def _reader(feats: tuple[str, ...]) -> Callable[[DecisionConfig], DecisionConfig]:
    def f(c: DecisionConfig) -> DecisionConfig:
        rest = tuple(x for x in c.allocator.untrusted if x not in NEWS_FEATURES)
        return dataclasses.replace(c, allocator=dataclasses.replace(c.allocator, untrusted=feats + rest))
    return f


for _r, _f in READERS.items():
    CONFIG_VARIANTS[f"V-reader-{_r}"] = _reader(_f)


def _drop(agent: str) -> Callable[[DecisionConfig], DecisionConfig]:
    """V-drop-{agent} (design §6.6, X1): leave one agent's features out of both learners, refit."""
    def f(c: DecisionConfig) -> DecisionConfig:
        keep = [x for x in c.allocator.trusted if CATALOG[x].agent != agent]
        keep_w = [x for x in c.allocator.untrusted if CATALOG[x].agent != agent]
        return dataclasses.replace(c, allocator=dataclasses.replace(c.allocator, trusted=tuple(keep), untrusted=tuple(keep_w)))
    return f


for _agent in dict.fromkeys(CATALOG[f].agent for f in TRUSTED_V1 + UNTRUSTED_V1):
    CONFIG_VARIANTS[f"V-drop-{_agent}"] = _drop(_agent)
COMPOSITION_VARIANTS = P.POOL_VARIANTS
ALL_VARIANTS = (*CONFIG_VARIANTS, *COMPOSITION_VARIANTS)


def variant_config(base: DecisionConfig, vid: str, learning: Path = LEARNING_YAML) -> DecisionConfig:
    """`base` with variant `vid` applied and named after it (V-static reads its fit window from learning_v1.yaml)."""
    if vid not in CONFIG_VARIANTS:
        raise KeyError(f"unknown configuration variant {vid!r}; known: {sorted(CONFIG_VARIANTS)}")
    cfg = CONFIG_VARIANTS[vid](base)
    if vid == "V-static":
        weeks = yaml.safe_load(Path(learning).read_text()).get("variants", {}).get("static_fit_weeks", 26)
        cfg = dataclasses.replace(cfg, freeze_after_weeks=int(weeks))
    return dataclasses.replace(cfg, name=vid)


def run_config_variant(base: DecisionConfig, vid: str, window: str, inputs: LayerBInputs | None = None, *,
                       root: Path = ROOT, register: bool = True, write: bool = True, **kw) -> EvalResult:
    return evaluate(variant_config(base, vid), window, inputs, root=root, register=register, write=write, **kw)


def pool_config(learning: Path = LEARNING_YAML, half_life_weeks: float | None = 52.0) -> P.PoolConfig:
    return P.PoolConfig.from_yaml(yaml.safe_load(Path(learning).read_text()), half_life_weeks)


def run_composition_variants(res: EvalResult, inputs: LayerBInputs, cfg: DecisionConfig, *,
                             pool_cfg: P.PoolConfig | None = None, variants: Sequence[str] = COMPOSITION_VARIANTS,
                             root: Path = ROOT, register: bool = True, system: Path = SYSTEM_YAML) -> dict:
    """Composition variants from a base run's logged heads and ledger: scores, the PD2 summary (IC, resolution-first
    scores, oracle-gap shares incl. the primary allocator), MA-6 pool scales, MA-7 dependence, MA-8 warm-up, and one
    registered D-sel-EW trial per variant (V-oracle is evaluation-only)."""
    log = res.log
    if log.heads is None:
        raise ValueError("the base run logged no agent heads (DecisionConfig.heads = False)")
    pool_cfg = pool_cfg or pool_config(half_life_weeks=cfg.allocator.half_life_weeks)
    labels = inputs.labels.restrict(log.dates[0], log.dates[-1])
    ledger = log.ledger.frame() if log.ledger is not None else None
    run = P.run_pools(log.heads, labels, pool_cfg, ledger=ledger, variants=(*variants, *P.EVAL_ONLY))
    refs = P.reference_forecasts(ledger, labels, prior_sd=cfg.allocator.prior_sd_beta,
                                 half_life_weeks=cfg.allocator.half_life_weeks) if ledger is not None else None
    summary = P.pool_summary(run, labels, primary=log.predictions[~log.predictions["cold"]], references=refs)
    out = {"summary": summary, "dependence": P.dependence_diagnostics(log.heads, labels),
           "warmup": {v: P.warmup_curve(s, labels) for v, s in run.scores.items() if v not in P.EVAL_ONLY},
           "diagnostics": run.diagnostics, "trials": [], "scores": run.scores}
    if inputs.prices is None:
        return out
    cold = log.predictions.groupby("t")["cold"].first().to_dict()
    reg = Path(root) / yaml.safe_load(Path(system).read_text()).get("registry", "results/trials.jsonl")
    for v, sc in run.scores.items():
        if v in P.EVAL_ONLY:
            continue
        books = P.pool_books(sc, cfg.K, cold, inputs.universe)
        bt = run_backtest(inputs.prices, books, cost_bps=cfg.cost_bps, cash_rate_daily=inputs.cash_rate_daily,
                          delisting_returns=dict(inputs.delisting_returns or {}), end=res.hold_end,
                          ma_exits=inputs.ma_exits)
        summary[v]["portfolio"] = M.summarize(bt.daily_returns, rf=bt.rf)
        if register:
            vsha = config_hash({"base_cfg_sha": res.cfg_sha, "variant": v, "pool_cfg": _plain(pool_cfg)})
            out["trials"].append(register_trial(
                reg, cfg_sha=vsha, name=f"{v}/sel_ew", window=res.window, returns=bt.daily_returns, rf=bt.rf,
                extra={"variant": v, "base_cfg_sha": res.cfg_sha, "gen_id": cfg.gen_id, "protocol": "P-clean",
                       "inputs_sha": res.inputs_sha, "hold_end": res.hold_end.date().isoformat() if res.hold_end is not None
                       else None, "kind": "composition"}))
    return out


def registered_names(variants: Mapping | Sequence = ALL_VARIANTS) -> list[str]:
    """Trial names a full variant sweep registers (both books for configuration variants, the signal book for
    composition variants): the MA-12 / PE-8 variant budget."""
    names = []
    for v in variants:
        names += [f"{v}/sel_ew", f"{v}/sized"] if v in CONFIG_VARIANTS else [f"{v}/sel_ew"]
    return names

