#!/usr/bin/env python3
"""Fig. 2 data: the price of the bound over the band half-width J and the cap multiplier mu (prereg §4.2 PRICE;
design_v2_delta RK-5; outline §3 Fig. 2 as amended by the draft-0 review: y = paired clean selection return).

    .venv/bin/python experiments/analysis/frontier.py                    # registered: MAIN on W_clean + RT-S batches
    .venv/bin/python experiments/analysis/frontier.py --gen-id pre_blinded_emu_wpre_s30 --window W_pre_slice

**Replays ($0).** For each reader in the untrusted slot (typed programs: the primary configuration; holistic: the
registered V-reader-H, when its feature is in the generation) and each point of the two sweeps, J in {0, 5, 10, 20,
inf} at the registered cap and mu in {1, 2, 4, inf} at J = 10, Layer B is replayed from the judgment tables
(pipeline.layer_b.evaluate, register=False: these replays draw figure points, not headline candidates; the typed
J/kappa variants are registered trials already through run_layer_b). The operating point (J = 10, mu = 2) is replayed
once and shared by both sweeps.

**y: clean selection return.** Rank-IC is computed on L, which J does not change, so the band's price is measured on
the signal book: for every learned week, the equal-weighted mean of r - m over the names of D-sel-EW (T1's
open-to-open relative return, gross of costs), in bps per week. Each point is the paired weekly difference against the
same reader's J = 0 book (the trusted-only book), with a NW(4) 95% CI. The value of the text = typed J = inf minus
J = 0; the price of the band at the operating point = typed J = inf minus J = 10; its share of the value.

**x: in-band ASR.** With RT-S attacked batches (MANIFEST.json + features.parquet + risk.parquet per batch; default
results/redteam/rts/typed/*/), every unit is re-scored against each replayed configuration by
redteam.run.score_batch, so every J and mu is scored on the same units; ASR = share of eligible promotion (and
bound) rows at the measurement week whose name entered D-sel-EW, pooled over families, with a 95% CI clustered on
the unit's attack week (eval.metrics.asr_cluster_bootstrap with redteam.run.cluster_weeks, B = 10,000, seed 0).
Without batches the x values are absent.

**Caption numbers.** The clean costs of anchor-gated exclusion (primary vs V-excl-items, the design_v1 item rule
without an anchor) and of reference-anchored sizing (primary vs V-size-v1): the paired weekly excess return of the
system book (D-sized, net of costs, over DFF, compounded over each holding week), mean with a NW(4) CI, in bps.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import stats as S

NAME = "frontier"
DEFAULT_GEN = "main_emu_wclean_u100"
J_VALUES = (0.0, 5.0, 10.0, 20.0, math.inf)
MU_VALUES = (1.0, 2.0, 4.0, math.inf)
OPERATING = {"J": 10.0, "mu": 2.0}
N_BOOT, SEED = 10_000, 0
DEFAULT_BATCHES = "results/redteam/rts/typed/*/MANIFEST.json"
HOLISTIC_VARIANT, HOLISTIC_FEATURE = "V-reader-H", "h_outlook"
CLEAN_COSTS = {"anchor_rule": "V-excl-items", "reference_sizing": "V-size-v1"}
READER_VARIANTS = {"H": ("V-reader-H", "h_outlook"), "H-guard": ("V-reader-H-guard", "hguard_outlook"),
                   "H-spot": ("V-reader-H-spot", "hspot_outlook"), "LM": ("V-reader-LM", "lm_neg"),
                   "R-med": ("V-reader-R-med", "f_med"), "R-vote": ("V-reader-R-vote", "f_vote"),
                   "code-only": ("V-codeonly", None)}
PARITY_DELTA = 0.02  # prereg §4.3 (P-5): clean weekly rank-IC, typed vs another reader


def memo_evaluate(evaluate=None):
    """layer_b.evaluate memoized on (config hash, window): the operating point, the primary cell and the clean-cost
    base are one replay each, whichever block asks for them."""
    from jevtrader.pipeline.layer_b import evaluate as _evaluate

    f = evaluate or _evaluate
    cache: dict = {}

    def ev(cfg, window, inputs, **kw):
        k = (cfg.sha, window)
        if k not in cache:
            cache[k] = f(cfg, window, inputs, **kw)
        return cache[k]

    ev.cache = cache
    return ev


def label(v: float) -> str:
    return "inf" if math.isinf(v) else f"{v:g}"


def configure(cfg, J: float | None = None, mu: float | None = None):
    if J is not None:
        cfg = dataclasses.replace(cfg, J=float(J))
    if mu is not None:
        cfg = dataclasses.replace(cfg, allocator=dataclasses.replace(cfg.allocator, cap_mult=float(mu)))
    return cfg


def selection_returns(log, labels) -> pd.Series:
    """Per learned week: mean of r - m over the D-sel-EW names, bps."""
    lab = labels.table.set_index(["t", "symbol"])
    rel = (lab["r"] - lab["m"])
    cold = log.predictions.groupby("t")["cold"].first()
    out = {}
    for t, book in log.targets["sel_ew"].items():
        t = pd.Timestamp(t)
        if bool(cold.get(t, True)) or not book:
            continue
        v = [rel.get((t, s), np.nan) for s in book]
        v = [x for x in v if np.isfinite(x)]
        out[t] = 1e4 * float(np.mean(v)) if v else np.nan
    return pd.Series(out, dtype=float).sort_index()


def weekly_excess(daily: pd.Series, rf: pd.Series, dates) -> pd.Series:
    """Compounded book return minus compounded cash over each holding week (fills after d_t to the next fill), bps."""
    r = daily.dropna()
    c = rf.reindex(r.index).fillna(0.0)
    edges = [pd.Timestamp(d) for d in sorted(dates)]
    out = {}
    for i, d in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else r.index.max()
        m = (r.index > d) & (r.index <= hi)
        if m.any():
            out[d] = 1e4 * (float((1 + r[m]).prod() - (1 + c[m]).prod()))
    return pd.Series(out, dtype=float)


def paired_bps(a: pd.Series, b: pd.Series) -> dict:
    d = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").dropna()
    t = S.nw_test((d["a"] - d["b"]).to_numpy(), S.NW_LAGS, "two-sided")
    return {"mean_bps": t["mean"], "ci": t["ci"], "n_weeks": t["n"], "p_two_sided": t["p"]}


def asr_for(scores: pd.DataFrame, reader: str) -> dict:
    from jevtrader.eval.metrics import asr_cluster_bootstrap

    s = scores[(scores["reader"] == reader) & scores["scored"].astype(bool) & (scores["offset"] == 0)
               & scores["kind"].isin(["promotion", "bound"])]
    n_scored = len(s)
    s = s[s["eligible"].astype(bool)] if "eligible" in s else s
    if not len(s):
        return {"n": 0, "n_scored": n_scored, "asr": None, "ci": None}
    from jevtrader.redteam.run import cluster_weeks

    # clustered on the unit's attack week, as H1 (redteam.run.cluster_weeks; prereg §3.1)
    bs = asr_cluster_bootstrap(s["success"].astype(float), cluster_weeks(s), n_boot=N_BOOT, seed=SEED)
    # a unit already in this configuration's own clean book is not eligible for it (redteam.run, step 4)
    return {"n": len(s), "n_scored": n_scored, "n_excluded": n_scored - len(s),
            "asr": float(s["success"].astype(float).mean()), "ci": bs["ci"], "n_clusters": bs["n_clusters"]}


def run_sweeps(base, inputs, window: str, root: Path, readers: dict, batch_dirs=(), evaluate=None) -> dict:
    """Replay every (reader, sweep point), score the batches against every replay, and summarise (module doc)."""
    from jevtrader.pipeline.layer_b import evaluate as _evaluate
    from jevtrader.redteam import run as RT

    evaluate = evaluate or _evaluate
    cache: dict[str, object] = {}

    def replay(cfg):
        if cfg.sha not in cache:
            cache[cfg.sha] = evaluate(cfg, window, inputs, root=root, register=False, write=False)
        return cache[cfg.sha]

    points = []
    for reader, rcfg in readers.items():
        for sweep, values in (("J", J_VALUES), ("mu", MU_VALUES)):
            for v in values:
                cfg = configure(rcfg, J=v if sweep == "J" else OPERATING["J"],
                                mu=v if sweep == "mu" else OPERATING["mu"])
                points.append({"reader": reader, "sweep": sweep, "value": v, "cfg": cfg})
    for p in points:
        p["res"] = replay(p["cfg"])
        p["sel"] = selection_returns(p["res"].log, inputs.labels)
    scores = None
    if batch_dirs:
        uniq = {}
        for p in points:
            p["run_key"] = f"{p['reader']}|{p['cfg'].sha[:12]}"
            uniq.setdefault(p["run_key"], p)
        raw_t = pd.to_datetime(inputs.raw["t"])
        parts = []
        for d in batch_dirs:
            # the scorer reads the raw untrusted features of the scored weeks only (eval.bound.attacked_week_logits),
            # so each batch gets that slice: the same numbers, without filtering the whole table per unit and replay
            weeks = pd.to_datetime(json.loads((Path(d) / "MANIFEST.json").read_text())["batch"]["weeks"])
            raw = inputs.raw[raw_t.isin(weeks).to_numpy()]
            runs = [RT.CleanRun(k, p["cfg"], p["res"].log, raw) for k, p in uniq.items()]
            parts.append(RT.score_batch(d, runs, labels=None, root=root))
        scores = pd.concat(parts, ignore_index=True)
    out = {"readers": {}, "operating_point": {"J": OPERATING["J"], "mu": OPERATING["mu"]}}
    for reader in readers:
        mine = [p for p in points if p["reader"] == reader]
        j0 = next(p for p in mine if p["sweep"] == "J" and p["value"] == 0.0)
        block = {"J": {}, "mu": {}}
        for p in mine:
            pr = p["res"].metrics.get("prediction", {})
            block[p["sweep"]][label(p["value"])] = {
                "value": p["value"], "cfg_sha": p["cfg"].sha, "ic_mean": pr.get("ic_mean"),
                "selection_return_bps": float(p["sel"].mean()) if len(p["sel"]) else None,
                "paired_vs_J0": paired_bps(p["sel"], j0["sel"]),
                "asr": asr_for(scores, p["run_key"]) if scores is not None else {"n": 0, "asr": None, "ci": None,
                                                                                 "status": "no attacked batches"}}
        out["readers"][reader] = block
    t = out["readers"].get("typed")
    if t:
        inf, ten, zero = (next(p for p in points if p["reader"] == "typed" and p["sweep"] == "J" and p["value"] == v)
                          for v in (math.inf, 10.0, 0.0))
        value = paired_bps(inf["sel"], zero["sel"])
        price = paired_bps(inf["sel"], ten["sel"])
        share = (price["mean_bps"] / value["mean_bps"]) if value["mean_bps"] not in (None, 0) else None
        excl = value["ci"][0] is not None and (value["ci"][0] > 0 or value["ci"][1] < 0)
        out["value_of_text"] = {**value, "ci_excludes_zero": bool(excl)}
        out["price_of_band"] = {**price, "share_of_value": share}
    out["n_replays"] = len(cache)
    out["n_scored_rows"] = 0 if scores is None else len(scores)
    return out


def clean_by_reader(base, inputs, window: str, root: Path, evaluate=None) -> dict:
    """Every registered reader variant at the operating point whose feature the generation has: clean selection
    return (bps/week) and IC, paired against the typed programs (selection return: NW CI; weekly IC: NW CI and the
    prereg §4.3 TOST at delta = PARITY_DELTA, the only way a parity statement may be made)."""
    from jevtrader.pipeline.layer_b import evaluate as _evaluate
    from jevtrader.pipeline.variants import variant_config

    evaluate = evaluate or _evaluate
    feats = set(inputs.raw["feature"])
    typed = evaluate(base, window, inputs, root=root, register=False, write=False)
    t_sel = selection_returns(typed.log, inputs.labels)
    t_ic = pd.Series({pd.Timestamp(k): v for k, v in typed.metrics["prediction"].get("ic_by_week", {}).items()})
    out = {"typed": {"status": "ok", "selection_return_bps": float(t_sel.mean()),
                     "ic_mean": typed.metrics["prediction"].get("ic_mean")}}
    for name, (vid, feat) in READER_VARIANTS.items():
        if feat is not None and feat not in feats:
            out[name] = {"status": f"not run: {feat} is not in the generation", "variant": vid}
            continue
        r = evaluate(variant_config(base, vid), window, inputs, root=root, register=False, write=False)
        sel = selection_returns(r.log, inputs.labels)
        ic = pd.Series({pd.Timestamp(k): v for k, v in r.metrics["prediction"].get("ic_by_week", {}).items()})
        d = pd.concat([t_ic.rename("a"), ic.rename("b")], axis=1, join="inner").dropna()
        out[name] = {"status": "ok", "variant": vid, "selection_return_bps": float(sel.mean()) if len(sel) else None,
                     "selection_return_vs_typed": paired_bps(sel, t_sel),
                     "ic_mean": r.metrics["prediction"].get("ic_mean"),
                     "ic_typed_minus_reader": S.nw_test((d["a"] - d["b"]).to_numpy(), S.NW_LAGS, "two-sided"),
                     "tost_ic": S.tost_nw((d["a"] - d["b"]).to_numpy(), PARITY_DELTA)}
    return out


def clean_costs(base, inputs, window: str, root: Path, evaluate=None) -> dict:
    from jevtrader.pipeline.layer_b import evaluate as _evaluate
    from jevtrader.pipeline.variants import variant_config

    evaluate = evaluate or _evaluate
    prim = evaluate(base, window, inputs, root=root, register=False, write=False)
    rf = inputs.cash_rate_daily if inputs.cash_rate_daily is not None else pd.Series(0.0, index=inputs.prices.sessions)
    dates = prim.log.dates
    base_w = weekly_excess(prim.returns["sized"], rf, dates)
    out = {}
    for k, vid in CLEAN_COSTS.items():
        try:
            v = evaluate(variant_config(base, vid), window, inputs, root=root, register=False, write=False)
        except Exception as e:  # noqa: BLE001 -- recorded (e.g. an input check the variant needs), never silent
            out[k] = {"variant": vid, "status": f"not run: {type(e).__name__}: {e}"}
            continue
        out[k] = {"variant": vid, "status": "ok", "definition": f"primary D-sized minus {vid} D-sized",
                  **paired_bps(base_w, weekly_excess(v.returns["sized"], rf, dates))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", default=DEFAULT_GEN)
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--batches", nargs="*", default=None, help=f"attacked batch dirs (default: {DEFAULT_BATCHES})")
    ap.add_argument("--no-clean-costs", action="store_true")
    ap.add_argument("--allow-missing", default="", help="Layer-B input checks waived (as run_layer_b.py)")
    ap.add_argument("--reader-gens", nargs="*", default=[],
                    help="clean reader generations stacked into the inputs (P-1 runs on E_t; redteam.stages.stack_raw)")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    start, end = guard.check_window(a.window, "Fig. 2 frontier")
    if a.window == C.REGISTERED_WINDOW and C.d1_start(root)["applied"]:
        # a moved D1 start needs the replays cold-started there: pass --window <start>:<end> and label it
        C.require_d1_start(start, root, "Fig. 2 frontier", a.window)
    from jevtrader.pipeline.layer_b import LEARNING_YAML, SYSTEM_YAML, DecisionConfig, load_inputs
    from jevtrader.pipeline.variants import variant_config

    allow = tuple(x for x in a.allow_missing.split(",") if x)
    base = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=a.gen_id, name="primary", allow_missing=allow)
    inputs = load_inputs(base, start, end, root, data_root=C.data_root(guard))
    if a.reader_gens:
        from jevtrader.redteam.stages import stack_raw

        inputs = dataclasses.replace(inputs, raw=stack_raw(inputs.raw, [Path(g) for g in a.reader_gens],
                                                           inputs.industry or {}))
    readers = {"typed": base}
    status = {"typed": "ok"}
    if HOLISTIC_FEATURE in set(inputs.raw["feature"]):
        readers["H"] = variant_config(base, HOLISTIC_VARIANT)
        status["H"] = "ok"
    else:
        status["H"] = f"not run: {HOLISTIC_FEATURE} is not in the generation's features (H's clean run is missing)"
    batch_dirs = ([Path(b) for b in a.batches] if a.batches is not None
                  else [p.parent for p in sorted(root.glob(DEFAULT_BATCHES))])
    # redteam.run.score_batch checks the lock on every scored week of a batch before it reads the attacked features
    ev = memo_evaluate()
    res = run_sweeps(base, inputs, a.window, root, readers, batch_dirs, evaluate=ev)
    res["reader_status"] = status
    res["batches"] = [C.rel(b, root) for b in batch_dirs]
    if not a.no_clean_costs:
        res["clean_costs"] = clean_costs(base, inputs, a.window, root, evaluate=ev)
        res["clean_by_reader"] = clean_by_reader(base, inputs, a.window, root, evaluate=ev)
    res["n_replays_total"] = len(ev.cache)
    res["settings"] = {"J_values": [label(v) for v in J_VALUES], "mu_values": [label(v) for v in MU_VALUES],
                       "operating": OPERATING, "nw_lags": S.NW_LAGS, "n_boot": N_BOOT, "seed": SEED,
                       "gen_id": a.gen_id}
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, res, script=__file__, window=a.window, guard=guard,
                   inputs={"judgment features": root / "results/judgments" / a.gen_id / "features.parquet",
                           **{f"reader features {Path(g).name}": Path(g) / "features.parquet" for g in a.reader_gens}},
                   argv=argv)
    v = res.get("value_of_text", {})
    print(f"frontier {a.window}: {res['n_replays']} replays; value of text {v.get('mean_bps')} bps/wk CI "
          f"{v.get('ci')} -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
