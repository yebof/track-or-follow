#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E1: why is the typed reader's attack success (ASR) above the holistic reader's?

    .venv/bin/python experiments/exploratory/e1_asr_mechanism.py      # -> results/exploratory/e1_asr_mechanism.json

Why exploratory. H1 (prereg §3.1) is decided: typed 10.5% vs holistic 0.1% (h1.json; Holm p = 1, the registered test
did not reject). Every decomposition below was specified after that outcome was known, so it explains the registered
estimate and never replaces it. No p-value is computed; intervals are descriptive. Nothing here changes a registered
configuration, threshold or variant (no tuning on W_clean): the replays reuse the registered clean runs and the RT-S
attacked batches as they are, and the counterfactual readers (rescaled or sign-flipped slopes) are diagnostic
constructions, never candidate systems.

Inputs ($0; files only, no model call). The clean Layer-B runs of every reader in the untrusted slot are replayed from
the judgment tables exactly as `experiments/redteam/rts.py score` builds them (pipeline.layer_b.evaluate,
register=False, write=False; the primary configuration with the logged DEV-1 `coverage` waiver; the H, H-spot and
H-guard clean generations on E_t stacked in, redteam.stages.stack_raw). Their predictions are checked against the
registered run directories. The attacked Layer-A features are the RT-S batch tables (typed arm: typed, H, LM, R-med,
R-vote; guards arm: H-spot, H-guard). Each promotion unit is replayed for each reader: the attacked week's untrusted
features with the unit's overrides, standardized exactly as eval.bound.attacked_week_logits (mirrored here so that
per-feature values are available; the mirror's L^W equals that function's output, checked), times the week's learned
gamma, capped at kappa_t; success = the name enters D-sel-EW (decide.select.select on L^T and L^T + Lbar^W, the only
input of the signal book), eligible = it is outside the reader's clean D-sel-EW. The as-logged replay must reproduce
the registered score tables' `eligible` and `success` exactly (checked; the script stops otherwise).

Decomposition (all per reader; E_t = the in-band names R^T <= K + J = 30 of the measurement week, where selection
happens and where the P-1 readers are covered):
1. Learned slopes: gamma per untrusted feature over the learned weeks (mean, share of weeks > 0, last week).
2. Clean text influence per learned week: sd over E_t of L^W and of the capped Lbar^W (also over U_t), sd(Lbar^W) /
   kappa_t, the share of names whose |L^W| exceeds kappa; the book's text dependence: the share of D-sel-EW that is
   not the trusted top-K (J = 0) book, and the share of E_t names whose D-sel-EW membership flips when their own
   Lbar^W is set to 0 (others unchanged).
3. ASR per unit of clean influence: ASR / mean over units of sd_E(Lbar^W)/kappa at the unit's week (week-cluster
   bootstrap of the ratio, B = 10,000, seed 0).
4. The attack's shift of the attacked name: dL^W (attacked - clean, before the cap) in kappa units and in units of the
   reader's clean sd_E(L^W) that week, its sign, and the combined-rank displacement.
5. Influence-matched replay: each reader's L^W (clean and attacked) is multiplied, per week, by
   target_sd / sd_E(clean L^W of that reader), with the target the typed reader's sd (A) or the holistic reader's (B),
   then capped at kappa_t and replayed (eligibility recomputed on the rescaled clean book).
6. Sign-aligned counterfactual (holistic family and the dictionary reader): the reader's own text feature's
   coefficient flipped in sign (gamma_j -> -gamma_j for h_outlook / hspot_outlook / hguard_outlook / lm_neg / f_med /
   f_vote), as learned and influence-matched to typed. It asks what the static promotional attack would do if the
   learned slope on the reader's reading had the typed reader's sign; it is not a system anyone would run.
7. Per-family ASR (A1-A7) of every reader and of nodef, from the registered score tables (attack-week-clustered
   bootstrap CI, B = 10,000, seed 0).
8. Which typed features the attacks moved: per-feature contribution change dc_j = gamma_j (x~_att - x~_clean) of the
   attacked name (all attacks, successes), the share of successes in which feature j is the largest positive
   contributor, raw feature changes, and a knock-out replay: the typed ASR when feature j's attacked values are
   reverted to clean (all names of the week), one feature at a time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402

NAME = "e1_asr_mechanism"
GEN = "main_emu_wclean_u100"
READER_GENS = ("h_emu_wclean_et", "hspot_emu_wclean_et", "hguard_emu_wclean_et")
READERS = {  # reader -> (variant, arm with its attacked tables, its own text feature(s))
    "typed": (None, "typed", ("f_imp", "f_cont", "f_rev", "f_sur", "f_net", "f_und", "f_story", "f_cov", "no_news")),
    "H": ("V-reader-H", "typed", ("h_outlook",)),
    "H-spot": ("V-reader-H-spot", "guards", ("hspot_outlook",)),
    "H-guard": ("V-reader-H-guard", "guards", ("hguard_outlook",)),
    "LM": ("V-reader-LM", "typed", ("lm_neg",)),
    "R-med": ("V-reader-R-med", "typed", ("f_med",)),
    "R-vote": ("V-reader-R-vote", "typed", ("f_vote",)),
}
REGISTERED_RUNS = {"typed": "primary", "H": "V-reader-H", "H-spot": "V-reader-H-spot", "H-guard": "V-reader-H-guard",
                   "LM": "V-reader-LM", "R-med": "V-reader-R-med", "R-vote": "V-reader-R-vote"}
HOLISTIC = ("H", "H-spot", "H-guard")
FAMILIES = ("A1", "A2", "A3", "A4", "A5", "A6", "A7")
N_BOOT, SEED = 10_000, 0
TOL = 1e-12


# ------------------------------------------------------------------------------------------------ clean runs
def clean_runs(root: Path, guard) -> tuple[object, object, dict]:
    from jevtrader.pipeline.layer_b import LEARNING_YAML, SYSTEM_YAML, DecisionConfig, evaluate, load_inputs
    from jevtrader.pipeline.variants import variant_config
    from jevtrader.redteam.stages import stack_raw

    start, end = guard.check_window("W_clean", "E1 clean runs")
    base = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=GEN, name="primary",
                                    allow_missing=("coverage",))
    inputs = load_inputs(base, start, end, root, data_root=C.data_root(guard))
    inputs = dataclasses.replace(inputs, raw=stack_raw(inputs.raw, [root / "results/judgments" / g for g in
                                                                     READER_GENS], inputs.industry or {}))
    runs = {}
    for reader, (vid, _, _) in READERS.items():
        cfg = base if vid is None else variant_config(base, vid)
        res = evaluate(cfg, "W_clean", inputs, root=root, register=False, write=False)
        runs[reader] = (cfg, res)
        print(f"  clean replay {reader}: IC {res.metrics['prediction']['ic_mean']:.6f}", flush=True)
    return base, inputs, runs


def check_registered(root: Path, guard, runs: dict) -> dict:
    """Max |diff| of L, L^W and Lbar^W between each replay and its registered run directory."""
    out = {}
    for reader, name in REGISTERED_RUNS.items():
        d = C.eval_dirs(root, "W_clean", name=name, gen_id=GEN)
        if len(d) != 1:
            out[reader] = {"status": f"registered run not unique ({len(d)})"}
            continue
        reg = C.read_eval_run(d[0], guard, f"E1 registered {name}").predictions
        mine = runs[reader][1].log.predictions.copy()
        mine["t"] = pd.to_datetime(mine["t"])
        m = mine.merge(reg, on=["t", "symbol"], suffixes=("", "_reg"))
        out[reader] = {"run_dir": C.rel(d[0], root), "rows": len(m),
                       **{f"max_abs_diff_{c}": float((m[c] - m[f"{c}_reg"]).abs().max()) for c in ("L", "L_W",
                                                                                                 "L_W_bar", "L_T")}}
    return out


# ------------------------------------------------------------------------------------------------ mirror of attacked_week_logits
def week_x(cfg, raw_t: pd.DataFrame, st, snap: dict, scales: dict, overrides: dict) -> tuple[pd.DataFrame, pd.Series]:
    """eval.bound.attacked_week_logits, returning the standardized untrusted features x (U_t x features) and gamma
    instead of their product (same operations, same order)."""
    from jevtrader.learn.features import CATALOG, rank_gaussian, wide_week

    feats = list(cfg.allocator.untrusted)
    uni = list(st.universe)
    val, cov = wide_week(raw_t, uni, feats)
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
    return x, gamma, val


def sel_names(L_T: pd.Series, lwb: pd.Series, uni, K: int, J: float) -> frozenset:
    from jevtrader.decide.select import select

    fr = pd.DataFrame({"L_T": L_T.reindex(uni).astype(float), "L": (L_T.reindex(uni) + lwb.reindex(uni).fillna(0.0))
                       .astype(float)})
    return frozenset(select(fr, K, J).names)


def cap(lw: pd.Series, kappa: float) -> pd.Series:
    return np.sign(lw) * np.minimum(np.abs(lw), kappa)


# ------------------------------------------------------------------------------------------------ bootstrap helpers
def ci_asr(success, cluster) -> dict:
    from jevtrader.eval.metrics import asr_cluster_bootstrap

    s = np.asarray(success, dtype=float)
    if not len(s):
        return {"n": 0, "asr": None, "ci": None}
    b = asr_cluster_bootstrap(s, np.asarray(cluster), n_boot=N_BOOT, seed=SEED)
    return {"n": int(len(s)), "asr": float(s.mean()), "ci": list(b["ci"]), "n_clusters": b["n_clusters"]}


def ci_paired(a, b, cluster) -> dict:
    """Paired difference mean(a) - mean(b) with the attack-week-clustered percentile CI (no p-value: exploratory)."""
    from jevtrader.eval.metrics import asr_cluster_bootstrap

    r = asr_cluster_bootstrap(np.asarray(a, float), np.asarray(cluster), np.asarray(b, float), n_boot=N_BOOT,
                              seed=SEED)
    return {"n": r["n_units"], "a": r["asr_a"], "b": r["asr_b"], "diff": r["estimate"], "ci": list(r["ci"]),
            "n_clusters": r["n_clusters"]}


def ci_ratio(num, den, cluster) -> dict:
    """mean(num) / mean(den) with an attack-week-clustered percentile CI."""
    num, den, cl = np.asarray(num, float), np.asarray(den, float), np.asarray(cluster)
    codes, idx = np.unique(cl, return_inverse=True)
    G = len(codes)
    sn, sd_, nc = (np.bincount(idx, weights=num, minlength=G), np.bincount(idx, weights=den, minlength=G),
                   np.bincount(idx, minlength=G).astype(float))
    rng = np.random.default_rng(SEED)
    boot = np.empty(N_BOOT)
    for i in range(0, N_BOOT, 1000):
        k = rng.integers(0, G, size=(min(1000, N_BOOT - i), G))
        boot[i:i + len(k)] = (sn[k].sum(1) / nc[k].sum(1)) / (sd_[k].sum(1) / nc[k].sum(1))
    est = float(num.mean() / den.mean())
    return {"estimate": est, "ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "numerator_mean": float(num.mean()), "denominator_mean": float(den.mean()), "n": int(len(num))}


def nw_mean(x) -> dict:
    import stats as S

    v = np.asarray(pd.Series(x, dtype=float).dropna())
    t = S.nw_test(v, S.NW_LAGS, "two-sided")
    return {"mean": t["mean"], "ci": t["ci"], "n": int(len(v))}


# ------------------------------------------------------------------------------------------------ clean influence
def clean_influence(cfg, res) -> tuple[dict, pd.DataFrame]:
    log = res.log
    p = log.predictions.copy()
    p["t"] = pd.to_datetime(p["t"])
    p = p[~p["cold"].astype(bool)]
    K, J = cfg.K, cfg.J
    rows = []
    for t, g in p.groupby("t"):
        g = g.set_index("symbol")
        E = g[g["R_T"] <= K + J]
        kappa = float(g["kappa"].iloc[0])
        uni = list(g.index)
        with_text = sel_names(g["L_T"], g["L_W_bar"], uni, K, J)
        trusted = sel_names(g["L_T"], pd.Series(0.0, index=uni), uni, K, J)
        flips = 0
        for s in E.index:
            lw0 = g["L_W_bar"].copy()
            lw0[s] = 0.0
            flips += int((s in with_text) != (s in sel_names(g["L_T"], lw0, uni, K, J)))
        rows.append({"t": t, "kappa": kappa, "sd_LW_E": float(E["L_W"].std(ddof=1)),
                     "sd_LWbar_E": float(E["L_W_bar"].std(ddof=1)), "sd_LW_U": float(g["L_W"].std(ddof=1)),
                     "sd_LWbar_U": float(g["L_W_bar"].std(ddof=1)), "sd_LT_U": float(g["L_T"].std(ddof=1)),
                     "capped_share_U": float((g["L_W"].abs() > kappa).mean()),
                     "book_text_swap_share": len(with_text - trusted) / K,
                     "pivotal_share_E": flips / len(E) if len(E) else np.nan})
    w = pd.DataFrame(rows).set_index("t")
    w["sd_LWbar_E_over_kappa"] = w["sd_LWbar_E"] / w["kappa"]
    summ = {c: nw_mean(w[c]) for c in ("sd_LW_E", "sd_LWbar_E", "sd_LWbar_E_over_kappa", "sd_LW_U", "sd_LWbar_U",
                                       "capped_share_U", "book_text_swap_share", "pivotal_share_E")}
    summ["kappa_mean"] = float(w["kappa"].mean())
    summ["n_weeks"] = int(len(w))
    return summ, w


def slopes(res, own: tuple[str, ...]) -> dict:
    snaps = [s for s in res.log.snapshots if s.get("n_train", 0) > 0]
    G = pd.DataFrame([s["gamma"] for s in snaps])
    SE = pd.DataFrame([s["se_gamma"] for s in snaps]) if snaps and "se_gamma" in snaps[0] else None
    out = {}
    for f in G.columns:
        out[f] = {"mean": float(G[f].mean()), "share_weeks_positive": float((G[f] > 0).mean()),
                  "last": float(G[f].iloc[-1]), "min": float(G[f].min()), "max": float(G[f].max()),
                  "mean_se": float(SE[f].mean()) if SE is not None and f in SE else None, "own_text_feature": f in own}
    return {"n_learned_weeks": int(len(G)), "gamma": out}


# ------------------------------------------------------------------------------------------------ attacked replays
def replay_arm(root: Path, arm: str, readers: list[str], runs: dict, raw_by_t: dict, clean_w: dict,
               max_batches: int | None) -> tuple[pd.DataFrame, dict]:
    """One row per (unit, family, reader) of a promotion batch of `arm` (offset 0), with the as-logged,
    influence-matched, sign-flipped and (typed) knock-out replays."""
    from jevtrader.decide.select import rank_desc
    from jevtrader.redteam import run as RT

    plan = json.loads((root / "results/redteam/rts" / arm / "plan.json").read_text())
    dirs = [root / "results/redteam/rts" / arm / b["batch_id"] for b in plan["batch_list"]]
    dirs = [d for d in dirs if (d / "MANIFEST.json").exists()][: max_batches or None]
    ctx = {}
    for r in readers:
        cfg, res = runs[r]
        log = res.log
        pred = log.predictions.copy()
        pred["t"] = pd.to_datetime(pred["t"])
        ctx[r] = {"cfg": cfg, "states": {pd.Timestamp(s.t): s for s in log.states},
                  "snaps": {x["t"]: x for x in log.snapshots}, "scales": {x["t"]: x for x in log.scales},
                  "pred": {t: g.set_index("symbol") for t, g in pred.groupby("t")}}
    typed_feats = list(runs["typed"][0].allocator.untrusted)
    rows, check = [], {"mirror_max_abs_diff_LW": 0.0, "clean_mirror_max_abs_diff_LW": 0.0}
    for bi, d in enumerate(dirs):
        man = json.loads((d / "MANIFEST.json").read_text())
        b = man["batch"]
        eval_t = pd.Timestamp(b["eval_t"])
        C.Guard(root).check([eval_t], f"E1 batch {b['batch_id']}")
        feats = RT.attacked_features(d)
        raw_t = raw_by_t.get(eval_t, pd.DataFrame(columns=["t", "symbol", "feature", "value", "covered"]))
        for r in readers:
            c = ctx[r]
            cfg, st = c["cfg"], c["states"].get(eval_t)
            if st is None:
                continue
            key = eval_t.date().isoformat()
            snap, scales, g = c["snaps"][key], c["scales"][key], c["pred"][eval_t]
            uni = list(st.universe)
            K, J, kappa = cfg.K, cfg.J, st.kappa
            E = [s for s in uni if int(g.at[s, "R_T"]) <= K + J]
            x0, gam, v0 = week_x(cfg, raw_t, st, snap, scales, {})
            LW0 = (x0 * gam).sum(axis=1)
            check["clean_mirror_max_abs_diff_LW"] = max(check["clean_mirror_max_abs_diff_LW"],
                                                        float((LW0 - g["L_W"].reindex(uni)).abs().max()))
            LW0 = g["L_W"].reindex(uni).astype(float)  # the logged clean L^W (the mirror equals it up to rounding)
            sd0 = float(LW0[E].std(ddof=1))
            clean_book = sel_names(st.L_T, st.L_W_bar, uni, K, J)
            rank_clean = rank_desc(st.L_T.reindex(uni) + st.L_W_bar.reindex(uni).fillna(0.0))
            own = [f for f in READERS[r][2] if f in gam.index]
            for u in b["units"]:
                if u["kind"] != RT.PROMOTION:
                    continue
                s = u["symbol"]
                ov = RT.unit_overrides(feats, cfg.allocator.untrusted, eval_t, u["affected"])
                x1, _, v1 = week_x(cfg, raw_t, st, snap, scales, ov)
                LW1 = (x1 * gam).sum(axis=1)
                if r == "typed" or bi % 25 == 0:  # the mirror vs the registered function, on a regular sample
                    from jevtrader.eval.bound import attacked_week_logits

                    ref = attacked_week_logits(cfg, raw_t, _Log(st, snap, scales), eval_t, ov)["L_W"]
                    check["mirror_max_abs_diff_LW"] = max(check["mirror_max_abs_diff_LW"],
                                                          float((ref.reindex(uni) - LW1).abs().max()))
                lwb1 = cap(LW1, kappa)
                att_book = sel_names(st.L_T, lwb1, uni, K, J)
                rank_att = rank_desc(st.L_T.reindex(uni) + lwb1.reindex(uni).fillna(0.0))
                row = {"unit_id": u["unit_id"], "family": b["family"], "t_attack": pd.Timestamp(b["t"]),
                       "eval_t": eval_t, "symbol": s, "reader": r, "R_T": int(g.at[s, "R_T"]), "kappa": kappa,
                       "eligible": s not in clean_book, "success": (s in att_book) and (s not in clean_book),
                       "rank_disp": int(rank_clean[s]) - int(rank_att[s]),
                       "dLW": float(LW1[s] - LW0[s]), "dLWbar": float(lwb1[s] - st.L_W_bar.get(s, 0.0)),
                       "sd_LW_E": sd0, "sd_LWbar_E": float(st.L_W_bar.reindex(E).std(ddof=1)),
                       "n_overrides": len(ov)}
                # influence-matched replays: target = typed's / H's clean sd over E_t of the same week
                for tag, ref_r in (("to_typed", "typed"), ("to_H", "H")):
                    tgt = clean_w[ref_r]["sd_LW_E"].get(eval_t, np.nan)
                    a = tgt / sd0 if sd0 > 0 and np.isfinite(tgt) else np.nan
                    row[f"scale_{tag}"] = a
                    if np.isfinite(a):
                        cb = sel_names(st.L_T, cap(a * LW0, kappa), uni, K, J)
                        ab = sel_names(st.L_T, cap(a * LW1, kappa), uni, K, J)
                        row[f"eligible_{tag}"], row[f"success_{tag}"] = s not in cb, (s in ab) and (s not in cb)
                # sign-aligned counterfactual: the reader's own text coefficient(s) flipped
                if r != "typed" and own:
                    flip0 = LW0 - 2 * (x0[own] * gam[own]).sum(axis=1)
                    flip1 = LW1 - 2 * (x1[own] * gam[own]).sum(axis=1)
                    sdf = float(flip0[E].std(ddof=1))
                    for tag, a in (("flip", 1.0), ("flip_to_typed", clean_w["typed"]["sd_LW_E"].get(eval_t, np.nan)
                                                    / sdf if sdf > 0 else np.nan)):
                        if np.isfinite(a):
                            cb = sel_names(st.L_T, cap(a * flip0, kappa), uni, K, J)
                            ab = sel_names(st.L_T, cap(a * flip1, kappa), uni, K, J)
                            row[f"eligible_{tag}"], row[f"success_{tag}"] = s not in cb, (s in ab) and (s not in cb)
                    row["d_own"] = float(((x1[own] - x0[own]) * gam[own]).sum(axis=1)[s])
                    row["d_own_raw"] = float(np.nansum([v1.at[s, f] - v0.at[s, f] for f in own
                                                        if np.isfinite(v1.at[s, f]) and np.isfinite(v0.at[s, f])]))
                    row["own_raw_clean"] = float(v0.at[s, own[0]]) if np.isfinite(v0.at[s, own[0]]) else np.nan
                    row["own_raw_att"] = float(v1.at[s, own[0]]) if np.isfinite(v1.at[s, own[0]]) else np.nan
                if r == "typed":
                    dc = (x1.loc[s] - x0.loc[s]) * gam
                    for f in typed_feats:
                        row[f"dc_{f}"] = float(dc[f])
                        dv = v1.at[s, f] - v0.at[s, f] if np.isfinite(v1.at[s, f]) and np.isfinite(v0.at[s, f]) \
                            else np.nan
                        row[f"draw_{f}"] = float(dv)
                        row[f"cov_changed_{f}"] = bool(np.isfinite(v1.at[s, f]) != np.isfinite(v0.at[s, f]))
                        if (x1[f] - x0[f]).abs().max() > 0:  # knock-out: feature f reverted to clean for all names
                            ko = LW1 - x1[f] * gam[f] + x0[f] * gam[f]
                            row[f"success_ko_{f}"] = (s in sel_names(st.L_T, cap(ko, kappa), uni, K, J)) and \
                                (s not in clean_book)
                        else:
                            row[f"success_ko_{f}"] = row["success"]
                rows.append(row)
        if (bi + 1) % 100 == 0:
            print(f"  {arm}: {bi + 1}/{len(dirs)} batches", flush=True)
    return pd.DataFrame(rows), check


class _Log:
    """The slice of a DecisionLog that eval.bound.attacked_week_logits reads, for one week."""

    def __init__(self, st, snap, scales):
        self.states, self.snapshots, self.scales = [st], [snap], [scales]


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--max-batches", type=int, default=None, help="development only: first N batches per arm")
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    guard.check_window("W_clean", "E1")
    print("E1: clean replays", flush=True)
    base, inputs, runs = clean_runs(root, guard)
    reg_check = check_registered(root, guard, runs)
    influence, weekly = {}, {}
    for r, (cfg, res) in runs.items():
        influence[r], weekly[r] = clean_influence(cfg, res)
    slope = {r: slopes(res, READERS[r][2]) for r, (cfg, res) in runs.items()}
    raw = inputs.raw.copy()
    raw["t"] = pd.to_datetime(raw["t"])
    raw_by_t = {t: g for t, g in raw.groupby("t")}
    print("E1: attacked replays", flush=True)
    parts, checks = [], {}
    for arm in ("typed", "guards"):
        rs = [r for r, v in READERS.items() if v[1] == arm]
        df, chk = replay_arm(root, arm, rs, runs, raw_by_t, weekly, a.max_batches)
        parts.append(df)
        checks[arm] = chk
    rep = pd.concat(parts, ignore_index=True)
    rep["cluster"] = pd.to_datetime(rep["t_attack"])

    # the as-logged replay must equal the registered score tables
    sc = pd.concat([pd.read_parquet(root / "results/redteam/rts" / arm / "scores.parquet") for arm in
                    ("typed", "guards")], ignore_index=True)
    sc = sc[(sc["kind"] == "promotion") & (sc["offset"] == 0) & sc["scored"].astype(bool)]
    m = rep.merge(sc[["unit_id", "family", "reader", "eligible", "success", "rank_L_clean", "rank_L_att"]],
                  on=["unit_id", "family", "reader"], how="left", suffixes=("", "_reg"))
    mism = m[(m["eligible"] != m["eligible_reg"].astype(bool)) | (m["success"] != m["success_reg"].astype(bool))
             | (m["rank_disp"] != (m["rank_L_clean"] - m["rank_L_att"]))]
    consistency = {"registered_runs": reg_check, "mirror": checks, "rows": int(len(rep)),
                   "rows_matched_to_scores": int(m["eligible_reg"].notna().sum()),
                   "mismatches_vs_registered_scores": int(len(mism))}
    if len(mism) or m["eligible_reg"].isna().any():
        print(mism.head(10).T)
        raise SystemExit(f"E1: the as-logged replay differs from the registered score tables ({len(mism)} rows)")

    out_readers = list(READERS)
    asr = {}
    for r in out_readers:
        g = rep[(rep["reader"] == r) & rep["eligible"]]
        asr[r] = {"as_logged": ci_asr(g["success"], g["cluster"])}
        for tag in ("to_typed", "to_H", "flip", "flip_to_typed"):
            if f"eligible_{tag}" in rep:
                h = rep[(rep["reader"] == r) & rep.get(f"eligible_{tag}", False).fillna(False).astype(bool)]
                if len(h):
                    asr[r][f"matched_{tag}" if tag.startswith("to_") else tag] = ci_asr(
                        h[f"success_{tag}"].astype(bool), h["cluster"])
    # paired typed - holistic at matched influence (same unit-families eligible for both)
    paired = {}
    for tag in ("as_logged", "to_typed", "to_H"):
        el, su = ("eligible", "success") if tag == "as_logged" else (f"eligible_{tag}", f"success_{tag}")
        for other in ("H", "H-spot", "H-guard", "LM", "R-med", "R-vote"):
            a_ = rep[rep["reader"] == "typed"].set_index(["unit_id", "family"])
            b_ = rep[rep["reader"] == other].set_index(["unit_id", "family"])
            j = a_[[el, su, "cluster"]].join(b_[[el, su]], rsuffix="_o", how="inner")
            j = j[j[el].fillna(False).astype(bool) & j[f"{el}_o"].fillna(False).astype(bool)]
            if len(j):
                paired[f"{tag}|typed-{other}"] = ci_paired(j[su].astype(float), j[f"{su}_o"].astype(float),
                                                           j["cluster"])
    # ASR per unit of clean influence (the unit's week: sd over E_t of the capped Lbar^W, in kappa units)
    per_infl = {}
    for r in out_readers:
        g = rep[(rep["reader"] == r) & rep["eligible"]]
        per_infl[r] = ci_ratio(g["success"].astype(float), g["sd_LWbar_E"] / g["kappa"], g["cluster"])
    # the attack's shift of the attacked name
    shift = {}
    for r in out_readers:
        g = rep[rep["reader"] == r]
        e = g[g["eligible"]]
        d = {"all": g, "eligible": e, "successes": e[e["success"]]}
        shift[r] = {k: {"n": int(len(v)), "dLW_over_kappa_mean": float((v["dLW"] / v["kappa"]).mean()) if len(v)
                        else None,
                        "dLW_over_sd_clean_mean": float((v["dLW"] / v["sd_LW_E"]).mean()) if len(v) else None,
                        "dLW_over_sd_clean_median": float((v["dLW"] / v["sd_LW_E"]).median()) if len(v) else None,
                        "share_dLW_positive": float((v["dLW"] > 0).mean()) if len(v) else None,
                        "share_dLW_negative": float((v["dLW"] < 0).mean()) if len(v) else None,
                        "rank_displacement_mean": float(v["rank_disp"].mean()) if len(v) else None}
                    for k, v in d.items()}
        if "d_own" in g:
            v = g.dropna(subset=["d_own"])
            shift[r]["own_feature"] = {
                "n": int(len(v)), "d_own_contribution_over_kappa_mean": float((v["d_own"] / v["kappa"]).mean()),
                "share_own_raw_up": float((v["d_own_raw"] > 0).mean()),
                "share_own_raw_down": float((v["d_own_raw"] < 0).mean()),
                "own_raw_delta_mean": float(v["d_own_raw"].mean()),
                "share_own_contribution_negative": float((v["d_own"] < 0).mean())}
        shift[r]["by_family"] = {f: {"n": int(len(v)), "dLW_over_sd_clean_mean": float((v["dLW"] / v["sd_LW_E"])
                                                                                     .mean()) if len(v) else None,
                                     "share_dLW_positive": float((v["dLW"] > 0).mean()) if len(v) else None}
                                 for f, v in g.groupby("family")}
    # per-family ASR from the registered score tables (every reader, nodef included)
    fam = {}
    allsc = pd.concat([pd.read_parquet(root / "results/redteam/rts" / arm / "scores.parquet") for arm in
                       ("typed", "guards", "nodef")], ignore_index=True)
    allsc = allsc[(allsc["kind"] == "promotion") & (allsc["offset"] == 0) & allsc["scored"].astype(bool)
                  & allsc["eligible"].astype(bool)]
    from jevtrader.redteam.run import cluster_weeks

    allsc["cluster"] = cluster_weeks(allsc)
    for f in FAMILIES:
        fam[f] = {r: ci_asr(g["success"].astype(float), g["cluster"])
                  for r, g in allsc[allsc["family"] == f].groupby("reader")}
    fam_paired = {}
    for f in FAMILIES:
        for other in ("H", "nodef"):
            a_ = allsc[(allsc["family"] == f) & (allsc["reader"] == "typed")].set_index("unit_id")
            b_ = allsc[(allsc["family"] == f) & (allsc["reader"] == other)].set_index("unit_id")
            j = a_[["success", "cluster"]].join(b_[["success"]], rsuffix="_o", how="inner")
            if len(j):
                fam_paired[f"{f}|typed-{other}"] = ci_paired(j["success"].astype(float),
                                                            j["success_o"].astype(float), j["cluster"])
    # which typed features the attacks moved
    ty = rep[rep["reader"] == "typed"]
    tf = list(runs["typed"][0].allocator.untrusted)
    succ = ty[ty["eligible"] & ty["success"]]
    el = ty[ty["eligible"]]
    dcs = succ[[f"dc_{f}" for f in tf]]
    largest = dcs.idxmax(axis=1).str.replace("dc_", "", regex=False) if len(dcs) else pd.Series(dtype=str)
    feats_out = {"n_eligible": int(len(el)), "n_successes": int(len(succ)), "features": {}}
    for f in tf:
        ko = el[f"success_ko_{f}"].astype(bool)
        feats_out["features"][f] = {
            "dc_mean_all_eligible_over_kappa": float((el[f"dc_{f}"] / el["kappa"]).mean()),
            "dc_mean_successes_over_kappa": float((succ[f"dc_{f}"] / succ["kappa"]).mean()) if len(succ) else None,
            "share_successes_dc_positive": float((succ[f"dc_{f}"] > TOL).mean()) if len(succ) else None,
            "share_successes_largest_positive_contributor": float((largest == f).mean()) if len(succ) else None,
            "raw_delta_mean_successes": float(succ[f"draw_{f}"].mean()) if len(succ) else None,
            "raw_delta_mean_all_eligible": float(el[f"draw_{f}"].mean()),
            "share_eligible_raw_changed": float((el[f"draw_{f}"].abs() > TOL).mean()),
            "share_eligible_coverage_changed": float(el[f"cov_changed_{f}"].mean()),
            "knockout_asr": ci_asr(ko, el["cluster"]),
            "successes_lost_by_knockout": int((el["success"] & ~ko).sum())}
    feats_out["by_family_largest_contributor"] = {
        fm: (largest[succ["family"] == fm].value_counts().to_dict() if len(succ) else {}) for fm in FAMILIES}
    payload = {
        "question": "Why is typed ASR above holistic (H1's registered estimate: 10.5% vs 0.1%)?",
        "definitions": {"E_t": "in-band names R^T <= K + J = 30 of the measurement week",
                        "influence": "sd over E_t of the capped untrusted logit Lbar^W, in kappa_t units",
                        "matched": "each reader's L^W (clean and attacked) x target sd / its own clean sd over E_t "
                                   "that week, capped at kappa_t; eligibility recomputed on the rescaled clean book",
                        "flip": "the reader's own text-feature coefficient(s) sign-flipped (diagnostic only)",
                        "cluster": "the unit's attack week (redteam.run.cluster_weeks), B = 10,000, seed 0"},
        "registered_reference": {"h1_confirmatory": C.read_json(root / "results/analysis/h1.json")["confirmatory"],
                                 "source": "results/analysis/h1.json (repeated, not recomputed)"},
        "slopes": slope, "clean_influence": influence, "asr": asr, "paired": paired,
        "asr_per_unit_influence": per_infl, "attack_shift": shift, "per_family_asr": fam,
        "per_family_paired": fam_paired, "typed_features": feats_out, "consistency": consistency,
        "weekly_influence": {r: w.reset_index().assign(t=lambda x: x["t"].dt.date.astype(str)).to_dict("records")
                             for r, w in weekly.items()},
        "settings": {"n_boot": N_BOOT, "seed": SEED, "K": base.K, "J": base.J, "readers": READERS,
                     "reader_gens": READER_GENS, "max_batches": a.max_batches}}
    inputs_ = {"typed scores": root / "results/redteam/rts/typed/scores.parquet",
               "guards scores": root / "results/redteam/rts/guards/scores.parquet",
               "nodef scores": root / "results/redteam/rts/nodef/scores.parquet",
               "typed plan": root / "results/redteam/rts/typed/plan.json",
               "guards plan": root / "results/redteam/rts/guards/plan.json",
               "MAIN features": root / "results/judgments" / GEN / "features.parquet",
               **{f"reader features {g}": root / "results/judgments" / g / "features.parquet" for g in READER_GENS},
               "h1": root / "results/analysis/h1.json"}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs_, argv=argv,
                why="decomposition of the registered H1 estimate specified after it was known")
    for r in out_readers:
        s = asr[r]
        print(f"{r:8s} ASR {s['as_logged']['asr']:.4f} matched->typed "
              f"{(s.get('matched_to_typed') or {}).get('asr')} flip->typed {(s.get('flip_to_typed') or {}).get('asr')}"
              f" infl {influence[r]['sd_LWbar_E_over_kappa']['mean']:.3f}")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
