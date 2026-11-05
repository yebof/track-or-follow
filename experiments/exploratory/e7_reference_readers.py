#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E7: reference readers and a paired reader contrast for the registered H2 frame
($0; local CPU; no model API call).

    .venv/bin/python experiments/exploratory/e7_reference_readers.py score   # FinBERT tone of every frame text
    .venv/bin/python experiments/exploratory/e7_reference_readers.py run     # -> results/exploratory/e7_reference_readers.json

Why exploratory. Specified on 2026-09-29 after the unlock, the registered H2 outcome and the round-1 paper reviews
(docs/reviews/paper_v2_r1_pc.md M3: "no learned reference reader"; paper_v2_r1_stats.md B3: the verbalized contrast
and emu were compared only by overlapping marginal intervals). Estimates and 95% intervals only; no p-value, no parity
claim (the registration allows none on per-implementation H2 rows, prereg §4.3); nothing is tuned.

1. FinBERT tone (a learned finance-sentiment reference reader). The pinned checkpoint yiyanghkust/finbert-tone
   (revision 4921590d...; its model card names Huang, Wang and Yang's FinBERT paper, huang2023finbert, and says it is
   FinBERT fine-tuned on analyst-report sentences labelled positive, negative or neutral) reads the raw headline + " " +
   summary of every frame text (max 128 tokens), zero-shot for this task, in the local-model environment
   (e7_finbert_worker.py, 4 threads). Tone = P(positive) - P(negative). On the registered H2 frame (as E5 and E6):
   the weekly statistic of tone vs AR[0,+1], on all units and on units without a price report (E5's regex), its
   placebo AR[-5,-1], the item AUC against the sign of AR[0,+1] (week-block bootstrap, as E5), and the Spearman
   correlation of tone with emu's E[effect] (agreement, not validity).
2. Paired contrast of the verbalized reader and emu on the verbalized reader's 2,000 registered sample units (the
   registered per-implementation rows' units): the pooled Spearman difference rho(E_VERB, AR) - rho(E_emu, AR) with a
   week-block bootstrap interval (2,000 draws, seed 0; the same resampled weeks for both readers), next to the VERB
   pooled MDE of the pre-registration's power output (results/prereg/power_prereg.json).

Texts are licensed: they are written only to data/cache/e7/ (gitignored) for the worker, never to results/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402
import e5_construct_checks as E5  # noqa: E402  (exploratory frame, text and classifier, read-only)
import h2_validity as H2  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402

NAME = "e7_reference_readers"
CACHE = C.ROOT / "data" / "cache" / "e7"
SCORES = X.OUT_DIR / "e7_finbert_scores.csv"
WORKER = Path(__file__).resolve().parent / "e7_finbert_worker.py"
NG_PY = C.ROOT / ".venv-ng" / "bin" / "python"
VERB_GEN = "verb_llm-verbalized_wclean_items2000"
N_BOOT, SEED, TOL = 2000, 0, 1e-12


def classified(root: Path, guard):
    spec, _ = E5.load_spec()
    f, start, end, d1 = E5.frame(root, guard)
    return E5.classify(E5.attach_text(f), spec), start, end


def cmd_score(root: Path, threads: int) -> int:
    guard = C.Guard(root)
    g, _, _ = classified(root, guard)  # Layer-A answers and texts only; no outcome
    t = g[g["has_text"]].drop_duplicates("eid")
    CACHE.mkdir(parents=True, exist_ok=True)
    inp = CACHE / "texts.jsonl"
    with open(inp, "w") as fh:
        for e, h, s in zip(t["eid"], t["headline"].fillna(""), t["summary"].fillna("")):
            fh.write(json.dumps({"eid": e, "text": f"{h} {s}".strip()}) + "\n")
    print(f"{len(t)} distinct texts -> worker", flush=True)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    rc = subprocess.call(["nice", "-n", "10", str(NG_PY), "-I", str(WORKER), str(inp), str(SCORES),
                          "--threads", str(threads)])
    inp.unlink(missing_ok=True)  # licensed text never stays on disk longer than the run
    return rc


def wk(f: pd.DataFrame, x: str, y: str = "ar") -> dict:
    return E5.wk(f.dropna(subset=[x, y]), x, y)


def verb_contrast(root: Path, frame: pd.DataFrame) -> dict:
    it = H2.implementation_items(root / "results/judgments" / VERB_GEN)
    e = frame.set_index("unit")
    it = it[it["unit"].isin(e.index)].copy()
    it["ar"] = e.loc[it["unit"], "ar"].to_numpy(dtype=float)
    it["E_emu"] = e.loc[it["unit"], "E_effect"].to_numpy(dtype=float)
    fin = np.isfinite(it["ar"].to_numpy(dtype=float))
    rep = {"rho_verb_own_units": S.spearman(it.loc[fin & np.isfinite(it["E_effect"]), "E_effect"],
                                            it.loc[fin & np.isfinite(it["E_effect"]), "ar"]),
           "rho_emu_own_units": S.spearman(it.loc[fin & np.isfinite(it["E_emu"]), "E_emu"],
                                           it.loc[fin & np.isfinite(it["E_emu"]), "ar"])}
    it = it[np.isfinite(it["E_effect"]) & np.isfinite(it["E_emu"]) & np.isfinite(it["ar"])]

    def diff(h: pd.DataFrame) -> float:
        return S.spearman(h["E_effect"], h["ar"]) - S.spearman(h["E_emu"], h["ar"])

    bs = S.week_block_bootstrap(it, "t", diff, n_boot=N_BOOT, seed=SEED)
    reg = C.read_json(root / "results/analysis/h2.json")["per_implementation"]["VERB"]
    power = C.read_json(root / "results/prereg/power_prereg.json")
    mde = power["baselines_reduced"]["VERB"]["reduced"]["spearman_pooled_mde"]
    out = {"n_units": int(len(it)), "n_weeks": int(it["t"].nunique()),
           "rho_verb": S.spearman(it["E_effect"], it["ar"]), "rho_emu": S.spearman(it["E_emu"], it["ar"]),
           "difference": bs["estimate"], "ci": bs["ci"], "n_boot": N_BOOT, "seed": SEED,
           "registered_rho_verb": reg["spearman_ar"]["rho"], "registered_rho_emu_same_units": reg["emu_same_units"]["rho"],
           "verb_pooled_mde_registered": mde,
           "reproduction": rep,
           "note": "estimate and interval only; the registration allows no parity claim on per-implementation rows"}
    for k, r in (("rho_verb_own_units", out["registered_rho_verb"]),
                 ("rho_emu_own_units", out["registered_rho_emu_same_units"])):
        if abs(rep[k] - r) > 1e-9:
            raise SystemExit(f"E7: VERB units do not reproduce the registered pooled rows ({k}: {rep[k]} vs {r})")
    return out


def cmd_run(root: Path, argv) -> int:
    guard = C.Guard(root)
    g, start, end = classified(root, guard)
    guard.check(g["t"].unique(), "E7 labels")
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    g = H2.add_abnormal_returns(g, px.book)
    reg = C.read_json(root / "results/analysis/h2.json")
    mine = H2.weekly_test(g)["mean"]
    if abs(mine - reg["confirmatory"]["mean"]) > TOL:
        raise SystemExit(f"E7: frame does not reproduce h2.json ({mine})")
    sc = pd.read_csv(SCORES).drop_duplicates("eid")
    sc["tone"] = sc["p_positive"] - sc["p_negative"]
    g = g.merge(sc[["eid", "tone", "p_neutral"]], on="eid", how="left")
    no = g[~g["price"]]
    fin = {"reader": "FinBERT (yiyanghkust/finbert-tone @ 4921590d), tone = P(positive) - P(negative)",
           "coverage_units": int(g["tone"].notna().sum()), "units": int(len(g)),
           "all": wk(g, "tone"), "no_price": wk(no, "tone"), "placebo_all": wk(g, "tone", "ar_placebo"),
           "item_auc_all": E5.auc_block(g.dropna(subset=["tone"]), "tone"),
           "item_auc_no_price": E5.auc_block(no.dropna(subset=["tone"]), "tone"),
           "agreement_with_emu_spearman": S.spearman(g.dropna(subset=["tone"])["tone"],
                                                     g.dropna(subset=["tone"])["E_effect"]),
           "share_top_neutral": float((g["p_neutral"] >= 0.5).mean())}
    payload = {"question": "a learned finance-sentiment reference reader on the H2 frame; the verbalized reader vs emu "
                           "on the same units",
               "frame": {"n_units": int(len(g)), "n_weeks": int(g["t"].nunique()), "h2_mean_recomputed": mine},
               "finbert": fin, "verb_minus_emu_same_units": verb_contrast(root, g),
               "settings": {"n_boot": N_BOOT, "seed": SEED, "nw_lags": S.NW_LAGS, "ci": "two-sided 95%"}}
    inputs = {"finbert scores": SCORES, "h2": root / "results/analysis/h2.json",
              "VERB answers": root / "results/judgments" / VERB_GEN / "answers.parquet"}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs, argv=argv,
                why="reference reader and paired contrast specified after the H2 outcome and the round-1 reviews")
    v = payload["verb_minus_emu_same_units"]
    print(f"FinBERT: all {fin['all']['mean']:.3f} {fin['all']['ci']}; no_price {fin['no_price']['mean']:.3f}; "
          f"AUC {fin['item_auc_all']['auc']:.3f}; VERB-emu {v['difference']:+.3f} {v['ci']}")
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("score", "run"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    return cmd_score(root, a.threads) if a.cmd == "score" else cmd_run(root, argv)


if __name__ == "__main__":
    sys.exit(main())
