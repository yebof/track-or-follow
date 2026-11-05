#!/usr/bin/env python3
"""End-to-end development run of every analysis entry point and report generator on the W_pre slice (P-8 test).

    nice -n 10 .venv/bin/python experiments/analysis/run_wpre.py [--n-random 200] [--skip frontier,pd1]

Inputs: the PRE generation `pre_blinded_emu_wpre_s30` (60 decisions x S30, blinded emu) and its Layer-B run
directories under results/eval/*/W_pre_slice__*/ (the primary cell and the registered variants). The W_pre slice is
the development window (contaminated for the emulator: no performance claim), so every number written here is a
pipeline check, never a result. Outputs go to results/analysis/W_pre_slice/ and the tables and figures to
results/analysis/W_pre_slice/report/ (never paper/).

W_pre has no RT-S run, no H clean run and no MEM run, so three analyses use stated stand-ins:
- H1 and the Fig. 2 ASR axis: feature-level synthetic plants (experiments/analysis/feature_plants.py) on units sampled
  by redteam.run.sample_units from the real clean logs, scored by redteam.run.score_batch; the holistic reader's
  place is taken by the registered V-reader-R-med (label "R-med"), whose clean book differs from the typed one;
- PD3: V-drop-news stands in for the all-stores-off replay;
- D1: the registered PROBE generations (probe_emu_d1d2 + probe_emu_d1_reask) answered the W_pre-slice schedule too, so
  the rule runs on real probe answers against the real W_pre labels, centred on 2023-07 (a development exercise of
  the registered code path; the registered rule is centred on 2024-07 and runs after the unlock);
- BOUND: the primary run's bound_check.json (the property tests run in the test suite, so --skip-tests here);
- localization: RT-S's W_pre mock rehearsal (data/cache/redteam/rts_wpre_mock; mock answers, mechanics only);
- RT-A-NG: the summary of the 2-item dry run (results/redteam/rta_ng/dry; not the registered run).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

WINDOW, GEN = "W_pre_slice", "pre_blinded_emu_wpre_s30"
PLANTS = Path("data/cache/analysis_wpre_plants")
STANDIN = ("R-med", "V-reader-R-med", ("f_med",))
N_PROMOTION, N_BOUND, UNIT_SEED = 60, 20, "RT-S-www27"
REPORTS = ("table1_operations", "table2_typed_interface", "table3_validity", "table4_portfolio", "fig2_frontier",
           "fig3_decision_card")


def plants(root: Path, out: Path) -> tuple[Path, list[Path]]:
    """Clean logs of the typed reader and the stand-in, sampled units, plant batches and their scores."""
    from feature_plants import score, write_batches

    from jevtrader.pipeline.layer_b import LEARNING_YAML, SYSTEM_YAML, DecisionConfig, evaluate, load_inputs
    from jevtrader.pipeline.variants import NEWS_FEATURES, variant_config
    from jevtrader.redteam import run as RT

    guard = C.Guard(root)
    start, end = guard.check_window(WINDOW, "W_pre plants")
    base = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=GEN, name="primary")
    inputs = load_inputs(base, start, end, root, data_root=C.data_root(guard))
    alt = variant_config(base, STANDIN[1])
    runs = {"typed": (base, evaluate(base, WINDOW, inputs, root=root, register=False, write=False).log),
            STANDIN[0]: (alt, evaluate(alt, WINDOW, inputs, root=root, register=False, write=False).log)}
    logs = {"typed": runs["typed"][1], "H": runs[STANDIN[0]][1]}
    units = RT.sample_units(logs, base.K, base.J, n_promotion=N_PROMOTION, n_bound=N_BOUND, n_demotion=0, n_anchor=0,
                            seed=UNIT_SEED, root=root)
    pdir = root / PLANTS
    if pdir.exists():
        import shutil

        shutil.rmtree(pdir)
    typed_feats = [f for f in base.allocator.untrusted if f in NEWS_FEATURES]
    dirs = write_batches(pdir / "batches", units, inputs.raw, runs["typed"][1], typed_feats, runs[STANDIN[0]][1],
                         list(STANDIN[2]))
    scores = score(dirs, runs, inputs.raw, inputs.labels, root)
    sp = pdir / "scores.parquet"
    scores.to_parquet(sp, index=False)
    units.to_parquet(pdir / "units.parquet", index=False)
    (out / "plants_note.json").parent.mkdir(parents=True, exist_ok=True)
    (out / "plants_note.json").write_text(json.dumps({
        "what": "feature-level synthetic plants on W_pre (experiments/analysis/feature_plants.py); not RT-S",
        "units": {"promotion": int((units["kind"] == "promotion").sum()), "bound": int((units["kind"] == "bound").sum()),
                  "seed": UNIT_SEED}, "standin_for_holistic": {"label": STANDIN[0], "variant": STANDIN[1],
                                                               "pushed_features": list(STANDIN[2])},
        "typed_pushed_features": typed_feats, "batches": len(dirs), "score_rows": len(scores)}, indent=1))
    return sp, dirs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--skip", default="")
    ap.add_argument("--root", default=str(C.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    skip = set(filter(None, a.skip.split(",")))
    out = C.out_path("x", WINDOW, None, root).parent
    t0 = time.time()
    log = {}

    def step(name, fn):
        if name in skip:
            log[name] = "skipped"
            return
        t = time.time()
        fn()
        log[name] = round(time.time() - t, 1)
        print(f"[{time.time() - t0:6.0f}s] {name} done ({log[name]}s)", flush=True)

    import confirmatory
    import d1_window_start
    import frontier
    import h1_asr
    import h2_validity
    import h3_signal
    import pd0_operations
    import pd1_portfolio
    import pd2_composition
    import pd3_stores

    common = ["--window", WINDOW, "--root", str(root)]
    step("h2", lambda: h2_validity.main(["--gen", GEN, "--ng", "", *common]))
    step("h3", lambda: h3_signal.main(["--gen-id", GEN, *common]))
    step("d1", lambda: d1_window_start.main([*common, "--center", "2023-07"]))
    step("d1_schedule", lambda: d1_window_start.main([*common, "--schedule"]))
    import d2_reid

    step("d2", lambda: d2_reid.main(common))
    import bound_claim
    import localization
    import rta_ng_summary

    step("bound", lambda: bound_claim.main(["--gen-id", GEN, "--skip-tests", *common]))
    mock = root / "data/cache/redteam/rts_wpre_mock"
    if (mock / "typed" / "scores.parquet").exists():
        step("localization", lambda: localization.main(["--arms", str(mock / "typed"), str(mock / "nodef"), *common]))
    dry = root / "results/redteam/rta_ng/dry"
    if (dry / "asr_budget.json").exists():
        step("rta_ng", lambda: rta_ng_summary.main(["--run-dir", str(dry), *common]))
    step("pd0", lambda: pd0_operations.main(["--typed", GEN, "--run-tag", "PRE/B/", "--reader", "",
                                             "--no-scaling", *common]))
    step("pd1", lambda: pd1_portfolio.main(["--gen-id", GEN, "--n-random", str(a.n_random), *common]))
    step("pd2", lambda: pd2_composition.main(["--gen-id", GEN, *common]))
    off = C.find_eval_dir(root, WINDOW, "V-drop-news", GEN)
    step("pd3", lambda: pd3_stores.main(["--gen-id", GEN, "--stores-off", str(off), *common]))
    holder: dict = {}
    step("plants", lambda: holder.update(zip(("scores", "dirs"), plants(root, out))))
    if "scores" in holder:
        step("h1", lambda: h1_asr.main(["--scores", str(holder["scores"]), "--holistic", STANDIN[0], *common]))
        step("frontier", lambda: frontier.main(["--gen-id", GEN, "--batches", *map(str, holder["dirs"]), *common]))
    step("confirmatory", lambda: confirmatory.main(common))
    rep = out / "report"
    sys.path.insert(0, str(C.ROOT / "experiments" / "report"))
    cards = C.find_eval_dir(root, WINDOW, "primary", GEN) / "cards.jsonl"
    for name in REPORTS:
        mod = __import__(name)
        extra = ["--cards", str(cards)] if name == "fig3_decision_card" else []
        step(name, lambda m=mod, e=extra: m.main(["--in-dir", str(out), "--out-dir", str(rep), *e]))
    (out / "run_wpre.json").write_text(json.dumps({"steps_s": log, "total_s": round(time.time() - t0, 1),
                                                   "finished_utc": pd.Timestamp.now(tz="UTC").isoformat()}, indent=1))
    print(json.dumps(log))
    return 0


if __name__ == "__main__":
    sys.exit(main())
