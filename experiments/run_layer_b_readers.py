"""Registering Layer-B replays of the reader variants whose clean reader generation lives outside MAIN (DEV-10,
docs/design/prereg_deviations.md). Written after the unlock; it adds no analysis and no variant.

    .venv/bin/python experiments/run_layer_b_readers.py --allow-missing coverage \
        --reader H=results/judgments/h_emu_wclean_et --reader H-spot=results/judgments/hspot_emu_wclean_et \
        [--reader H-guard=results/judgments/hguard_emu_wclean_et]

Why. The registered trial set (prereg §4.4, M = 73) contains V-reader-H, V-reader-H-guard and V-reader-H-spot (both
books). Their readers' clean runs are, by the registered deviation P-1, separate generations on E_t
(`rts.py clean-readers`), so `run_layer_b.py --variants all` over MAIN's generation skipped them (no `*_outlook`
feature) and no registered entry point writes their trial rows. `pd1_portfolio.py` then exits by its registered strict
trial rule. This driver does what `run_layer_b.py` does for every other configuration variant, on the inputs that
`redteam.stages.reader_log` (hashed, used by rts.py for H1) and `frontier.py --reader-gens` (DEV-5) already use:

  cfg    = DecisionConfig.from_yaml(learning, system, gen_id=MAIN, name="primary", allow_missing=--allow-missing)
           (run_layer_b.py's construction; with `coverage` waived it is the registered primary 4ea23931..., DEV-1)
  inputs = load_inputs(cfg, W_clean) with the reader generation stacked in (redteam.stages.stack_raw)
  evaluate(variant_config(cfg, "V-reader-<reader>"), "W_clean", inputs, register=True, write=True)

`evaluate` appends the two books' rows (V-reader-<reader>/sel_ew, /sized) to results/trials.jsonl and writes the run
directory results/eval/<cfg16>/W_clean__<inputs12>/ (the stacked inputs have their own inputs hash, so no existing run
directory is touched). A summary with this script's SHA-256 and argv goes to results/eval/reader_variants_W_clean.json.
No model is called; the W_clean lock applies inside `evaluate`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.pipeline.layer_b import (  # noqa: E402
    LEARNING_YAML,
    SYSTEM_YAML,
    DecisionConfig,
    evaluate,
    load_inputs,
    resolve_window,
)
from jevtrader.pipeline.registry import _plain  # noqa: E402
from jevtrader.pipeline.variants import variant_config  # noqa: E402
from jevtrader.redteam.stages import stack_raw  # noqa: E402

READERS = ("H", "H-spot", "H-guard")  # the registered reader variants covered on E_t only (prereg §5, P-1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gen-id", default="main_emu_wclean_u100")
    ap.add_argument("--window", default="W_clean")
    ap.add_argument("--reader", action="append", default=[], help="READER=generation dir (repeatable)")
    ap.add_argument("--allow-missing", default="")
    ap.add_argument("--root", default=str(ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    allow = tuple(x for x in a.allow_missing.split(",") if x)
    readers = dict(x.split("=", 1) for x in a.reader)
    bad = sorted(set(readers) - set(READERS))
    if bad or not readers:
        raise SystemExit(f"--reader must name some of {READERS}; got {sorted(readers)}")
    cfg = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=a.gen_id, name="primary", allow_missing=allow)
    start, end = resolve_window(a.window)
    base = load_inputs(cfg, start, end, root)
    out = {}
    for reader, gdir in readers.items():
        g = Path(gdir)
        if not (g / "features.parquet").exists():
            raise SystemExit(f"{reader}: no features.parquet in {g} (its clean run did not complete)")
        inputs = dataclasses.replace(base, raw=stack_raw(base.raw, [g], base.industry or {}))
        vcfg = variant_config(cfg, f"V-reader-{reader}")
        res = evaluate(vcfg, a.window, inputs, root=root, register=True, write=True)
        out[f"V-reader-{reader}"] = {
            "reader_generation": str(g.relative_to(root) if g.is_absolute() else g),
            "reader_features_sha256": hashlib.sha256((g / "features.parquet").read_bytes()).hexdigest(),
            "cfg_sha": res.cfg_sha, "inputs_sha": res.inputs_sha, "out_dir": str(res.out_dir.relative_to(root)),
            "prediction": {k: res.metrics["prediction"].get(k) for k in ("ic_mean", "ic_t_nw4", "log_loss", "auc",
                                                                         "n_weeks_scored")},
            "portfolio_sized": res.metrics.get("portfolio_sized"),
            "portfolio_sel_ew": res.metrics.get("portfolio_sel_ew"),
            "input_problems_waived": res.metrics["inputs"]["problems"]}
        print(f"V-reader-{reader}: cfg {res.cfg_sha[:16]} inputs {res.inputs_sha[:12]} IC "
              f"{res.metrics['prediction'].get('ic_mean'):.4f} -> {res.out_dir}", flush=True)
    rec = {"script": "experiments/run_layer_b_readers.py",
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "argv": list(argv if argv is not None else sys.argv[1:]), "gen_id": a.gen_id, "window": a.window,
           "base_cfg_sha": cfg.sha, "allow_missing": list(allow), "deviation": "DEV-10",
           "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "variants": out}
    p = root / "results/eval" / f"reader_variants_{a.window}.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    rec["variants"] = {**old.get("variants", {}), **out}
    p.write_text(json.dumps(_plain(rec), indent=1, sort_keys=True, default=str))
    print(f"-> {p.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
