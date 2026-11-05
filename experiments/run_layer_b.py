"""Run Layer B from files (Day-1 integration open item 4): every input is read from disk, nothing is passed by hand.

    .venv/bin/python experiments/run_layer_b.py --gen-id <gen_id> [--window W_pre_slice] [--variants all|none|V-a,V-b]
        [--composition] [--bound-check] [--attribution] [--sharpe-tests] [--no-register] [--root .]
        [--allow-missing check1,check2]

Inputs (pipeline.layer_b.load_inputs + pipeline/inputs.py): the judgment tables and code features of
results/judgments/<gen_id>/, risk.parquet (Layer A's sentinel columns) and firm_info.parquet (industry, earnings),
the adjusted and raw bars, the point-in-time universe, the DFF cash rate (FRED archive), the M&A exits (SEC 8-K
archive) and the trusted anchor panel (prices + 8-K item codes). A missing input stops the run (InputError) unless it
is waived with --allow-missing, and the waiver is recorded with every trial. When the primary cell is refused, the
script first prints the failed checks as JSON; for `layer_a_weeks` (a Layer-A program of the configuration has no row
in some evaluated week, dev1_report §11.4) it names each program, its features and its missing weeks.

Outputs go to evaluate()'s directory results/eval/<cfg_sha16>/<window>__<inputs_sha12>/ (evaluate() first removes any
Layer-B output an earlier run left there) plus the files below; `manifest.json` lists every file this run wrote with
its SHA-256, the code hash and this script's hash, and report scripts read only files listed there:
  inputs_manifest.json   the input files read, with SHA-256
  variants.json          metrics of every configuration variant run (design §6.6, BG-2 V-size-v1, BG-3c rules)
  composition.json       PD2 composition summary, MA-6 scales, MA-7 dependence, MA-8 warm-up (MA-1..MA-8)
  bound_check.json       BG-5 worst-case reader (pass criterion: 0 violations) and BG-6 constructed witnesses
  exclusion_rules.json   BG-3c clean cost of the exclusion rules (needs the V-excl-* variants)
  attribution.json       PE-3 own-factor style attribution; PE-4 FF5 + momentum when data/raw/factors exists
  sharpe_tests.json      PE-7 Ledoit-Wolf studentized block-bootstrap Sharpe differences vs EW-U100
A configuration variant whose inputs fail the input checks (e.g. V-excl-collapsed without Layer A's n_prob_collapsed
column) is not run; variants.json records it under "skipped" with the reason, and exclusion_rules.json is not written.
The W_clean test lock applies (pipeline/lock.py): a window touching W_clean is refused before the pre-registration
unlock. Layer B makes no network call and no paid API call.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from jevtrader.eval import bound as B  # noqa: E402
from jevtrader.eval import metrics as M  # noqa: E402
from jevtrader.pipeline import inputs as I  # noqa: E402
from jevtrader.pipeline.layer_b import (  # noqa: E402
    LEARNING_YAML,
    SYSTEM_YAML,
    DecisionConfig,
    InputError,
    evaluate,
    input_report,
    load_inputs,
    resolve_window,
    write_manifest,
)
from jevtrader.pipeline.lock import LOCKED_WINDOW, assert_unlocked_for, range_touches_locked  # noqa: E402
from jevtrader.pipeline.registry import _plain, file_sha256  # noqa: E402
from jevtrader.pipeline.variants import CONFIG_VARIANTS, run_composition_variants, variant_config  # noqa: E402

BG3C = {"anchored": None, "bylines": "V-excl-bylines", "items": "V-excl-items", "anchored_collapsed": "V-excl-collapsed"}


def _rf(inputs, r: pd.Series) -> pd.Series:
    """The book's daily cash rate on its return dates (0 when the cash-rate check was waived)."""
    if inputs.cash_rate_daily is None:
        return pd.Series(0.0, index=r.index)
    return inputs.cash_rate_daily.reindex(r.index).fillna(0.0)


def _dump(path: Path, obj, written: list[str] | None = None) -> None:
    path.write_text(json.dumps(_plain(obj), indent=1, sort_keys=True, default=str))
    if written is not None:
        written.append(path.name)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gen-id", required=True)
    ap.add_argument("--window", default="W_pre_slice")
    ap.add_argument("--variants", default="none", help="all | none | comma-separated configuration variant ids")
    ap.add_argument("--composition", action="store_true")
    ap.add_argument("--bound-check", action="store_true")
    ap.add_argument("--attribution", action="store_true")
    ap.add_argument("--sharpe-tests", action="store_true")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--allow-missing", default="")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--trusted", default="", help="development only: comma-separated trusted feature set (config hash)")
    ap.add_argument("--untrusted", default="", help="development only: comma-separated untrusted feature set ('none')")
    a = ap.parse_args(argv)
    root = Path(a.root)
    t0 = time.time()
    allow = tuple(x for x in a.allow_missing.split(",") if x)
    cfg = DecisionConfig.from_yaml(LEARNING_YAML, SYSTEM_YAML, gen_id=a.gen_id, name="primary", allow_missing=allow)
    if a.trusted or a.untrusted:
        import dataclasses

        def feats(x: str) -> tuple[str, ...]:
            return () if x == "none" else tuple(v for v in x.split(",") if v)

        alloc = dataclasses.replace(cfg.allocator, **({"trusted": feats(a.trusted)} if a.trusted else {}),
                                    **({"untrusted": feats(a.untrusted)} if a.untrusted else {}))
        cfg = dataclasses.replace(cfg, allocator=alloc)
    start, end = resolve_window(a.window)
    if range_touches_locked(start, end + pd.Timedelta(days=7)):  # before loading anything
        assert_unlocked_for([max(start, LOCKED_WINDOW[0])], root)
    inputs = load_inputs(cfg, start, end, root)
    register = not a.no_register
    try:
        res = evaluate(cfg, a.window, inputs, root=root, register=register)
    except InputError:
        # a clear report before the traceback: which checks failed, and for missing Layer-A weeks which program,
        # which features and which weeks (dev1_report §11.4); nothing was written and no trial was registered
        dates = [pd.Timestamp(t) for t in sorted(inputs.universe) if start <= pd.Timestamp(t) <= end]
        rep = input_report(cfg, inputs, dates)
        bad = sorted(set(rep["problems"]) - set(allow))
        print(json.dumps(_plain({"status": "refused by the Layer-B input gate", "gen_id": a.gen_id, "window": a.window,
                                 "n_decisions": len(dates), "failed_checks": bad,
                                 "problems": {k: rep["problems"][k] for k in bad},
                                 "layer_a_weeks": rep["layer_a_weeks"], "allow_missing": list(allow),
                                 "waiver": "--allow-missing " + ",".join(bad) + " (recorded with every trial)"}),
                         indent=1, default=str), flush=True)
        raise
    res_problems = res.metrics["inputs"]["problems"]
    if res_problems:
        print(json.dumps(_plain({"input_problems_waived": res_problems,
                                 "layer_a_weeks": res.metrics["inputs"].get("layer_a_weeks")}), indent=1, default=str))
    out = res.out_dir
    summary = {"out_dir": str(out), "cfg_sha": res.cfg_sha, "inputs_sha": res.inputs_sha, "window": a.window,
               "n_decisions": res.metrics["n_decisions"], "prediction": {k: res.metrics["prediction"].get(k) for k in
                                                                         ("ic_mean", "ic_t_nw4", "log_loss", "auc")}}
    written: list[str] = []
    _dump(out / "inputs_manifest.json", {**I.input_manifest(root, a.gen_id), "allow_missing": list(allow),
                                         "problems": res.metrics["inputs"]["problems"]}, written)
    wanted = list(CONFIG_VARIANTS) if a.variants == "all" else [v for v in a.variants.split(",") if v and v != "none"]
    runs, skipped = {}, {}
    for v in wanted:
        try:
            runs[v] = evaluate(variant_config(cfg, v), a.window, inputs, root=root, register=register)
        except InputError as e:  # recorded, never a silent row: e.g. a BG-3c rule without its key column
            skipped[v] = str(e)
    if skipped:
        summary["variants_skipped"] = sorted(skipped)
    if runs or skipped:
        _dump(out / "variants.json", {**{v: {"cfg_sha": r.cfg_sha, "out_dir": str(r.out_dir),
                                             "prediction": {k: r.metrics["prediction"].get(k) for k in
                                                            ("ic_mean", "ic_t_nw4", "log_loss", "auc", "n_weeks_scored")},
                                             "portfolio_sized": r.metrics.get("portfolio_sized"),
                                             "portfolio_sel_ew": r.metrics.get("portfolio_sel_ew")}
                                         for v, r in runs.items()},
                                      **({"skipped": skipped} if skipped else {})}, written)
        rules = {k: (res if v is None else runs.get(v)) for k, v in BG3C.items()}
        if all(r is not None for r in rules.values()):
            _dump(out / "exclusion_rules.json", B.exclusion_rule_report(rules, inputs.labels), written)
    if a.composition:
        comp = run_composition_variants(res, inputs, cfg, root=root, register=register)
        _dump(out / "composition.json", {k: v for k, v in comp.items() if k != "scores"}, written)
    if a.bound_check:
        bc = B.bound_check(res.log, cfg)
        _dump(out / "bound_check.json", {"worst_case_reader": bc, "constructed_witnesses": B.constructed_witnesses()},
              written)
        summary["bound_check_passed"] = bc["passed"]
    if a.attribution:
        from jevtrader.eval import attribution as A

        gen = root / "results/judgments" / a.gen_id
        att: dict = {}
        code = gen / "code_features.parquet"
        if code.exists():
            raw = pd.read_parquet(code)
            raw["t"] = pd.to_datetime(raw["t"])
            wide = A.factor_inputs(raw, I.load_firm_info(gen))
            fac = A.own_factor_returns(inputs.prices, wide, inputs.universe, res.log.dates, end=res.hold_end,
                                       cash_rate_daily=inputs.cash_rate_daily,
                                       delisting_returns=inputs.delisting_returns, ma_exits=inputs.ma_exits)
            att["own_factors"] = {b: A.style_attribution(res.returns[b], _rf(inputs, res.returns[b]), fac)
                                  for b in ("sized", "sel_ew") if b in res.returns}
        fdir = root / A.FACTORS_DIR
        if fdir.exists():
            fr = A.load_french(fdir)
            att["ff5_mom"] = {b: A.french_attribution(res.returns[b], _rf(inputs, res.returns[b]), fr, window_end=end)
                              for b in ("sized", "sel_ew") if b in res.returns}
        _dump(out / "attribution.json", att, written)
    if a.sharpe_tests and "ew_u100" in res.returns:
        _dump(out / "sharpe_tests.json", {b: M.sharpe_diff_lw(res.returns[b], res.returns["ew_u100"],
                                                              _rf(inputs, res.returns[b]))
                                          for b in ("sized", "sel_ew") if b in res.returns}, written)
    write_manifest(out, written, script_sha=file_sha256(Path(__file__)), argv=list(argv if argv is not None else
                                                                                    sys.argv[1:]))
    summary["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(_plain(summary), indent=1, default=str))
    return summary


if __name__ == "__main__":
    main()
