"""Identities the pre-registration fixes (docs/design/preregistration_v1.md §11): analysis-code hashes, the attack
templates hash, configuration and bank hashes, the frozen generations' table hashes, the registered variant budget and
the lock state. Read-only; imports no model client and never touches `pipeline/lock.py` beyond hashing the file.

    .venv/bin/python experiments/prereg/prereg_hashes.py --out results/prereg/hashes.json

Run it again at commit time: the values in the committed pre-registration must be the ones printed then.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ANALYSIS_GROUPS = {
    "layer_b_decision": ["src/jevtrader/pipeline/layer_b.py", "src/jevtrader/pipeline/variants.py",
                         "src/jevtrader/pipeline/registry.py", "src/jevtrader/pipeline/inputs.py",
                         "src/jevtrader/learn/allocator.py", "src/jevtrader/learn/features.py",
                         "src/jevtrader/learn/heads.py", "src/jevtrader/learn/labels.py", "src/jevtrader/learn/online.py",
                         "src/jevtrader/fusion/calibration.py", "src/jevtrader/fusion/logpool.py",
                         "src/jevtrader/fusion/pools.py", "src/jevtrader/decide/select.py", "src/jevtrader/decide/risk.py",
                         "src/jevtrader/decide/cards.py", "src/jevtrader/features/anchors.py",
                         "src/jevtrader/backtest/engine.py"],
    "statistics": ["src/jevtrader/eval/metrics.py", "src/jevtrader/eval/bound.py", "src/jevtrader/eval/attribution.py"],
    "red_team": ["src/jevtrader/redteam/run.py", "src/jevtrader/redteam/inject.py", "src/jevtrader/redteam/attacks.py",
                 "src/jevtrader/redteam/adaptive.py", "src/jevtrader/redteam/display_names_v1.json"],
    "baselines_and_parser": ["src/jevtrader/baselines/classic.py", "src/jevtrader/baselines/ml.py",
                             "src/jevtrader/baselines/text.py", "src/jevtrader/baselines/llm.py",
                             "src/jevtrader/baselines/parse.py"],
    "drivers": ["experiments/run_layer_b.py", "experiments/run_classic_baselines.py", "experiments/pre/d3_pre.py",
                "experiments/pre/caproot.py", "experiments/gens/run_gen.py", "experiments/ng/items500.py",
                "experiments/report/make_numbers.py", "experiments/report/style.py"],
    "lock": ["src/jevtrader/pipeline/lock.py"],
    "configs": ["configs/system_v1.yaml", "configs/learning_v1.yaml", "configs/backends.yaml",
                "configs/availability_v1.yaml", "pyproject.toml"],
    "prereg_tools": ["experiments/prereg/audit_generations.py", "experiments/prereg/power_prereg.py",
                     "experiments/prereg/prereg_hashes.py"],
}
# Entry points the design names for H1-H3 / PD0-PD3 that do not exist yet: they must be written, tested on W_pre or
# synthetic data, and hashed here before the commit.
TO_WRITE = {
    "H2 (C2) analysis": "weekly Spearman of E[effect] with AR[0,+1] (learn.labels.abnormal_returns), NW(4), placebo "
                        "AR[-5,-1], pre-09:30 and prior-look controls, 8-K agreement; per implementation on the NG and "
                        "VERB samples",
    "Table 1 (PD0)": "experiments/report/table1_operations.py (OPS-1)",
    "Table 2 (H1 + TI)": "experiments/report/table2_typed_interface.py (RB-16)",
    "Table 3 (H2, H3, PD2, PD3)": "experiments/report/table3_validity.py",
    "Table 4 (PD1)": "experiments/report/table4_portfolio.py (PE-16)",
    "Fig. 2": "frontier replays over J and kappa (RK-5)",
    "Fig. 3": "experiments/report/fig3_decision_card.py (XP-4)",
    "D1 window-start rule": "the monthly AUC with 3-month pooling and the start-month rule (prereg §6.2)",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(root: Path) -> dict:
    spec = importlib.util.spec_from_file_location("wfm", ROOT / "scripts/write_freeze_manifest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.tree(root)


def group_hash(files: dict[str, str | None]) -> str:
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def prior_looks() -> dict:
    """The W_clean stock-weeks of the prior looks that can be excluded by stock-week (design_v1 §13.6): the pilot's 600
    (date, symbol) keys (key columns only; its outcome column is not read) and the wording smoke test's items (AMD, JPM,
    GILD, created 2025-04-01..2025-05-31; every decision week whose news window can hold such an item). Written to
    results/prereg/prior_look_stock_weeks.csv; its SHA-256 is registered."""
    import pandas as pd
    from jevtrader.programs.context import decision_dates, universe_asof

    d = pd.read_parquet(ROOT / "data/raw/bars/etf_daily_2018_2026.parquet", columns=["date"])["date"]
    ses = pd.DatetimeIndex(sorted(pd.to_datetime(d.unique())))
    wc = decision_dates(ses, "2024-07-05", "2026-09-11")
    mem = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet", columns=["date", "symbol", "rank"])
    pilot = pd.read_parquet(ROOT / "results/pilot/news_signal_pilot.parquet", columns=["date", "symbol"]).drop_duplicates()
    pilot["t"] = pd.to_datetime(pilot["date"])
    rows = [(t, s, "pilot") for t, s in zip(pilot["t"], pilot["symbol"])]
    smoke_weeks = [t for t in decision_dates(ses, "2025-04-01", "2025-06-13") if t >= pd.Timestamp("2025-04-04")]
    smoke_weeks = [t for t in smoke_weeks if t <= pd.Timestamp("2025-06-06")]
    rows += [(t, s, "smoke") for t in smoke_weeks for s in ("AMD", "JPM", "GILD")]
    df = pd.DataFrame(rows, columns=["t", "symbol", "source"])
    df["is_decision_date"] = df["t"].isin(wc)
    df["in_U_t"] = [s in universe_asof(mem, t, 100) for t, s in zip(df["t"], df["symbol"])]
    df = df.sort_values(["t", "symbol", "source"]).reset_index(drop=True)
    path = ROOT / "results/prereg/prior_look_stock_weeks.csv"
    df.assign(t=df["t"].dt.date.astype(str)).to_csv(path, index=False)
    return {"file": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows": int(len(df)),
            "by_source": df["source"].value_counts().to_dict(),
            "pilot_dates_all_decision_dates": bool(df.loc[df["source"] == "pilot", "is_decision_date"].all()),
            "stock_weeks_in_MAIN_frame": int((df["is_decision_date"] & df["in_U_t"]).sum()),
            "smoke_decision_weeks": [str(t.date()) for t in smoke_weeks],
            "not_excludable_by_stock_week": ["classical-factor ICs on W_clean (evaluation_first §7.4)",
                                             "results/baselines_classic.json (B1-B7 over 2020-01..2026-09-18)"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/prereg/hashes.json"))
    a = ap.parse_args()
    from jevtrader.fusion import pools as P
    from jevtrader.pipeline import variants as V
    from jevtrader.programs.bank_v1 import FROZEN_RUN_TAGS, FROZEN_SHA256
    from jevtrader.redteam import attacks as A
    from jevtrader.redteam import run as R

    freeze = json.loads((ROOT / "results/freeze/FREEZE.json").read_text())
    out: dict = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                 "script": "experiments/prereg/prereg_hashes.py"}
    groups = {}
    for g, files in ANALYSIS_GROUPS.items():
        fh = {f: (sha256_file(ROOT / f) if (ROOT / f).exists() else None) for f in files}
        groups[g] = {"sha256": group_hash(fh), "files": fh, "missing": [f for f, h in fh.items() if h is None]}
    out["analysis_code"] = groups
    out["trees"] = {d: {k: v for k, v in tree(ROOT / d).items() if k != "files"}
                    for d in ("src", "experiments", "scripts", "tests")}
    out["to_write_before_commit"] = TO_WRITE
    out["attack_templates"] = {"templates_sha256": A.templates_sha256(), "version": A.TEMPLATES_VERSION,
                               "selection_salt": A.SELECTION_SALT, "c1_promotion_families": list(A.C1_PROMOTION),
                               "demotion_families": list(A.DEMOTION), "a4_subvariants": list(A.A4_SUBVARIANTS),
                               "unit_sampling_seed": R.SEED}
    out["bank"] = {"frozen_sha256": FROZEN_SHA256, "frozen_run_tags": sorted(FROZEN_RUN_TAGS),
                   "file_sha256": sha256_file(ROOT / "docs/design/question_bank_v1.json"),
                   "freeze_json_file_sha256": freeze["bank"]["file_sha256"]}
    out["freeze"] = {"describe": freeze["describe"], "FREEZE.json_sha256": sha256_file(ROOT / "results/freeze/FREEZE.json"),
                     "emulator_cfg_hash": freeze["emulator"]["cfg_hash"], "aliases": freeze["blinder"]["aliases_sha256"]}
    gens = {}
    for name, d in (("MAIN", "results/judgments/main_emu_wclean_u100"), ("PRE/B", "results/judgments/pre_blinded_emu_wpre_s30"),
                    ("PRE/U", "results/judgments/pre_unblinded_emu_wpre_s30"),
                    ("TAL", "results/gens/tal_ta-lite_wclean-last60_s30")):
        gens[name] = {p.name: sha256_file(p) for p in sorted((ROOT / d).glob("*.parquet"))}
        gens[name]["MANIFEST.json"] = sha256_file(ROOT / d / "MANIFEST.json")
    ng = ROOT / "results/judgments/ng_typed-local_wclean_items500"
    gens["NG"] = {"sample.parquet": sha256_file(ng / "sample.parquet"), "sample.json": sha256_file(ng / "sample.json"),
                  "complete": (ng / "MANIFEST.json").exists()}
    tin = ROOT / "data/cache/gens/tal_inputs.jsonl.gz"
    gens["TAL"]["inputs(data/cache/gens/tal_inputs.jsonl.gz)"] = sha256_file(tin) if tin.exists() else None
    out["generations"] = gens
    names = V.registered_names()
    out["registered_variants"] = {"configuration": list(V.CONFIG_VARIANTS), "composition": list(V.COMPOSITION_VARIANTS),
                                  "evaluation_only": list(P.EVAL_ONLY), "trial_series_per_full_sweep": len(names),
                                  "trial_series_names": names}
    trials = [json.loads(line) for line in (ROOT / "results/trials.jsonl").read_text().splitlines() if line.strip()]
    out["trial_registry"] = {"rows": len(trials), "by_window": dict(Counter(t.get("window") for t in trials)),
                             "note": "window counts only (no Sharpe read)"}
    out["prior_look_stock_weeks"] = prior_looks()
    lock_key = ROOT / "results/locks/clean_unlocked"
    out["lock"] = {"lock_py_sha256": sha256_file(ROOT / "src/jevtrader/pipeline/lock.py"),
                   "lock_py_mtime": dt.datetime.fromtimestamp((ROOT / "src/jevtrader/pipeline/lock.py").stat().st_mtime)
                   .isoformat(timespec="seconds"),
                   "unlock_key_present": lock_key.exists(),
                   "prereg_yaml_present": (ROOT / "configs/prereg_v1.yaml").exists()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("attack_templates", "bank", "freeze", "lock", "trial_registry")}, indent=1))
    print({g: (v["sha256"][:16], v["missing"]) for g, v in groups.items()})
    print({d: v["sha256"][:16] for d, v in out["trees"].items()})


if __name__ == "__main__":
    main()
