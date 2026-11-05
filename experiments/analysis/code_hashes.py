#!/usr/bin/env python3
"""SHA-256 manifest of the analysis entry points and every repository file they import (prereg §14.3, P-8).

    .venv/bin/python experiments/analysis/code_hashes.py            # writes results/analysis/CODE_HASHES.json
    .venv/bin/python experiments/analysis/code_hashes.py --check    # exit 1 if a hashed file changed since
    .venv/bin/python experiments/analysis/code_hashes.py --generations   # results/analysis/GENERATION_HASHES.json

For each entry point the import closure is computed statically: every `import x` / `from x import y` statement in the
file (module level or inside a function) is resolved against the repository (src/ for `jevtrader.*`; the entry
point's own directory, experiments/analysis, experiments/report, experiments/pre, experiments/redteam,
experiments/ng and experiments/dev for script-local modules), and the resolution is repeated on every repository file found, so the
closure is transitive. Three groups are registered: the analysis and report entry points, the red-team and FAITH
drivers (RT-S, RT-A-NG, FAITH, their calibration and hash scripts), and the D1 probe re-ask runner; the development
tools are hashed too but outside the registered hash. Standard-library and
third-party modules are not hashed; the versions of the numerical packages are recorded instead. The configuration
files the analyses read are hashed as a separate group. Each entry point gets its own SHA-256 and a closure hash
(sha256 over the sorted "path<TAB>sha256" lines, the convention of experiments/prereg/prereg_hashes.py); the union of
all files gets one overall hash, which the pre-registration quotes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "analysis" / "CODE_HASHES.json"
ENTRY_POINTS = (
    "experiments/analysis/h1_asr.py", "experiments/analysis/h2_validity.py", "experiments/analysis/h3_signal.py",
    "experiments/analysis/d1_window_start.py", "experiments/analysis/confirmatory.py",
    "experiments/analysis/pd0_operations.py", "experiments/analysis/pd1_portfolio.py",
    "experiments/analysis/pd2_composition.py", "experiments/analysis/pd3_stores.py", "experiments/analysis/frontier.py",
    "experiments/report/table1_operations.py", "experiments/report/table2_typed_interface.py",
    "experiments/report/table3_validity.py", "experiments/report/table4_portfolio.py",
    "experiments/report/fig2_frontier.py", "experiments/report/fig3_decision_card.py",
    "experiments/report/make_numbers.py",
    "experiments/analysis/bound_claim.py", "experiments/analysis/localization.py",
    "experiments/analysis/rta_ng_summary.py", "experiments/analysis/d2_reid.py",
)
REDTEAM_FAITH = (
    "experiments/redteam/rts.py", "experiments/redteam/rta_ng.py", "experiments/redteam/calibrate_tokens.py",
    "experiments/redteam/precommit_hashes.py", "experiments/faith/faith.py", "experiments/run_gen_probe_reask.py",
)
DEVELOPMENT = ("experiments/analysis/run_wpre.py", "experiments/analysis/feature_plants.py",
               "experiments/analysis/code_hashes.py", "experiments/analysis/sim_null_rates.py",
               "experiments/analysis/budget_snapshot.py")
CONFIGS = ("configs/system_v1.yaml", "configs/learning_v1.yaml", "configs/backends.yaml", "pyproject.toml")
LOCAL_DIRS = ("experiments/analysis", "experiments/report", "experiments/pre", "experiments/redteam", "experiments/ng",
              "experiments/dev")
GENERATIONS = (  # every generation the registered analyses read (prereg §14.1)
    "results/judgments/main_emu_wclean_u100", "results/judgments/pre_blinded_emu_wpre_s30",
    "results/judgments/pre_unblinded_emu_wpre_s30", "results/gens/tal_ta-lite_wclean-last60_s30",
    "results/judgments/ng_typed-local_wclean_items500", "results/judgments/mcot_mono-cot_wclean-last60_s30",
    "results/judgments/verb_llm-verbalized_wclean_items2000", "results/judgments/probe_emu_d1d2",
    "results/judgments/probe_emu_d1_reask", "results/judgments/fid_emu_wclean_v1-150_v2-100x3",
    "results/judgments/mem_emu_mepi-off_wclean_s30", "results/judgments/mem_emu_mwork-off_wclean-last98_s30",
)
GEN_OUT = ROOT / "results" / "analysis" / "GENERATION_HASHES.json"
PACKAGES = ("numpy", "pandas", "scipy", "pyarrow", "lightgbm", "scikit-learn", "matplotlib", "statsmodels", "pyyaml",
            "fonttools")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def group_hash(files: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()


def _module_file(mod: str, here: Path) -> Path | None:
    """The repository file of a dotted module name, or None (stdlib / third party)."""
    parts = mod.split(".")
    if parts[0] == "jevtrader":
        base = ROOT / "src" / Path(*parts)
        for cand in (base.with_suffix(".py"), base / "__init__.py"):
            if cand.exists():
                return cand
        return None
    if len(parts) == 1:
        for d in (here, *[ROOT / x for x in LOCAL_DIRS]):
            cand = Path(d) / f"{parts[0]}.py"
            if cand.exists():
                return cand
    return None


def imports_of(path: Path) -> set[Path]:
    """Repository files imported by `path` (every import statement, including function-level ones)."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    here = Path(path).parent
    out: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                f = _module_file(a.name, here)
                if f:
                    out.add(f)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            f = _module_file(node.module, here)
            if f:
                out.add(f)
            for a in node.names:  # `from jevtrader.eval import metrics` imports a submodule
                g = _module_file(f"{node.module}.{a.name}", here)
                if g:
                    out.add(g)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__" and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            f = _module_file(node.args[0].value, here)
            if f:
                out.add(f)
    # a package's __init__ runs on import of any of its submodules
    src = ROOT / "src"
    for f in list(out):
        if src in f.parents:
            p = f.parent
            while p != src:
                if (p / "__init__.py").exists():
                    out.add(p / "__init__.py")
                p = p.parent
    out.discard(Path(path))
    return out


def closure(entry: Path) -> list[Path]:
    seen, todo = set(), [Path(entry)]
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        todo.extend(imports_of(f) - seen)
    seen.discard(Path(entry))
    return sorted(seen)


def rel(p: Path) -> str:
    return str(Path(p).resolve().relative_to(ROOT))


def manifest() -> dict:
    eps, union = {}, {}
    for group, names in (("entry_points", ENTRY_POINTS), ("redteam_faith", REDTEAM_FAITH),
                         ("development", DEVELOPMENT)):
        for e in names:
            p = ROOT / e
            if not p.exists():
                continue
            files = {rel(f): sha256_file(f) for f in closure(p)}
            own = sha256_file(p)
            union[e] = own
            union.update(files)
            eps[e] = {"group": group, "sha256": own, "imports": files,
                      "closure_sha256": group_hash({e: own, **files})}
    cfg = {c: sha256_file(ROOT / c) for c in CONFIGS if (ROOT / c).exists()}
    pk = {}
    for name in PACKAGES:
        try:
            pk[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pk[name] = None
    def members(entries) -> dict:
        return {k: v for k, v in union.items() if any(k == e or k in eps[e]["imports"] for e in entries if e in eps)}

    return {"what": "SHA-256 of every analysis entry point, red-team/FAITH driver and development tool and of every "
                    "repository file it imports (static, transitive); experiments/analysis/code_hashes.py",
            "entry_points": eps, "configs": {"files": cfg, "sha256": group_hash(cfg)},
            "files": dict(sorted(union.items())),
            "analysis_sha256": group_hash(members(ENTRY_POINTS)),
            "redteam_faith_sha256": group_hash(members(REDTEAM_FAITH)),
            "registered_sha256": group_hash(members((*ENTRY_POINTS, *REDTEAM_FAITH))),
            "all_sha256": group_hash(union),
            "src_since_freeze": src_since_freeze(),
            "python": platform.python_version(), "packages": pk}


def src_since_freeze() -> dict:
    """src/ files changed or added since the bank/code freeze (results/freeze/FREEZE.json code.src)."""
    f = ROOT / "results" / "freeze" / "FREEZE.json"
    if not f.exists():
        return {"status": "no FREEZE.json"}
    fz = json.loads(f.read_text())
    frozen = fz["code"]["src"]
    cur = {rel(p): sha256_file(p) for p in (ROOT / "src").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    return {"freeze": fz.get("freeze"), "freeze_src_tree_sha256": fz["code"].get("src_tree_sha256"),
            "changed": sorted(k for k, v in frozen.items() if k in cur and cur[k] != v),
            "removed": sorted(k for k in frozen if k not in cur), "added": sorted(set(cur) - set(frozen))}


def generation_hashes(gens=GENERATIONS) -> dict:
    """SHA-256 of every generation's MANIFEST.json, top-level tables (*.parquet, *.json) and blackboard
    (memory/*.jsonl), and one hash per generation over its sorted "path<TAB>sha256" lines."""
    out = {}
    for g in gens:
        d = ROOT / g
        if not d.exists():
            out[g] = {"missing": True}
            continue
        files = sorted([*d.glob("*.parquet"), *d.glob("*.json"), *d.glob("memory/*.jsonl"), *d.glob("plan/*")])
        fh = {rel(f): sha256_file(f) for f in files if f.is_file()}
        out[g] = {"files": fh, "sha256": group_hash(fh)}
    return {"what": "SHA-256 of the generations the registered analyses read; experiments/analysis/code_hashes.py "
                    "--generations", "generations": out, "all_sha256": group_hash(
                        {g: v["sha256"] for g, v in out.items() if "sha256" in v})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the written manifest; exit 1 if stale")
    ap.add_argument("--generations", action="store_true", help="hash the registered generations instead")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.generations:
        import datetime as dt

        g = generation_hashes()
        g["created_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        out = Path(a.out) if a.out else GEN_OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(g, indent=1))
        print(f"{len(g['generations'])} generations, all {g['all_sha256'][:16]} -> {out}")
        return 0
    a.out = a.out or str(OUT)
    m = manifest()
    out = Path(a.out)
    if a.check:
        if not out.exists():
            print(f"{out} missing")
            return 1
        old = json.loads(out.read_text())
        stale = sorted(k for k in set(old["files"]) | set(m["files"]) if old["files"].get(k) != m["files"].get(k))
        cfg = sorted(k for k in set(old["configs"]["files"]) | set(m["configs"]["files"])
                     if old["configs"]["files"].get(k) != m["configs"]["files"].get(k))
        print(json.dumps({"stale_files": stale, "stale_configs": cfg}, indent=1))
        return 1 if stale or cfg else 0
    import datetime as dt

    m["created_utc"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1))
    print(f"{len(m['files'])} files, registered {m['registered_sha256'][:16]}, all {m['all_sha256'][:16]} -> "
          f"{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
