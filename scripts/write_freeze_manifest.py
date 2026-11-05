"""Tag the frozen configuration (docs/impl/freeze_report.md; docs/design/decisions.md 2026-09-28): write
results/freeze/FREEZE.json with the identities of everything a MAIN or PRE run depends on. There is no git in this
workflow, so the "git describe" equivalent is the SHA-256 of the source tree.

    .venv/bin/python scripts/write_freeze_manifest.py [--tests "<pytest summary line>"] [--leak-scan <json>]

Recorded:
- bank: version, canonical SHA-256 (= bank_v1.FROZEN_SHA256, checked), raw-file and rendered-Markdown SHA-256, the
  rc7 hash whose content it freezes, the run tags that must load it;
- emulator: backend id, pinned model, prompt version, `cfg_hash` (the one in every request key, computed offline with a
  stub client: no network, no key), the canonical SHA-256 of the emu-v1 prompt, the configuration block of
  configs/backends.yaml (prompt-cache routing included) and its hash;
- blinder: text, masking and filing rules, the aliases identity that enters every context description
  (`aliases_sha256`), the SHA-256 of every curated table and of the 10-K signature/officer store, row counts by role;
- code: SHA-256 per file and a tree hash for src/, and tree hashes of scripts/, experiments/, tests/ and configs/;
- the test-suite summary and the leak-scan summary, when given.
Nothing here reads prices, labels or any W_clean outcome."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "freeze" / "FREEZE.json"
RC7_SHA256 = "c44c1ec005cd14f081403fe021809851466054d934488e9d58859993050cbf4f"
SKIP_DIRS = {"__pycache__", ".pytest_cache", "out", "shards"}
CODE_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".toml", ".txt", ".md", ".sql", ".cfg", ".ini", ".sh"}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tree(root: Path, suffixes: set[str] | None = CODE_SUFFIXES) -> dict:
    """{"sha256": tree hash, "n_files": n, "files": {relative path: sha256}} over the files under `root` (sorted paths;
    caches and generated output directories skipped)."""
    files = {}
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            continue
        if suffixes is not None and p.suffix not in suffixes:
            continue
        files[str(p.relative_to(ROOT))] = sha256_file(p)
    h = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in files.items()).encode()).hexdigest()
    return {"sha256": h, "n_files": len(files), "files": files}


def bank_block() -> dict:
    from jevtrader.programs import bank_v1

    b = bank_v1.load_bank()
    if b.sha256 != bank_v1.FROZEN_SHA256:
        raise SystemExit(f"the bank file ({b.sha256}) is not the frozen bank ({bank_v1.FROZEN_SHA256})")
    b = bank_v1.load_bank(run_tag="MAIN")  # the check a MAIN run makes
    return {"path": str(bank_v1.BANK_PATH.relative_to(ROOT)), "version": b.version, "canonical_sha256": b.sha256,
            "frozen_sha256": bank_v1.FROZEN_SHA256, "file_sha256": sha256_file(bank_v1.BANK_PATH),
            "markdown": str(bank_v1.BANK_MD_PATH.relative_to(ROOT)), "markdown_sha256": sha256_file(bank_v1.BANK_MD_PATH),
            "markdown_prints": bank_v1.md_sha256(), "frozen_content_of": {"version": "v1.0-rc7", "sha256": RC7_SHA256},
            "frozen_run_tags": sorted(bank_v1.FROZEN_RUN_TAGS),
            "n_templates": sum(len(v) for v in b.programs.values())}


def emulator_block() -> dict:
    from jevtrader.programs.bank_v1 import load_bank
    from jevtrader.s1.backends import LogprobEmulator, config_hash, load_backend_config
    from jevtrader.s1.types import canonical_json

    cfg = load_backend_config()
    blk = dict(cfg["backends"]["llm-logprob"])
    kw = {k: v for k, v in blk.items() if k in ("model", "prompt_version", "degraded_retries", "max_concurrency",
                                                "assert_response_model", "prompt_cache")}
    emu = LogprobEmulator(client=object(), bank_hash=load_bank().sha256, **kw)  # a stub client: never called
    prompt = load_bank().prompts[blk["prompt_version"]]
    return {"backend": emu.backend_id, "s1_backend_default": cfg["s1"]["backend"], "model_pinned": emu.model,
            "prompt_version": emu.prompt_version, "cfg_hash": emu.cfg_hash,
            "prompt_sha256": hashlib.sha256(canonical_json(prompt).encode()).hexdigest(),
            "config_block": blk, "config_block_sha256": config_hash(blk),
            "prompt_cache_routing_enabled": bool(emu.prompt_cache.enabled),
            "backends_yaml_sha256": sha256_file(ROOT / "configs" / "backends.yaml"),
            "request_key": "request_key(backend, model, prompt_version, cfg_hash, bank_sha256, state, questions)"}


def blinder_block() -> dict:
    import pandas as pd

    from jevtrader.data import blinding as B
    from jevtrader.programs import context as C

    sub = ROOT / "data" / "raw" / "sec" / "submissions"
    adir = B.aliases_dir(sub)
    files = {p.name: sha256_file(p) for p in sorted(adir.glob("*.csv"))}
    a = pd.read_csv(adir / B.EXECUTIVES_FILE, dtype=str, keep_default_na=False)
    store = tree(adir / "tenk_officers", suffixes={".json"})
    return {"text_rules": B.TEXT_RULES, "masking": C.MASKING_RULE, "filing_masking": C.FILING_MASKING,
            "discovery_rule_if_on": B.DISCOVERY_RULE, "alias_discovery_default": False,
            "aliases_sha256": B.aliases_sha256(sub), "aliases_sha256_covers": [B.MANUAL_ALIASES_FILE, B.EXECUTIVES_FILE],
            "tables_sha256": files, "tenk_officers_store": {"sha256": store["sha256"], "n_files": store["n_files"]},
            "person_rows_by_role": {k: int(v) for k, v in a.roles.value_counts().sort_index().items()},
            "person_rows": len(a), "exec_mask": B.EXEC_MASK, "salt_in_code": True}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tests", default=None, help="the full test-suite summary line")
    ap.add_argument("--leak-scan", default=None, help="a JSON file whose 'summary' is recorded")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    import pandas as pd

    src = tree(ROOT / "src")
    m = {
        "freeze": "v1.0",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "decision": "docs/design/decisions.md, 2026-09-28: freeze v1.0 = rc7 content (label only), prompt-cache routing "
                    "OFF (rule O-5), founders/directors in the blinder data",
        "report": "docs/impl/freeze_report.md",
        "describe": f"freeze-v1.0-src-{src['sha256'][:12]}",  # the git-describe equivalent (no git in this workflow)
        "bank": bank_block(),
        "emulator": emulator_block(),
        "blinder": blinder_block(),
        "code": {"src_tree_sha256": src["sha256"], "src_files": src["n_files"], "src": src["files"],
                 "scripts_tree_sha256": tree(ROOT / "scripts")["sha256"],
                 "experiments_tree_sha256": tree(ROOT / "experiments")["sha256"],
                 "tests_tree_sha256": tree(ROOT / "tests")["sha256"],
                 "configs": {p.name: sha256_file(p) for p in sorted((ROOT / "configs").glob("*.yaml"))},
                 "pyproject_sha256": sha256_file(ROOT / "pyproject.toml")},
        "tests": a.tests,
        "leak_scan": json.loads(Path(a.leak_scan).read_text()).get("summary") if a.leak_scan else None,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=1, sort_keys=False, default=str))
    print(f"wrote {out}: {m['describe']}; bank {m['bank']['canonical_sha256'][:16]}; emu cfg_hash "
          f"{m['emulator']['cfg_hash']}; aliases {m['blinder']['aliases_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
