"""SHA-256 of the red-team and FAITH entry points and specifications, for the pre-registration commit (P-4, P-8).

Writes `results/redteam/precommit_hashes.json`: every file's SHA-256, the group hash (over the sorted per-file
hashes), and the frozen specification hashes (RT-A-NG `NGSpec.sha256()`, FAITH `faith_sha256()`, the attack templates
`attacks.templates_sha256()`, RB-9 `AdaptiveSpec().sha256()` for reference), and the RT-A-NG operator pool on its own:
sha256 of the canonical JSON of `adaptive_ng.pool_description(NGSpec())` (the 27 payloads, the 16 operators rendered
on a probe sentence, the 4 positions and the proposal schedule; the spec hash covers it too). Reads source files
only.

    .venv/bin/python experiments/redteam/precommit_hashes.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

FILES = (
    "src/jevtrader/redteam/adaptive_ng.py", "src/jevtrader/redteam/stages.py", "src/jevtrader/redteam/projection.py",
    "src/jevtrader/redteam/faith.py", "src/jevtrader/redteam/run.py", "src/jevtrader/redteam/adaptive.py",
    "src/jevtrader/redteam/attacks.py", "src/jevtrader/redteam/inject.py",
    "src/jevtrader/redteam/display_names_v1.json",
    "experiments/redteam/rta_ng.py", "experiments/redteam/rts.py", "experiments/redteam/calibrate_tokens.py",
    "experiments/redteam/precommit_hashes.py", "experiments/faith/faith.py",
    "results/redteam/token_calibration.json",
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    from jevtrader.redteam import adaptive as AD
    from jevtrader.redteam import adaptive_ng as NG
    from jevtrader.redteam import attacks as A
    from jevtrader.redteam import faith as F

    files = {f: sha(ROOT / f) for f in FILES if (ROOT / f).exists()}
    missing = [f for f in FILES if f not in files]
    group = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
    res = {"files": files, "missing": missing, "group_sha256": group,
           "specs": {"rta_ng_spec_sha256": NG.NGSpec().sha256(),
                     "rta_ng_operator_pool_sha256": hashlib.sha256(json.dumps(
                         NG.pool_description(NG.NGSpec()), sort_keys=True, ensure_ascii=True).encode()).hexdigest(),
                     "rta_ng_pool_size": {"payloads": len(NG.payloads()),
                                          "operators": sum(len(v) for v in NG.operators().values()),
                                          "positions": len(NG.NGSpec().positions)},
                     "faith_sha256": F.faith_sha256(),
                     "templates_sha256": A.templates_sha256(), "rb9_adaptive_spec_sha256": AD.AdaptiveSpec().sha256()}}
    out = ROOT / "results" / "redteam" / "precommit_hashes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"group_sha256": group, **res["specs"], "missing": missing}, indent=1))
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
