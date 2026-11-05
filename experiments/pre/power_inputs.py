"""Power inputs from the W_pre slice (design_v1 §13.6: sigma_IC and rho are re-estimated on the W_pre slice and W_dev
only, never on W_clean): the weekly IC sd of the primary cell and of the C3 contrast V-codeonly, their correlation and
the sd of the weekly Delta IC, read from the Layer-B outputs of the PRE blinded generation (results/eval/...).

    .venv/bin/python experiments/pre/power_inputs.py --eval-dir results/eval/<cfg16>/W_pre_slice__<inputs12> \
        --out results/pre/power_inputs.json

Caveat carried into the output: on the S30 slice only 30 of U_t's 100 names have S1 judgments, so the primary and
code-only logits coincide on 70 names; rho is therefore higher, and sd(Delta IC) lower, than on U100.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

Z_HOLM, Z2 = 2.128 + 0.8416, 1.96 + 0.8416  # experiments/plan/plan_v1.py: z_holm (one-sided alpha/3, power .8), z2


def weekly(metrics_path: Path) -> pd.Series:
    m = json.loads(Path(metrics_path).read_text())["prediction"]["ic_by_week"]
    return pd.Series({pd.Timestamp(k): v for k, v in m.items()})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.eval_dir)
    prim = weekly(d / "metrics.json")
    variants = json.loads((d / "variants.json").read_text())
    out = {"eval_dir": str(d), "weeks": int(len(prim)), "sd_ic_primary": float(prim.std(ddof=1)), "contrasts": {},
           "caveat": "S30 slice: 70 of 100 names have no S1 judgment, so rho is inflated relative to U100"}
    for name in ("V-codeonly", "V-joint", "V-nocap"):
        v = weekly(Path(variants[name]["out_dir"]) / "metrics.json")
        k = prim.index.intersection(v.index)
        dlt = prim[k] - v[k]
        out["contrasts"][name] = {"weeks": int(len(k)), "sd_ic_variant": float(v[k].std(ddof=1)),
                                  "rho": float(np.corrcoef(prim[k], v[k])[0, 1]), "sd_delta": float(dlt.std(ddof=1)),
                                  "mde_115_weeks_holm": float(Z_HOLM * dlt.std(ddof=1) / math.sqrt(115))}
    out["single_system_ic_mde_115_weeks"] = float(Z2 * out["sd_ic_primary"] / math.sqrt(115))
    out["plan_v1_assumptions"] = {"sigma_ic": 0.12, "rho": [0.5, 0.8]}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    main()
