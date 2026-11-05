"""Cost projection of the reduced TAL / MCOT generations from their free dry runs (`run_gen.py --dry`).

The dry run sends every request the paid run would send to a fake client whose usage comes from `run_gen.TokenModel`
(calibrated on the only measured TA-lite / Mono-CoT calls, the DEV0/WS-C2 smoke: prompt tokens per UTF-8 byte of
system + user, and output tokens per role; docs/impl/day2_wsC2.md §3). Its paid rows (response-cache misses) are
priced with `s1.pricing.cost_usd` (gpt-4.1-mini-2025-04-14, $0.40 / $1.60 per 1M tokens, no provider caching assumed).

- point: the calibrated means;
- high: the largest per-role ratio and the largest per-role output of the smoke, plus 10% on the downstream prompts
  (bull, bear, trader, risk carry the analysts' reports, whose length follows the analyst outputs).

    .venv/bin/python experiments/gens/projection.py --dry-dir tal=<scratch>/tal_dry --dry-dir mcot=<scratch>/mcot_dry \
        --out results/gens/projection.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "gens"))

from jevtrader.s1.pricing import cost_usd  # noqa: E402

from run_gen import TokenModel  # noqa: E402

MODEL = "gpt-4.1-mini-2025-04-14"
# the smoke's largest observed values (docs/impl/day2_wsC2.md §3; scratch smoke_llm/requests.parquet)
HIGH_IN_PER_BYTE = {"fundamentals": 0.2735, "technical": 0.2779, "news": 0.2639, "macro_policy": 0.2410,
                    "bull": 0.2110, "bear": 0.2110, "trader": 0.2132, "risk": 0.2031, "mono": 0.2567}
HIGH_OUT = {"analyst": 226, "bull": 189, "bear": 180, "trader": 74, "risk": 80, "mono": 476}
DOWNSTREAM = {"bull", "bear", "trader", "risk"}
ANALYSTS = {"fundamentals", "technical", "news", "macro_policy"}


def project(dry: Path) -> dict:
    req = pd.read_parquet(dry / "requests.parquet") if (dry / "requests.parquet").exists() else pd.concat(
        [pd.read_parquet(p) for p in sorted((dry / "shards").glob("*/requests.parquet"))], ignore_index=True)
    paid = req[~req.cached].copy()
    tm = TokenModel()
    rows = []
    for _, r in paid.iterrows():
        role = r.role
        kind = "analyst" if role in ANALYSTS else role
        point = cost_usd(MODEL, int(r.input_tokens), int(r.output_tokens))
        ratio = HIGH_IN_PER_BYTE[role] / tm.IN_PER_BYTE[role]
        n_in = r.input_tokens * ratio * (1.10 if role in DOWNSTREAM else 1.0)
        high = cost_usd(MODEL, int(round(n_in)), HIGH_OUT[kind])
        rows.append((role, point, high, int(r.input_tokens), int(r.output_tokens)))
    df = pd.DataFrame(rows, columns=["role", "point", "high", "in", "out"])
    n_sw = int(req.groupby(["t", "symbol"]).ngroups)
    by_role = df.groupby("role").agg(paid=("point", "size"), usd=("point", "sum"), high=("high", "sum"),
                                     in_tokens=("in", "sum"), out_tokens=("out", "sum"))
    return {"dry_dir": str(dry), "stock_weeks": n_sw, "requests": int(len(req)), "paid_requests": int(len(paid)),
            "response_cache_hits": int(req.cached.sum()),
            "projected_usd": round(float(df.point.sum()), 4), "high_usd": round(float(df.high.sum()), 4),
            "usd_per_stock_week": round(float(df.point.sum()) / max(n_sw, 1), 6),
            "by_role": {k: {c: (round(float(v), 4) if c in ("usd", "high") else int(v)) for c, v in row.items()}
                        for k, row in by_role.iterrows()},
            "basis": "run_gen.TokenModel (DEV0/WS-C2 smoke calibration) over the dry run's paid rows; high = the "
                     "smoke's largest per-role ratio and output, +10% on downstream prompts"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-dir", action="append", required=True, help="reader=path")
    ap.add_argument("--out", default=str(ROOT / "results/gens/projection.json"))
    a = ap.parse_args()
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    for spec in a.dry_dir:
        reader, path = spec.split("=", 1)
        res[reader] = project(Path(path))
        print(reader, {k: v for k, v in res[reader].items() if k != "by_role"})
        for role, v in res[reader]["by_role"].items():
            print("   ", role, v)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
