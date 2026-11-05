"""Free checks of the TAL/MCOT inputs (data/cache/gens/tal_inputs.jsonl.gz) before any paid call:
- shape: 60 weeks x 30 names, hashes intact (`run_gen.check_inputs`);
- leak: the focal ticker (whole token) or one of the focal firm's own names (feed short names, the distinctive first
  word of its SEC name: `data.leak_probe`, the blinding regression test's patterns) in any block, as in the DEV0
  smoke (scripts/smoke_llm_baselines.py);
- sources: technical / fundamentals present or "not available" counts; T states carry a market regime from M;
- sizes: UTF-8 bytes per block (mean, p95, max) and news items per stock-week; macro blocks per week (distinct).
No outcome is read.

    .venv/bin/python experiments/gens/check_inputs.py --out results/gens/inputs_check.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "gens"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_gen as RG  # noqa: E402
import tal_inputs as TI  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(TI.DEFAULT_OUT))
    ap.add_argument("--out", default=str(ROOT / "results/gens/inputs_check.json"))
    a = ap.parse_args()
    recs = TI.read_inputs(Path(a.inputs))
    by_week = RG.check_inputs(recs)
    from jevtrader.programs.context import RealData
    from smoke_llm_baselines import leak_probes

    data = RealData()
    syms = sorted({r["symbol"] for r in recs})
    probe = leak_probes(syms, data.news_table)
    leaks = []
    for r in recs:
        for b in TI.BLOCKS:
            text = r[b]
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(r['symbol'])}(?![A-Za-z0-9])", text):
                leaks.append((r["t"], r["symbol"], b, "ticker"))
            leaks += [(r["t"], r["symbol"], b, "name") for p in probe.get(r["symbol"], []) if p.search(text)]
    sizes = {b: np.array([len(r[b].encode()) for r in recs]) for b in TI.BLOCKS}
    n_items = np.array([len(json.loads(r["news"]).get("news_items", [])) for r in recs])
    na = {b: int(sum(r[b].strip() in ("not available", '"not available"') or r["sources"].get(b) == "not available"
                     for r in recs)) for b in ("technical", "fundamentals")}
    regime_na = int(sum('"market regime": "not available"' in r["technical"] for r in recs))
    res = {"stock_weeks": len(recs), "weeks": len(by_week), "first": min(by_week), "last": max(by_week),
           "leaks": len(leaks), "leak_examples": leaks[:10], "leak_kinds": dict(Counter(x[3] for x in leaks)),
           "not_available": na, "t_regime_not_available": regime_na,
           "macro_blocks_per_week": dict(Counter(len({r["macro_policy"] for r in v}) for v in by_week.values())),
           "bytes": {b: {"mean": round(float(v.mean()), 1), "p95": int(np.percentile(v, 95)), "max": int(v.max())}
                     for b, v in sizes.items()},
           "news_items": {"mean": round(float(n_items.mean()), 2), "zero": int((n_items == 0).sum()),
                          "max": int(n_items.max()), "at_cap_40": int((n_items >= 40).sum())}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if not leaks else 1


if __name__ == "__main__":
    sys.exit(main())
