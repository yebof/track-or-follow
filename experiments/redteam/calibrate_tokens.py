"""Token calibration of the free dry-run projections (`jevtrader.redteam.projection`): emu prompt bytes -> billed
prompt tokens, output tokens per question and the provider prompt-cache share, from MEASURED paid requests.

- Input: the NG item sample's rebuilt states (`results/judgments/ng_typed-local_wclean_items500/states/*.jsonl`, whose
  emu request keys were verified equal to MAIN's) give the exact emu-v1 prompts of up to `--n` items' N0 and five N1
  requests; MAIN's `requests.parquet` gives the prompt tokens OpenAI billed for those keys. Fit per stage family
  (N0; N1 and every other JSON state): tokens = a * UTF-8 bytes + b (least squares), and the ratio's p95 for `high`.
- Output tokens per question and the cached share: MAIN's paid rows by stage (median and p90 of output tokens per
  question; sum cached / sum input).

Reads Layer-A states and request token counts only (operational data: no answer, return, label or coefficient). $0:
the prompts are rendered by the emulator class with a dummy client (no network).

    .venv/bin/python experiments/redteam/calibrate_tokens.py [--n 150]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "ng"))

OUT = ROOT / "results" / "redteam" / "token_calibration.json"
MAIN = ROOT / "results" / "judgments" / "main_emu_wclean_u100"
STATES = ROOT / "results" / "judgments" / "ng_typed-local_wclean_items500" / "states"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args(argv)
    import items500 as I5

    from jevtrader.programs import news as N
    from jevtrader.programs.bank_v1 import load_bank
    from jevtrader.s1.backends import LogprobEmulator

    bank = load_bank()
    emu = LogprobEmulator(client=object(), prompt_cache=False, bank_hash=bank.sha256)
    recs = []
    for p in sorted(STATES.glob("*.jsonl")):
        recs += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    recs = sorted(recs, key=lambda r: r["rank"])[: a.n]
    plan = N.n1_plan(bank)
    rows = []
    for r in recs:
        qs = I5.n0_questions(bank, r["n_sentences"])
        rows.append(("N0", r["emu_keys"]["N0"], r["n0_state"], {n: q for n, _, q in qs}))
        for g in plan.groups():
            qs = I5.n1_questions(bank, g.ask)
            rows.append(("N1", r["emu_keys"][f"N1.{g.name}"], r["n1_state"], {n: q for n, _, q in qs}))
    req = pd.read_parquet(MAIN / "requests.parquet", columns=["request_key", "stage", "input_tokens",
                                                              "cached_tokens", "output_tokens", "n_questions",
                                                              "cached", "attempts"])
    paid = req[~req["cached"].astype(bool) & (req["attempts"] == 1)]
    tok = paid.drop_duplicates("request_key").set_index("request_key")["input_tokens"]
    data = []
    for fam, key, state, qs in rows:
        if key not in tok.index:
            continue
        msgs, _ = emu.messages(state, qs)
        nbytes = sum(len(m["content"].encode("utf-8")) for m in msgs)
        data.append((fam, nbytes, int(tok[key])))
    df = pd.DataFrame(data, columns=["family", "bytes", "tokens"])
    fits = {}
    for fam, g in df.groupby("family"):
        x, y = g["bytes"].to_numpy(float), g["tokens"].to_numpy(float)
        A = np.vstack([x, np.ones_like(x)]).T
        (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
        ratio = y / x
        fits[fam] = {"n": int(len(g)), "tokens_per_byte": float(slope), "intercept": float(icpt),
                     "ratio_median": float(np.median(ratio)), "ratio_p95": float(np.quantile(ratio, 0.95)),
                     "max_abs_rel_error": float(np.max(np.abs((slope * x + icpt) - y) / y))}
    by = paid.assign(opq=paid["output_tokens"] / paid["n_questions"].clip(lower=1)).groupby("stage")
    out_q = {s: {"median": float(g["opq"].median()), "p90": float(g["opq"].quantile(0.9)),
                 "cached_share": float(g["cached_tokens"].sum() / max(g["input_tokens"].sum(), 1)),
                 "n": int(len(g))}
             for s, g in by if len(g) >= 50}
    res = {"source": {"states": str(STATES.relative_to(ROOT)), "main_requests": str((MAIN / "requests.parquet")
                                                                                   .relative_to(ROOT)),
                      "items": len(recs), "requests_matched": int(len(df))},
           "input": fits, "output_per_question": out_q,
           "note": "prompt tokens billed = tokens_per_byte * UTF-8 bytes of the emu-v1 messages + intercept; N1's fit "
                   "is used for every non-N0 JSON state (N2, N3, T, readers); high = ratio_p95 * bytes"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["input"], indent=1))
    print({k: v for k, v in out_q.items() if k in ("N0", "N1", "N2", "N3", "T")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
