"""Validation of a reduced TAL / MCOT generation (`run_gen.py`). Operational checks only: nothing here reads a return,
a label or any W_clean outcome.

    .venv/bin/python experiments/gens/validate_gen.py --reader tal --gen results/gens/<gen_id> \
        [--ledger data/cache/spend.sqlite] --out results/gens/<gen_id>_validation.json

Checks (all must pass):
- `complete`: every (week, name) of the inputs (the last 60 W_clean decisions x S30) has one answer row, and TA-lite
  8 request rows (one per role) or Mono-CoT 1;
- `weeks`: the weeks are exactly the last 60 W_clean decisions, and each week's names are S30 of U_t;
- `pinned`: every request row was answered by gpt-4.1-mini-2025-04-14, with prompt ta-lite-v1 / mono-cot-v1, parser
  gen-parse-v1 with this repository's parser hash, the frozen bank v1.0;
- `keys`: every analyst request key (TA-lite) / the Mono-CoT key equals the key recomputed from the stored inputs and
  the frozen prompt (the run read exactly these inputs);
- `degraded`: no call still degraded after its retries (else they are listed; a rerun re-asks them);
- `ledger` (paid runs): the ledger's rows under the run tag cover the generation's paid rows (ledger >= paid rows,
  the difference being paid work redone free from the response cache after an interruption).
Reported, not gated: invalid-output rates per role, finish reasons, decision and source counts (no outcome).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "gens"))

from jevtrader.baselines import llm as L  # noqa: E402
from jevtrader.baselines.parse import PARSER_SHA256, PARSER_VERSION  # noqa: E402
from jevtrader.programs import bank_v1  # noqa: E402
from jevtrader.programs.context import SUBSETS, subset_names, universe_asof  # noqa: E402

import run_gen as RG  # noqa: E402
import tal_inputs as TI  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=sorted(RG.READERS), required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--inputs", default=str(TI.DEFAULT_OUT))
    ap.add_argument("--ledger", default=None, help="spend ledger to reconcile (omit for a dry generation)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = RG.READERS[a.reader]
    gen = Path(a.gen)
    ans, req = pd.read_parquet(gen / "answers.parquet"), pd.read_parquet(gen / "requests.parquet")
    recs = TI.read_inputs(Path(a.inputs))
    by_week = RG.check_inputs(recs)
    bank = bank_v1.load_bank()
    checks, info = {}, {}

    # complete
    want = {(pd.Timestamp(r["t"]), r["symbol"]) for r in recs}
    got = set(zip(pd.to_datetime(ans.t), ans.symbol))
    per_sw = req.groupby([pd.to_datetime(req.t), req.symbol]).role.apply(sorted)
    roles = sorted([*L.ROLES, "bull", "bear", "trader", "risk"]) if a.reader == "tal" else ["mono"]
    bad_roles = [k for k, v in per_sw.items() if v != roles]
    checks["complete"] = got == want and len(ans) == len(want) and set(per_sw.index) == want and not bad_roles
    info["complete"] = {"stock_weeks": len(want), "answers": len(ans), "requests": len(req),
                        "missing": len(want - got), "extra": len(got - want), "bad_role_sets": len(bad_roles)}

    # weeks and names
    from jevtrader.programs.context import RealData

    data = RealData()
    weeks = TI.last_weeks(data.sessions)
    salt, n = SUBSETS["S30"]
    bad_names = [w for w in weeks if tuple(sorted(r["symbol"] for r in by_week[str(w.date())]))
                 != tuple(sorted(subset_names(universe_asof(data.membership, w), n, salt)))]
    checks["weeks"] = sorted(by_week) == [str(w.date()) for w in weeks] and not bad_names
    info["weeks"] = {"first": str(weeks[0].date()), "last": str(weeks[-1].date()), "n": len(weeks),
                     "weeks_with_other_names": [str(w.date()) for w in bad_names]}

    # pinned
    pv = R["prompt"]
    checks["pinned"] = bool((req.response_model == L.MODEL).all() and (req.model_pinned == L.MODEL).all()
                            and (req.prompt_version == pv).all() and (req.parser_version == PARSER_VERSION).all()
                            and (req.parser_sha256 == PARSER_SHA256).all()
                            and (req.bank_hash == bank_v1.FROZEN_SHA256).all() and bank.sha256 == bank_v1.FROZEN_SHA256)
    info["pinned"] = {"response_models": req.response_model.value_counts().to_dict(), "parser": PARSER_VERSION,
                      "parser_sha256": PARSER_SHA256[:16], "bank": bank.sha256[:16]}

    # keys: recomputed from the stored inputs and the frozen prompt
    llm = L.GenLLM(cache=False)
    keyed = req.set_index([pd.to_datetime(req.t), req.symbol, req.role]).request_key
    mism = 0
    if a.reader == "tal":
        p = bank.prompts["ta-lite-v1"]
        for r in recs:
            for role in L.ROLES:
                user = L._fill(p["analyst_template"], role=p["roles"][role]["role"], input=r[role])
                k = llm.key(p["system"], user, int(p["max_tokens"]["analyst"]), float(p["decoding"]["temperature"]))
                mism += keyed.get((pd.Timestamp(r["t"]), r["symbol"], role)) != k
    else:
        p = bank.prompts["mono-cot-v1"]
        for r in recs:
            user = L._fill(p["user_template"], fundamentals=r["fundamentals"], technicals=r["technical"],
                           news=r["news"], macro_policy=r["macro_policy"])
            k = llm.key(p["system"], user, int(p["decoding"]["max_tokens"]), float(p["decoding"]["temperature"]))
            mism += keyed.get((pd.Timestamp(r["t"]), r["symbol"], "mono")) != k
    checks["keys"] = mism == 0
    info["keys"] = {"mismatches": int(mism)}

    # degraded
    checks["degraded"] = int(req.degraded.sum()) == 0
    info["degraded"] = {"calls": int(req.degraded.sum()), "attempts_gt1": int((req.attempts > 1).sum())}

    # ledger
    paid = req[~req.cached]
    info["paid"] = {"rows": int(len(paid)), "usd": round(float(paid.usd.sum()), 4),
                    "input_tokens": int(paid.input_tokens.sum()), "output_tokens": int(paid.output_tokens.sum()),
                    "cached_input_tokens": int(paid.cached_tokens.sum()), "cache_hit_rows": int(req.cached.sum()),
                    "usd_per_stock_week": round(float(paid.usd.sum()) / max(len(want), 1), 6),
                    "latency_p50_s": round(float(paid.latency_s.quantile(.5)), 3) if len(paid) else None,
                    "latency_p95_s": round(float(paid.latency_s.quantile(.95)), 3) if len(paid) else None}
    if a.ledger:
        c = sqlite3.connect(f"file:{a.ledger}?mode=ro", uri=True)
        n, usd = c.execute("SELECT COUNT(*), COALESCE(SUM(usd), 0) FROM spend WHERE tag = ?", (R["tag"],)).fetchone()
        c.close()
        checks["ledger"] = n >= len(paid) and usd >= float(paid.usd.sum()) - 1e-9
        info["ledger"] = {"tag": R["tag"], "rows": int(n), "usd": round(float(usd), 4),
                          "minus_paid_rows_usd": round(float(usd) - float(paid.usd.sum()), 4)}

    # reported, not gated
    if a.reader == "tal":
        inv = {k: int(ans[f"invalid_{k}"].sum()) for k in ("analyst_stance", "trader", "risk", "final")}
        info["outputs"] = {"invalid": inv, "source": ans.source.value_counts().to_dict(),
                           "decision": ans.decision.value_counts(dropna=False).to_dict()}
    else:
        info["outputs"] = {"invalid": int(ans.invalid.sum())}
    info["finish_reason"] = req.finish_reason.value_counts().to_dict()
    res = {"reader": a.reader, "gen": str(gen), "ok": all(checks.values()), "checks": checks, "info": info}
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
