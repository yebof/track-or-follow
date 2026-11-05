"""Reduced pre-unlock generative baselines (2026-09-28 budget re-plan): TAL (TA-lite, B14) and MCOT (Mono-CoT, B15)
on S30 x the LAST 60 weekly decisions of W_clean, frozen bank v1.0 prompts (ta-lite-v1 with QB-9's reason-first
trader and risk templates; mono-cot-v1), pinned gpt-4.1-mini-2025-04-14, gen-parse-v1 (`baselines/llm.py`).

    # free rehearsal: fake OpenAI client, scratch ledger, scratch response cache, scratch output
    .venv/bin/python experiments/gens/run_gen.py --reader tal --dry --out <scratch>/tal_dry
    # paid (preflight first; exits 3 if the projection does not fit the sub-cap or the $75 reserve)
    .venv/bin/python experiments/gens/run_gen.py --reader tal --projection results/gens/projection.json

Inputs: `experiments/gens/tal_inputs.py` (data/cache/gens/tal_inputs.jsonl.gz, 1,800 stock-weeks, checked here).
Layer-A judgments only: nothing here reads a return, a label or any W_clean outcome.

Budget (CLAUDE.md, user instruction 2026-09-28): the ledger's hard cap and the run's sub-cap apply to every call
(SpendLedger), and this process also stops before the ledger total can pass $250 - $75 = $175 (the reserve for the
post-unlock C1 red team, readers on attacked units, faithfulness and live): GenLLM's guard reserves each call's
conservative projection (bytes/3.5 input, max_tokens output) before sending it, so the reserve holds even if the
projection was low. Weeks run newest first, so a guard stop leaves "the last k weeks" complete.

Resumable: one shard per decision week (`shards/<t>/`, DONE.json with a fingerprint of reader, bank, prompt, parser,
model and the week's input hashes). A rerun skips DONE weeks; an interrupted week reruns, and its finished calls are
response-cache hits (free). Degraded responses (empty, abnormal finish) are never cached and are retried twice.

TA-lite's macro-and-policy analyst sees the same input for every name of a week (M indicators and the week's P1
headlines carry no focal firm), so its request is identical across the week's 30 names: it is sent once per week
before the week's stock-weeks run, and the other 29 are response-cache hits of that identical request (8 calls per
stock-week, 7 x 30 + 1 = 211 paid per week). Stated as a deviation in docs/impl/gens_report.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "gens"))

from jevtrader.baselines import llm as L  # noqa: E402
from jevtrader.baselines.parse import PARSER_SHA256, PARSER_VERSION  # noqa: E402
from jevtrader.programs import bank_v1  # noqa: E402
from jevtrader.s1.budget import BudgetExceeded as LedgerBudgetExceeded  # noqa: E402
from jevtrader.s1.budget import SpendLedger  # noqa: E402

import tal_inputs as TI  # noqa: E402

log = logging.getLogger("run_gen")
USER_BUDGET = 250.0
RESERVE = 75.0  # the post-unlock reserve (task of 2026-09-28): the ledger total must stay <= 250 - 75
READERS = {
    "tal": {"run": "TAL", "tag": "TAL/ta-lite", "gen_id": "tal_ta-lite_wclean-last60_s30", "prompt": "ta-lite-v1"},
    "mcot": {"run": "MCOT", "tag": "MCOT/mono-cot", "gen_id": "mcot_mono-cot_wclean-last60_s30",
             "prompt": "mono-cot-v1"},
}
N_WEEKS, N_NAMES = 60, 30


# ------------------------------------------------------------------------------------------ fake client (dry runs)
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


# Output lengths of the DEV0/WS-C2 smoke (the only measured TA-lite / Mono-CoT calls; docs/impl/day2_wsC2.md §3):
# mean bytes of the raw text per role, so a dry run's downstream prompts (bull, bear, trader, risk) have real sizes.
ANALYST_ROLES = frozenset({"fundamentals", "technical", "news", "macro_policy"})
SMOKE_OUT_BYTES = {"analyst": 991, "bull": 917, "bear": 854, "trader": 305, "risk": 330, "mono": 2088}


def _role_of(user: str) -> str:
    if user.startswith("ROLE: Risk manager"):
        return "risk"
    if user.startswith("ROLE: Trader"):
        return "trader"
    if user.startswith("ROLE: Bull researcher"):
        return "bull"
    if user.startswith("ROLE: Bear researcher"):
        return "bear"
    for prefix, role in (("ROLE: Fundamentals analyst", "fundamentals"), ("ROLE: Technical analyst", "technical"),
                         ("ROLE: News analyst", "news"), ("ROLE: Macro and policy analyst", "macro_policy")):
        if user.startswith(prefix):
            return role
    if user.startswith("ROLE: "):
        raise ValueError(f"unknown TA-lite role: {user[:60]!r}")
    return "mono"


class FakeCompletions:
    """Deterministic stand-in for `client.chat.completions` (no network): answers in the parser's formats, with
    the smoke's output sizes and usage from the calibrated token model, answered by the pinned model id."""

    def __init__(self, tok_model):
        self.tok = tok_model
        self.n = 0

    async def create(self, model, messages, temperature, max_completion_tokens):
        system, user = messages[0]["content"], messages[1]["content"]
        role = _role_of(user)
        h = int(hashlib.sha256(user.encode()).hexdigest()[:8], 16)
        dec = ("BUY", "HOLD", "SELL")[h % 3]
        p = round(0.3 + (h % 400) / 1000, 3)
        tag = hashlib.sha256(user.encode()).hexdigest()[:12]  # input-specific text, as real outputs are
        pad = lambda n: (tag + " " + "lorem ipsum dolor sit amet " * (n // 27 + 1))[:n]  # noqa: E731
        if role in ANALYST_ROLES:
            text = pad(SMOKE_OUT_BYTES["analyst"] - 16) + "\nStance: " + ("bullish", "neutral", "bearish")[h % 3]
        elif role in ("bull", "bear"):
            text = pad(SMOKE_OUT_BYTES[role])
        elif role == "trader":
            text = json.dumps({"reason": pad(SMOKE_OUT_BYTES["trader"] - 80), "decision": dec, "p_outperform": p,
                               "cited": ["news"]})
        elif role == "risk":
            text = json.dumps({"reason": pad(SMOKE_OUT_BYTES["risk"] - 100), "decision": dec, "p_outperform": p,
                               "cited": ["news"], "action": "approve"})
        else:
            text = pad(SMOKE_OUT_BYTES["mono"] - 40) + "\n" + json.dumps({"p_outperform": p})
        n_in = self.tok.input_tokens(role, system + user)
        n_out = self.tok.output_tokens(role)
        self.n += 1
        await asyncio.sleep(0)
        return _Obj(model=model, choices=[_Obj(message=_Obj(content=text), finish_reason="stop")],
                    usage=_Obj(prompt_tokens=n_in, completion_tokens=n_out,
                               prompt_tokens_details=_Obj(cached_tokens=0)))


class FakeClient:
    def __init__(self, tok_model):
        self.chat = _Obj(completions=FakeCompletions(tok_model))


class TokenModel:
    """Calibrated token counts (the DEV0/WS-C2 smoke: measured prompt tokens / UTF-8 bytes of system + user, and
    mean output tokens per role; docs/impl/day2_wsC2.md §3). The risk output is the trader's plus `action`, because
    the frozen v1.0 risk template makes it repeat every trader key (the rc3 smoke's approvals omitted them)."""

    IN_PER_BYTE = {"fundamentals": 0.2708, "technical": 0.2774, "news": 0.2628, "macro_policy": 0.2379,
                   "analyst": 0.2628, "bull": 0.2030, "bear": 0.2030, "trader": 0.2062, "risk": 0.1960,
                   "mono": 0.2534}
    OUT = {"analyst": 196.0, "bull": 179.0, "bear": 175.5, "trader": 71.0, "risk": 77.0, "mono": 438.0}

    def input_tokens(self, role: str, text: str) -> int:
        return int(round(self.IN_PER_BYTE[role] * len(text.encode("utf-8"))))

    def output_tokens(self, role: str) -> int:
        return int(round(self.OUT["analyst" if role in ANALYST_ROLES else role]))


# ------------------------------------------------------------------------------------------ helpers
def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs/backends.yaml").read_text())


def ledger_totals(path: Path) -> tuple[float, dict[str, float]]:
    import sqlite3

    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    runs: dict[str, float] = {}
    for tag, usd in c.execute("SELECT tag, COALESCE(SUM(usd), 0) FROM spend GROUP BY tag"):
        r = (tag or "").split("/", 1)[0]
        runs[r] = runs.get(r, 0.0) + float(usd)
    c.close()
    return sum(runs.values()), runs


def week_fingerprint(reader: str, bank, recs: list[dict]) -> dict:
    prompt = bank.prompts[READERS[reader]["prompt"]]
    return {"reader": reader, "bank_sha256": bank.sha256, "prompt_sha256": hashlib.sha256(
        json.dumps(prompt, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "parser_version": PARSER_VERSION, "parser_sha256": PARSER_SHA256, "model": L.MODEL,
            "inputs_sha256": hashlib.sha256("".join(sorted(r["sha256"] for r in recs)).encode()).hexdigest(),
            "n": len(recs)}


def to_inputs(r: dict) -> L.StockWeekInputs:
    return L.StockWeekInputs(symbol=r["symbol"], t=pd.Timestamp(r["t"]), code_name=r["code_name"],
                             fundamentals=r["fundamentals"], technical=r["technical"], news=r["news"],
                             macro_policy=r["macro_policy"], sources=r.get("sources", {}))


def check_inputs(recs: list[dict]) -> dict[str, list[dict]]:
    """1,800 stock-weeks = the last 60 W_clean decisions x 30 names, each record's hash intact."""
    by_week: dict[str, list[dict]] = {}
    for r in recs:
        if TI.sw_sha(r) != r["sha256"]:
            raise SystemExit(f"input record {r['t']} {r['symbol']}: hash mismatch")
        by_week.setdefault(r["t"], []).append(r)
    if len(by_week) != N_WEEKS or any(len(v) != N_NAMES for v in by_week.values()):
        raise SystemExit(f"inputs: {len(by_week)} weeks, sizes {Counter(len(v) for v in by_week.values())}; "
                         f"expected {N_WEEKS} x {N_NAMES}")
    if len({(r['t'], r['symbol']) for r in recs}) != len(recs):
        raise SystemExit("inputs: duplicate stock-weeks")
    return dict(sorted(by_week.items()))


async def _settle() -> None:
    """A budget refusal inside TaLite.run leaves its concurrent calls running: let every call in flight settle (and be
    booked in the ledger) before stopping, so no answered request goes unbooked."""
    me = asyncio.current_task()
    await asyncio.gather(*[t for t in asyncio.all_tasks() if t is not me], return_exceptions=True)


async def run_week(reader: str, llm: L.GenLLM, bank, recs: list[dict]) -> list:
    xs = [to_inputs(r) for r in recs]
    lead: list[L.StockWeekInputs] = []
    first: list = []
    if reader == "tal":
        tal = L.TaLite(llm, bank.prompts["ta-lite-v1"])
        # the week's first stock-week (per distinct macro input) runs alone, so its macro-and-policy analyst request
        # is answered and cached before the other names send the identical request (module doc); its row is the
        # week's one paid macro row
        seen: set[str] = set()
        for x in xs:
            if x.macro_policy not in seen:
                seen.add(x.macro_policy)
                lead.append(x)
        try:
            for x in lead:
                first.append(await tal.run(x))
        finally:
            await _settle()
        coros = [tal.run(x) for x in xs if all(x is not y for y in lead)]
    else:
        mc = L.MonoCoT(llm, bank.prompts["mono-cot-v1"])
        coros = [mc.run(x) for x in xs]
    res = await asyncio.gather(*coros, return_exceptions=True)
    await _settle()
    errs = [r for r in res if isinstance(r, BaseException)]
    budget = [e for e in errs if isinstance(e, (L.BudgetExceeded, LedgerBudgetExceeded))]
    if budget:
        raise budget[0]
    if errs:
        raise errs[0]
    rank = {x.symbol: i for i, x in enumerate(xs)}
    return sorted([*first, *res], key=lambda r: rank[r.symbol])


def write_shard(d: Path, frames: dict[str, pd.DataFrame], marker: dict) -> None:
    """A week with a call still degraded after its retries is written but not marked DONE: a rerun re-asks only
    that call (every other answer is a response-cache hit)."""
    d.mkdir(parents=True, exist_ok=True)
    for k, df in frames.items():
        df.to_parquet(d / f"{k}.parquet", index=False)
    if marker["degraded_calls"] == 0:
        (d / "DONE.json").write_text(json.dumps(marker, indent=1))


def finalize(out: Path, reader: str, weeks: list[str], meta: dict) -> dict:
    parts = {k: [] for k in ("answers", "requests", "outputs")}
    for w in weeks:
        for k in parts:
            parts[k].append(pd.read_parquet(out / "shards" / w / f"{k}.parquet"))
    frames = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    sha = {}
    for k, df in frames.items():
        df.to_parquet(out / f"{k}.parquet", index=False)
        sha[k] = hashlib.sha256((out / f"{k}.parquet").read_bytes()).hexdigest()
    man = {"gen_id": out.name, "reader": reader, "weeks": weeks, "n_weeks": len(weeks),
           "rows": {k: len(v) for k, v in frames.items()}, "sha256": sha,
           "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), **meta}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    return man


# ------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", choices=sorted(READERS), required=True)
    ap.add_argument("--inputs", default=str(TI.DEFAULT_OUT))
    ap.add_argument("--out", default=None, help="default results/gens/<gen_id>")
    ap.add_argument("--dry", action="store_true", help="fake client, scratch ledger and cache (no network, $0)")
    ap.add_argument("--projection", default=str(ROOT / "results/gens/projection.json"))
    ap.add_argument("--max-usd", type=float, default=None, help="this process's own cap (never above the budget)")
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    R = READERS[args.reader]
    out = Path(args.out) if args.out else ROOT / "results/gens" / R["gen_id"]
    if args.dry and args.out is None:
        raise SystemExit("--dry needs --out (a scratch directory)")
    out.mkdir(parents=True, exist_ok=True)

    bank = bank_v1.load_bank()
    if bank.sha256 != bank_v1.FROZEN_SHA256:
        raise SystemExit(f"bank {bank.sha256[:12]} is not the frozen v1.0 bank {str(bank_v1.FROZEN_SHA256)[:12]}")
    by_week = check_inputs(TI.read_inputs(Path(args.inputs)))
    weeks_desc = sorted(by_week, reverse=True)  # newest first (module doc)

    cfg = load_cfg()
    lc = cfg["ledger"]
    sub_caps = {k: float(v) for k, v in (lc.get("sub_caps_usd") or {}).items()}
    sub_cap = sub_caps.get(R["run"])
    if not sub_cap:
        raise SystemExit(f"run {R['run']} has no positive sub-cap in configs/backends.yaml (0 = disabled)")
    hard_cap = float(lc["hard_cap_usd"])
    real_ledger = ROOT / lc["path"]
    total, runs = ledger_totals(real_ledger)
    spent_run = runs.get(R["run"], 0.0)

    # ---------------------------------------------------------------- preflight (printed before any paid call)
    proj = json.loads(Path(args.projection).read_text())[args.reader] if Path(args.projection).exists() else None
    done_weeks = [w for w in weeks_desc if (out / "shards" / w / "DONE.json").exists()]
    frac_left = 1 - len(done_weeks) / len(weeks_desc)
    reserve_room = USER_BUDGET - RESERVE - total
    guard = min(sub_cap - spent_run, reserve_room, hard_cap - total)
    if args.max_usd is not None:
        guard = min(guard, args.max_usd)
    if proj is None:
        print(f"PREFLIGHT {R['run']}: no projection at {args.projection}; refusing a paid run", flush=True)
        if not args.dry:
            return 3
    else:
        p_left, p_hi = proj["projected_usd"] * frac_left, proj["high_usd"] * frac_left
        print(f"PREFLIGHT {R['run']} ({args.reader}, {len(weeks_desc) - len(done_weeks)} of {len(weeks_desc)} weeks "
              f"left): projected ${p_left:.2f} (high ${p_hi:.2f}) vs sub-cap ${sub_cap:.2f} (run spent "
              f"${spent_run:.2f}); ledger total ${total:.2f}, hard cap ${hard_cap:.2f}; user budget ${USER_BUDGET:.0f} "
              f"remaining ${USER_BUDGET - total:.2f}, after the run ${USER_BUDGET - total - p_left:.2f} (reserve "
              f"${RESERVE:.0f}); process guard ${guard:.2f}", flush=True)
        if not args.dry:
            if spent_run + p_left > sub_cap:
                print(f"ABORT: projected ${p_left:.2f} exceeds the {R['run']} sub-cap headroom "
                      f"${sub_cap - spent_run:.2f}", flush=True)
                return 3
            if USER_BUDGET - total - p_left < RESERVE:
                print(f"ABORT: the remaining total budget would fall below ${RESERVE:.0f}", flush=True)
                return 3
    if guard <= 0 and not args.dry:
        print("ABORT: no budget headroom", flush=True)
        return 3

    if args.dry:
        tok = TokenModel()
        # the dry ledger measures (fake money): no caps, so a dry run always prices the whole generation
        ledger = SpendLedger(out / "dry_spend.sqlite", hard_cap_usd=float("inf"), sub_caps={})
        llm = L.GenLLM(client=FakeClient(tok), ledger=ledger, cache=out / "dry_gen_cache.sqlite", tag=R["tag"],
                       budget_usd=args.max_usd, max_concurrency=args.concurrency)  # --max-usd tests the guard
    else:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("no OPENAI_API_KEY")
        ledger = SpendLedger(real_ledger, hard_cap_usd=hard_cap, sub_caps=sub_caps,
                             batch_size=int(lc.get("batch_size", 32)))
        llm = L.GenLLM(ledger=ledger, tag=R["tag"], budget_usd=guard, max_concurrency=args.concurrency)

    t0 = time.time()
    state = {"status": "complete", "n_run": 0}

    async def run_all() -> None:  # one event loop for every week (GenLLM's semaphore is per loop)
        for w in weeks_desc:
            d = out / "shards" / w
            fp = week_fingerprint(args.reader, bank, by_week[w])
            if (d / "DONE.json").exists():
                if json.loads((d / "DONE.json").read_text())["fingerprint"] != fp:
                    raise SystemExit(f"{w}: DONE shard has another fingerprint; refusing to mix")
                continue
            try:
                res = await run_week(args.reader, llm, bank, by_week[w])
            except (L.BudgetExceeded, LedgerBudgetExceeded) as e:
                print(f"BUDGET STOP at week {w}: {e}", flush=True)
                state["status"] = "budget_stop"
            finally:
                await asyncio.to_thread(ledger.flush)
            if state["status"] == "budget_stop":
                return
            frames = L.result_frames(res, bank_hash=bank.sha256, tag=R["tag"])
            req = frames["requests"]
            if (req.response_model != L.MODEL).any():
                raise SystemExit(f"{w}: a response was not answered by {L.MODEL}")
            marker = {"fingerprint": fp, "paid_calls": int((~req.cached).sum()),
                      "paid_usd": float(req.usd[~req.cached].sum()), "degraded_calls": int(req.degraded.sum()),
                      "written_utc": pd.Timestamp.now(tz="UTC").isoformat()}
            await asyncio.to_thread(write_shard, d, frames, marker)
            state["n_run"] += 1
            print(f"{w} {'DONE' if marker['degraded_calls'] == 0 else 'NOT DONE (degraded calls)'}: {marker['paid_calls']} paid calls ${marker['paid_usd']:.4f}; process spent "
                  f"${llm.spent:.4f} ({time.time() - t0:.0f}s)", flush=True)

    asyncio.run(run_all())
    ledger.flush()
    status, n_run = state["status"], state["n_run"]
    done = sorted([w for w in weeks_desc if (out / "shards" / w / "DONE.json").exists()])
    info = {"status": status, "weeks_run_this_launch": n_run, "process_spent_usd": llm.spent,
            "process_calls": llm.calls, "degraded_calls": llm.degraded_calls, "dry": args.dry,
            "guard_usd": guard, "tag": R["tag"], "run": R["run"], "wall_s": time.time() - t0}
    (out / f"launch_{int(time.time())}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    if len(done) == len(weeks_desc):
        man = finalize(out, args.reader, done, {"bank_sha256": bank.sha256, "parser_version": PARSER_VERSION,
                                                "parser_sha256": PARSER_SHA256, "model": L.MODEL, "tag": R["tag"],
                                                "inputs": str(Path(args.inputs).relative_to(ROOT))
                                                if Path(args.inputs).is_relative_to(ROOT) else args.inputs})
        print(f"finalized {man['rows']}", flush=True)
        return 0
    return 3 if status == "budget_stop" else 1


if __name__ == "__main__":
    sys.exit(main())
