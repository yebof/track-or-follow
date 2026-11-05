"""DEV #2 ops: measure provider prompt caching on 50 real DEV N1 requests (docs/impl/dev2_ops.md §1.4).

Paid (tag DEV2OPS/N1-<arm>; ledger sub-cap DEV2OPS = $0.50 in configs/backends.yaml; own guard $0.20). The requests
are 50 distinct N1 states of the DEV #1 generation (2021 W_dev weeks; seeded sample), asked with the live bank's N1
questions through the production emulator (`make_backend`), with the response cache off so that every request reaches
the provider. Each arm routes on its own fresh key namespace (a nonce), so no arm can reuse another arm's or an
earlier run's cache entries (a `prompt_cache_key` separates cache reuse between groups of requests):

  A  "DEV #1 regime": 17 requests, one key, no warm-up, all sent at once (concurrency 64): what DEV #1's N1 traffic
     looked like to the provider (one routing group far above ~15 requests per minute);
  B  "router, burst": 17 requests through the router as the OpenAI guide advises (warm-up gate, keys partitioned at
     ~15 rpm), all sent at once: the start of a production run;
  C  "router, steady": 16 requests through the same router, one every 4.1 s (<= 15 rpm on one key): the per-key
     regime the guide's partitioning aims for in a long run.

Per request it records tokens (input, provider-cached, output), $, latency, the prompt_cache_key and the top labels,
and per arm the cached share of input tokens, the hit rate and the realization (cached tokens over the cacheable
ceiling: the static prefix, rounded down to 128 tokens). No state text is written to the repository (request keys
only). Answers are compared with DEV #1's joint answers as a sanity check that routing changes nothing.

    .venv/bin/python experiments/plan/measure_prompt_cache.py [--dry-run] [--states <states_dev1.jsonl.gz>]
        -> results/dev2_ops/prompt_cache_measure.json   (--dry-run: fake provider, $0, output not written)
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import sqlite3
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

import jevtrader.s1.backends as SB
from jevtrader.programs.bank_v1 import load_bank

RUN = "DEV2OPS"
GUARD_USD = 0.20
GEN = ROOT / "results/judgments/dev1_emu_wdev10_s50"
STATES = Path("/tmp/jevtrader-scratch/dev1/"
              "states_dev1.jsonl.gz")
OUT = ROOT / "results/dev2_ops/prompt_cache_measure.json"
# B and C ran (09-28, nonce 20260927T163216) with the router defaults of that time, written out here: partitioning on
# at 15 rpm per key (max_shards 256) and the warm-up gate; the default became max_shards = 1 after this measurement
PARTITIONED = {"rpm_per_key": 15, "max_shards": 256, "warm": True}
ARMS = (("A", 17, {"warm": False, "max_shards": 1}, "burst"),
        ("B", 17, PARTITIONED, "burst"),
        ("C", 16, PARTITIONED, "paced"))
PACE_S = 4.1  # just under 15 requests per minute (16 requests span 61.5 s: never 16 in one minute)


def run_spent(path: Path) -> float:
    with sqlite3.connect(path) as c:
        return float(c.execute("SELECT COALESCE(SUM(usd), 0) FROM spend WHERE tag = ? OR tag LIKE ?",
                               (RUN, RUN + "/%")).fetchone()[0])


def sample_states(states_path: Path, n: int) -> list[dict]:
    """n distinct DEV #1 N1 states (seeded), with their DEV #1 request keys."""
    req = pd.read_parquet(GEN / "requests.parquet")
    paid = set(req[(req.program == "N1") & ~req.cached].request_key)
    seen: dict[str, dict] = {}
    with gzip.open(states_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["tag"] == "DEV1/N1" and r["request_key"] in paid and r["request_key"] not in seen:
                seen[r["request_key"]] = json.loads(r["state"])
    keys = random.Random(RUN).sample(sorted(seen), n)
    return [{"dev1_key": k, "state": seen[k]} for k in keys]


def dev1_top_labels() -> dict[str, dict[str, str | None]]:
    ans = pd.read_parquet(GEN / "answers.parquet", columns=["request_key", "qid", "probs_json"])
    out: dict[str, dict[str, str | None]] = {}
    for r in ans.drop_duplicates(["request_key", "qid"]).itertuples():
        p = json.loads(r.probs_json) if isinstance(r.probs_json, str) else None
        out.setdefault(r.request_key, {})[r.qid] = None if p is None else max(p, key=p.get)
    return out


def fake_client():
    from jevtrader.s1.fakes import chat_completion, letter_answers, openai_client

    def handler(body):
        n = sum(1 for ln in body["messages"][1]["content"].split("\n") if ln.startswith("[q"))
        return chat_completion(letter_answers([{"A": 0.9, "B": 0.1}] * n), SB.EMULATOR_MODEL, (2100, 50, 1792))

    return openai_client(handler)


async def run_arm(arm: str, rows: list[dict], pc: dict, mode: str, questions: dict, nonce: str, dry: bool) -> list[dict]:
    ns = f"dev2ops-{arm}-{nonce}"
    kw = {"client": fake_client()} if dry else {}
    s1 = SB.make_backend(backend="llm-logprob", cache=False, ledger=not dry, prompt_cache={**pc, "namespace": ns}, **kw)
    t0 = time.monotonic()
    out: list[dict] = []

    async def one(i: int, row: dict) -> None:
        if mode == "paced":
            await asyncio.sleep(max(0.0, t0 + i * (0.01 if dry else PACE_S) - time.monotonic()))
        sent = time.monotonic() - t0
        r = await s1.ask(row["state"], questions, tag=f"{RUN}/N1-{arm}")
        out.append({"arm": arm, "i": i, "dev1_key": row["dev1_key"], "request_key": r.request_key,
                    "same_key_as_dev1": r.request_key == row["dev1_key"], "prompt_cache_key": r.meta.get("prompt_cache_key"),
                    "sent_s": round(sent, 3), "done_s": round(time.monotonic() - t0, 3),
                    "input_tokens": r.input_tokens, "cached_tokens": r.cached_tokens, "output_tokens": r.output_tokens,
                    "usd": r.cost_usd, "latency_s": round(r.latency_s, 3), "attempts": r.attempts,
                    "degraded": list(r.degraded),
                    "top": {q: a.top for q, a in r.answers.items()}})

    await asyncio.gather(*(one(i, row) for i, row in enumerate(rows)))
    if s1.ledger is not None:
        s1.ledger.flush()
    out.sort(key=lambda x: x["i"])
    for x in out:
        x["router"] = {"shards": s1._router.shards(), "stats": dict(s1._router.stats)} if x["i"] == 0 else None
    return out


def summarize(rows: list[dict], ceiling: int) -> dict:
    n = len(rows)
    inp = sum(r["input_tokens"] for r in rows)
    cached = sum(r["cached_tokens"] for r in rows)
    hits = [r for r in rows if r["cached_tokens"] > 0]
    return {"n": n, "input_tokens": inp, "cached_tokens": cached, "cached_share": cached / inp if inp else None,
            "hit_rate": len(hits) / n if n else None,
            "cached_per_hit": statistics.median([r["cached_tokens"] for r in hits]) if hits else 0,
            "realization": cached / (n * ceiling) if n else None,
            "usd": sum(r["usd"] for r in rows), "latency_p50_s": statistics.median(r["latency_s"] for r in rows),
            "wall_s": max(r["done_s"] for r in rows), "keys": sorted({r["prompt_cache_key"] for r in rows}),
            "degraded": sum(len(r["degraded"]) for r in rows)}


async def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--states", default=str(STATES))
    a = ap.parse_args(argv)
    cfg = SB.load_backend_config()
    led_path = ROOT / cfg["ledger"]["path"]
    cap = float(cfg["ledger"]["sub_caps_usd"][RUN])
    before = run_spent(led_path)
    if not a.dry_run and cap - before < GUARD_USD:
        raise SystemExit(f"{RUN} headroom {cap - before:.4f} < guard {GUARD_USD}")
    bank = load_bank()
    questions = {s.name: s.to_question() for s in bank.program("N1").values()}
    rows = sample_states(Path(a.states), sum(n for _, n, _, _ in ARMS))
    probe = SB.make_backend(backend="llm-logprob", cache=False, ledger=False, client=fake_client())
    prefix = probe.static_prefix(questions)
    req = pd.read_parquet(GEN / "requests.parquet")
    hits = req[(req.program == "N1") & ~req.cached & (req.cached_tokens > 0)].cached_tokens
    ceiling = int(hits.mode().iloc[0])  # DEV #1's N1 cached prefix (the static prefix rounded down to 128 tokens)
    nonce = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    results, k = [], 0
    for arm, n, pc, mode in ARMS:
        results += await run_arm(arm, rows[k:k + n], pc, mode, questions, nonce, a.dry_run)
        k += n
        await asyncio.sleep(0 if a.dry_run else 5)
    spent = run_spent(led_path) - before
    top1 = dev1_top_labels()
    agree = [r["top"][q] == top1.get(r["dev1_key"], {}).get(q) for r in results for q in r["top"]
             if top1.get(r["dev1_key"], {}).get(q) is not None and r["top"][q] is not None]
    out = {
        "run_tag": RUN, "nonce": nonce, "dry_run": a.dry_run, "bank_sha256": bank.sha256,
        "model": SB.EMULATOR_MODEL, "n_requests": len(results),
        "static_prefix": {"chars": len(prefix), "proxy_tokens": SB.estimate_tokens(prefix),
                          "dev1_cached_prefix_tokens": ceiling,
                          "dev1_n1_input_tokens_mean": float(req[(req.program == "N1") & ~req.cached].input_tokens.mean())},
        "request_keys_equal_dev1": sum(r["same_key_as_dev1"] for r in results),
        "arms": {arm: {"design": {"n": n, "prompt_cache": pc, "mode": mode},
                       **summarize([r for r in results if r["arm"] == arm], ceiling)} for arm, n, pc, mode in ARMS},
        "answers_agree_with_dev1": {"n": len(agree), "share": sum(agree) / len(agree) if agree else None},
        "ledger_spend_usd": spent,
        "requests": [{k2: v for k2, v in r.items() if k2 != "top"} for r in results],
    }
    print(json.dumps({k2: v for k2, v in out.items() if k2 != "requests"}, indent=1, default=str))
    if not a.dry_run:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    asyncio.run(main())
