"""Shared machinery of the pre-commit paid generations of 2026-09-28 (PROBE, VERB, FID, MEM; MCOT runs through
experiments/gens/run_gen.py via experiments/run_gen_mcot.py). docs/impl/precommit_paid.md describes the runs.

Budget rules (CLAUDE.md; decisions.md 2026-09-28 P-2; the operator task):
- every paid launch prints a preflight first: the projection of a free dry run (point and high) against the run's
  sub-cap in configs/backends.yaml, the ledger total, the $245 hard cap and the task's stop-list threshold ($225:
  the list stops if the ledger would exceed it); it aborts (exit 3) if the point projection exceeds the sub-cap
  headroom or would take the ledger above $225 or the hard cap;
- the paid process is also bounded by a guard = min(sub-cap headroom, $225 - ledger, hard cap - ledger): each call
  reserves a conservative estimate (bytes/3.5 input + 400, the decoding's output bound) before it is sent
  (`run_dev.SpendGuard`); the ledger's own sub-cap and hard-cap checks apply to every call as well;
- no paid smoke test: runners are tested with the fake OpenAI transport below ($0) and by cache replay.

Dry runs: the real backend classes (`LogprobEmulator`, `VerbalizedLLM`) run end to end on a fake OpenAI transport
(`s1.fakes.openai_client`), a scratch ledger without caps, and the real response cache opened READ-ONLY with an
in-memory overlay (so a request the paid run would get from the cache is free in the projection too, and the real
cache is never written). Usage is calibrated: input tokens = the family's tokens-per-byte ratio x the UTF-8 bytes of
the prompt (ratios measured on rebuilt MAIN/DEV prompts against their ledger tokens, see `CALIB`), output tokens =
4 per question (emu; MAIN measured 3.9 per question) or 9 per option + 6 per question + 8 (verbalized). The high
projection prices every call at bytes/3.5 input tokens and the decoding's maximum output.

W_clean rule: nothing here reads a return, a label or any W_clean outcome.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import re
import sqlite3
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "experiments" / "dev", ROOT / "experiments" / "pre"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from jevtrader.programs import bank_v1  # noqa: E402
from jevtrader.programs.base import ProgramOutput, Request, ask  # noqa: E402
from jevtrader.programs.runner import COLUMNS, SORT, _frames  # noqa: E402
from jevtrader.s1.budget import BudgetExceeded, SpendLedger  # noqa: E402
from jevtrader.s1.pricing import cost_usd  # noqa: E402
from jevtrader.s1.types import S1Response  # noqa: E402

log = logging.getLogger("run_gen_common")
USER_BUDGET = 250.0
STOP_LEDGER = 225.0  # the task's stop rule: the list stops if the ledger would exceed $225
MODEL = "gpt-4.1-mini-2025-04-14"
TABLES = ("answers", "requests", "evidence")
# tokens per UTF-8 byte of system + user text (point projection); measured in `calibrate_*` below and recorded in each
# projection JSON. Defaults are the o200k-typical 1/3.9 for JSON-heavy English until a run measures its own.
CALIB = {"default": 1 / 3.9}


# ------------------------------------------------------------------------------------------ config and ledger
def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "backends.yaml").read_text())


def ledger_path() -> Path:
    return ROOT / load_cfg()["ledger"]["path"]


def ledger_totals(path: Path | None = None) -> tuple[float, dict[str, float]]:
    c = sqlite3.connect(f"file:{path or ledger_path()}?mode=ro", uri=True)
    runs: dict[str, float] = {}
    for tag, usd in c.execute("SELECT tag, COALESCE(SUM(usd), 0) FROM spend GROUP BY tag"):
        r = (tag or "").split("/", 1)[0]
        runs[r] = runs.get(r, 0.0) + float(usd)
    c.close()
    return sum(runs.values()), runs


def ledger_run_rows(run: str, path: Path | None = None, since: float | None = None) -> dict:
    c = sqlite3.connect(f"file:{path or ledger_path()}?mode=ro", uri=True)
    q = "SELECT COUNT(*), COALESCE(SUM(usd), 0), COALESCE(SUM(input_tokens), 0), COALESCE(SUM(output_tokens), 0), " \
        "MIN(ts), MAX(ts) FROM spend WHERE (tag = ? OR tag LIKE ?)"
    args: list[Any] = [run, f"{run}/%"]
    if since is not None:
        q += " AND ts >= ?"
        args.append(since)
    n, usd, i, o, t0, t1 = c.execute(q, args).fetchone()
    by = {tag: {"rows": k, "usd": round(float(u), 6)} for tag, k, u in c.execute(
        "SELECT tag, COUNT(*), SUM(usd) FROM spend WHERE (tag = ? OR tag LIKE ?) GROUP BY tag", [run, f"{run}/%"])}
    c.close()
    return {"rows": int(n), "usd": float(usd), "input_tokens": int(i), "output_tokens": int(o), "first_ts": t0,
            "last_ts": t1, "by_tag": by}


def sub_cap(run: str) -> float:
    caps = {k: float(v) for k, v in (load_cfg()["ledger"].get("sub_caps_usd") or {}).items()}
    if not caps.get(run):
        raise SystemExit(f"run {run} has no positive sub-cap in configs/backends.yaml (0 = disabled)")
    return caps[run]


@dataclasses.dataclass
class Preflight:
    run: str
    ok: bool
    guard: float
    line: str
    total: float
    spent_run: float


def preflight(run: str, projected: float, high: float, what: str, dry: bool = False) -> Preflight:
    """Print the preflight line; `ok` is False (abort) if the point projection does not fit the sub-cap headroom, the
    $225 stop threshold or the hard cap. The guard bounds the paid process by the same three limits."""
    cap = sub_cap(run)
    lc = load_cfg()["ledger"]
    hard = float(lc["hard_cap_usd"])
    total, runs = ledger_totals()
    spent = runs.get(run, 0.0)
    guard = min(cap - spent, STOP_LEDGER - total, hard - total)
    line = (f"PREFLIGHT {run} ({what}): projected ${projected:.2f} (high ${high:.2f}) vs sub-cap ${cap:.2f} (run spent "
            f"${spent:.2f}, headroom ${cap - spent:.2f}); ledger total ${total:.2f}, hard cap ${hard:.2f}, stop-list "
            f"threshold ${STOP_LEDGER:.0f}: after the run ${total + projected:.2f}; user budget ${USER_BUDGET:.0f} "
            f"remaining after the run ${USER_BUDGET - total - projected:.2f}; process guard ${guard:.2f}")
    print(line, flush=True)
    ok = True
    if spent + projected > cap:
        print(f"ABORT: projected ${projected:.2f} exceeds the {run} sub-cap headroom ${cap - spent:.2f}", flush=True)
        ok = False
    elif total + projected > STOP_LEDGER:
        print(f"ABORT: the ledger would exceed ${STOP_LEDGER:.0f} (stop the list)", flush=True)
        ok = False
    elif total + projected > hard:
        print(f"ABORT: the ledger would exceed the hard cap ${hard:.2f}", flush=True)
        ok = False
    elif guard <= 0:
        print("ABORT: no budget headroom", flush=True)
        ok = False
    return Preflight(run, ok or dry, guard, line, total, spent)


# ------------------------------------------------------------------------------------------ read-only cache overlay
class ReadOnlyCache:
    """The real response cache opened read-only, with an in-memory overlay for what a dry run 'caches'. Same interface
    as `s1.cache.ResponseCache` for the backends (aget/aput)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False) \
            if self.path.exists() else None
        self._overlay: dict[str, dict] = {}
        self.hits_real = 0

    def get(self, key: str) -> S1Response | None:
        raw = self._overlay.get(key)
        if raw is None and self._db is not None:
            row = self._db.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
            if row is not None:
                raw = json.loads(row[0])
                self.hits_real += 1
        if raw is None:
            return None
        r = S1Response.from_json(raw)
        r.cached, r.request_key = True, key
        return r

    def put(self, key: str, response: S1Response) -> None:
        if response.degraded or any(a.probs is None for a in response.answers.values()):
            raise ValueError("degraded responses are never cached")
        self._overlay[key] = response.to_json()

    async def aget(self, key: str) -> S1Response | None:
        return self.get(key)

    async def aput(self, key: str, response: S1Response) -> None:
        self.put(key, response)


# ------------------------------------------------------------------------------------------ fake OpenAI transport
_Q = re.compile(r"^\[q(\d+)\] ")
_OPT = re.compile(r"^  ([A-T]): ")


def question_shape(user: str) -> list[int]:
    """Number of options of each question of an emu-v1 / verb-v1 user message (render order)."""
    block = user.split("QUESTIONS:\n", 1)[1]
    block = block.split("\nReply with", 1)[0].split("\nFor every question give", 1)[0]
    shape: list[int] = []
    for ln in block.split("\n"):
        if _Q.match(ln):
            shape.append(0)
        elif _OPT.match(ln) and shape:
            shape[-1] += 1
    return shape


class FakeOpenAI:
    """Deterministic stand-in for the provider (no network): emu answers with letter log-probabilities, verbalized
    answers with a JSON object; usage from the calibration; every call recorded for the projection."""

    def __init__(self, ratio: Callable[[str, str], float] | float = CALIB["default"], degrade_every: int = 0):
        self.ratio = ratio
        self.calls: list[dict] = []
        self.degrade_every = degrade_every  # test hook: every n-th call returns a malformed answer

    def handler(self, body: dict) -> dict:
        from jevtrader.s1.fakes import chat_completion, letter_answers

        msgs = body["messages"]
        system, user = msgs[0]["content"], msgs[-1]["content"]
        text = system + user
        nbytes = len(text.encode("utf-8"))
        shape = question_shape(user)
        verb = "response_format" in body
        h = hashlib.sha256(text.encode()).digest()
        r = self.ratio(system, user) if callable(self.ratio) else float(self.ratio)
        n_in = int(round(r * nbytes))
        max_out = int(body.get("max_completion_tokens") or 0)
        n_opts = sum(shape)
        bad = self.degrade_every and (len(self.calls) + 1) % self.degrade_every == 0
        if verb:
            n_out = min(max_out, 9 * n_opts + 6 * len(shape) + 8)
            obj = {}
            for i, k in enumerate(shape, 1):
                w = [1 + h[(i * 7 + j) % len(h)] for j in range(k)]
                obj[f"q{i}"] = {"ABCDEFGHIJKLMNOPQRST"[j]: round(w[j] / sum(w), 3) for j in range(k)}
            content = "not json" if bad else json.dumps(obj)
            resp = chat_completion([(content, [])], MODEL, usage=(n_in, n_out, 0))
            resp["choices"][0]["logprobs"] = None
        else:
            n_out = min(max_out, 4 * len(shape))
            dists = []
            for i, k in enumerate(shape, 1):
                w = [math.exp(3 * (h[(i * 5 + j) % len(h)] / 255 - 0.5)) for j in range(k)]
                dists.append(None if bad and i == 1 else {"ABCDEFGHIJKLMNOPQRST"[j]: w[j] / sum(w) for j in range(k)})
            resp = chat_completion(letter_answers(dists), MODEL, usage=(n_in, n_out, 0))
        self.calls.append({"bytes": nbytes, "n_q": len(shape), "n_opts": n_opts, "in": n_in, "out": n_out,
                           "max_out": max_out, "verb": verb, "usd": cost_usd(MODEL, n_in, n_out),
                           "usd_high": cost_usd(MODEL, math.ceil(nbytes / 3.5) + 7, max_out)})
        return resp

    def client(self):
        from jevtrader.s1.fakes import openai_client

        return openai_client(self.handler)

    def summary(self) -> dict:
        c = self.calls
        return {"calls": len(c), "in_tokens": sum(x["in"] for x in c), "out_tokens": sum(x["out"] for x in c),
                "usd": round(sum(x["usd"] for x in c), 4), "usd_high": round(sum(x["usd_high"] for x in c), 4)}


def make_s1(backend: str, dry: bool, scratch: Path | None = None, fake: FakeOpenAI | None = None,
            max_concurrency: int | None = None):
    """The backend of a run. Paid: `make_backend` with the configured cache and ledger (sub-caps, hard cap). Dry: the
    same class on the fake transport, the read-only cache overlay and an uncapped scratch ledger."""
    from jevtrader.s1.backends import make_backend

    cfg = load_cfg()
    kw = {} if max_concurrency is None else {"max_concurrency": max_concurrency}
    if not dry:
        s1 = make_backend(backend=backend, **kw)
        if s1.ledger is None or s1.cache is None:
            raise SystemExit("paid backend without ledger or cache")
        return s1
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    led = SpendLedger(scratch / "dry_spend.sqlite", hard_cap_usd=float("inf"), sub_caps={})
    cache = ReadOnlyCache(ROOT / cfg["cache"]["path"])
    s1 = make_backend(backend=backend, cache=False, ledger=led, client=(fake or FakeOpenAI()).client(), **kw)
    s1.cache = cache
    return s1


class Tagged:
    """Backend proxy that prefixes every ledger tag with the run id ("PROBE/D")."""

    def __init__(self, s1, run: str):
        self._s1, self.run = s1, run

    def __getattr__(self, k):
        return getattr(self._s1, k)

    async def ask(self, state, questions, tag: str = "", **kw):
        return await self._s1.ask(state, questions, tag=f"{self.run}/{tag}", **kw)


def guarded(s1, run: str, max_usd: float):
    from run_dev import SpendGuard

    return SpendGuard(Tagged(s1, run), max_usd)


def guarded_raw(s1, max_usd: float):
    """The guard for runs through `programs.runner.generate`, whose own proxy already prefixes the run tag."""
    from run_dev import SpendGuard

    return SpendGuard(s1, max_usd)


# ------------------------------------------------------------------------------------------ jobs and shards
@dataclasses.dataclass
class Job:
    uid: str  # unique within the generation
    request: Request
    meta: dict  # written to jobs.parquet (stratum, level, source, ...)


def _jsonable(x):
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return x


async def _ask_one(s1, job: Job, sem: asyncio.Semaphore):
    async with sem:
        a = await ask(s1, job.request)
    return job, a


async def run_shard(s1, jobs: list[Job], concurrency: int) -> tuple[list, BaseException | None]:
    """Ask every job; on a budget refusal let the calls in flight settle (and be booked) and report it."""
    sem = asyncio.Semaphore(concurrency)
    res = await asyncio.gather(*(_ask_one(s1, j, sem) for j in jobs), return_exceptions=True)
    errs = [r for r in res if isinstance(r, BaseException)]
    budget = [e for e in errs if isinstance(e, BudgetExceeded)]
    if budget:
        return [r for r in res if not isinstance(r, BaseException)], budget[0]
    if errs:
        raise errs[0]
    return list(res), None


def frames_of(asked: list, run_tag: str, data_freeze: str) -> dict[str, pd.DataFrame]:
    outs = [ProgramOutput(j.request.program, j.request.entity, pd.Timestamp(j.request.t), {}, a.trace, [a.request],
                          a.evidence) for j, a in asked]
    fr = _frames(outs, "D", run_tag, data_freeze)
    fr.pop("features")
    jobs = pd.DataFrame([{"uid": j.uid, "request_key": a.request.request_key, "program": j.request.program,
                          "stage": j.request.stage, "t": pd.Timestamp(j.request.t), "symbol": j.request.entity,
                          "state_sha256": a.request.state_sha256,
                          "meta_json": json.dumps({k: _jsonable(v) for k, v in j.meta.items()}, sort_keys=True,
                                                  default=str)}
                         for j, a in asked])
    fr["jobs"] = jobs
    return fr


def run_shards(out: Path, shards: dict[str, list[Job]], s1, run_tag: str, data_freeze: str, fingerprint: dict,
               concurrency: int = 32, order: Iterable[str] | None = None) -> dict:
    """Ask the jobs shard by shard (in `order`), writing `shards/<name>/{answers,requests,evidence,jobs}.parquet` and
    DONE.json (with the fingerprint). A rerun skips DONE shards (and refuses one with another fingerprint); the response
    cache makes redoing an interrupted shard free except for degraded answers."""
    state = {"status": "complete", "shards_run": 0, "budget_error": None}
    order = list(order or shards)
    t0 = time.time()

    async def go():
        for name in order:
            d = out / "shards" / name
            fp = {**fingerprint, "shard": name, "uids_sha256": hashlib.sha256(
                "\n".join(sorted(j.uid for j in shards[name])).encode()).hexdigest(), "n": len(shards[name])}
            if (d / "DONE.json").exists():
                if json.loads((d / "DONE.json").read_text())["fingerprint"] != fp:
                    raise SystemExit(f"shard {name}: DONE with another fingerprint; refusing to mix")
                continue
            asked, err = await run_shard(s1, shards[name], concurrency)
            led = getattr(s1, "ledger", None)
            if led is not None:
                await asyncio.to_thread(led.flush)
            if err is not None:
                print(f"BUDGET STOP in shard {name}: {err}", flush=True)
                state["status"], state["budget_error"] = "budget_stop", str(err)
                return
            fr = frames_of(asked, run_tag, data_freeze)
            d.mkdir(parents=True, exist_ok=True)
            for k, df in fr.items():
                df.to_parquet(d / f"{k}.parquet", index=False)
            req = fr["requests"]
            marker = {"fingerprint": fp, "paid_calls": int((~req.cached.astype(bool)).sum()),
                      "paid_usd": float(req.usd[~req.cached.astype(bool)].sum()),
                      "degraded_requests": int((req.degraded > 0).sum()),
                      "written_utc": pd.Timestamp.now(tz="UTC").isoformat()}
            (d / "DONE.json").write_text(json.dumps(marker, indent=1))
            state["shards_run"] += 1
            spent = getattr(s1, "spent", float("nan"))
            print(f"{name}: {len(asked)} requests, {marker['paid_calls']} paid ${marker['paid_usd']:.4f}, "
                  f"{marker['degraded_requests']} degraded; process spent ${spent:.4f} ({time.time() - t0:.0f}s)",
                  flush=True)

    asyncio.run(go())
    return state


def finalize(out: Path, order: list[str], meta: dict) -> dict:
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in (*TABLES, "jobs")}
    for name in order:
        d = out / "shards" / name
        if not (d / "DONE.json").exists():
            raise RuntimeError(f"shard {name} is not DONE")
        for k in parts:
            df = pd.read_parquet(d / f"{k}.parquet")
            if len(df):
                parts[k].append(df)
    sha, rows = {}, {}
    for k, v in parts.items():
        df = pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=COLUMNS.get(k, []))
        if k in SORT:
            df = df[COLUMNS[k]].sort_values(SORT[k], kind="stable").reset_index(drop=True)
        else:
            df = df.sort_values("uid", kind="stable").reset_index(drop=True)
        df.to_parquet(out / f"{k}.parquet", index=False)
        sha[k] = hashlib.sha256((out / f"{k}.parquet").read_bytes()).hexdigest()
        rows[k] = len(df)
    req = pd.read_parquet(out / "requests.parquet")
    ans = pd.read_parquet(out / "answers.parquet")
    paid = ~req.cached.astype(bool)
    man = {"gen_id": out.name, "shards": order, "rows": rows, "sha256": sha,
           "validity_rate": float(1 - ans.probs_json.isna().mean()) if len(ans) else None,
           "paid_requests": int(paid.sum()), "paid_usd_rows": float(req.usd[paid].sum()),
           "bank_sha256": bank_v1.FROZEN_SHA256, "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), **meta}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1, default=str))
    return man


def launch_record(out: Path, info: dict) -> None:
    (out / f"launch_{int(time.time())}.json").write_text(json.dumps(info, indent=1, default=str))


def projection_record(path: Path, run: str, fake: FakeOpenAI, s1, extra: dict) -> dict:
    """Projection of a dry run: point = the scratch ledger (fake usage), high = every paid call at bytes/3.5 and the
    maximum output; cache hits of the real cache are free in both."""
    led = s1.ledger
    led.flush()
    c = sqlite3.connect(f"file:{led.path}?mode=ro", uri=True)
    n, usd = c.execute("SELECT COUNT(*), COALESCE(SUM(usd), 0) FROM spend").fetchone()
    by = {tag: {"calls": k, "usd": round(float(u), 4)} for tag, k, u in
          c.execute("SELECT tag, COUNT(*), SUM(usd) FROM spend GROUP BY tag")}
    c.close()
    rec = {"run": run, "paid_calls": int(n), "projected_usd": round(float(usd), 4),
           "high_usd": round(sum(x["usd_high"] for x in fake.calls), 4), "by_tag": by, "fake": fake.summary(),
           "real_cache_hits": getattr(s1.cache, "hits_real", None), "calibration": extra.pop("calibration", None),
           "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), **extra}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1, default=str))
    return rec


def prompt_bytes(s1, state, questions) -> int:
    msgs, _ = s1.messages(state, questions)
    return sum(len(m["content"].encode("utf-8")) for m in msgs)


def calibrate(s1, samples: list[tuple[Any, dict, int]]) -> dict:
    """tokens-per-byte of the emu/verb prompt from (state, questions, measured input tokens) triples."""
    b = [prompt_bytes(s1, st, qs) for st, qs, _ in samples]
    t = [int(n) for _, _, n in samples]
    if not samples:
        return {"n": 0, "ratio": CALIB["default"]}
    return {"n": len(samples), "ratio": sum(t) / sum(b), "bytes_mean": sum(b) / len(b), "tokens_mean": sum(t) / len(t),
            "ratio_min": min(x / y for x, y in zip(t, b)), "ratio_max": max(x / y for x, y in zip(t, b))}


# ------------------------------------------------------------------------------------------ MAIN replay ($0)
MAIN_DIR = ROOT / "results" / "judgments" / "main_emu_wclean_u100"
W_CLEAN = ("2024-07-05", "2026-09-11")


def main_weeks() -> list[pd.Timestamp]:
    """MAIN's decision weeks inside W_clean (115; MAIN's pseudo-forward 2026-09-18 excluded)."""
    man = json.loads((MAIN_DIR / "MANIFEST.json").read_text())
    ws = [pd.Timestamp(w) for w in man["spec"]["weeks"] if W_CLEAN[0] <= w <= W_CLEAN[1]]
    if len(ws) != 115:
        raise SystemExit(f"MAIN has {len(ws)} W_clean weeks, expected 115")
    return ws


def main_context(subset: str | None, symbols=None):
    """A context builder that renders exactly MAIN's states (MAIN's fact symbols and archive); `subset` only selects
    the focus names (cross-sectional buckets are over the full U_t, so a subset run renders a full run's states)."""
    from jevtrader.programs.context import ContextBuilder, RealData

    man = json.loads((MAIN_DIR / "MANIFEST.json").read_text())
    dev = json.loads((MAIN_DIR / "dev_run.json").read_text())
    data = RealData()
    data.fact_symbols = data.symbols_for_weeks([pd.Timestamp(w) for w in dev["weeks"]])
    main_builder = ContextBuilder(data, subset=None)
    from jevtrader.s1.types import canonical_json

    if hashlib.sha256(canonical_json(main_builder.describe()).encode()).hexdigest()[:10] != dev["context_id"]:
        raise SystemExit("context differs from MAIN's")
    if data.archive_id() != man["spec"]["data_freeze"]:
        raise SystemExit("archive differs from MAIN's data freeze")
    return ContextBuilder(data, subset=subset, symbols=symbols), data


def work_records(gen: Path, prefixes: tuple[str, ...]):
    """Records of a generation's M-work blackboard whose key starts with one of `prefixes` (streamed)."""
    with open(gen / "memory" / "work.jsonl") as f:
        for line in f:
            if not any(f'"key": "{p}' in line for p in prefixes):
                continue
            d = json.loads(line)
            if d["record"]["key"].startswith(prefixes):
                yield d


class ReplayN01:
    """N01's blackboard rows of a generation, replayed without asking anything: `N01.items.<s>`, `N01.errors.<s>` and
    `N01.sentinel.<s>` of each (week, focus name), exactly as MAIN's N01 wrote them. N23 (news.NewsEventProgram) reads
    only these rows from N01 and re-checks that every item rebuilds to the N1 state N01 judged (`n1_sha256`)."""

    name = "N01"
    taint = "W"
    depends_on: tuple[str, ...] = ()
    week_ordered = False
    memory_reads: tuple[str, ...] = ()

    def __init__(self, gen: Path, weeks: Iterable[pd.Timestamp], symbols: dict[pd.Timestamp, set[str]] | None = None):
        self.version = f"N01-replay-{gen.name}"
        want = {str(pd.Timestamp(w).date()) for w in weeks}
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}
        for d in work_records(gen, ("N01.",)):
            rec = d["record"]
            t = rec["t"][:10]
            if t not in want:
                continue
            prefix, sym = rec["key"].rsplit(".", 1)[0] + ".", rec["key"].rsplit(".", 1)[1]
            if symbols is not None and sym not in symbols.get(pd.Timestamp(t), set()):
                continue
            self.rows.setdefault((t, sym), {})[rec["key"]] = rec["value"]

    async def run(self, ctx, s1) -> list[ProgramOutput]:
        t = pd.Timestamp(ctx.t)
        out = []
        for sym in ctx.stock_names():
            bb = self.rows.get((str(t.date()), sym))
            if bb is None:
                raise RuntimeError(f"no N01 rows of {sym} {t.date()} in the replayed generation")
            out.append(ProgramOutput("N01", sym, t, {}, blackboard=dict(bb)))
        return out


class MainReplay:
    """A $0 backend that answers every request from MAIN's answers table by request key: the exact answers MAIN got
    (degraded ones included). A request MAIN never sent raises `KeyError`: the state was not reproduced. Captures
    every (state, questions) it answers. Same identity as MAIN's emulator, so `key()` computes MAIN's keys."""

    def __init__(self, programs: Iterable[str], symbols: Iterable[str] | None = None, gen: Path = MAIN_DIR):
        from jevtrader.s1.types import request_key

        b = json.loads((gen / "MANIFEST.json").read_text())["backend"]
        self._b, self._rk = b, request_key
        self.backend_id, self.model, self.prompt_version = b["backend"], b["model_pinned"], b["prompt_version"]
        self.dump: Path | None = None
        self.cfg_hash, self.bank_hash = b["cfg_hash"], b["bank_hash"]
        self.ledger, self.cache, self.coalesced = None, None, 0
        programs = list(programs)
        ans = pd.read_parquet(gen / "answers.parquet", columns=["symbol", "program", "qid", "kind", "probs_json",
                                                                 "request_key"])
        ans = ans[ans.program.isin(programs)]
        if symbols is not None:
            ans = ans[ans.symbol.isin(set(symbols))]
        self.table: dict[str, dict[str, tuple[str, str | None]]] = {}
        for _, prog, qid, kind, pj, k in ans.itertuples(index=False):
            self.table.setdefault(k, {})[qid] = (kind, pj)
        self.captured: list[tuple[str, Any, dict, str]] = []
        self.missing = 0
        self.missing_states: list[tuple[str, Any, str]] = []

    def key(self, state, questions) -> str:
        return self._rk(self.backend_id, self.model, self.prompt_version, self.cfg_hash, self.bank_hash, state,
                        questions)

    def describe(self) -> dict:
        return {k: self._b[k] for k in ("backend", "model_pinned", "prompt_version", "cfg_hash", "bank_hash")} | {
            "identity": {"replay": "MAIN answers.parquet"}}

    def estimate_cost(self, state, questions) -> float:
        return 0.0

    async def ask(self, state, questions, tag: str = "", **kw) -> S1Response:
        from jevtrader.s1.types import Answer

        k = self.key(state, questions)
        row = self.table.get(k)
        if row is None or set(questions) - set(row):
            self.missing += 1
            self.missing_states.append((k, state, tag))
            if self.dump is not None:
                with open(self.dump, "a") as fh:
                    fh.write(json.dumps({"key": k, "tag": tag, "state": state}, ensure_ascii=False) + "\n")
            raise KeyError(f"MAIN never sent this {tag} request ({k[:12]})")
        answers = {n: Answer(row[n][0], json.loads(row[n][1]) if isinstance(row[n][1], str) else None)
                   for n in questions}
        self.captured.append((k, state, dict(questions), tag))
        return S1Response(answers=answers, model=self.model, backend=self.backend_id, response_model=self.model,
                          prompt_version=self.prompt_version, request_key=k, cached=True,
                          degraded=[n for n, a in answers.items() if a.probs is None])
