"""NG item-level sample: N0 and N1 on 500 seeded MAIN items with the open typed decision model (typed-local, $0 API).

design_v2_delta IF-1 / VC-13 / IF-4, reduced by the 2026-09-28 re-plan (user budget $250; NG-C2's 4,000 items and
NG-C1i are replaced by this 500-item sample). Purpose (computed after the W_clean unlock, not here): H2 concept
validity per implementation (VC-13: the item set is defined by emu's N1 answers; NG's own eligibility is reported) and
emulator-vs-NG item-level agreement on identical requests.

    .venv/bin/python experiments/ng/items500.py sample     # the pre-registered sample (no model, no outcome)
    .venv/bin/python experiments/ng/items500.py prepare    # rebuild MAIN's states, verify them against MAIN's keys
    .venv/bin/python experiments/ng/items500.py run        # ask NG (resumable; one checkpoint line per item)
    .venv/bin/python experiments/ng/items500.py finalize   # parquet tables + MANIFEST.json
    .venv/bin/python experiments/ng/items500.py verify [--first 20]   # checks + throughput / ETA
    experiments/ng/run_items500.sh                         # all four steps under nohup (the long job)

**Sample (pre-registered; `sample`).** Frame: every MAIN (`results/judgments/main_emu_wclean_u100`) N01 item row whose
decision date t lies in W_clean (2024-07-05..2026-09-11; MAIN's pseudo-forward week 2026-09-18 is outside) and that is
C2-eligible by emu's own answers (design_v1 §13.1): the item was kept for events (P(about_focal) >= .5,
P(new_development) >= .5, not quarantined, attributes answered: `status.kept`) and its top `event_type` (argmax of
emu's distribution; ties to the first label in bank order) is not `stock_activity_or_opinion`. Unit = (t, focal
symbol, evidence id), one row per (item, focal) as C2. Strata = year of t x emu's top event type. Allocation:
proportional to stratum size, Hamilton largest remainder (ties: stratum name). Within a stratum the units with the
smallest rank are taken, rank = sha256("NG-C2-www27|<t>|<symbol>|<eid>") (the design's NG-C2 seed name). The rule
reads Layer-A judgments only: no return, label, IC or learned coefficient. The sample's SHA-256 (over the sorted unit
ids) goes to sample.json and the report, for the pre-registration.

**States (`prepare`).** Rebuilt exactly as MAIN built them: MAIN's context builder (`ContextBuilder(RealData(),
subset=None)`, fact symbols of MAIN's weeks, context id checked = MAIN's), `programs.news.ItemSource`, `n0_request`,
and N1's item state from MAIN's N0 decision (`kept_sentences`; `n1_item_state`), so NG and emu read the same text.
Every rebuilt request is checked against MAIN: the emu request key (which hashes the state and the questions in render
order) must equal MAIN's key for N0 and for each of the five N1 requests, `item_hash` must equal MAIN's `n1_sha256`,
and the archived text hash must equal MAIN's evidence row. Any mismatch stops the step.

**Requests (`run`).** Per item: the N0 request, and the frozen bank's N1 request plan (v1.0 `code_constants
["N1.requests"]`, `programs.news.n1_plan`): stage 1 about_focal, instructs_reader, promotional (each alone); stage 2
new_development alone and the rc5 13-question request read for the nine attributes (its four gate answers are
relabelled `N1-joint`, as in production). Every sampled item passed emu's stage-2 gate, so these are exactly the six
requests MAIN sent for the item. Stage 2 is asked for every item (the H2 item set is emu's); NG's own gate
(`stage2_gate` on NG's stage-1 answers), NG's triage and NG's C2 eligibility are recorded per item. NG's N0 answers
are recorded but do not change the N1 state (identical states across implementations). No retries; a worker error or
timeout abstains (probs None) and is counted (`typed_local.TypedDecisionLocal`). The backend is `typed-local`
(prompt ng-v1, `flock-io/this-that-model-1.0` at the pinned revision, 6 threads, float32; configs/backends.yaml): no
ledger, no API key, no network (the worker loads the local snapshot offline).

Resume: `run` appends one JSON line per finished item to `checkpoint.jsonl` (fsync) and skips finished units on
restart; items run in rank order, so any prefix is a random subsample of the stratified sample. `progress.json` holds
the running throughput and ETA.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.programs import news as N  # noqa: E402
from jevtrader.programs.bank_v1 import FROZEN_SHA256, load_bank  # noqa: E402
from jevtrader.programs.base import Request, ask  # noqa: E402
from jevtrader.s1.types import canonical_json, labels, request_key, state_sha256  # noqa: E402

log = logging.getLogger("ng_items500")
MAIN_GEN = "main_emu_wclean_u100"
MAIN_DIR = ROOT / "results" / "judgments" / MAIN_GEN
GEN_ID = "ng_typed-local_wclean_items500"
GEN_DIR = ROOT / "results" / "judgments" / GEN_ID
SEED = "NG-C2-www27"
N_ITEMS = 500
W_CLEAN = ("2024-07-05", "2026-09-11")
EXCLUDED_TYPE = "stock_activity_or_opinion"
RUN_TAG = "NG"
N1_KEYS = ("N1.about_focal", "N1.instructs_reader", "N1.promotional", "N1.new_development", "N1.attributes")
TABLE_COLUMNS = {
    "answers": ["t", "symbol", "program", "stage", "qid", "template", "kind", "probs_json", "request_key"],
    "requests": ["request_key", "t", "symbol", "program", "stage", "backend", "model_pinned", "response_model",
                 "prompt_version", "bank_hash", "data_freeze", "input_tokens", "cached_tokens", "output_tokens", "usd",
                 "latency_s", "cached", "degraded", "attempts", "usage_estimated", "state_sha256", "tag",
                 "n_questions", "lint_warnings", "backend_meta"],
    "evidence": ["request_key", "eid", "available_at_utc", "kept", "quarantined", "text_sha256", "d0_flags"],
}


def unit_id(t: str, symbol: str, eid: str) -> str:
    return f"{t}|{symbol}|{eid}"


def rank_of(uid: str) -> str:
    return hashlib.sha256(f"{SEED}|{uid}".encode()).hexdigest()


def top_label(probs: dict | None) -> str | None:
    return None if not probs else max(probs, key=probs.get)  # max keeps the first maximal key: bank order


def fsync_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def main_manifest() -> dict:
    return json.loads((MAIN_DIR / "MANIFEST.json").read_text())


def main_item_rows() -> dict[str, dict]:
    """MAIN's N01 item rows (M-work blackboard `N01.items.<symbol>`), keyed by unit id, for weeks in W_clean."""
    out = {}
    with open(MAIN_DIR / "memory" / "work.jsonl") as f:
        for line in f:
            rec = json.loads(line)["record"]
            if not rec["key"].startswith(N.ITEMS_KEY):
                continue
            t = rec["t"][:10]
            if not (W_CLEAN[0] <= t <= W_CLEAN[1]):
                continue
            sym = rec["key"][len(N.ITEMS_KEY):]
            for r in rec["value"]:
                uid = unit_id(t, sym, r["eid"])
                if uid in out:
                    raise RuntimeError(f"duplicate MAIN item row {uid}")
                out[uid] = {"t": t, "symbol": sym, **r}
    return out


# ------------------------------------------------------------------------------------------ sample
def hamilton(sizes: dict[str, int], n: int) -> dict[str, int]:
    tot = sum(sizes.values())
    quota = {k: n * v / tot for k, v in sizes.items()}
    alloc = {k: math.floor(q) for k, q in quota.items()}
    left = n - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k))[:left]:
        alloc[k] += 1
    return alloc


def cmd_sample(a) -> int:
    rows = main_item_rows()
    frame = []
    for uid, r in rows.items():
        top = top_label(r.get("event_type"))
        if r["status"]["kept"] and top is not None and top != EXCLUDED_TYPE:
            frame.append({"unit": uid, "t": r["t"], "symbol": r["symbol"], "eid": r["eid"], "year": r["t"][:4],
                          "event_type": top, "stratum": f"{r['t'][:4]}|{top}", "rank": rank_of(uid)})
    fr = pd.DataFrame(frame)
    sizes = fr.groupby("stratum").size().to_dict()
    alloc = hamilton(sizes, N_ITEMS)
    parts = [g.sort_values("rank").head(alloc[s]) for s, g in fr.groupby("stratum")]
    smp = pd.concat(parts).sort_values("rank").reset_index(drop=True)
    smp["order"] = range(len(smp))
    sha = hashlib.sha256("\n".join(sorted(smp["unit"])).encode()).hexdigest()
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    smp.to_parquet(GEN_DIR / "sample.parquet", index=False)
    rec = {
        "sample": "NG item-level sample (design_v2_delta IF-1/VC-13; 2026-09-28 re-plan)", "n": len(smp),
        "seed": SEED, "rank": "sha256('NG-C2-www27|<t>|<symbol>|<eid>'), smallest first",
        "frame": {"source": f"results/judgments/{MAIN_GEN}/memory/work.jsonl (N01.items rows)",
                  "window": list(W_CLEAN), "rule": "status.kept and top event_type != stock_activity_or_opinion "
                  "(emu answers; design_v1 13.1 C2 items)", "items_in_window": len(rows), "eligible": len(fr)},
        "strata": "year of t x emu top event_type", "allocation": "proportional, Hamilton largest remainder",
        "cells": {s: {"frame": int(sizes[s]), "sample": int(alloc[s])} for s in sorted(sizes)},
        "by_year": smp.groupby("year").size().to_dict(), "by_event_type": smp.groupby("event_type").size().to_dict(),
        "weeks": int(smp["t"].nunique()), "symbols": int(smp["symbol"].nunique()),
        "sample_sha256": sha, "main_manifest_created_utc": main_manifest()["created_utc"],
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    old = GEN_DIR / "sample.json"
    if old.exists():
        prev = json.loads(old.read_text())
        if prev["sample_sha256"] != sha:
            raise SystemExit(f"sample.json exists with another sample ({prev['sample_sha256'][:12]} != {sha[:12]})")
        rec["created_utc"] = prev["created_utc"]
    fsync_write(old, json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("n", "frame", "by_year", "by_event_type", "weeks", "symbols",
                                          "sample_sha256")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------ prepare
def emu_key(emu: dict, state, questions) -> str:
    return request_key("llm-logprob", emu["model_pinned"], emu["prompt_version"], emu["cfg_hash"], emu["bank_hash"],
                       state, questions)


def n0_questions(bank, n: int) -> list:
    specs = bank.program("N0")
    qs = []
    for i in range(n):
        for tmpl in ("reports_i", "judges_i"):
            name, q = specs[tmpl].expand(index=i)
            qs.append((name, specs[tmpl], q))
    return qs


def n1_questions(bank, names) -> list:
    specs = bank.program("N1")
    order = list(specs)
    return [(n, specs[n], specs[n].to_question()) for n in sorted(names, key=order.index)]


def cmd_prepare(a) -> int:
    from jevtrader.programs.context import ContextBuilder, RealData

    smp = pd.read_parquet(GEN_DIR / "sample.parquet")
    bank = load_bank(run_tag="MAIN")  # refuses anything but the frozen v1.0 bank
    man = main_manifest()
    emu = man["backend"]
    if bank.sha256 != emu["bank_hash"] or bank.sha256 != FROZEN_SHA256:
        raise SystemExit("bank is not MAIN's frozen v1.0 bank")
    rows = main_item_rows()
    dev = json.loads((MAIN_DIR / "dev_run.json").read_text())
    data = RealData()
    data.fact_symbols = data.symbols_for_weeks([pd.Timestamp(w) for w in dev["weeks"]])
    builder = ContextBuilder(data, subset=None)
    ctx_id = hashlib.sha256(canonical_json(builder.describe()).encode()).hexdigest()[:10]
    if ctx_id != dev["context_id"] or data.archive_id() != man["spec"]["data_freeze"]:
        raise SystemExit(f"context {ctx_id} / archive {data.archive_id()} differ from MAIN's "
                         f"{dev['context_id']} / {man['spec']['data_freeze']}")
    plan = N.n1_plan(bank)
    ev = pd.read_parquet(MAIN_DIR / "evidence.parquet", columns=["request_key", "eid", "text_sha256"])
    n0keys = {rows[u]["request_keys"]["N0"] for u in smp["unit"]}
    ev = ev[ev["request_key"].isin(n0keys)]
    ev_sha = {(k, e): s for k, e, s in ev.itertuples(index=False)}
    sdir = GEN_DIR / "states"
    sdir.mkdir(exist_ok=True)
    t_all = time.time()
    for week, g in smp.groupby("t", sort=True):
        path = sdir / f"{week}.jsonl"
        if path.exists() and sum(1 for _ in open(path)) == len(g):
            continue
        t0 = time.time()
        wt = pd.Timestamp(week)
        ctx = builder(wt)
        src = N.ItemSource(ctx, bank)
        lines = []
        for u in g.itertuples(index=False):
            main = rows[u.unit]
            items = {it.eid: it for it in src.items(u.symbol)}
            it = items.get(u.eid)
            if it is None:
                raise SystemExit(f"{u.unit}: item not rebuilt")
            fc = N.focal(bank, ctx, u.symbol)
            r0 = N.n0_request(bank, it, wt)
            k0 = emu_key(emu, r0.state, r0.questions)
            keep = [i in set(main["kept_sentences"]) for i in range(len(it.n0_sentences))]
            n1 = N.n1_item_state(it, keep)
            checks = {"N0_key": k0 == main["request_keys"]["N0"],
                      "n_sentences": len(it.n0_sentences) == main["n_sentences"],
                      "n1_sha256": N.item_hash(n1) == main["n1_sha256"],
                      "text_sha256": ev_sha.get((main["request_keys"]["N0"], it.eid)) == it.raw_sha256}
            n1_state = None
            for grp in plan.groups():
                rq = N.n1_group_request(bank, it, fc, n1, wt, grp.ask)
                n1_state = rq.state if n1_state is None else n1_state
                checks[f"N1.{grp.name}_key"] = (rq.state == n1_state and
                                                emu_key(emu, rq.state, rq.questions) == main["request_keys"].get(
                                                    f"N1.{grp.name}"))
            if not all(checks.values()):
                raise SystemExit(f"{u.unit}: rebuilt state differs from MAIN's: {checks}")
            evd = it.evidence()
            lines.append({
                "unit": u.unit, "order": int(u.order), "rank": u.rank, "t": week, "symbol": u.symbol, "eid": u.eid,
                "stratum": u.stratum, "n_sentences": len(it.n0_sentences), "n0_state": r0.state, "n1_state": n1_state,
                "evidence": {"eid": evd.eid, "available_at_utc": pd.Timestamp(evd.available_at_utc).isoformat(),
                             "d0_flags": evd.d0_flags, "raw_sha256": evd.raw_sha256},
                "emu_keys": {k: main["request_keys"][k] for k in ("N0", *N1_KEYS)},
                "n0_state_sha256": state_sha256(r0.state), "n1_state_sha256": state_sha256(n1_state),
                "checks": checks, "main_item": main,
            })
        fsync_write(path, "".join(json.dumps(x) + "\n" for x in lines))
        print(f"prepare {week}: {len(lines)} item(s) verified against MAIN keys, {time.time() - t0:.1f} s", flush=True)
    n = sum(sum(1 for _ in open(p)) for p in sdir.glob("*.jsonl"))
    print(f"prepare done: {n} of {len(smp)} states, {time.time() - t_all:.0f} s", flush=True)
    return 0 if n == len(smp) else 2


def load_states() -> list[dict]:
    out = []
    for p in sorted((GEN_DIR / "states").glob("*.jsonl")):
        out += [json.loads(x) for x in open(p)]
    return sorted(out, key=lambda x: x["order"])


# ------------------------------------------------------------------------------------------ run
def iso(x) -> str | None:
    return None if x is None else pd.Timestamp(x).isoformat()


def asked_rows(a) -> dict:
    ans = []
    for r in a.trace:
        d = dataclasses.asdict(r)
        d["t"] = iso(d["t"])
        d["probs_json"] = None if d.pop("probs") is None else json.dumps(r.probs, sort_keys=True)
        ans.append(d)
    req = dataclasses.asdict(a.request)
    req["t"] = iso(req["t"])
    evs = []
    for e in a.evidence:
        d = dataclasses.asdict(e)
        d["available_at_utc"] = iso(d["available_at_utc"])
        evs.append(d)
    return {"answers": ans, "request": req, "evidence": evs}


def pval(a) -> float | None:
    return None if a is None or a.probs is None else float(a.probs["yes"])


def dist(a) -> dict | None:
    return None if a is None or a.probs is None else {k: float(v) for k, v in a.probs.items()}


async def run_item(s1, bank, plan, emu: dict, st: dict) -> dict:
    from jevtrader.programs.news import NewsEvidence

    t = pd.Timestamp(st["t"])
    e = st["evidence"]
    evid = NewsEvidence(eid=e["eid"], available_at_utc=pd.Timestamp(e["available_at_utc"]), kept=True,
                        quarantined=False, text=None, d0_flags=e["d0_flags"], raw_sha256=e["raw_sha256"])
    t0 = time.perf_counter()
    reqs = [("N0", None, Request.build("N0", "N0", st["symbol"], t, st["n0_state"],
                                       n0_questions(bank, st["n_sentences"]), [evid], tag="N0"))]
    for g in plan.groups():
        reqs.append((f"N1.{g.name}", g, Request.build("N1", "N1", st["symbol"], t, st["n1_state"],
                                                      n1_questions(bank, g.ask), [evid], tag="N1")))
    out, keys, answers, stage1, n0_ans = [], {}, {}, {}, None
    for name, g, rq in reqs:
        if emu_key(emu, rq.state, rq.questions) != st["emu_keys"][name]:  # same (state, questions) as MAIN sent
            raise RuntimeError(f"{st['unit']} {name}: request differs from MAIN's")
        a = await ask(s1, rq)
        if g is None:
            n0_ans = a.response.answers
        else:
            got = {q: v for q, v in a.response.answers.items() if q in g.use}
            answers.update(got)
            if g in plan.stage1:
                stage1.update(got)
            a = N._relabel_unread(a, g)
        keys[name] = a.request.request_key
        out.append((name, a))
    keep = N.n0_keep(n0_ans, st["n_sentences"])
    tri = N.triage(answers, n0_quarantined=False)
    ev_type = top_label(dist(answers.get("event_type")))
    for _, a in out:  # the item's status under NG's own answers (production sets it after triage)
        for r in a.evidence:
            r.kept, r.quarantined = tri.kept, tri.quarantined
    meta = [json.loads(a.request.backend_meta or "{}") for _, a in out]
    item = {
        "unit": st["unit"], "order": st["order"], "t": st["t"], "symbol": st["symbol"], "eid": st["eid"],
        "stratum": st["stratum"], "ng_keys": keys, "emu_keys": st["emu_keys"],
        "ng_n0_keep": keep, "ng_n0_quarantined": not any(keep),
        "ng_gate": N.stage2_gate(stage1), "ng_status": dataclasses.asdict(tri),
        "ng_c2_eligible": bool(tri.kept and ev_type is not None and ev_type != EXCLUDED_TYPE),
        "ng_c2_eligible_own_n0": bool(tri.kept and any(keep) and ev_type is not None and ev_type != EXCLUDED_TYPE),
        "ng_p": {k: pval(answers.get(q)) for k, q in (
            ("about", "about_focal"), ("new", "new_development"), ("instructs", "instructs_reader"),
            ("promotional", "promotional"), ("important", "important"), ("scheduled", "scheduled"),
            ("unconfirmed", "unconfirmed"), ("serious", "serious_problem"), ("unresolved", "problem_unresolved"))},
        "ng_event_type": dist(answers.get("event_type")), "ng_effect": dist(answers.get("effect")),
        "ng_vs_expectations": dist(answers.get("vs_expectations")),
        "ng_problem_kind": dist(answers.get("problem_kind")),
        "n_requests": len(out), "n_abstained": sum(1 for _, a in out if a.request.degraded),
        "worker_s": round(sum(float(m.get("wall_s") or 0) for m in meta), 3),
        "item_s": round(time.perf_counter() - t0, 3), "maxrss_gb": max(float(m.get("maxrss_gb") or 0) for m in meta),
        "finished_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    return {"item": item, "asked": {name: asked_rows(a) for name, a in out}}


def read_checkpoint(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in open(path):
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a line cut by a kill mid-write; the item reruns
            done[rec["item"]["unit"]] = rec
    return done


async def run_all(a, states: list[dict]) -> int:
    from jevtrader.s1.backends import make_backend

    bank = load_bank()
    if bank.sha256 != FROZEN_SHA256:
        raise SystemExit("bank is not the frozen v1.0 bank")
    emu = main_manifest()["backend"]
    plan = N.n1_plan(bank)
    ck = GEN_DIR / "checkpoint.jsonl"
    done = read_checkpoint(ck)
    if done:  # rewrite without any cut line
        fsync_write(ck, "".join(json.dumps(done[s["unit"]]) + "\n" for s in states if s["unit"] in done))
    todo = [s for s in states if s["unit"] not in done]
    if a.limit:
        todo = todo[: max(0, a.limit - len(done))]
    print(f"run: {len(done)} done, {len(todo)} to do (of {len(states)}); mock={a.mock}", flush=True)
    if not todo:
        return 0
    s1 = make_backend(backend="typed-local", cache=False, ledger=False, mock=a.mock,
                      log_path=str(GEN_DIR / "worker.log"))
    if s1.backend_id != "typed-local" or s1.ledger is not None:
        raise SystemExit("not the typed-local backend, or a ledger is attached")
    from jevtrader.s1.typed_local import NG_REPO, NG_REVISION

    started, recent, held = time.time(), [], []
    prog = GEN_DIR / "progress.json"

    def write(f, rec) -> None:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())

    try:
        with open(ck, "a") as f:
            for k, st in enumerate(todo, 1):
                rec = await run_item(s1, bank, plan, emu, st)
                info = getattr(s1._transport, "info", {}) or {}
                if not (info.get("ready") and info.get("model") == NG_REPO and info.get("revision") == NG_REVISION
                        and bool(info.get("mock")) == a.mock):
                    raise SystemExit(f"worker is not the pinned model (ready record {info}); nothing written for "
                                     f"{st['unit']}")
                # an abstention (worker error or timeout) is recorded and counted, as in production; but a streak
                # means the worker is broken: the streak's items are not written (they rerun on resume) and the run
                # stops instead of recording 500 abstentions
                if rec["item"]["n_abstained"]:
                    held.append(rec)
                    if len(held) >= 3:
                        raise SystemExit(f"{len(held)} consecutive items with abstaining requests (worker failures "
                                         f"{s1.failures}); stopped, see {GEN_DIR / 'worker.log'}")
                    continue
                for h in held:  # isolated abstentions before a clean item: recorded
                    write(f, h)
                held = []
                write(f, rec)
                it = rec["item"]
                recent = (recent + [it["item_s"]])[-25:]
                n_done = len(done) + k
                per = sum(recent) / len(recent)
                left = len(states) - n_done
                eta = pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=per * left)
                print(f"[{n_done}/{len(states)}] {it['unit']} {it['item_s']:.1f} s (worker {it['worker_s']:.1f} s; "
                      f"abstained {it['n_abstained']}; ng_gate {it['ng_gate']}) | {per:.1f} s/item (last "
                      f"{len(recent)}) | ETA {eta.strftime('%Y-%m-%d %H:%M')} UTC", flush=True)
                fsync_write(prog, json.dumps({
                    "done": n_done, "total": len(states), "s_per_item_recent": round(per, 2),
                    "s_per_item_session": round((time.time() - started) / k, 2), "eta_utc": eta.isoformat(),
                    "pid": os.getpid(), "session_started_utc": pd.Timestamp(started, unit="s", tz="UTC").isoformat(),
                    "updated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "mock": a.mock,
                    "worker": getattr(s1._transport, "info", {}), "backend_failures": s1.failures}, indent=1))
            for h in held:  # fewer than 3 abstaining items at the very end: recorded
                write(f, h)
    finally:
        info = dict(getattr(s1._transport, "info", {}) or {})
        restarts = getattr(s1._transport, "restarts", 0)
        s1.close()
        (GEN_DIR / "backend.json").write_text(json.dumps({**s1.describe(), "worker": info, "restarts": restarts},
                                                         indent=1, default=str))
    return 0


def cmd_run(a) -> int:
    states = load_states()
    n = len(pd.read_parquet(GEN_DIR / "sample.parquet"))
    if len(states) != n:
        raise SystemExit(f"{len(states)} prepared states for {n} sampled items: run `prepare` first")
    return asyncio.run(run_all(a, states))


# ------------------------------------------------------------------------------------------ finalize / verify
def cmd_finalize(a) -> int:
    from jevtrader.s1.backends import make_backend

    states = load_states()
    done = read_checkpoint(GEN_DIR / "checkpoint.jsonl")
    man = main_manifest()
    rows = {k: [] for k in TABLE_COLUMNS}
    items = []
    for st in states:
        rec = done.get(st["unit"])
        if rec is None:
            continue
        items.append(rec["item"])
        for x in rec["asked"].values():
            rows["answers"] += x["answers"]
            rows["requests"].append({**x["request"], "tag": f"{RUN_TAG}/{x['request']['tag']}",
                                     "data_freeze": man["spec"]["data_freeze"]})
            rows["evidence"] += x["evidence"]
    tables = {}
    for k, cols in TABLE_COLUMNS.items():
        df = pd.DataFrame(rows[k], columns=cols)
        if "t" in df and len(df):
            df["t"] = pd.to_datetime(df["t"]).astype("datetime64[ms]")  # MAIN's unit
        if k == "evidence" and len(df):
            df["available_at_utc"] = pd.to_datetime(df["available_at_utc"], utc=True)
        df.to_parquet(GEN_DIR / f"{k}.parquet", index=False)
        tables[k] = df
    it = pd.DataFrame(items)
    for c in it.columns:
        if it[c].map(lambda v: isinstance(v, (dict, list))).any():
            it[c] = it[c].map(lambda v: json.dumps(v, sort_keys=True))
    it.to_parquet(GEN_DIR / "items.parquet", index=False)
    req, ans = tables["requests"], tables["answers"]
    s1 = make_backend(backend="typed-local", cache=False, ledger=False, mock=a.mock)  # identity only; no worker
    bj = GEN_DIR / "backend.json"
    backend = json.loads(bj.read_text()) if bj.exists() else s1.describe()
    s1.close()
    meta = [json.loads(m or "{}") for m in req["backend_meta"]] if len(req) else []
    walls = pd.Series([m.get("wall_s") for m in meta], dtype=float)
    smp = json.loads((GEN_DIR / "sample.json").read_text())
    m = {
        "gen_id": GEN_ID, "run_tag": RUN_TAG, "complete": len(items) == len(states),
        "spec": {"script": "experiments/ng/items500.py", "sample": smp["sample"], "seed": SEED,
                 "sample_sha256": smp["sample_sha256"], "from_gen": MAIN_GEN, "window": list(W_CLEAN),
                 "programs": ["N0", "N1"], "n1_plan": "bank v1.0 code_constants['N1.requests'], every group asked",
                 "data_freeze": man["spec"]["data_freeze"], "context_id": json.loads(
                     (MAIN_DIR / "dev_run.json").read_text())["context_id"]},
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "bank": {"version": load_bank().version, "sha256": FROZEN_SHA256},
        "backend": backend,
        "counts": {"items": len(items), "items_planned": len(states), "requests": len(req), "answers": len(ans),
                   "answers_degraded": int(ans["probs_json"].isna().sum()) if len(ans) else 0,
                   "requests_abstained": int((req["degraded"] > 0).sum()) if len(req) else 0,
                   "evidence": len(tables["evidence"]),
                   "by_program": req.groupby("program").size().to_dict() if len(req) else {}},
        "validity_rate": (1 - ans["probs_json"].isna().mean()) if len(ans) else None,
        "spend_usd": 0.0,
        "worker_s": {"total": float(walls.sum()), "median_per_request": float(walls.median()) if len(walls) else None},
        "response_models": req["response_model"].value_counts().to_dict() if len(req) else {},
    }
    (GEN_DIR / "MANIFEST.json").write_text(json.dumps(m, indent=1, default=str))
    print(json.dumps({k: m[k] for k in ("complete", "counts", "validity_rate", "worker_s", "response_models")},
                     indent=1, default=str))
    return 0


def cmd_verify(a) -> int:
    """Checks on the first `--first` finished items (in run order) and the throughput / ETA of all finished ones."""
    states = load_states()
    done = read_checkpoint(GEN_DIR / "checkpoint.jsonl")
    order = [s for s in states if s["unit"] in done]
    first = order[: a.first] if a.first else order
    bank = load_bank()
    n1 = bank.program("N1")
    pinned = "flock-io/this-that-model-1.0@3d927195c4f9845efe66c5715883a7a0f42b1239"
    fails = Counter()
    per_prog = defaultdict(list)
    for st in first:
        rec = done[st["unit"]]
        it, asked = rec["item"], rec["asked"]
        if set(asked) != {"N0", *N1_KEYS}:
            fails["request set"] += 1
        for name, x in asked.items():
            r = x["request"]
            meta = json.loads(r["backend_meta"] or "{}")
            per_prog["N0" if name == "N0" else ("N1-joint(13q)" if name == "N1.attributes" else "N1-single")].append(
                float(meta.get("wall_s") or 0))
            fails["backend"] += r["backend"] != "typed-local"
            fails["prompt_version"] += r["prompt_version"] != "ng-v1"
            fails["response_model != pin"] += r["response_model"] != pinned
            fails["model_pinned != pin"] += r["model_pinned"] != pinned
            fails["bank_hash"] += r["bank_hash"] != FROZEN_SHA256
            fails["usd != 0"] += r["usd"] != 0
            fails["abstained"] += r["degraded"] > 0
            fails["truncated"] += bool(meta.get("truncated"))
            fails["threads != 6"] += meta.get("threads") != 6 and not a.mock
            want_state = st["n0_state"] if name == "N0" else st["n1_state"]
            fails["state hash"] += r["state_sha256"] != state_sha256(want_state)
            want_n = 2 * st["n_sentences"] if name == "N0" else (13 if name == "N1.attributes" else 1)
            fails["n_questions"] += r["n_questions"] != want_n or len(x["answers"]) != want_n
            for row in x["answers"]:
                p = None if row["probs_json"] is None else json.loads(row["probs_json"])
                if p is None or abs(sum(p.values()) - 1) > 1e-6 or min(p.values()) < 0:
                    fails["invalid distribution"] += 1
                    continue
                if row["program"] == "N1":  # a distribution over exactly the declared options
                    fails["labels"] += set(p) != set(labels(n1[row["qid"]].to_question()))
            if name == "N1.attributes":
                relab = [r2 for r2 in x["answers"] if r2["program"] == N.N1_JOINT_LABEL]
                fails["N1-joint relabel"] += len(relab) != 4
    items = [done[s["unit"]]["item"] for s in order]
    secs = pd.Series([i["item_s"] for i in items], dtype=float)
    rest = len(states) - len(items)
    per = float(secs.tail(25).mean()) if len(secs) else float("nan")
    out = {
        "checked_items": len(first), "finished_items": len(items), "of": len(states),
        "failures": {k: int(v) for k, v in fails.items() if v},
        "worker_s_median": {k: round(float(pd.Series(v).median()), 2) for k, v in per_prog.items()},
        "worker_s_p90": {k: round(float(pd.Series(v).quantile(0.9)), 2) for k, v in per_prog.items()},
        "requests_checked": {k: len(v) for k, v in per_prog.items()},
        "item_s": {"mean": round(float(secs.mean()), 2), "median": round(float(secs.median()), 2),
                   "recent25_mean": round(per, 2)} if len(secs) else None,
        "items_per_hour": round(3600 / per, 1) if len(secs) else None,
        "eta_hours": round(rest * per / 3600, 2) if len(secs) else None,
        "eta_utc": (pd.Timestamp.now(tz="UTC") + pd.Timedelta(seconds=rest * per)).isoformat() if len(secs) else None,
        "ng_gate_pass_rate": round(sum(i["ng_gate"] for i in items) / len(items), 3) if items else None,
        "maxrss_gb": max((i["maxrss_gb"] for i in items), default=None),
    }
    print(json.dumps(out, indent=1))
    return 0 if not out["failures"] else 2


def main(argv=None) -> int:
    global GEN_DIR
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["sample", "prepare", "run", "finalize", "verify"])
    ap.add_argument("--gen-dir", default=None, help="output directory (default results/judgments/<gen_id>)")
    ap.add_argument("--mock", action="store_true", help="typed-local mock worker (pipeline test; not a result)")
    ap.add_argument("--limit", type=int, default=None, help="run: stop after this many finished items in total")
    ap.add_argument("--first", type=int, default=20, help="verify: check the first N finished items")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if a.gen_dir:
        GEN_DIR = Path(a.gen_dir).resolve()
    if a.mock and GEN_DIR == ROOT / "results" / "judgments" / GEN_ID and a.cmd in ("run", "finalize"):
        raise SystemExit("--mock never writes the real generation: give --gen-dir")
    return {"sample": cmd_sample, "prepare": cmd_prepare, "run": cmd_run, "finalize": cmd_finalize,
            "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
