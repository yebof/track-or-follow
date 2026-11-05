"""VERB (X5; design_v2_delta IF-2, VC-11, VC-13; preregistration_v1 DRAFT §3.2, §6): the verbalized backend on the
pre-registered aligned 2,000-item N1 sample. Raw answers only.

    .venv/bin/python experiments/run_gen_verb.py sample     # the 2,000 units (no model, no outcome)
    .venv/bin/python experiments/run_gen_verb.py prepare    # MAIN's N1 states, verified against MAIN's keys
    .venv/bin/python experiments/run_gen_verb.py dry        # fake transport projection ($0)
    .venv/bin/python experiments/run_gen_verb.py run        # preflight, then paid (tag VERB/N1)
    .venv/bin/python experiments/run_gen_verb.py validate

**Sample** (draft §3.2 "VERB"): the NG 500 units of `ng_typed-local_wclean_items500` (sample SHA-256 0c6c3da2...)
plus 1,500 more drawn by the same rule: frame = MAIN's N01 item rows in W_clean that are C2-eligible by emu's answers
(`status.kept` and top event_type != stock_activity_or_opinion), strata = year x emu top event type, extra per cell
= max(0, Hamilton allocation of 2,000 over the frame - NG units already in the cell), taken by the smallest rank
sha256("NG-C2-www27|<t>|<symbol>|<eid>") among the cell's units not in NG (`experiments/ng/items500.py` helpers).

**States** (`prepare`): NG's verified states for its 500 units; the 1,500 others rebuilt exactly as MAIN built them
(MAIN's context builder, `ItemSource`, N1 item state from MAIN's N0 decision) and checked: emu request keys of the
five N1 requests = MAIN's, `item_hash` = MAIN's `n1_sha256`, archived text hash = MAIN's evidence row.

**Requests** (`run`): per item the frozen bank's five N1 requests (v1.0 plan: stage 1 about_focal, instructs_reader,
promotional alone; stage 2 new_development alone and the 13-question attributes request), exactly MAIN's (state,
questions), sent to `llm-verbalized` / verb-v1 (pinned gpt-4.1-mini-2025-04-14; JSON probabilities; degraded ->
2 retries -> abstain; provider prompt-cache routing off, P-6). Stage 2 is asked for every item (the H2 item set is
emu's, VC-13). `--attributes-only` sends only the attributes request (the reduction used if the five requests do not
fit the sub-cap; recorded in MANIFEST). Items run NG's 500 first (NG order), then the rest by rank, in shards of 100
items, so a budget stop leaves the NG-aligned items complete first.

Output: results/judgments/verb_llm-verbalized_wclean_items2000/. No W_clean outcome is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gen_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "experiments" / "ng"))
import items500 as NG  # noqa: E402

from jevtrader.programs import news as N  # noqa: E402
from jevtrader.programs.bank_v1 import FROZEN_SHA256, load_bank  # noqa: E402
from jevtrader.programs.base import Request  # noqa: E402
from jevtrader.s1.types import canonical_json, state_sha256  # noqa: E402

log = logging.getLogger("run_gen_verb")
ROOT = C.ROOT
RUN = "VERB"
GEN_ID = "verb_llm-verbalized_wclean_items2000"
OUT = ROOT / "results" / "judgments" / GEN_ID
N_ITEMS = 2000
SHARD = 100
SCRATCH = Path("/tmp/jevtrader-scratch/paid")
GROUPS_ALL = ("about_focal", "instructs_reader", "promotional", "new_development", "attributes")
SINGLES = ("about_focal", "instructs_reader", "promotional", "new_development")
# shard prefix and projection file per request set (a launch asks one set; `finalize` merges every DONE shard)
SETS = {("attributes",): ("b", "projection.json"), SINGLES: ("s", "projection_singles.json"),
        GROUPS_ALL: ("g", "projection_all_groups.json")}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


_EMU: dict = {}


def main_emu() -> dict:
    if not _EMU:
        _EMU.update(NG.main_manifest()["backend"])
    return _EMU


# ------------------------------------------------------------------------------------------ sample
def cmd_sample(a) -> int:
    ng = pd.read_parquet(NG.GEN_DIR / "sample.parquet")
    ng_sha = hashlib.sha256("\n".join(sorted(ng["unit"])).encode()).hexdigest()
    if ng_sha != json.loads((NG.GEN_DIR / "sample.json").read_text())["sample_sha256"]:
        raise SystemExit("NG sample hash mismatch")
    rows = NG.main_item_rows()
    frame = []
    for uid, r in rows.items():
        top = NG.top_label(r.get("event_type"))
        if r["status"]["kept"] and top is not None and top != NG.EXCLUDED_TYPE:
            frame.append({"unit": uid, "t": r["t"], "symbol": r["symbol"], "eid": r["eid"], "year": r["t"][:4],
                          "event_type": top, "stratum": f"{r['t'][:4]}|{top}", "rank": NG.rank_of(uid)})
    fr = pd.DataFrame(frame)
    if len(fr) != json.loads((NG.GEN_DIR / "sample.json").read_text())["frame"]["eligible"]:
        raise SystemExit("frame size differs from NG's")
    if not set(ng["unit"]) <= set(fr["unit"]):
        raise SystemExit("NG units outside the frame")
    sizes = fr.groupby("stratum").size().to_dict()
    alloc = NG.hamilton(sizes, N_ITEMS)
    ng_cell = ng.groupby("stratum").size().to_dict()
    extra = {s: max(0, alloc[s] - ng_cell.get(s, 0)) for s in sizes}
    ngset = set(ng["unit"])
    parts = [g[~g.unit.isin(ngset)].sort_values("rank").head(extra[s]) for s, g in fr.groupby("stratum")]
    add = pd.concat(parts).sort_values("rank")
    ngp = fr[fr.unit.isin(ngset)].set_index("unit").loc[ng.sort_values("order")["unit"]].reset_index()
    smp = pd.concat([ngp.assign(in_ng=True), add.assign(in_ng=False)], ignore_index=True)
    smp["order"] = range(len(smp))
    uids = sorted(smp["unit"])
    rec = {"sample": "VERB aligned N1 sample (draft §3.2; IF-2 nested on NG)", "n": len(smp),
           "n_ng": int(smp.in_ng.sum()), "n_extra": int((~smp.in_ng).sum()), "seed": NG.SEED,
           "rule": "NG 500 + per cell max(0, Hamilton(2000) - NG cell count), smallest rank first",
           "cells": {s: {"frame": int(sizes[s]), "hamilton_2000": int(alloc[s]), "ng": int(ng_cell.get(s, 0)),
                         "extra": int(extra[s])} for s in sorted(sizes)},
           "cells_where_ng_exceeds_hamilton": sorted(s for s in sizes if ng_cell.get(s, 0) > alloc[s]),
           "by_year": smp.groupby("year").size().to_dict(), "weeks": int(smp.t.nunique()),
           "ng_sample_sha256": ng_sha, "sample_sha256": sha("\n".join(uids)),
           "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    OUT.mkdir(parents=True, exist_ok=True)
    old = OUT / "sample.json"
    if old.exists() and json.loads(old.read_text())["sample_sha256"] != rec["sample_sha256"]:
        raise SystemExit("sample.json exists with another sample")
    smp.to_parquet(OUT / "sample.parquet", index=False)
    old.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("n", "n_ng", "n_extra", "weeks", "sample_sha256",
                                          "cells_where_ng_exceeds_hamilton")}))
    return 0


# ------------------------------------------------------------------------------------------ prepare
def cmd_prepare(a) -> int:
    from jevtrader.programs.context import ContextBuilder, RealData

    smp = pd.read_parquet(OUT / "sample.parquet")
    bank = load_bank(run_tag=RUN)
    man = NG.main_manifest()
    emu = man["backend"]
    if bank.sha256 != emu["bank_hash"] or bank.sha256 != FROZEN_SHA256:
        raise SystemExit("bank is not MAIN's frozen v1.0 bank")
    sdir = OUT / "states"
    sdir.mkdir(exist_ok=True)
    # NG's verified states for its 500 units
    ngst = {}
    for p in sorted((NG.GEN_DIR / "states").glob("*.jsonl")):
        for line in open(p):
            x = json.loads(line)
            ngst[x["unit"]] = x
    rows = NG.main_item_rows()
    todo = smp[~smp.unit.isin(ngst)]
    have = {}
    for p in sdir.glob("*.jsonl"):
        for line in open(p):
            x = json.loads(line)
            have[x["unit"]] = x
    ng_path = sdir / "ng_units.jsonl"
    if not ng_path.exists():
        with open(ng_path, "w") as fh:
            for u in smp[smp.in_ng].unit:
                x = ngst[u]
                fh.write(json.dumps({k: x[k] for k in ("unit", "t", "symbol", "eid", "stratum", "n_sentences",
                                                        "n1_state", "evidence", "emu_keys", "n1_state_sha256",
                                                        "checks")} | {"source": "ng_states"}) + "\n")
    dev = json.loads((NG.MAIN_DIR / "dev_run.json").read_text())
    data = RealData()
    data.fact_symbols = data.symbols_for_weeks([pd.Timestamp(w) for w in dev["weeks"]])
    builder = ContextBuilder(data, subset=None)
    ctx_id = sha(canonical_json(builder.describe()))[:10]
    if ctx_id != dev["context_id"] or data.archive_id() != man["spec"]["data_freeze"]:
        raise SystemExit("context or archive differs from MAIN's")
    plan = N.n1_plan(bank)
    ev = pd.read_parquet(NG.MAIN_DIR / "evidence.parquet", columns=["request_key", "eid", "text_sha256"])
    n0keys = {rows[u]["request_keys"]["N0"] for u in todo["unit"]}
    ev = ev[ev["request_key"].isin(n0keys)]
    ev_sha = {(k, e): s for k, e, s in ev.itertuples(index=False)}
    t_all = time.time()
    for week, g in todo.groupby("t", sort=True):
        path = sdir / f"{week}.jsonl"
        if path.exists() and sum(1 for _ in open(path)) == len(g):
            continue
        wt = pd.Timestamp(week)
        ctx = builder(wt)
        src = N.ItemSource(ctx, bank)
        lines = []
        for u in g.itertuples(index=False):
            main = rows[u.unit]
            it = {i.eid: i for i in src.items(u.symbol)}.get(u.eid)
            if it is None:
                raise SystemExit(f"{u.unit}: item not rebuilt")
            fc = N.focal(bank, ctx, u.symbol)
            r0 = N.n0_request(bank, it, wt)
            keep = [i in set(main["kept_sentences"]) for i in range(len(it.n0_sentences))]
            n1 = N.n1_item_state(it, keep)
            checks = {"N0_key": NG.emu_key(emu, r0.state, r0.questions) == main["request_keys"]["N0"],
                      "n_sentences": len(it.n0_sentences) == main["n_sentences"],
                      "n1_sha256": N.item_hash(n1) == main["n1_sha256"],
                      "text_sha256": ev_sha.get((main["request_keys"]["N0"], it.eid)) == it.raw_sha256}
            n1_state = None
            for grp in plan.groups():
                rq = N.n1_group_request(bank, it, fc, n1, wt, grp.ask)
                n1_state = rq.state if n1_state is None else n1_state
                checks[f"N1.{grp.name}_key"] = (rq.state == n1_state and NG.emu_key(emu, rq.state, rq.questions)
                                                == main["request_keys"].get(f"N1.{grp.name}"))
            if not all(checks.values()):
                raise SystemExit(f"{u.unit}: rebuilt state differs from MAIN's: {checks}")
            e = it.evidence()
            lines.append({"unit": u.unit, "t": week, "symbol": u.symbol, "eid": u.eid, "stratum": u.stratum,
                          "n_sentences": len(it.n0_sentences), "n1_state": n1_state,
                          "evidence": {"eid": e.eid, "available_at_utc": pd.Timestamp(e.available_at_utc).isoformat(),
                                       "d0_flags": e.d0_flags, "raw_sha256": e.raw_sha256},
                          "emu_keys": {k: main["request_keys"][k] for k in ("N0", *NG.N1_KEYS)},
                          "n1_state_sha256": state_sha256(n1_state), "checks": checks, "source": "rebuilt"})
        NG.fsync_write(path, "".join(json.dumps(x) + "\n" for x in lines))
        print(f"prepare {week}: {len(lines)} item(s) verified ({time.time() - t_all:.0f}s)", flush=True)
    n = sum(sum(1 for _ in open(p)) for p in sdir.glob("*.jsonl"))
    print(f"prepare done: {n} of {len(smp)} states", flush=True)
    return 0 if n == len(smp) else 2


def load_states() -> dict[str, dict]:
    out = {}
    for p in sorted((OUT / "states").glob("*.jsonl")):
        for line in open(p):
            x = json.loads(line)
            out[x["unit"]] = x
    return out


# ------------------------------------------------------------------------------------------ run
def jobs_for(bank, st: dict, groups: tuple[str, ...]) -> list[C.Job]:
    from jevtrader.programs.news import NewsEvidence

    plan = N.n1_plan(bank)
    e = st["evidence"]
    evid = NewsEvidence(eid=e["eid"], available_at_utc=pd.Timestamp(e["available_at_utc"]), kept=True,
                        quarantined=False, text=None, d0_flags=e["d0_flags"], raw_sha256=e["raw_sha256"])
    t = pd.Timestamp(st["t"])
    out = []
    for g in plan.groups():
        if g.name not in groups:
            continue
        rq = Request.build("N1", f"N1.{g.name}", st["symbol"], t, st["n1_state"], NG.n1_questions(bank, g.ask),
                           [evid], tag="N1")
        if NG.emu_key(main_emu(), rq.state, rq.questions) != st["emu_keys"][f"N1.{g.name}"]:
            raise SystemExit(f"{st['unit']} {g.name}: not MAIN's request")
        out.append(C.Job(f"{st['unit']}|{g.name}", rq, {"unit": st["unit"], "group": g.name,
                                                        "use": list(g.use), "stratum": st["stratum"]}))
    return out


def build_shards(bank, groups) -> dict[str, list[C.Job]]:
    smp = pd.read_parquet(OUT / "sample.parquet").sort_values("order")
    states = load_states()
    if set(smp.unit) - set(states):
        raise SystemExit(f"{len(set(smp.unit) - set(states))} states missing; run prepare")
    prefix = SETS[tuple(groups)][0]
    sh = {}
    for i, u in enumerate(smp.unit):
        sh.setdefault(f"{prefix}{i // SHARD:02d}", []).extend(jobs_for(bank, states[u], groups))
    return sh


def done_shards(out: Path) -> list[str]:
    return sorted(d.name for d in (out / "shards").iterdir() if (d / "DONE.json").exists())


def done_groups(out: Path) -> list[str]:
    got = set()
    for name in done_shards(out):
        got |= set(json.loads((out / "shards" / name / "DONE.json").read_text())["fingerprint"]["groups"])
    return [g for g in GROUPS_ALL if g in got]


def run(dry: bool, groups: tuple[str, ...], concurrency: int) -> int:
    bank = load_bank(run_tag=RUN)
    smp_info = json.loads((OUT / "sample.json").read_text())
    fp = {"gen_id": GEN_ID, "bank": bank.sha256, "sample_sha256": smp_info["sample_sha256"], "groups": list(groups),
          "backend": "llm-verbalized/verb-v1"}
    shards = build_shards(bank, groups)
    n_req = sum(len(v) for v in shards.values())
    if dry:
        scratch = SCRATCH / f"{GEN_ID}_dry_{int(time.time())}"
        cal = calibration(bank)
        fake = C.FakeOpenAI(ratio=cal["ratio"])
        s1 = C.make_s1("llm-verbalized", True, scratch, fake)
        out = scratch / GEN_ID
        guard = float("inf")
    else:
        proj = json.loads((OUT / SETS[tuple(groups)][1]).read_text())
        if proj.get("groups") != list(groups):
            raise SystemExit(f"projection.json is for groups {proj.get('groups')}, not {list(groups)}")
        pf = C.preflight(RUN, proj["projected_usd"], proj["high_usd"], f"{len(shards)} shards, {n_req} requests, "
                                                                          f"groups {','.join(groups)}")
        if not pf.ok:
            return 3
        s1 = C.make_s1("llm-verbalized", False)
        if s1.prompt_cache.enabled:
            raise SystemExit("verb routing must be off (P-6)")
        out, guard = OUT, pf.guard
    g = C.guarded(s1, RUN, guard)
    t0 = time.time()
    st = C.run_shards(out, shards, g, RUN, "archive-a377a4d6abd3bfe3", fp, concurrency)
    s1.ledger.flush()
    info = {**st, "dry": dry, "groups": list(groups), "process_spent_usd": g.spent, "process_calls": g.calls,
            "refused": g.refused, "wall_s": round(time.time() - t0, 1)}
    C.launch_record(out, info)
    print(json.dumps(info), flush=True)
    if dry:
        rec = C.projection_record(OUT / SETS[tuple(groups)][1], RUN, fake, s1,
                                  {"requests": n_req, "groups": list(groups), "dry_dir": str(scratch),
                                   "calibration": cal})
        print(json.dumps({k: rec[k] for k in ("paid_calls", "projected_usd", "high_usd", "by_tag")}), flush=True)
    if st["status"] != "complete":
        return 3
    man = C.finalize(out, done_shards(out), {"run": RUN, "backend": s1.describe(), "groups": done_groups(out),
                                             "sample_sha256": smp_info["sample_sha256"], "tags": ["VERB/N1"],
                                             "dry": dry})
    print(f"finalized {man['rows']} validity {man['validity_rate']}", flush=True)
    return 0


def calibration(bank) -> dict:
    """Tokens per byte of the verb-v1 prompt, measured on the emu-v1 prompts of the same states against MAIN's ledger
    tokens (the two prompts share the questions block and the state; verb-v1's instructions differ by ~200 bytes)."""
    from jevtrader.s1.backends import LogprobEmulator

    emu = LogprobEmulator(client=object(), bank_hash=bank.sha256, prompt_cache=False)
    states = load_states()
    smp = pd.read_parquet(OUT / "sample.parquet").sort_values("rank").head(300)
    keys = {}
    for u in smp.unit:
        s = states[u]
        keys[s["emu_keys"]["N1.attributes"]] = (s, "attributes")
        keys[s["emu_keys"]["N1.about_focal"]] = (s, "about_focal")
    req = pd.read_parquet(NG.MAIN_DIR / "requests.parquet", columns=["request_key", "input_tokens"])
    req = req[req.request_key.isin(keys) & (req.input_tokens > 0)].drop_duplicates("request_key")
    plan = {g.name: g for g in N.n1_plan(bank).groups()}
    samples = []
    for k, n in zip(req.request_key, req.input_tokens):
        s, grp = keys[k]
        qs = {name: q for name, _, q in NG.n1_questions(bank, plan[grp].ask)}
        samples.append((s["n1_state"], qs, int(n)))
    return C.calibrate(emu, samples) | {"source": "MAIN N1.attributes / N1.about_focal rows of 300 sampled items"}


def cmd_validate(a) -> int:
    smp = pd.read_parquet(OUT / "sample.parquet")
    man = json.loads((OUT / "MANIFEST.json").read_text())
    groups = tuple(man["groups"])
    jobs = pd.read_parquet(OUT / "jobs.parquet")
    req = pd.read_parquet(OUT / "requests.parquet")
    ans = pd.read_parquet(OUT / "answers.parquet")
    led = C.ledger_run_rows(RUN)
    paid = ~req.cached.astype(bool)
    want = {f"{u}|{g}" for u in smp.unit for g in groups}
    meta = pd.DataFrame([json.loads(x) for x in jobs.meta_json])
    chk = {
        "complete": set(jobs.uid) == want,
        "pinned": bool((req.model_pinned == C.MODEL).all() and (req.response_model[paid] == C.MODEL).all()
                       and (req.prompt_version == "verb-v1").all() and (req.backend == "llm-verbalized").all()
                       and (req.bank_hash == FROZEN_SHA256).all()),
        "tags": sorted(req.tag.unique()) == ["VERB/N1"],
        # identical requests of two items share a key: answers per key = questions x request rows with that key
        "answers_per_request": bool((ans.groupby("request_key").size() == req.groupby("request_key").n_questions
                                     .sum()).all()) and len(ans) == int(req.n_questions.sum()),
        "validity_ge_98pct": bool(1 - ans.probs_json.isna().mean() >= 0.98),
        "ledger_covers_paid_rows": led["usd"] + 1e-9 >= float(req.usd[paid].sum()),
    }
    res = {"gen_id": GEN_ID, "checks": chk, "pass": all(chk.values()),
           "summary": {"items": int(meta.unit.nunique()), "requests": len(req), "paid": int(paid.sum()),
                       "degraded_requests": int((req.degraded > 0).sum()),
                       "degraded_answers": int(ans.probs_json.isna().sum()), "by_group": meta.group.value_counts()
                       .to_dict(), "paid_usd_rows": float(req.usd[paid].sum()), "ledger": led},
           "manifest_sha256": man["sha256"], "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"pass": res["pass"], "checks": chk, "summary": {k: v for k, v in res["summary"].items()
                                                                      if k != "ledger"}}, default=str))
    return 0 if res["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "prepare", "dry", "run", "validate"])
    ap.add_argument("--attributes-only", action="store_true")
    ap.add_argument("--singles", action="store_true", help="the four stage-1/2 single-question requests only")
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    groups = ("attributes",) if a.attributes_only else SINGLES if a.singles else GROUPS_ALL
    if a.cmd == "sample":
        return cmd_sample(a)
    if a.cmd == "prepare":
        return cmd_prepare(a)
    if a.cmd in ("dry", "run"):
        return run(a.cmd == "dry", groups, a.concurrency)
    return cmd_validate(a)


if __name__ == "__main__":
    sys.exit(main())
