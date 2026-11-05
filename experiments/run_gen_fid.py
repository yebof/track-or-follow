"""FID (S8 interface conformance; design_v1 §13.9; preregistration_v1 DRAFT §9 "FID"): V1 joint vs single-question
answering on 150 W_clean stock-weeks (T + one N1 item each) and the V2 re-ask floor, 100 requests x 3 repeats.
emu-v1, frozen bank v1.0, pinned gpt-4.1-mini-2025-04-14. Raw answers only (no statistic is computed here).

    .venv/bin/python experiments/run_gen_fid.py prepare     # sample + MAIN's states, verified against MAIN's keys ($0)
    .venv/bin/python experiments/run_gen_fid.py dry         # fake transport projection ($0)
    .venv/bin/python experiments/run_gen_fid.py run         # preflight, then paid (tags FID/T, FID/N1, FID/N3)
    .venv/bin/python experiments/run_gen_fid.py validate

**Sample.** Frame: MAIN's W_clean stock-weeks (115 decisions x U100) with at least one N01 item whose 13-question
attributes request was asked (stage 2). The 150 with the smallest sha256("FID-www27|<t>|<symbol>"); per stock-week
the item with the smallest sha256("FID-www27|<t>|<symbol>|<eid>") among those items.

**V1** (joint vs single). The joint answers are MAIN's (T's 6-question request; N1's 13-question attributes request).
The single-question requests re-ask each question alone on the identical state: T 6 per stock-week; N1 13 per item,
of which the 4 gate questions alone ARE production's stage-1/2 requests (identical key: MAIN's answer from the
response cache, $0) and the 9 attributes are new. States are rebuilt as MAIN built them and checked against MAIN's
request keys (T: MAIN's M.regime from its blackboard; N1: `n1_sha256` and the attributes key).
**V2** (re-ask floor). 100 production requests re-asked 3 times each with a throwaway `uid` field appended to the
state (design_v1 §13.9, §10.3: bypasses the response cache; the same mechanism as the FAITH null arms): 34 T
requests and 33 N1 attributes requests of the first V1 stock-weeks by rank, and 33 N3 requests of MAIN (S30 names;
states from the MEM replay, `mem_emu_mepi-off_wclean_s30/main_n3_states.jsonl.gz`, whose keys equal MAIN's), the
smallest sha256("FID-www27|N3|<t>|<symbol>"). uid = sha256("FID-V2|<MAIN key>|<repeat>")[:12].
Order and label-swap diagnostics are not run (draft §9: only if headroom remains; not in this task).

Output: results/judgments/fid_emu_wclean_v1-150_v2-100x3/. No W_clean outcome is read.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gen_common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "experiments" / "ng"))
import items500 as NG  # noqa: E402

from jevtrader.programs import news as N  # noqa: E402
from jevtrader.programs.bank_v1 import FROZEN_SHA256, load_bank  # noqa: E402
from jevtrader.programs.base import Evidence, Request  # noqa: E402
from jevtrader.programs.technical import TechnicalProgram  # noqa: E402

log = logging.getLogger("run_gen_fid")
ROOT = C.ROOT
RUN = "FID"
GEN_ID = "fid_emu_wclean_v1-150_v2-100x3"
OUT = ROOT / "results" / "judgments" / GEN_ID
SEED = "FID-www27"
N_V1, N_V2 = 150, {"T": 34, "N1": 33, "N3": 33}
REPEATS = 3
SCRATCH = Path("/tmp/jevtrader-scratch/paid")
N3_STATES = ROOT / "results/judgments/mem_emu_mepi-off_wclean_s30/main_n3_states.jsonl.gz"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def emu() -> dict:
    return json.loads((C.MAIN_DIR / "MANIFEST.json").read_text())["backend"]


# ------------------------------------------------------------------------------------------ prepare
def cmd_prepare(a) -> int:
    from jevtrader.memory.store import MemoryStore

    bank = load_bank(run_tag=RUN)
    if bank.sha256 != FROZEN_SHA256:
        raise SystemExit("not the frozen bank")
    e = emu()
    weeks = {str(w.date()) for w in C.main_weeks()}
    rows = NG.main_item_rows()
    by_sw: dict[tuple[str, str], list[dict]] = {}
    for uid, r in rows.items():
        if r["t"] in weeks and "N1.attributes" in (r.get("request_keys") or {}):
            by_sw.setdefault((r["t"], r["symbol"]), []).append(r)
    frame = sorted(by_sw, key=lambda k: sha(f"{SEED}|{k[0]}|{k[1]}"))
    chosen = frame[:N_V1]
    req = pd.read_parquet(C.MAIN_DIR / "requests.parquet", columns=["t", "symbol", "program", "request_key",
                                                                    "input_tokens"])
    tk = req[req.program == "T"].copy()
    tk["d"] = pd.to_datetime(tk.t).dt.strftime("%Y-%m-%d")
    tkey = {(d, s): k for d, s, k in zip(tk.d, tk.symbol, tk.request_key)}
    tok = req[req.input_tokens > 0].groupby("request_key").input_tokens.first().to_dict()
    builder, data = C.main_context(None)
    tmp = Path(tempfile.mkdtemp(prefix="fid_mem_"))
    with open(tmp / "work.jsonl", "w") as fh:
        for d in C.work_records(C.MAIN_DIR, ("M.",)):
            fh.write(json.dumps(d) + "\n")
    store = MemoryStore(tmp)
    tprog = TechnicalProgram(bank)
    tq = {s.name: s.to_question() for s in bank.program("T").values()}
    plan = {g.name: g for g in N.n1_plan(bank).groups()}
    out = []
    t0 = time.time()
    by_week: dict[str, list[tuple[str, str]]] = {}
    for d, s in chosen:
        by_week.setdefault(d, []).append((d, s))
    for d, sws in sorted(by_week.items()):
        t = pd.Timestamp(d)
        ctx0 = builder(t)
        view = store.view(t, ctx0.cutoff_utc, deps=["M"], history=[], readable={"M"}, clearance="T")
        ctx = dataclasses.replace(ctx0, memory=view)
        src = N.ItemSource(ctx0, bank)
        for _, s in sws:
            state, ev = tprog.states(ctx, names=[s])[s]
            if NG.emu_key(e, state, tq) != tkey[(d, s)]:
                raise SystemExit(f"{d} {s}: T state differs from MAIN's")
            items = sorted(by_sw[(d, s)], key=lambda r: sha(f"{SEED}|{d}|{s}|{r['eid']}"))
            main = items[0]
            it = {i.eid: i for i in src.items(s)}.get(main["eid"])
            if it is None:
                raise SystemExit(f"{d} {s} {main['eid']}: item not rebuilt")
            keep = [i in set(main["kept_sentences"]) for i in range(len(it.n0_sentences))]
            n1 = N.n1_item_state(it, keep)
            fc = N.focal(bank, ctx0, s)
            rq = N.n1_group_request(bank, it, fc, n1, t, plan["attributes"].ask)
            if N.item_hash(n1) != main["n1_sha256"] or NG.emu_key(e, rq.state, rq.questions) != \
                    main["request_keys"]["N1.attributes"]:
                raise SystemExit(f"{d} {s} {main['eid']}: N1 state differs from MAIN's")
            evn = it.evidence()
            out.append({"t": d, "symbol": s, "rank": sha(f"{SEED}|{d}|{s}"), "eid": main["eid"],
                        "t_state": state, "t_evidence": [{"eid": x.eid, "at": pd.Timestamp(x.available_at_utc)
                                                          .isoformat(), "sha": x.text_sha256} for x in ev],
                        "t_key": tkey[(d, s)], "t_tokens": int(tok.get(tkey[(d, s)], 0)),
                        "n1_state": rq.state, "n1_evidence": {"eid": evn.eid, "at": pd.Timestamp(
                            evn.available_at_utc).isoformat(), "sha": evn.text_sha256, "d0": evn.d0_flags},
                        "n1_keys": {k: v for k, v in main["request_keys"].items() if k.startswith("N1.")},
                        "n1_tokens": int(tok.get(main["request_keys"]["N1.attributes"], 0))})
        print(f"prepare {d}: {len(sws)} stock-weeks verified ({time.time() - t0:.0f}s)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    out.sort(key=lambda x: x["rank"])
    # V2 N3: MAIN's N3 states (S30), from the MEM replay
    n3 = []
    with gzip.open(N3_STATES, "rt") as fh:
        for line in fh:
            x = json.loads(line)
            if x["t"] in weeks:
                n3.append(x | {"rank": sha(f"{SEED}|N3|{x['t']}|{x['symbol']}"),
                               "tokens": int(tok.get(x["request_key"], 0))})
    n3 = sorted(n3, key=lambda x: x["rank"])[: N_V2["N3"]]
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "states.jsonl.gz", "wt") as fh:
        for x in out:
            fh.write(json.dumps({"kind": "sw", **x}, ensure_ascii=False) + "\n")
        for x in n3:
            fh.write(json.dumps({"kind": "n3", **x}, ensure_ascii=False) + "\n")
    info = {"frame_stock_weeks": len(frame), "v1": len(out), "v2": N_V2, "repeats": REPEATS, "seed": SEED,
            "sample_sha256": sha("\n".join(sorted(f"{x['t']}|{x['symbol']}|{x['eid']}" for x in out))),
            "n3_sha256": sha("\n".join(sorted(x["request_key"] for x in n3))),
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / "sample.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)
    return 0


def load_states() -> tuple[list[dict], list[dict]]:
    sw, n3 = [], []
    with gzip.open(OUT / "states.jsonl.gz", "rt") as fh:
        for line in fh:
            x = json.loads(line)
            (sw if x["kind"] == "sw" else n3).append(x)
    return sw, n3


# ------------------------------------------------------------------------------------------ jobs
@dataclasses.dataclass(frozen=True)
class StoredEvidence(Evidence):
    stored_sha256: str | None = None

    @property
    def text_sha256(self) -> str | None:
        return self.stored_sha256


def build_jobs(bank) -> dict[str, list[C.Job]]:
    sw, n3 = load_states()
    e = emu()
    tspecs = bank.program("T")
    n1specs = bank.program("N1")
    attrs = [g for g in N.n1_plan(bank).groups() if g.name == "attributes"][0]
    n3specs = bank.program("N3")
    sh: dict[str, list[C.Job]] = {"V1_T": [], "V1_N1": [], "V2_T": [], "V2_N1": [], "V2_N3": []}
    for i, x in enumerate(sw):
        t = pd.Timestamp(x["t"])
        tev = [StoredEvidence(v["eid"], pd.Timestamp(v["at"]), stored_sha256=v["sha"]) for v in x["t_evidence"]]
        nev = [StoredEvidence(x["n1_evidence"]["eid"], pd.Timestamp(x["n1_evidence"]["at"]),
                              d0_flags=x["n1_evidence"]["d0"], stored_sha256=x["n1_evidence"]["sha"])]
        base = {"t": x["t"], "symbol": x["symbol"], "eid": x["eid"], "rank": x["rank"]}
        for s in tspecs.values():
            r = Request.build("T", f"V1:T:{s.name}", x["symbol"], t, x["t_state"], [(s.name, s, s.to_question())],
                              tev, tag="T")
            sh["V1_T"].append(C.Job(f"V1|T|{x['t']}|{x['symbol']}|{s.name}", r, base | {"arm": "V1", "q": s.name,
                                                                                         "joint_key": x["t_key"]}))
        for name in attrs.ask:
            s = n1specs[name]
            r = Request.build("N1", f"V1:N1:{name}", x["symbol"], t, x["n1_state"], [(name, s, s.to_question())],
                              nev, tag="N1")
            prod = x["n1_keys"].get(f"N1.{name}")  # the production single of a gate question (same key)
            if prod is not None and NG.emu_key(e, r.state, r.questions) != prod:
                raise SystemExit(f"{x['t']} {x['symbol']} {name}: single request is not production's")
            sh["V1_N1"].append(C.Job(f"V1|N1|{x['t']}|{x['symbol']}|{x['eid']}|{name}", r,
                                     base | {"arm": "V1", "q": name, "joint_key": x["n1_keys"]["N1.attributes"],
                                             "production_single": prod is not None}))
        for rep in range(1, REPEATS + 1):
            if i < N_V2["T"]:
                uid = sha(f"FID-V2|{x['t_key']}|{rep}")[:12]
                r = Request.build("T", f"V2:T:r{rep}", x["symbol"], t, {**x["t_state"], "uid": uid},
                                  [(s.name, s, s.to_question()) for s in tspecs.values()], tev, tag="T")
                sh["V2_T"].append(C.Job(f"V2|T|{x['t']}|{x['symbol']}|r{rep}", r,
                                        base | {"arm": "V2", "repeat": rep, "uid": uid, "main_key": x["t_key"]}))
            if i < N_V2["N1"]:
                k = x["n1_keys"]["N1.attributes"]
                uid = sha(f"FID-V2|{k}|{rep}")[:12]
                r = Request.build("N1", f"V2:N1:r{rep}", x["symbol"], t, {**x["n1_state"], "uid": uid},
                                  [(n, n1specs[n], n1specs[n].to_question()) for n in attrs.ask], nev, tag="N1")
                sh["V2_N1"].append(C.Job(f"V2|N1|{x['t']}|{x['symbol']}|{x['eid']}|r{rep}", r,
                                         base | {"arm": "V2", "repeat": rep, "uid": uid, "main_key": k}))
    for x in n3:
        t = pd.Timestamp(x["t"])
        for rep in range(1, REPEATS + 1):
            uid = sha(f"FID-V2|{x['request_key']}|{rep}")[:12]
            r = Request.build("N3", f"V2:N3:r{rep}", x["symbol"], t, {**x["state"], "uid": uid},
                              [(s.name, s, s.to_question()) for s in n3specs.values()], [], tag="N3")
            sh["V2_N3"].append(C.Job(f"V2|N3|{x['t']}|{x['symbol']}|r{rep}", r,
                                     {"t": x["t"], "symbol": x["symbol"], "arm": "V2", "repeat": rep, "uid": uid,
                                      "main_key": x["request_key"], "rank": x["rank"]}))
    # the V2 N3 states must be MAIN's (their key without the uid)
    for x in n3:
        q = {s.name: s.to_question() for s in n3specs.values()}
        if NG.emu_key(e, x["state"], q) != x["request_key"]:
            raise SystemExit(f"N3 {x['t']} {x['symbol']}: state is not MAIN's")
    return sh


def calibration(bank) -> dict:
    from jevtrader.s1.backends import LogprobEmulator

    em = LogprobEmulator(client=object(), bank_hash=bank.sha256, prompt_cache=False)
    sw, n3 = load_states()
    tq = {s.name: s.to_question() for s in bank.program("T").values()}
    attrs = [g for g in N.n1_plan(bank).groups() if g.name == "attributes"][0]
    aq = {n: bank.program("N1")[n].to_question() for n in attrs.ask}
    n3q = {s.name: s.to_question() for s in bank.program("N3").values()}
    t_ = C.calibrate(em, [(x["t_state"], tq, x["t_tokens"]) for x in sw if x["t_tokens"]])
    n1 = C.calibrate(em, [(x["n1_state"], aq, x["n1_tokens"]) for x in sw if x["n1_tokens"]])
    n3_ = C.calibrate(em, [(x["state"], n3q, x["tokens"]) for x in n3 if x["tokens"]])
    return {"T": t_, "N1": n1, "N3": n3_, "source": "MAIN's measured input tokens of the same (joint) states"}


def run(dry: bool, concurrency: int) -> int:
    bank = load_bank(run_tag=RUN)
    info = json.loads((OUT / "sample.json").read_text())
    shards = build_jobs(bank)
    n_req = sum(len(v) for v in shards.values())
    fp = {"gen_id": GEN_ID, "bank": bank.sha256, "sample_sha256": info["sample_sha256"],
          "n3_sha256": info["n3_sha256"], "backend": "llm-logprob/emu-v1"}
    if dry:
        scratch = SCRATCH / f"{GEN_ID}_dry_{int(time.time())}"
        cal = calibration(bank)
        rt, rn, r3 = cal["T"]["ratio"], cal["N1"]["ratio"], cal["N3"]["ratio"]
        fake = C.FakeOpenAI(ratio=lambda system, user: r3 if '"this_week_events"' in user else
                            (rn if '"item"' in user else rt))
        s1 = C.make_s1("llm-logprob", True, scratch, fake)
        out, guard = scratch / GEN_ID, float("inf")
    else:
        proj = json.loads((OUT / "projection.json").read_text())
        pf = C.preflight(RUN, proj["projected_usd"], proj["high_usd"],
                         f"V1 {len(shards['V1_T']) + len(shards['V1_N1'])} + V2 "
                         f"{len(shards['V2_T']) + len(shards['V2_N1']) + len(shards['V2_N3'])} requests")
        if not pf.ok:
            return 3
        s1 = C.make_s1("llm-logprob", False)
        out, guard = OUT, pf.guard
    g = C.guarded(s1, RUN, guard)
    t0 = time.time()
    st = C.run_shards(out, shards, g, RUN, "archive-a377a4d6abd3bfe3", fp, concurrency)
    s1.ledger.flush()
    rec = {**st, "dry": dry, "process_spent_usd": g.spent, "process_calls": g.calls, "refused": g.refused,
           "wall_s": round(time.time() - t0, 1)}
    C.launch_record(out, rec)
    print(json.dumps(rec), flush=True)
    if dry:
        p = C.projection_record(OUT / "projection.json", RUN, fake, s1, {"requests": n_req, "dry_dir": str(scratch),
                                                                          "calibration": cal})
        print(json.dumps({k: p[k] for k in ("paid_calls", "projected_usd", "high_usd", "by_tag")}), flush=True)
    if st["status"] != "complete":
        return 3
    man = C.finalize(out, list(shards), {"run": RUN, "backend": s1.describe(), "sample": info,
                                         "tags": ["FID/T", "FID/N1", "FID/N3"], "dry": dry})
    print(f"finalized {man['rows']} validity {man['validity_rate']}", flush=True)
    return 0


def cmd_validate(a) -> int:
    bank = load_bank(run_tag=RUN)
    shards = build_jobs(bank)
    want = {j.uid for v in shards.values() for j in v}
    jobs = pd.read_parquet(OUT / "jobs.parquet")
    req = pd.read_parquet(OUT / "requests.parquet")
    ans = pd.read_parquet(OUT / "answers.parquet")
    man = json.loads((OUT / "MANIFEST.json").read_text())
    led = C.ledger_run_rows(RUN)
    paid = ~req.cached.astype(bool)
    meta = pd.DataFrame([json.loads(x) for x in jobs.meta_json])
    v1n1 = jobs[jobs.uid.str.startswith("V1|N1|")].merge(req[["request_key", "cached"]].drop_duplicates(
        "request_key"), on="request_key")
    prod = v1n1[[json.loads(m)["production_single"] for m in v1n1.meta_json]]
    main_keys = set(pd.read_parquet(C.MAIN_DIR / "requests.parquet", columns=["request_key"]).request_key)
    chk = {
        "complete": set(jobs.uid) == want,
        "pinned": bool((req.model_pinned == C.MODEL).all() and (req.response_model[paid] == C.MODEL).all()
                       and (req.prompt_version == "emu-v1").all() and (req.bank_hash == FROZEN_SHA256).all()),
        "tags": set(req.tag.unique()) <= {"FID/T", "FID/N1", "FID/N3"},
        "production_singles_are_main_keys": bool(prod.request_key.isin(main_keys).all()),
        "v2_repeats_distinct_keys": bool(jobs[jobs.uid.str.startswith("V2|")].request_key.nunique() ==
                                          (jobs.uid.str.startswith("V2|")).sum()),
        "v2_not_main_keys": not jobs[jobs.uid.str.startswith("V2|")].request_key.isin(main_keys).any(),
        "validity_ge_98pct": bool(1 - ans.probs_json.isna().mean() >= 0.98),
        "ledger_covers_paid_rows": led["usd"] + 1e-9 >= float(req.usd[paid].sum()),
    }
    res = {"gen_id": GEN_ID, "checks": chk, "pass": all(chk.values()),
           "summary": {"requests": len(req), "paid": int(paid.sum()), "cached": int((~paid).sum()),
                       "by_arm_prog": meta.groupby("arm").size().to_dict(),
                       "degraded_requests": int((req.degraded > 0).sum()),
                       "production_singles_cached": int(prod.cached.sum()), "production_singles": len(prod),
                       "paid_usd_rows": float(req.usd[paid].sum()), "ledger": led},
           "manifest_sha256": man["sha256"], "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"pass": res["pass"], "checks": chk, "paid": res["summary"]["paid"]}, default=str))
    return 0 if res["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "dry", "run", "validate"])
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd == "prepare":
        return cmd_prepare(a)
    if a.cmd in ("dry", "run"):
        return run(a.cmd == "dry", a.concurrency)
    return cmd_validate(a)


if __name__ == "__main__":
    sys.exit(main())
