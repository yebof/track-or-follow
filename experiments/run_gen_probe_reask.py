"""PROBE re-ask of the scheduled D1 probes the PROBE generation did not answer (pre-commit audit 2026-09-28).

    .venv/bin/python experiments/run_gen_probe_reask.py plan       # the missing scheduled probes (free; no model call)
    .venv/bin/python experiments/run_gen_probe_reask.py dry        # fake transport, read-only cache: projection ($0)
    .venv/bin/python experiments/run_gen_probe_reask.py run        # preflight vs the PROBE sub-cap, then paid
    .venv/bin/python experiments/run_gen_probe_reask.py validate

**Why.** The registered D1 schedule (preregistration_v1 §7.2; experiments/analysis/d1_window_start.py `schedule`) takes
the weekly decisions of a month from the full session calendar: the last session of each ISO week, in the month of
its own date. `probe_emu_d1d2` cut the calendar at each month's edge instead, so a week that straddles a month end gave
a month-end pseudo-decision (e.g. 2024-07-31, a Wednesday) that has no T1 label, and the true decision the hash rule
picks was not asked. In the rule period 40 scheduled probes had no answer (2024-07-26, 2024-12-06, 2025-03-28,
2025-04-17; 10 names each), and 20 more in the W_pre-slice months (2023-07-21, 2023-10-13). This script asks exactly
those (t, symbol) probes, nothing else, in a separate generation `probe_emu_d1_reask`; d1_window_start.py reads the
union of the two generations. The off-schedule answers of `probe_emu_d1d2` stay in its tables and are ignored by the
rule (listed in its output).

**Requests.** Identical to `probe_emu_d1d2`'s D1 requests (experiments/run_gen_probe.py): `RecallProbeProgram` stage D1
(`D.recall_beats_median`, unblinded, no evidence), state {company_name, period = (e_t, e_{t+1})} from the full session
calendar, backend `llm-logprob` / emu-v1, pinned gpt-4.1-mini-2025-04-14, frozen bank v1.0, ledger tag PROBE/D (run
PROBE, sub-cap in configs/backends.yaml). The names are checked against both universe sources (the membership table
the probe runner uses and the monthly universe file the D1 script uses).

**Budget.** `run` prints the preflight (run_gen_common.preflight: the dry-run projection, point and high, against the
PROBE sub-cap headroom, the ledger total, the $245 hard cap and the $225 stop threshold) and aborts (exit 3) if it
does not fit; the paid process is bounded by the preflight's guard. No W_clean outcome is read: the plan uses the
session calendar, the universe and the PROBE answers table only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
import run_gen_common as C  # noqa: E402

from jevtrader.programs import probes as P  # noqa: E402
from jevtrader.programs.bank_v1 import FROZEN_SHA256, load_bank  # noqa: E402
from jevtrader.programs.base import Request, WeekContext  # noqa: E402
from jevtrader.programs.context import RealData, subset_names, universe_asof  # noqa: E402
from jevtrader.data.news import decision_cutoff  # noqa: E402

log = logging.getLogger("run_gen_probe_reask")
ROOT = C.ROOT
RUN = "PROBE"
GEN_ID = "probe_emu_d1_reask"
SOURCE_GEN = "probe_emu_d1d2"
OUT = ROOT / "results" / "judgments" / GEN_ID
SRC = ROOT / "results" / "judgments" / SOURCE_GEN
D1_SEED = "D1-www27"
SEGMENTS = (("rule", "2024-01", "2025-06", "2024-01-01"), ("wpre", "2023-04", "2023-12", "2023-04-07"))
DATA_FREEZE = "archive-a377a4d6abd3bfe3"


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def missing_probes(root: Path = ROOT) -> pd.DataFrame:
    """The registered D1 schedule (d1_window_start.schedule on its own calendar) minus the (t, symbol) probes that
    `probe_emu_d1d2` answered: t, symbol, month, pos, segment."""
    import d1_window_start as D1

    ans = D1.probe_answers(SRC)
    parts = []
    for seg, m0, m1, lo in SEGMENTS:
        cal = D1.rule_calendar(root, pd.Timestamp(lo))
        s = D1.schedule(cal, D1._universe(root, cal), D1.months_between(m0, m1)).assign(segment=seg)
        parts.append(s)
    sched = pd.concat(parts, ignore_index=True)
    m = sched.merge(ans, on=["t", "symbol"], how="left")
    return m[m["p"].isna()].drop(columns="p").reset_index(drop=True)


def job_record(uid: str, req: Request, meta: dict) -> dict:
    return {"uid": uid, "program": req.program, "stage": req.stage, "entity": req.entity, "t": str(req.t.date()),
            "state": req.state, "qspec": {"kind": "bank", "qid": "D.recall_beats_median"}, "evidence": [],
            "tag": req.tag, "meta": meta}


def build_request(bank, rec: dict) -> Request:
    spec = bank.q(rec["qspec"]["qid"])
    return Request.build(rec["program"], rec["stage"], rec["entity"], pd.Timestamp(rec["t"]), rec["state"],
                         [(spec.name, spec, spec.to_question())], [], tag=rec["tag"])


def d1_jobs(data: RealData, bank, miss: pd.DataFrame) -> list[dict]:
    """The D1 requests of the missing probes, built exactly as run_gen_probe.d1_jobs builds them."""
    prog = P.RecallProbeProgram(bank, stages=("D1",))
    recs = []
    for t, g in miss.groupby("t"):
        t = pd.Timestamp(t)
        uni = universe_asof(data.membership, t, 100)
        names = subset_names(uni, 10, D1_SEED)
        want = sorted(g["symbol"])
        if not set(want) <= set(names):
            raise SystemExit(f"{t.date()}: scheduled names {want} are not the probe runner's {sorted(names)}")
        ctx = WeekContext(t=t, cutoff_utc=decision_cutoff(t), universe=uni, focus=tuple(want), sessions=data.sessions)
        reqs = prog.requests(ctx)
        if sorted(r.entity for r in reqs) != want:
            raise SystemExit(f"{t.date()}: {len(reqs)} requests for {len(want)} names")
        for r in reqs:
            row = g[g["symbol"] == r.entity].iloc[0]
            recs.append(job_record(f"D1|{t.date()}|{r.entity}", r,
                                   {"month": row["month"], "segment": row["segment"], "source": "D1",
                                    "pos": int(row["pos"]), "reask_of": SOURCE_GEN}))
    return recs


def load_plan() -> list[dict]:
    with gzip.open(OUT / "plan" / "jobs.jsonl.gz", "rt") as fh:
        return [json.loads(x) for x in fh]


def cmd_plan(a) -> int:
    bank = load_bank(run_tag=RUN)
    if bank.sha256 != FROZEN_SHA256:
        raise SystemExit("not the frozen bank")
    miss = missing_probes()
    recs = d1_jobs(RealData(), bank, miss)
    uids = [r["uid"] for r in recs]
    if len(set(uids)) != len(uids):
        raise SystemExit("duplicate job ids")
    (OUT / "plan").mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "plan" / "jobs.jsonl.gz", "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = miss.groupby(["segment", "month"])["t"].agg(lambda x: sorted({str(pd.Timestamp(v).date()) for v in x}))
    info = {"n_jobs": len(recs), "jobs_sha256": sha("\n".join(sorted(uids))), "bank_sha256": bank.sha256,
            "source_gen": SOURCE_GEN, "missing_by_month": {f"{s}|{m}": v for (s, m), v in by.items()},
            "rule": "registered D1 schedule (d1_window_start.schedule on rule_calendar) minus the answered probes",
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / "plan" / "plan.json").write_text(json.dumps(info, indent=1, default=str))
    print(json.dumps({k: info[k] for k in ("n_jobs", "missing_by_month", "jobs_sha256")}, indent=1), flush=True)
    return 0


def shards_of(bank, recs: list[dict]) -> dict[str, list[C.Job]]:
    sh: dict[str, list[C.Job]] = {}
    for r in recs:
        sh.setdefault(f"D1_{r['meta']['segment']}", []).append(C.Job(r["uid"], build_request(bank, r), r["meta"]))
    return dict(sorted(sh.items()))


def run(dry: bool, concurrency: int, scratch: Path | None) -> int:
    bank = load_bank(run_tag=RUN)
    recs = load_plan()
    plan = json.loads((OUT / "plan" / "plan.json").read_text())
    fp = {"gen_id": GEN_ID, "bank": bank.sha256, "jobs_sha256": plan["jobs_sha256"], "backend": "llm-logprob/emu-v1"}
    ratio = json.loads((SRC / "projection.json").read_text())["calibration"]["D1"]["ratio"]
    if dry:
        if scratch is None:
            raise SystemExit("dry needs --scratch (a directory outside the repository)")
        sdir = Path(scratch) / f"{GEN_ID}_dry_{int(time.time())}"
        fake = C.FakeOpenAI(ratio=ratio)
        s1 = C.make_s1("llm-logprob", True, sdir, fake)
        out = sdir / GEN_ID
    else:
        proj = json.loads((OUT / "projection.json").read_text())
        if proj.get("jobs_sha256") != plan["jobs_sha256"]:
            raise SystemExit("the projection is for another plan; rerun dry")
        pf = C.preflight(RUN, proj["projected_usd"], proj["high_usd"], f"D1 re-ask {plan['n_jobs']} requests")
        if not pf.ok:
            return 3
        s1 = C.make_s1("llm-logprob", False)
        out = OUT
    g = C.guarded(s1, RUN, float("inf") if dry else pf.guard)
    shards = shards_of(bank, recs)
    t0 = time.time()
    st = C.run_shards(out, shards, g, RUN, DATA_FREEZE, fp, concurrency)
    s1.ledger.flush()
    info = {**st, "dry": dry, "process_spent_usd": g.spent, "process_calls": g.calls, "refused": g.refused,
            "wall_s": round(time.time() - t0, 1), "started_ts": t0}
    if not dry:
        C.launch_record(out, info)
    print(json.dumps(info), flush=True)
    if dry:
        rec = C.projection_record(OUT / "projection.json", RUN, fake, s1,
                                  {"requests": len(recs), "jobs_sha256": plan["jobs_sha256"], "dry_dir": str(sdir),
                                   "calibration": {"D1": {"ratio": ratio, "source": f"{SOURCE_GEN}/projection.json"}}})
        print(json.dumps({k: rec[k] for k in ("paid_calls", "projected_usd", "high_usd", "by_tag")}), flush=True)
        return 0
    if st["status"] != "complete":
        return 3
    man = C.finalize(out, list(shards), {"run": RUN, "backend": s1.describe(), "plan": {k: plan[k] for k in (
        "n_jobs", "jobs_sha256")}, "tags": ["PROBE/D"], "source_gen": SOURCE_GEN, "dry": dry,
        "launch_ts": t0})
    print(f"finalized {man['rows']} validity {man['validity_rate']}", flush=True)
    return 0


def cmd_validate(a) -> int:
    recs = {r["uid"]: r for r in load_plan()}
    jobs = pd.read_parquet(OUT / "jobs.parquet")
    req = pd.read_parquet(OUT / "requests.parquet")
    ans = pd.read_parquet(OUT / "answers.parquet")
    man = json.loads((OUT / "MANIFEST.json").read_text())
    led = C.ledger_run_rows(RUN, since=float(man["launch_ts"]))
    paid = ~req.cached.astype(bool)
    import d1_window_start as D1

    still = missing_probes()
    got = D1.probe_answers(OUT)
    chk = {
        "complete": set(jobs.uid) == set(recs) and len(jobs) == len(recs),
        "one_answer_per_request": len(ans) == len(req) and ans.request_key.nunique() == req.request_key.nunique(),
        "pinned": bool((req.model_pinned == C.MODEL).all() and (req.response_model[paid] == C.MODEL).all()
                       and (req.prompt_version == "emu-v1").all() and (req.bank_hash == FROZEN_SHA256).all()),
        "tags": sorted(req.tag.unique()) == ["PROBE/D"],
        "validity_ge_98pct": bool(1 - ans.probs_json.isna().mean() >= 0.98),
        "schedule_closed": bool(still.merge(got, on=["t", "symbol"], how="left")["p"].notna().all()),
        "ledger_covers_paid_rows": led["usd"] + 1e-9 >= float(req.usd[paid].sum()),
    }
    res = {"gen_id": GEN_ID, "checks": chk, "pass": all(chk.values()),
           "summary": {"requests": len(req), "paid": int(paid.sum()), "paid_usd_rows": float(req.usd[paid].sum()),
                       "degraded_requests": int((req.degraded > 0).sum()), "ledger_since_launch": led},
           "manifest_sha256": man["sha256"], "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"pass": res["pass"], "checks": chk, "paid": int(paid.sum()), "ledger_usd": led["usd"]},
                     default=str))
    return 0 if res["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "dry", "run", "validate"])
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--scratch", default=None, help="dry: a scratch directory outside the repository")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd in ("dry", "run"):
        return run(a.cmd == "dry", a.concurrency, Path(a.scratch) if a.scratch else None)
    return cmd_validate(a)


if __name__ == "__main__":
    sys.exit(main())
