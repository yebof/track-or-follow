"""PROBE (design_v1 §13.7 D1 + D2; preregistration_v1 DRAFT §7.2, §8; decisions 2026-09-28 P-3): raw answers only.

    .venv/bin/python experiments/run_gen_probe.py plan       # schedule, D2 sample and states (free; no model call)
    .venv/bin/python experiments/run_gen_probe.py dry        # fake transport, read-only cache: projection ($0)
    .venv/bin/python experiments/run_gen_probe.py run        # preflight, then paid (tag PROBE/D, PROBE/D2)
    .venv/bin/python experiments/run_gen_probe.py validate

Backend: `llm-logprob` / emu-v1 (production probes stay on emu-v1; bank prompts `emu-v1-probe` is LA-2 validation
only), pinned gpt-4.1-mini-2025-04-14, frozen bank v1.0 (run tag PROBE is in FROZEN_RUN_TAGS, P-7).

**D1** (`D.recall_beats_median`, unblinded, evidence-free; `programs.probes.RecallProbeProgram` stage D1 only):
- rule months 2024-01..2025-06 (18) and, as the draft registers ("the same schedule runs on the W_pre slice"), the
  W_pre-slice months 2023-04..2023-12 not already in that range (9; decisions inside the W_pre slice window);
- weeks: the 2 decisions of the month with the smallest sha256("D1-www27|" + ISO date);
- names: the 10 of U_t with the smallest sha256("D1-www27" + symbol) (`subset_names(U_t, 10, "D1-www27")`);
- state {company_name, period = (e_t, e_{t+1})}, exactly `RecallProbeProgram.requests`. 27 x 2 x 10 = 540 requests.
  No AUC is computed here: the months from 2024-06 on would read W_clean labels (computed after the commit, §7.2).

**D2** (re-identification; 600 states = 2 periods x 3 blinding levels x 100):
- periods: the W_pre slice (PRE/B's 60 decisions, S30) and W_clean (MAIN's 115 decisions; the S30 names of U_t, as
  PRE); per period 100 base states = 34 N1 item states + 33 T states + 33 F-B states, the smallest
  rank = sha256("D2-www27|<period>|<source>|<t>|<symbol>[|<eid>]") among the stock-weeks (items) the generation
  asked (T/FB request rows; N01 item rows with an N1 state);
- `full` level = the exact production state, rebuilt with the generation's context and verified against its request
  key (T, FB: the program's key; N1: the about_focal group's key and `n1_sha256`);
- `names` level = the focal firm's SEC names and historical tickers masked (plus levels and times), no brand list, no
  feed short names, no executives, no masking of other U members (day3_blind §8 item 1); T and F-B states carry no
  free text, so their `names` state is the `full` state (byte-identical; the request is a cache hit);
- `none` level = identity revealed at the backend boundary exactly as PRE's unblinded arm (pre_report §1.1,
  `experiments/pre/unblind.py`): every study-firm code name -> SEC registrant name valid at d_t, the code-name note
  replaced; persons, levels and relative times stay masked;
- questions: `D.reid_company` (true name + 4 same-FF12 adjacent-size decoys, shuffled by sha256(D2-www27|t|symbol),
  `programs.probes.reid_question`) on every state; `D.reid_year` on every N1 state, each in its own request.

Output: results/judgments/probe_emu_d1d2/{answers,requests,evidence,jobs}.parquet, MANIFEST.json, plan/,
projection.json, validation.json. No W_clean outcome is read.
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
from functools import cached_property
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_gen_common as C  # noqa: E402

from jevtrader.programs import news as N  # noqa: E402
from jevtrader.programs import probes as P  # noqa: E402
from jevtrader.programs.bank_v1 import FROZEN_SHA256, load_bank  # noqa: E402
from jevtrader.programs.base import Evidence, Request, WeekContext  # noqa: E402
from jevtrader.programs.context import (  # noqa: E402
    ContextBuilder,
    PitBlinder,
    RealData,
    decision_dates,
    subset_names,
    universe_asof,
)
from jevtrader.data.news import decision_cutoff  # noqa: E402
from jevtrader.s1.types import canonical_json, request_key  # noqa: E402

log = logging.getLogger("run_gen_probe")
ROOT = C.ROOT
RUN = "PROBE"
GEN_ID = "probe_emu_d1d2"
OUT = ROOT / "results" / "judgments" / GEN_ID
SCRATCH = Path("/tmp/jevtrader-scratch/paid")
D1_SEED = "D1-www27"
D2_SEED = "D2-www27"
RULE_MONTHS = pd.period_range("2024-01", "2025-06", freq="M")
WPRE_MONTHS = pd.period_range("2023-04", "2023-12", freq="M")
W_PRE = ("2023-04-07", "2024-05-31")
W_CLEAN = ("2024-07-05", "2026-09-11")
N_WEEKS_PER_MONTH, N_NAMES = 2, 10
N_BASE = {"N1": 34, "T": 33, "FB": 33}
LEVELS = ("full", "names", "none")
GENS = {"W_pre": ROOT / "results/judgments/pre_blinded_emu_wpre_s30",
        "W_clean": ROOT / "results/judgments/main_emu_wclean_u100"}
CONTEXT_ID = {"W_pre": "74f1467179"}  # PRE/B's MANIFEST notes; MAIN's is in its dev_run.json


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ------------------------------------------------------------------------------------------ names-level blinder
class NamesOnlyData(RealData):
    """RealData whose blinder masks only SEC names (current, former, distinctive first word) and historical tickers:
    no brand list, no feed short names, no manual aliases, no executives; the week's view masks the focal firm only."""

    @cached_property
    def blinder(self):
        from datetime import date

        from jevtrader.data.blinding import load_blinder
        from jevtrader.data.universe import SymbolResolver, load_membership, load_renames

        m = load_membership(self.root / "data" / "raw" / "universe" / "sp500_hist_components.csv", "2018-01-01")
        resolver = SymbolResolver(load_renames(self.root / "data" / "raw" / "universe" / "alpaca_name_changes.json", m))
        return load_blinder(self.root / "data" / "raw" / "sec" / "submissions", self.study_symbols, resolver=resolver,
                            history=(date(2018, 1, 1), date(2026, 12, 31)), lives=self.lives, brand_aliases={},
                            manual_aliases={}, executives={})

    def pit_blinder(self, cutoff_utc):
        return FocalOnlyBlinder(self, cutoff_utc)


class FocalOnlyBlinder(PitBlinder):
    def blinder_at(self, cutoff=None, symbol: str | None = None):
        return self._data.blinder_for({symbol} if symbol else set())


# ------------------------------------------------------------------------------------------ serialization
@dataclasses.dataclass(frozen=True)
class StoredEvidence(Evidence):
    stored_sha256: str | None = None

    @property
    def text_sha256(self) -> str | None:
        return self.stored_sha256


def ev_json(evs) -> list[dict]:
    return [{"eid": e.eid, "available_at_utc": pd.Timestamp(e.available_at_utc).isoformat(), "kept": e.kept,
             "quarantined": e.quarantined, "text_sha256": e.text_sha256} for e in evs]


def ev_objs(js: list[dict]) -> list[Evidence]:
    return [StoredEvidence(e["eid"], pd.Timestamp(e["available_at_utc"]), e["kept"], e["quarantined"], None,
                           None, e["text_sha256"]) for e in js]


def job_record(uid: str, req: Request, qspec: dict, meta: dict) -> dict:
    return {"uid": uid, "program": req.program, "stage": req.stage, "entity": req.entity, "t": str(req.t.date()),
            "state": req.state, "qspec": qspec, "evidence": ev_json(req.evidence), "tag": req.tag, "meta": meta}


def build_request(bank, rec: dict) -> Request:
    q = rec["qspec"]
    t = pd.Timestamp(rec["t"])
    if q["kind"] == "bank":
        spec = bank.q(q["qid"])
        items = [(spec.name, spec, spec.to_question())]
    else:  # reid_company with its options
        spec = bank.q("D.reid_company")
        items = [(spec.name, spec, P.reid_question(bank, q["true"], q["decoys"], q["seed"]))]
    return Request.build(rec["program"], rec["stage"], rec["entity"], t, rec["state"], items,
                         ev_objs(rec["evidence"]), tag=rec["tag"])


def load_plan() -> list[dict]:
    with gzip.open(OUT / "plan" / "jobs.jsonl.gz", "rt") as fh:
        return [json.loads(x) for x in fh]


# ------------------------------------------------------------------------------------------ plan: D1
def d1_jobs(data: RealData, bank) -> tuple[list[dict], dict]:
    sessions, membership = data.sessions, data.membership
    prog = P.RecallProbeProgram(bank, stages=("D1",))
    recs, sched = [], {}
    for seg, months in (("rule", RULE_MONTHS), ("wpre", WPRE_MONTHS)):
        for m in months:
            lo, hi = m.start_time.normalize(), m.end_time.normalize()
            if seg == "wpre":
                lo = max(lo, pd.Timestamp(W_PRE[0]))
            decs = decision_dates(sessions, lo, hi)
            chosen = sorted(decs, key=lambda d: sha(f"{D1_SEED}|{d.date().isoformat()}"))[:N_WEEKS_PER_MONTH]
            sched[str(m)] = {"segment": seg, "decisions": [str(d.date()) for d in decs],
                             "chosen": sorted(str(d.date()) for d in chosen)}
            for t in sorted(chosen):
                uni = universe_asof(membership, t, 100)
                names = subset_names(uni, N_NAMES, D1_SEED)
                ctx = WeekContext(t=t, cutoff_utc=decision_cutoff(t), universe=uni, focus=names, sessions=sessions)
                reqs = prog.requests(ctx)
                if len(reqs) != N_NAMES:
                    raise SystemExit(f"D1 {t.date()}: {len(reqs)} requests (probe period missing?)")
                for r in reqs:
                    recs.append(job_record(f"D1|{t.date()}|{r.entity}", r,
                                           {"kind": "bank", "qid": "D.recall_beats_median"},
                                           {"month": str(m), "segment": seg, "source": "D1"}))
    return recs, sched


# ------------------------------------------------------------------------------------------ plan: D2
def gen_weeks(period: str) -> list[pd.Timestamp]:
    man = json.loads((GENS[period] / "MANIFEST.json").read_text())
    lo, hi = (W_PRE if period == "W_pre" else W_CLEAN)
    return [pd.Timestamp(w) for w in man["spec"]["weeks"] if lo <= w <= hi]


def work_records(gen: Path, prefixes: tuple[str, ...]):
    with open(gen / "memory" / "work.jsonl") as f:
        for line in f:
            if not any(f'"key": "{p}' in line for p in prefixes):
                continue
            d = json.loads(line)
            if d["record"]["key"].startswith(prefixes):
                yield d


def d2_sample(period: str, weeks: list[pd.Timestamp], data: RealData) -> list[dict]:
    gen = GENS[period]
    s30 = {t: set(subset_names(universe_asof(data.membership, t, 100), 30, "S30-www27")) for t in weeks}
    wk = {str(t.date()): t for t in weeks}
    req = pd.read_parquet(gen / "requests.parquet", columns=["t", "symbol", "program", "request_key", "input_tokens"])
    req = req[req.program.isin(["T", "FB"])]
    req["d"] = pd.to_datetime(req["t"]).dt.strftime("%Y-%m-%d")
    out = []
    for src in ("T", "FB"):
        g = req[(req.program == src) & req.d.isin(wk)]
        g = g[[s in s30[wk[d]] for d, s in zip(g.d, g.symbol)]].drop_duplicates(["d", "symbol"])
        fr = [{"source": src, "t": d, "symbol": s, "eid": None, "gen_key": k, "gen_input_tokens": int(n),
               "rank": sha(f"{D2_SEED}|{period}|{src}|{d}|{s}")}
              for d, s, k, n in zip(g.d, g.symbol, g.request_key, g.input_tokens)]
        out += sorted(fr, key=lambda x: x["rank"])[: N_BASE[src]]
    items = []
    for d in work_records(gen, (N.ITEMS_KEY,)):
        rec = d["record"]
        t = rec["t"][:10]
        sym = rec["key"][len(N.ITEMS_KEY):]
        if t not in wk or sym not in s30[wk[t]]:
            continue
        for r in rec["value"]:
            if r.get("n1_sha256"):
                items.append({"source": "N1", "t": t, "symbol": sym, "eid": r["eid"], "main_item": r,
                              "rank": sha(f"{D2_SEED}|{period}|N1|{t}|{sym}|{r['eid']}")})
    out += sorted(items, key=lambda x: x["rank"])[: N_BASE["N1"]]
    for x in out:
        x["period"] = period
    return out


def regime_store(gen: Path, tmp: Path):
    from jevtrader.memory.store import MemoryStore

    tmp.mkdir(parents=True, exist_ok=True)
    with open(tmp / "work.jsonl", "w") as fh:
        for d in work_records(gen, ("M.",)):
            fh.write(json.dumps(d) + "\n")
    return MemoryStore(tmp)


def emu_desc(gen: Path) -> dict:
    return json.loads((gen / "MANIFEST.json").read_text())["backend"]


def emu_key(emu: dict, state, questions) -> str:
    return request_key("llm-logprob", emu["model_pinned"], emu["prompt_version"], emu["cfg_hash"], emu["bank_hash"],
                       state, questions)


def all_questions(bank, prog: str) -> dict:
    return {s.name: s.to_question() for s in bank.program(prog).values()}


def d2_jobs(bank) -> tuple[list[dict], dict]:
    from jevtrader.programs.fundamentals import FundamentalsProgram
    from jevtrader.programs.technical import TechnicalProgram
    from unblind import Unblinder, blinded_notes

    recs, report, calib = [], {}, []
    tprog, fprog = TechnicalProgram(bank), FundamentalsProgram(bank)
    for period in ("W_pre", "W_clean"):
        gen = GENS[period]
        weeks = gen_weeks(period)
        data = RealData()
        data.fact_symbols = data.symbols_for_weeks(json.loads((gen / "MANIFEST.json").read_text())["spec"]["weeks"])
        builder = ContextBuilder(data, subset="S30" if period == "W_pre" else None)
        ctx_id = sha(canonical_json(builder.describe()))[:10]
        want = CONTEXT_ID.get(period) or json.loads((gen / "dev_run.json").read_text())["context_id"]
        if ctx_id != want:
            raise SystemExit(f"{period}: context {ctx_id} != the generation's {want}")
        ndata = NamesOnlyData()
        ndata.fact_symbols = data.fact_symbols
        nbuilder = ContextBuilder(ndata, subset="S30" if period == "W_pre" else None)
        unb = Unblinder({v: k for k, v in data.blinder.code_names.items()}, P.company_name, blinded_notes(bank))
        emu = emu_desc(gen)
        smp = d2_sample(period, weeks, data)
        tmp = Path(tempfile.mkdtemp(prefix="probe_mem_"))
        store = regime_store(gen, tmp)
        checks = {"T_key": 0, "FB_key": 0, "N1_key": 0, "N1_hash": 0, "N1_names_sentences_differ": 0}
        by_week: dict[str, list[dict]] = {}
        for x in smp:
            by_week.setdefault(x["t"], []).append(x)
        t0 = time.time()
        for d, xs in sorted(by_week.items()):
            t = pd.Timestamp(d)
            ctx0 = builder(t)
            view = store.view(t, ctx0.cutoff_utc, deps=["M"], history=[], readable={"M"}, clearance="T")
            ctx = dataclasses.replace(ctx0, memory=view)
            nctx = None
            src = N.ItemSource(ctx0, bank)
            for x in xs:
                sym = x["symbol"]
                base = []  # (level, state, evidence)
                if x["source"] in ("T", "FB"):
                    prog = tprog if x["source"] == "T" else fprog
                    st = prog.states(ctx, names=[sym]).get(sym)
                    if st is None:
                        raise SystemExit(f"{period} {d} {sym}: no {x['source']} state")
                    state, ev = st
                    if emu_key(emu, state, all_questions(bank, x["source"])) != x["gen_key"]:
                        raise SystemExit(f"{period} {d} {sym}: rebuilt {x['source']} state differs from the gen's")
                    checks[f"{x['source']}_key"] += 1
                    if x["gen_input_tokens"] > 0:
                        calib.append({"program": x["source"], "state": state, "tokens": x["gen_input_tokens"]})
                    base = [("full", state, ev), ("names", state, ev)]
                else:
                    main = x["main_item"]
                    it = {i.eid: i for i in src.items(sym)}.get(x["eid"])
                    if it is None:
                        raise SystemExit(f"{period} {d} {sym} {x['eid']}: item not rebuilt")
                    keep = [i in set(main["kept_sentences"]) for i in range(len(it.n0_sentences))]
                    n1 = N.n1_item_state(it, keep)
                    if N.item_hash(n1) != main["n1_sha256"]:
                        raise SystemExit(f"{period} {d} {sym} {x['eid']}: n1_sha256 differs")
                    checks["N1_hash"] += 1
                    fc = N.focal(bank, ctx0, sym)
                    rq = N.n1_group_request(bank, it, fc, n1, t, ("about_focal",))
                    if emu_key(emu, rq.state, rq.questions) != main["request_keys"]["N1.about_focal"]:
                        raise SystemExit(f"{period} {d} {sym} {x['eid']}: N1 key differs from the gen's")
                    checks["N1_key"] += 1
                    calib.append({"program": "N1", "state": rq.state, "key": main["request_keys"]["N1.about_focal"],
                                  "gen": period})
                    ev = [it.evidence()]
                    if nctx is None:
                        nctx = nbuilder(t)
                        nsrc = N.ItemSource(nctx, bank)
                    nit = {i.eid: i for i in nsrc.items(sym)}.get(x["eid"])
                    if nit is None:
                        raise SystemExit(f"{period} {d} {sym} {x['eid']}: names-level item not rebuilt")
                    if len(nit.n0_sentences) != len(it.n0_sentences):
                        checks["N1_names_sentences_differ"] += 1
                    nkeep = [i in set(main["kept_sentences"]) for i in range(len(nit.n0_sentences))]
                    nstate = N.n1_group_request(bank, nit, N.focal(bank, nctx, sym), N.n1_item_state(nit, nkeep), t,
                                                ("about_focal",)).state
                    base = [("full", rq.state, ev), ("names", nstate, ev)]
                # decoys and the reid question are the same at every level
                names = [P.company_name(s, t) for s in [sym, *P.decoys(ctx0, sym)]]
                seed = f"{P.D2_SALT}|{t.date()}|{sym}"
                qspec = {"kind": "reid", "true": names[0], "decoys": names[1:], "seed": seed}
                full_state = base[0][1]
                levels = [*base, ("none", unb.state(full_state, t), base[0][2])]
                for level, state, evs in levels:
                    meta = {"period": period, "source": x["source"], "level": level, "rank": x["rank"],
                            "eid": x["eid"], "true_name": names[0]}
                    uid = f"D2|{period}|{x['source']}|{d}|{sym}|{x['eid'] or '-'}|{level}"
                    r = Request.build("D2", f"D2:{x['source']}:{level}", sym, t, state,
                                      [("reid_company", bank.q("D.reid_company"),
                                        P.reid_question(bank, names[0], names[1:], seed))], evs, tag="D2")
                    recs.append(job_record(uid, r, qspec, meta))
                    if x["source"] == "N1":
                        ry = P.reid_year_request(bank, t, sym, state, stage=f"D2y:N1:{level}")
                        ry = dataclasses.replace(ry, evidence=tuple(evs))
                        recs.append(job_record(uid.replace("D2|", "D2y|", 1), ry,
                                               {"kind": "bank", "qid": "D.reid_year"}, {**meta, "question": "reid_year"}))
            print(f"D2 {period} {d}: {len(xs)} base states ({time.time() - t0:.0f}s)", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        report[period] = {"base_states": len(smp), "weeks": len(by_week), "checks": checks,
                          "by_source": pd.Series([x["source"] for x in smp]).value_counts().to_dict(),
                          "unblinder": unb.report() | {"names": None}}
    # MAIN / PRE input tokens of the N1 about_focal requests whose states were rebuilt (calibration only)
    for period in ("W_pre", "W_clean"):
        keys = {c["key"] for c in calib if c.get("gen") == period}
        r = pd.read_parquet(GENS[period] / "requests.parquet", columns=["request_key", "input_tokens"])
        tok = r[r.request_key.isin(keys) & (r.input_tokens > 0)].groupby("request_key").input_tokens.first().to_dict()
        for c in calib:
            if c.get("gen") == period:
                c["tokens"] = int(tok.get(c["key"], 0))
    report["calibration_samples"] = [c for c in calib if c.get("tokens")]
    return recs, report


def cmd_plan(a) -> int:
    bank = load_bank(run_tag=RUN)
    if bank.sha256 != FROZEN_SHA256:
        raise SystemExit("not the frozen bank")
    (OUT / "plan").mkdir(parents=True, exist_ok=True)
    data = RealData()
    d1, sched = d1_jobs(data, bank)
    print(f"D1: {len(d1)} requests over {len(sched)} months", flush=True)
    d2, rep = d2_jobs(bank)
    calib = rep.pop("calibration_samples")
    with gzip.open(OUT / "plan" / "calibration.jsonl.gz", "wt") as fh:
        for c in calib:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"D2: {len(d2)} requests; {json.dumps(rep, default=str)[:600]}", flush=True)
    recs = d1 + d2
    uids = [r["uid"] for r in recs]
    if len(set(uids)) != len(uids):
        raise SystemExit("duplicate job ids")
    with gzip.open(OUT / "plan" / "jobs.jsonl.gz", "wt") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"d1_schedule": sched, "d2": rep, "n_jobs": len(recs), "n_d1": len(d1), "n_d2": len(d2),
            "jobs_sha256": sha("\n".join(sorted(uids))), "bank_sha256": bank.sha256,
            "d2_sample": {"seed": D2_SEED, "base_per_period": N_BASE, "levels": LEVELS},
            "d1_rule": {"seed": D1_SEED, "weeks_per_month": N_WEEKS_PER_MONTH, "names": N_NAMES,
                        "rule_months": [str(m) for m in RULE_MONTHS], "wpre_months": [str(m) for m in WPRE_MONTHS]},
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / "plan" / "plan.json").write_text(json.dumps(info, indent=1, default=str))
    return 0


# ------------------------------------------------------------------------------------------ calibration
def calibration(bank) -> dict:
    """Tokens per prompt byte: D1 from DEV #2's D1 rows (bank rc7 = v1.0 content, so the prompt text is identical; each
    rebuilt key is checked against DEV #2's), D2 from the MAIN / PRE T, F-B and N1 states rebuilt by `plan`."""
    from jevtrader.s1.backends import LogprobEmulator

    emu = LogprobEmulator(client=object(), bank_hash=bank.sha256, prompt_cache=False)
    dev = ROOT / "results/judgments/dev2_emu_wdev10_s50"
    r = pd.read_parquet(dev / "requests.parquet", columns=["t", "symbol", "program", "stage", "input_tokens",
                                                          "request_key"])
    r = r[(r.program == "D") & (r.stage == "D1")]
    ed = emu_desc(dev)
    sessions = RealData().sessions
    q = {"recall_beats_median": bank.q("D.recall_beats_median").to_question()}
    s1, ok = [], 0
    for t, sym, n, k in zip(r.t, r.symbol, r.input_tokens, r.request_key):
        t = pd.Timestamp(t)
        st = {"company_name": P.company_name(sym, t), "period": P.probe_period(sessions, t)}
        ok += emu_key(ed, st, q) == k
        s1.append((st, q, int(n)))
    d1 = C.calibrate(emu, s1) | {"keys_matched": ok, "source": "DEV2 D1 rows"}
    s2 = []
    with gzip.open(OUT / "plan" / "calibration.jsonl.gz", "rt") as fh:
        for line in fh:
            c = json.loads(line)
            qs = {"about_focal": bank.q("N1.about_focal").to_question()} if c["program"] == "N1" else \
                all_questions(bank, c["program"])
            s2.append((c["state"], qs, c["tokens"]))
    d2 = C.calibrate(emu, s2) | {"source": "MAIN/PRE T, FB, N1.about_focal rows of the rebuilt base states"}
    return {"D1": d1, "D2": d2}


# ------------------------------------------------------------------------------------------ run
def shards_of(bank, recs: list[dict]) -> dict[str, list[C.Job]]:
    sh: dict[str, list[C.Job]] = {}
    for r in recs:
        m = r["meta"]
        name = f"D1_{m['month']}" if r["program"] == "D" else f"D2_{m['period']}_{m['source']}"
        sh.setdefault(name, []).append(C.Job(r["uid"], build_request(bank, r), m))
    return dict(sorted(sh.items()))


def run(dry: bool, concurrency: int) -> int:
    bank = load_bank(run_tag=RUN)
    recs = load_plan()
    plan = json.loads((OUT / "plan" / "plan.json").read_text())
    fp = {"gen_id": GEN_ID, "bank": bank.sha256, "jobs_sha256": plan["jobs_sha256"], "backend": "llm-logprob/emu-v1"}
    if dry:
        scratch = SCRATCH / f"{GEN_ID}_dry_{int(time.time())}"
        calib = calibration(bank)
        r1, r2 = calib["D1"]["ratio"], calib["D2"]["ratio"]
        fake = C.FakeOpenAI(ratio=lambda system, user: r1 if '"company_name"' in user.split("STATE:", 1)[-1] else r2)
        s1 = C.make_s1("llm-logprob", True, scratch, fake)
        out = scratch / GEN_ID
    else:
        proj = json.loads((OUT / "projection.json").read_text())
        pf = C.preflight(RUN, proj["projected_usd"], proj["high_usd"], f"D1 {plan['n_d1']} + D2 {plan['n_d2']} requests")
        if not pf.ok:
            return 3
        s1 = C.make_s1("llm-logprob", False)
        out = OUT
    g = C.guarded(s1, RUN, float("inf") if dry else pf.guard)
    shards = shards_of(bank, recs)
    t0 = time.time()
    st = C.run_shards(out, shards, g, RUN, "archive-a377a4d6abd3bfe3", fp, concurrency)
    s1.ledger.flush()
    info = {**st, "dry": dry, "process_spent_usd": g.spent, "process_calls": g.calls, "refused": g.refused,
            "wall_s": round(time.time() - t0, 1)}
    C.launch_record(out, info)
    print(json.dumps(info), flush=True)
    if dry:
        rec = C.projection_record(OUT / "projection.json", RUN, fake, s1,
                                  {"requests": len(recs), "dry_dir": str(scratch), "calibration": calib})
        print(json.dumps({k: rec[k] for k in ("paid_calls", "projected_usd", "high_usd", "by_tag")}), flush=True)
    if st["status"] != "complete":
        return 3
    man = C.finalize(out, list(shards), {"run": RUN, "backend": s1.describe(), "plan": {k: plan[k] for k in (
        "n_jobs", "n_d1", "n_d2", "jobs_sha256")}, "tags": ["PROBE/D", "PROBE/D2"], "dry": dry})
    print(f"finalized {man['rows']} validity {man['validity_rate']}", flush=True)
    return 0


def cmd_validate(a) -> int:
    recs = {r["uid"]: r for r in load_plan()}
    jobs = pd.read_parquet(OUT / "jobs.parquet")
    req = pd.read_parquet(OUT / "requests.parquet")
    ans = pd.read_parquet(OUT / "answers.parquet")
    man = json.loads((OUT / "MANIFEST.json").read_text())
    led = C.ledger_run_rows(RUN)
    paid = ~req.cached.astype(bool)
    chk = {
        "complete": set(jobs.uid) == set(recs) and len(jobs) == len(recs),
        "one_answer_per_request": len(ans) == len(req) and ans.request_key.nunique() == req.request_key.nunique(),
        "pinned": bool((req.model_pinned == C.MODEL).all() and (req.response_model[paid] == C.MODEL).all()
                       and (req.prompt_version == "emu-v1").all() and (req.bank_hash == FROZEN_SHA256).all()),
        "tags": sorted(req.tag.unique()) == ["PROBE/D", "PROBE/D2"],
        "validity_ge_98pct": bool(1 - ans.probs_json.isna().mean() >= 0.98),
        "ledger_covers_paid_rows": led["usd"] + 1e-9 >= float(req.usd[paid].sum()),
    }
    j = jobs.merge(req[["request_key", "degraded"]].drop_duplicates("request_key"), on="request_key", how="left")
    meta = pd.json_normalize([json.loads(x) for x in j.meta_json])
    summ = {"requests": len(req), "paid": int(paid.sum()), "cached": int((~paid).sum()),
            "degraded_requests": int((req.degraded > 0).sum()), "d1": int((req.program == "D").sum()),
            "d2": int((req.program == "D2").sum()),
            "by_stage": req.groupby("stage").size().to_dict(),
            "d2_by_period_level_source": {"|".join(k): int(v) for k, v in meta[meta.source != "D1"].groupby(
                ["period", "level", "source"]).size().items()} if "level" in meta else {},
            "paid_usd_rows": float(req.usd[paid].sum()), "ledger": led}
    res = {"gen_id": GEN_ID, "checks": chk, "pass": all(chk.values()), "summary": summ,
           "manifest_sha256": man["sha256"], "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (OUT / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"pass": res["pass"], "checks": chk, "paid": summ["paid"], "ledger_usd": led["usd"]}, default=str))
    return 0 if res["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "dry", "run", "validate"])
    ap.add_argument("--concurrency", type=int, default=32)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "dry":
        return run(True, a.concurrency)
    if a.cmd == "run":
        return run(False, a.concurrency)
    return cmd_validate(a)


if __name__ == "__main__":
    sys.exit(main())
