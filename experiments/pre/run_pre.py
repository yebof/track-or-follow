"""PRE Layer-A run (design_v1 §13.2 PRE): the W_pre slice (2023-04-07..2024-05-31) x S30, blinded and unblinded.

    .venv/bin/python experiments/pre/run_pre.py --arm blinded   --backend llm-logprob --max-usd 45
    .venv/bin/python experiments/pre/run_pre.py --arm unblinded --backend llm-logprob --max-usd 45
    .venv/bin/python experiments/pre/run_pre.py --arm unblinded --backend mock --n-weeks 3 --out-root <scratch>

Both arms are the weekly decision run of `experiments/dev/run_dev.py` (programs M, T, FA, FB, P1, PX and the
discovered news programs N01, N23 and the code-only LM reader; no P2, no baseline readers, no diagnostics), on the S30
subset of U_t, through the same resumable runner (`programs.runner.generate`; `--no-resume` starts over), the response
cache and the spend ledger. The ledger run tag is `PRE/B` (blinded) or `PRE/U` (unblinded): both count against the
`PRE` sub-cap of configs/backends.yaml (a hard stop across processes; this script refuses to start without it) and both
are frozen-bank runs (`bank_v1.FROZEN_RUN_TAGS` matches "PRE"). `--max-usd` bounds this process's own spend as well
(run_dev's SpendGuard).

The unblinded arm (`experiments/pre/unblind.py`) runs the same blinded pipeline and reveals firm identity at the
backend boundary: study-firm code names in states and questions become the SEC registrant names valid at d_t, and the
note's code-name clause is dropped. Its variant set and context description carry `unblind.VERSION`, so a blinded
shard is never resumed into it (and vice versa).

No window may touch W_clean (2024-07-05..2026-09-11): the script refuses it, whatever the arguments.

Outputs: results/judgments/<gen_id>/ (judgment tables, MANIFEST.json, memory/, availability.json, risk.parquet) plus
`pre_run.json` (arguments, context, guard spend, unblinder report). `--state-log` writes every request actually sent
(request key, tag, cached, state text, question texts) to a gzipped JSONL, for the arm checks (scratch, not committed).
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import json
import logging
import sqlite3
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "dev"))
sys.path.insert(0, str(ROOT / "experiments" / "pre"))

import run_dev  # noqa: E402
import unblind as UB  # noqa: E402

from jevtrader.programs.context import ContextBuilder, RealData  # noqa: E402
from jevtrader.programs.runner import GeneratorSpec, generate, resolve_program  # noqa: E402
from jevtrader.s1.types import canonical_json, question_to_json  # noqa: E402

log = logging.getLogger("run_pre")
RUN = "PRE"
ARM_TAG = {"blinded": "PRE/B", "unblinded": "PRE/U"}
GEN_ID = {"blinded": "pre_blinded_emu_wpre_s30", "unblinded": "pre_unblinded_emu_wpre_s30"}
PAID = {"llm-logprob", "llm-verbalized", "jev"}


class UnblindedContextBuilder(ContextBuilder):
    """The S30 ContextBuilder whose description names the unblinding (resume never mixes the arms)."""

    def describe(self):
        return {**super().describe(), "unblinding": UB.VERSION}


class StateLog:
    """Backend proxy that logs what is sent (after any unblinding): request key, tag, cached, state and questions."""

    def __init__(self, s1, path: Path):
        self._s1, self._lock = s1, threading.Lock()
        self._fh = gzip.open(path, "at", encoding="utf-8")
        self.n = 0

    def __getattr__(self, k):
        return getattr(self._s1, k)

    async def ask(self, state, questions, tag: str = "", **kw):
        resp = await self._s1.ask(state, questions, tag=tag, **kw)
        from jevtrader.s1.backends import state_text

        rec = {"request_key": resp.request_key, "tag": tag, "cached": bool(resp.cached), "state": state_text(state),
               "questions": canonical_json([[n, question_to_json(q)] for n, q in questions.items()])}
        with self._lock:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.n += 1
        return resp

    def close(self):
        with self._lock:
            self._fh.close()


def run_spent(ledger_path: Path, prefix: str) -> float:
    with sqlite3.connect(ledger_path) as c:
        return float(c.execute("SELECT COALESCE(SUM(usd),0) FROM spend WHERE tag LIKE ? OR tag = ?",
                               (prefix + "/%", prefix)).fetchone()[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm", required=True, choices=sorted(ARM_TAG))
    ap.add_argument("--window", default="W_pre_slice", help="a configs/system_v1.yaml window or START:END")
    ap.add_argument("--n-weeks", type=int, default=None, help="first n weeks only (rehearsal)")
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--max-usd", type=float, default=None)
    ap.add_argument("--gen-id", default=None)
    ap.add_argument("--out-root", default=str(ROOT / "results" / "judgments"))
    ap.add_argument("--state-log", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-parallel-units", type=int, default=16)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    paid = a.backend in PAID
    if paid and a.max_usd is None:
        raise SystemExit(f"--backend {a.backend} is paid: give --max-usd")
    from jevtrader.s1.backends import load_backend_config, make_backend

    cfg = load_backend_config()
    ledger_path = ROOT / cfg["ledger"]["path"]
    cap = float(cfg["ledger"]["sub_caps_usd"][RUN])  # KeyError if the PRE sub-cap is gone: refuse to run
    before_run, before_arm = run_spent(ledger_path, RUN), run_spent(ledger_path, ARM_TAG[a.arm])
    print(f"{RUN} spent before ${before_run:.4f} (arm {ARM_TAG[a.arm]} ${before_arm:.4f}); ledger sub-cap {RUN} "
          f"${cap:.2f}", flush=True)
    t0 = time.time()
    data = RealData()
    ns = argparse.Namespace(weeks=None, window=a.window, n_weeks=a.n_weeks, allow_wclean=False)
    weeks = run_dev.resolve_weeks(ns, data.sessions)
    programs = run_dev.resolve_programs(argparse.Namespace(programs="all", with_p2=False))
    run_dev.check_wclean(weeks, programs, data.sessions, allow=False)  # never on W_clean, whatever the arguments
    data.fact_symbols = data.symbols_for_weeks(weeks)
    builder = (UnblindedContextBuilder if a.arm == "unblinded" else ContextBuilder)(data, subset="S30")
    context = builder.describe()
    context_id = hashlib.sha256(canonical_json(context).encode()).hexdigest()[:10]
    s1 = make_backend(backend=a.backend, cache=paid, ledger=paid)
    guard = run_dev.SpendGuard(s1, a.max_usd if a.max_usd is not None else float("inf"))
    front = guard
    slog = None
    if a.state_log:
        slog = front = StateLog(front, Path(a.state_log))
    unblinder = None
    objects = None
    variant: frozenset[str] = frozenset()
    if a.arm == "unblinded":
        from jevtrader.programs.bank_v1 import load_bank
        from jevtrader.programs.probes import company_name

        code_to_symbol = {v: k for k, v in data.blinder.code_names.items()}
        unblinder = UB.Unblinder(code_to_symbol, company_name, UB.blinded_notes(load_bank()))
        front = UB.UnblindingBackend(front, unblinder)
        objects = UB.week_scoped(programs, resolve_program)
        variant = frozenset({UB.VERSION})
    gen_id = a.gen_id or GEN_ID[a.arm]
    spec = GeneratorSpec(gen_id=gen_id, weeks=weeks, programs=programs, run_tag=ARM_TAG[a.arm], variant=variant,
                         data_freeze=data.archive_id(), backend=a.backend, out_root=Path(a.out_root),
                         max_parallel_units=a.max_parallel_units,
                         notes=f"experiments/pre/run_pre.py; arm={a.arm}; subset=S30; window={a.window}; "
                               f"context={context_id}", allow_missing_deps=True, context_factory=builder,
                         program_objects=objects, s1=front)
    print(f"{gen_id}: arm {a.arm}, {len(weeks)} week(s) {weeks[0].date()}..{weeks[-1].date()}, programs {programs}, "
          f"backend {a.backend}, tag {ARM_TAG[a.arm]}", flush=True)
    err = None
    try:
        out = generate(spec, resume=not a.no_resume)
    except BaseException as e:  # noqa: BLE001 (the spend summary is printed whatever happened)
        err = e
    finally:
        if slog is not None:
            slog.close()
        if hasattr(s1, "close"):
            s1.close()
    after_run, after_arm = run_spent(ledger_path, RUN), run_spent(ledger_path, ARM_TAG[a.arm])
    summary = {"arm": a.arm, "gen_id": gen_id, "run_tag": ARM_TAG[a.arm], "guard_spend_usd": guard.spent,
               "guard_calls": guard.calls, "guard_refused": guard.refused,
               "ledger_this_process_usd": after_arm - before_arm, "ledger_arm_total_usd": after_arm,
               "ledger_PRE_total_usd": after_run, "PRE_sub_cap_usd": cap, "wall_s": round(time.time() - t0, 1),
               "error": None if err is None else f"{type(err).__name__}: {err}"}
    print(json.dumps(summary, indent=1), flush=True)
    if err is not None:
        raise err
    (out / "availability.json").write_text(json.dumps([r.to_json() for _, r in sorted(builder.reports.items())],
                                                      indent=1))
    from jevtrader.programs.sentinel import write_risk_table

    write_risk_table(out)
    (out / "pre_run.json").write_text(json.dumps({
        "args": vars(a), "context": context, "context_id": context_id, "programs": programs,
        "weeks": [str(w.date()) for w in weeks], "summary": summary,
        "unblinder": unblinder.report() if unblinder is not None else None,
        "state_log_requests": slog.n if slog is not None else None}, indent=1, default=str))
    m = json.loads((out / "MANIFEST.json").read_text())
    print(json.dumps({k: m[k] for k in ("counts", "validity_rate", "spend_usd", "tokens", "wall_time_s")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
