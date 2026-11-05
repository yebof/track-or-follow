"""Inputs of the reduced TAL/MCOT generations (2026-09-28 budget re-plan): S30 x the LAST 60 weekly decisions of
W_clean, built exactly as `baselines.llm.build_inputs` builds them (blinded, point in time), with the T state's
market regime read from MAIN's M-work blackboard (M.regime of the same week, as the typed T program saw it).

Free: no API call, no ledger. Reads MAIN's memory (work layer only, from a private copy so nothing is written to the
generation) and the archive. Never reads a return, label or any W_clean outcome: the blocks are the Layer-A inputs.

    .venv/bin/python experiments/gens/tal_inputs.py [--weeks N] [--out data/cache/gens/tal_inputs.jsonl.gz]

Output: one JSON line per stock-week {t, symbol, code_name, sources, fundamentals, technical, news, macro_policy,
sha256}; a stock-week already in the file is not rebuilt (resumable, append-only).
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.baselines import llm as L  # noqa: E402
from jevtrader.memory.store import MemoryStore  # noqa: E402
from jevtrader.programs.bank_v1 import load_bank  # noqa: E402
from jevtrader.programs.context import ContextBuilder, RealData, decision_dates  # noqa: E402

W_CLEAN = ("2024-07-05", "2026-09-11")  # 115 decisions (design_v1 §2.5)
N_LAST = 60
MAIN_MEMORY = ROOT / "results/judgments/main_emu_wclean_u100/memory"
DEFAULT_OUT = ROOT / "data/cache/gens/tal_inputs.jsonl.gz"
BLOCKS = ("fundamentals", "technical", "news", "macro_policy")
log = logging.getLogger("tal_inputs")


def last_weeks(sessions, n: int = N_LAST) -> list[pd.Timestamp]:
    w = decision_dates(sessions, pd.Timestamp(W_CLEAN[0]), pd.Timestamp(W_CLEAN[1]))
    if len(w) != 115:
        raise SystemExit(f"W_clean has {len(w)} decisions, expected 115")
    return list(w[-n:])


def sw_sha(rec: dict) -> str:
    blob = json.dumps({k: rec[k] for k in ("t", "symbol", "code_name", *BLOCKS)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def read_inputs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=N_LAST, help="the last N weeks of W_clean (default 60)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["t"], r["symbol"]) for r in read_inputs(out)}
    bank = load_bank()
    t0 = time.time()
    data = RealData()
    build = ContextBuilder(data, subset="S30")
    weeks = last_weeks(data.sessions, args.weeks)
    print(f"weeks {weeks[0].date()}..{weeks[-1].date()} ({len(weeks)}); already built {len(done)} stock-weeks")
    tmp = Path(tempfile.mkdtemp(prefix="tal_mem_"))
    shutil.copy2(MAIN_MEMORY / "work.jsonl", tmp / "work.jsonl")  # private copy: MAIN's memory is never written
    store = MemoryStore(tmp)
    n_new = 0
    with gzip.open(out, "at", encoding="utf-8") as fh:
        for t in weeks:
            ctx0 = build(t)
            view = store.view(t, ctx0.cutoff_utc, deps=["M"], history=[], readable={"M"}, clearance="T")
            if view.work("M.regime", max_taint="T") is None:
                raise SystemExit(f"{t.date()}: no M.regime on MAIN's blackboard")
            ctx = dataclasses.replace(ctx0, memory=view)
            names = ctx.stock_names()
            if len(names) != 30:
                raise SystemExit(f"{t.date()}: S30 has {len(names)} names")
            for sym in names:
                if (str(t.date()), sym) in done:
                    continue
                x = L.build_inputs(ctx, sym, bank)
                rec = {"t": str(t.date()), "symbol": sym, "code_name": x.code_name, "sources": x.sources,
                       **{b: x.block(b) for b in BLOCKS}}
                rec["sha256"] = sw_sha(rec)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_new += 1
            fh.flush()
            print(f"{t.date()} done ({time.time() - t0:.0f}s)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"built {n_new} stock-weeks in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
