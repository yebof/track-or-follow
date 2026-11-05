"""Checks that the two PRE arms differ only in firm identity (no outcome is read).

    .venv/bin/python experiments/pre/arm_checks.py --blinded-log <b.jsonl.gz> --unblinded-log <u.jsonl.gz> \
        --out results/pre/arm_checks.json

From the state logs (every request as actually sent, `run_pre.py --state-log`) and the two generations:
- blinded arm: no revealed-name note in any request; the share of requests carrying study-firm code names;
- unblinded arm: no study-firm code name in any state or question text; the code-name note appears only in requests
  that reveal no name (sent byte-identical to the blinded arm); every study firm's name is the SEC name valid at d_t
  (names that fell back to the ticker are listed);
- M (no identity) has identical answers in both arms (the x_mac placebo of D3 rests on it);
- per program, the requests of the unblinded arm that reveal no name are response-cache hits of the blinded arm's.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "pre"))

import unblind as UB  # noqa: E402

GENS = {"blinded": ROOT / "results/judgments/pre_blinded_emu_wpre_s30",
        "unblinded": ROOT / "results/judgments/pre_unblinded_emu_wpre_s30"}


def scan(path: Path, study: set[str]) -> dict:
    n, with_code, new_note, old_note, cached = 0, 0, 0, 0, 0
    by_prog = Counter()
    by_prog_code = Counter()
    unchanged_cached = Counter()
    unchanged = Counter()
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line)
        n += 1
        prog = r["tag"].split("/")[-1]
        by_prog[prog] += 1
        text = r["state"] + r["questions"]
        codes = [c for c in UB.CODE_NAME.findall(text) if c in study]
        if codes:
            with_code += 1
            by_prog_code[prog] += 1
        has_new = UB.UNBLINDED_NOTE in r["state"]
        has_old = "replaced by code names" in r["state"]
        new_note += has_new
        old_note += has_old
        cached += bool(r["cached"])
        if not has_new:
            unchanged[prog] += 1
            unchanged_cached[prog] += bool(r["cached"])
    return {"requests": n, "with_study_code_names": with_code, "with_unblinded_note": new_note,
            "with_code_name_note": old_note, "cached_or_coalesced": cached, "by_program": dict(by_prog),
            "by_program_with_code_names": dict(by_prog_code), "without_unblinded_note_by_program": dict(unchanged),
            "without_unblinded_note_cached_by_program": dict(unchanged_cached)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blinded-log", required=True)
    ap.add_argument("--unblinded-log", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from jevtrader.programs.context import RealData

    study = set(RealData().blinder.code_names.values())
    out = {"blinded": scan(Path(a.blinded_log), study), "unblinded": scan(Path(a.unblinded_log), study)}
    rep = json.loads((GENS["unblinded"] / "pre_run.json").read_text())["unblinder"]
    names = rep.pop("names")
    sym_name = {k.split("@")[0]: v for k, v in names.items()}
    fallback = sorted(s for s, v in sym_name.items() if v == s)
    renamed = sorted({s for s in sym_name if len({v for k, v in names.items() if k.split("@")[0] == s}) > 1})
    out["unblinder"] = {**rep, "firms": len(sym_name), "ticker_fallback": fallback,
                        "renamed_within_window": {s: sorted({v for k, v in names.items() if k.split("@")[0] == s})
                                                  for s in renamed},
                        "all_caps_names": sum(1 for v in set(sym_name.values()) if v.isupper())}
    ans = {k: pd.read_parquet(g / "answers.parquet") for k, g in GENS.items()}
    m = {k: v[v["program"] == "M"].sort_values(["t", "qid"])[["t", "qid", "probs_json", "request_key"]]
         .reset_index(drop=True) for k, v in ans.items()}
    out["M_identical_across_arms"] = bool(m["blinded"].equals(m["unblinded"]))
    req = {k: pd.read_parquet(g / "requests.parquet") for k, g in GENS.items()}
    shared = set(req["blinded"]["request_key"]) & set(req["unblinded"]["request_key"])
    out["shared_request_keys"] = {"n": len(shared), "by_program": dict(Counter(
        req["unblinded"][req["unblinded"]["request_key"].isin(shared)]["program"]))}
    b, u = out["blinded"], out["unblinded"]
    out["pass"] = bool(b["with_unblinded_note"] == 0 and u["with_study_code_names"] == 0
                       and u["with_unblinded_note"] + u["with_code_name_note"] <= u["requests"]
                       and out["M_identical_across_arms"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("pass", "M_identical_across_arms")} | {
        "blinded": {k: b[k] for k in ("requests", "with_study_code_names", "with_unblinded_note")},
        "unblinded": {k: u[k] for k in ("requests", "with_study_code_names", "with_unblinded_note")}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
