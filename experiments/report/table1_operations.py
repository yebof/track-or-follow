#!/usr/bin/env python3
"""Table 1: operating profile and deployment check (PD0; outline_v1 §3 Table 1).

    .venv/bin/python experiments/report/table1_operations.py            # results/analysis/pd0.json -> paper/tables/
    .venv/bin/python experiments/report/table1_operations.py --in-dir results/analysis/W_pre_slice --out-dir X

Upper block: the question programs grouped by agent (news with the sentinel, technical, fundamentals, policy, macro)
and their total, then the comparator readers (holistic, holistic + guards, holistic + marking, the TA-style pipeline
and Mono-CoT on the S30 names, marked with a dagger). Columns: requests per stock-week; input tokens per request
(thousands); response-cache share of requests (%); $ per stock-week (x 10^-3, paid rows at the pinned model's list
price); p50 / p95 latency per paid request (s); degraded (S1) or invalid (generative) outputs (%).
Lower block (label: value): MAIN wall-clock (h), $ per weekly rebalance from the ledger (system; TA-style), re-evaluation
identical hashes, scaling per 10 names, $ per stock-week per news item, the screening side costs, and the deployment
check (decisions, pre-trade violations, shortfall, cutoff-to-order latency, $ per rebalance), "---" where not run.
Every number comes from pd0.json (experiments/analysis/pd0_operations.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _latex as L

NAME = "table1_operations"
AGENT_ROWS = (("news", "News (N0--N3, sentinel)"), ("technical", "Technical (T)"),
              ("fundamentals", "Fundamentals (F-A, F-B)"), ("policy", "Policy (P1, P2)"), ("macro", "Macro (M)"))
COMPARATORS = (("H", "Holistic reader"), ("H-guard", "Holistic + guards"), ("H-spot", "Holistic + marking"),
               ("TA-lite", "TA-style pipeline$^\\dagger$"), ("Mono-CoT", "Mono-CoT$^\\dagger$"))


def prof_row(label: str, p: dict | None, invalid: float | None = None) -> str:
    if not p:
        return L.row(label, *([L.NOT_RUN] * 6))
    lat = f"{L.num(p.get('latency_p50_s'), 2)} / {L.num(p.get('latency_p95_s'), 2)}"
    bad = invalid if invalid is not None else p.get("degraded_share")
    return L.row(label, L.num(p.get("requests_per_stock_week"), 2), L.num(p.get("input_tokens_per_request"), 2, 1e-3),
                 L.pct(p.get("response_cache_share"), 1), L.num(p.get("usd_per_stock_week"), 3, 1e3), lat,
                 L.pct(bad, 2))


def build(d: dict) -> str:
    t = d["typed"]
    body = [prof_row(lab, L.get(t, "agents", k)) for k, lab in AGENT_ROWS]
    body.append(prof_row("\\textit{Question programs, total}", t.get("total")))
    body.append("\\midrule")
    for k, lab in COMPARATORS:
        c = L.get(d, "comparators", k)
        if not c or c.get("status") == "not run":
            body.append(prof_row(lab, None))
            continue
        inv = L.get(c, "invalid_share", "invalid_final")
        body.append(prof_row(lab, c.get("total"), inv))
    body.append("\\midrule")
    led = t.get("ledger", {})
    tal = L.get(d, "comparators", "TA-lite", "ledger") or {}
    sc = d.get("scaling", {})
    scr = d.get("screening", {})
    rep = d.get("replay_determinism", {})
    dep = d.get("deployment", {})
    reeval = (f"{rep.get('identical')} of {rep.get('runs')}" if rep.get("status") != "not run" and "runs" in rep
              else L.NOT_RUN)
    lower = [
        ("Reading-layer wall-clock, all weeks (h)", L.num(led.get("wall_clock_h"), 1)),
        ("\\$ per weekly rebalance, system (ledger)", L.num(led.get("usd_per_week"), 2)),
        ("\\$ per weekly rebalance, TA-style$^\\dagger$ (ledger)", L.num(tal.get("usd_per_week"), 3)),
        ("Re-evaluation from stored judgments: identical output hashes", reeval),
        ("Requests / \\$ per week per 10 more names", f"{L.num(sc.get('requests_per_week_per_10_names'), 0)} / "
                                                         f"{L.num(sc.get('usd_per_week_per_10_names'), 3)}"),
        ("\\$ per stock-week per news item ($\\times 10^{-3}$)", L.num(sc.get("usd_per_stock_week_per_item"), 3, 1e3)),
        ("N0 / guard quarantine; promotional flag (\\%)",
         f"{L.pct(scr.get('n0_quarantine_rate'), 2)} / {L.pct(scr.get('guard_quarantine_rate_of_asked'), 2)}; "
         f"{L.pct(scr.get('promotional_rate_of_non_quarantined'), 2)}"),
        ("Sentinel escalation, stock-weeks (\\%)", L.pct(scr.get("sentinel_escalation_rate"), 2)),
        ("Deployment check: decisions", str(dep.get("n")) if dep.get("n") else L.NOT_RUN),
    ]
    for k, lab in (("pretrade_violations", "Deployment: pre-trade violations (count)"),
                   ("shortfall_bps", "Deployment: implementation shortfall (bps)"),
                   ("cutoff_to_order_h", "Deployment: cutoff-to-order latency (h)"),
                   ("usd_per_rebalance", "Deployment: \\$ per rebalance")):
        vals = [v.get(k) for v in (dep.get("decisions") or {}).values() if isinstance(v, dict)]
        vals = [v for v in vals if L.finite(v)]
        if k == "pretrade_violations":  # a count over the decisions, never a mean (audit 2026-09-28)
            lower.append((lab, str(int(round(sum(vals)))) if vals else L.NOT_RUN))
        else:  # rates, latencies and costs: the mean per decision
            lower.append((lab, L.num(sum(vals) / len(vals), 2) if vals else L.NOT_RUN))
    for lab, val in lower:
        body.append(L.row(f"\\multicolumn{{5}}{{l}}{{{lab}}}", f"\\multicolumn{{2}}{{r}}{{{val}}}"))
    header = ("& Req./sw & In tok./req (k) & Cached (\\%) & \\$/sw ($\\times10^{-3}$) & p50 / p95 (s) & "
              "Invalid (\\%) \\\\")
    note = (f"{t.get('stock_weeks')} stock-weeks in {t.get('weeks')} weeks; latency over paid requests; "
            "\\$ at the pinned model's list price; ledger rows are the spend of record.")
    return L.tabular("lrrrrrr", header, body, note)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    d = L.load("pd0", a.in_dir)
    if d is None:
        sys.exit("table1: pd0.json missing (run experiments/analysis/pd0_operations.py)")
    out = L.check_inputs([d], a.out_dir)
    p = L.write(out, NAME, build(d))
    print(f"table1 -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
