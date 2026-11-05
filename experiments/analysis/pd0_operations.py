#!/usr/bin/env python3
"""PD0: the operating profile per question program and reader (prereg §4.1; design_v2_delta OPS-1..OPS-4).

    .venv/bin/python experiments/analysis/pd0_operations.py                        # MAIN + TAL (+ --reader X=dir)
    .venv/bin/python experiments/analysis/pd0_operations.py --typed pre_blinded_emu_wpre_s30 --window W_pre_slice \
        --reader "" --run-tag PRE/B

Operational metrics only (Layer-A request tables, manifests, the spend ledger read-only): no return, label or learned
coefficient, so nothing here needs the W_clean lock. Units are per stock-week and per weekly rebalance only.

**Per program** (typed generation; agents: news = N0, N1, N1-joint, N2, N3 with the sentinel inside N1; technical =
T; fundamentals = F-A, F-B; policy = P1, P2; macro = M) and **per comparator reader** (H, H-guard, H-spot as S1
readers; the TA-style pipeline and Mono-CoT by role): requests per stock-week, input tokens per request (all rows),
output tokens per paid request, the response-cache share of requests, the provider prompt-cache share of paid input
tokens, $ per stock-week (paid rows at the pinned model's list price), p50/p95 latency of paid requests, and the
degraded (S1) or invalid/abstained (generative, `gen-parse-v1`) share. The stock-week count is the number of (t, focal
name) pairs with a technical request (typed) or an answer row (generative readers).

**System rows.** Spend of record = the ledger (data/cache/spend.sqlite) by run tag: $ per weekly rebalance and per
stock-week, next to the tables' paid rows (MAIN's difference is paid work of interrupted units redone from the cache,
main_report §4.3). Wall-clock = the summed spans of the run tag's ledger activity, split where no paid call was booked
for WALL_GAP (the stop and the dry resume are gaps). Replay determinism (OPS-3): results/ops/replay_check.json if
written. Scaling (OPS-2): per-week requests and $ on the S30, S50 and U100 names of the same weeks, with the fitted
slope per 10 names (weekly MARKET requests are fixed per week), and by news-volume decile of stock-weeks with the
slope per item. Screening side costs (prereg §12): N0 quarantine, instruction-guard quarantine, promotional flags,
sentence removal (items of the M-work blackboard) and sentinel escalation (risk.parquet, h >= .5). Deployment check
(OPS-4): every results/live/<t>/deployment.json, passed through.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NAME = "pd0"
AGENTS = {"news": ("N0", "N1", "N1-joint", "N2", "N3"), "technical": ("T",), "fundamentals": ("FA", "FB"),
          "policy": ("P1", "P2"), "macro": ("M",)}
LEDGER = Path("data/cache/spend.sqlite")
WALL_GAP = pd.Timedelta(minutes=30)
SENTINEL_ESCALATION = 0.5
SUBSETS = ("S30", "S50")
DECILES = 10
DEFAULT_READERS = {"TA-lite": "results/gens/tal_ta-lite_wclean-last60_s30"}


def _q(x: pd.Series, q: float) -> float | None:
    x = pd.Series(x, dtype=float).dropna()
    return float(x.quantile(q)) if len(x) else None


def request_profile(req: pd.DataFrame, stock_weeks: int, weeks: int) -> dict:
    """The OPS-1 columns for one set of request rows."""
    paid = req[~req["cached"].astype(bool)]
    in_tok = req["input_tokens"].astype(float)
    ptok = paid["input_tokens"].astype(float)
    deg = req["degraded"].astype(float) > 0 if "degraded" in req else pd.Series(False, index=req.index)
    return {"requests": len(req), "paid": len(paid),
            "requests_per_stock_week": len(req) / stock_weeks if stock_weeks else None,
            "requests_per_week": len(req) / weeks if weeks else None,
            "input_tokens_per_request": float(in_tok.mean()) if len(req) else None,
            "output_tokens_per_paid_request": float(paid["output_tokens"].astype(float).mean()) if len(paid) else None,
            "response_cache_share": float(req["cached"].astype(bool).mean()) if len(req) else None,
            "provider_cached_share_paid": float(paid["cached_tokens"].astype(float).sum() / ptok.sum())
            if ptok.sum() > 0 else None,
            "usd_paid": float(paid["usd"].astype(float).sum()),
            "usd_per_stock_week": float(paid["usd"].astype(float).sum()) / stock_weeks if stock_weeks else None,
            "usd_per_week": float(paid["usd"].astype(float).sum()) / weeks if weeks else None,
            "latency_p50_s": _q(paid["latency_s"], 0.5), "latency_p95_s": _q(paid["latency_s"], 0.95),
            "degraded_share": float(deg.mean()) if len(req) else None}


def typed_profile(gen: Path) -> dict:
    req = pd.read_parquet(gen / "requests.parquet")
    req["t"] = pd.to_datetime(req["t"])
    tt = req[req["program"] == "T"]
    sw = int(tt[["t", "symbol"]].drop_duplicates().shape[0])
    if not sw and (gen / "features.parquet").exists():  # an S1 reader run on E_t (H, H-spot): its feature rows
        sw = int(pd.read_parquet(gen / "features.parquet", columns=["t", "symbol"]).drop_duplicates().shape[0])
    weeks = int(req["t"].nunique())
    progs = {p: request_profile(g, sw, weeks) for p, g in req.groupby("program")}
    agents = {a: request_profile(req[req["program"].isin(ps)], sw, weeks) for a, ps in AGENTS.items()}
    other = sorted(set(req["program"]) - {p for ps in AGENTS.values() for p in ps})
    return {"gen": gen.name, "stock_weeks": sw, "weeks": weeks, "programs": progs, "agents": agents,
            "unmapped_programs": other, "total": request_profile(req, sw, weeks)}


def generative_profile(gen: Path) -> dict:
    req = pd.read_parquet(gen / "requests.parquet")
    ans = pd.read_parquet(gen / "answers.parquet")
    sw = int(ans[["t", "symbol"]].drop_duplicates().shape[0])
    weeks = int(pd.to_datetime(ans["t"]).nunique())
    roles = {r: request_profile(g, sw, weeks) for r, g in req.groupby("role")} if "role" in req else {}
    inv = {c: float(ans[c].astype(float).mean()) for c in ("invalid", "invalid_final") if c in ans}
    inv_lines = {c: int(ans[c].astype(float).sum()) for c in ans.columns if c.startswith("invalid_") and c != "invalid_final"}
    return {"gen": gen.name, "stock_weeks": sw, "weeks": weeks, "roles": roles, "total": request_profile(req, sw, weeks),
            "invalid_share": inv, "invalid_lines": inv_lines,
            "decisions": ans["decision"].value_counts().to_dict() if "decision" in ans else {}}


def ledger_tag(root: Path, prefix: str) -> dict:
    """Spend of record and activity spans for a run tag (read-only)."""
    p = Path(root) / LEDGER
    if not p.exists():
        return {"status": "no ledger"}
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("select ts, tag, usd, input_tokens, output_tokens from spend where tag like ?", con,
                               params=(f"{prefix}%",))
        total = con.execute("select count(*), sum(usd) from spend").fetchone()
    finally:
        con.close()
    if not len(df):
        return {"status": f"no ledger rows for {prefix}", "ledger_total_usd": total[1]}
    ts = pd.to_datetime(df["ts"], unit="s", utc=True).sort_values()
    gaps = ts.diff() > WALL_GAP
    span_id = gaps.cumsum()
    spans = [(g.min(), g.max()) for _, g in ts.groupby(span_id)]
    wall = sum((b - a).total_seconds() for a, b in spans) / 3600.0
    return {"tag_prefix": prefix, "rows": len(df), "usd": float(df["usd"].sum()),
            "input_tokens": int(df["input_tokens"].sum()), "output_tokens": int(df["output_tokens"].sum()),
            "wall_clock_h": wall, "spans": [[a.isoformat(), b.isoformat()] for a, b in spans],
            "wall_gap_min": WALL_GAP.total_seconds() / 60, "ledger_total_usd": float(total[1]),
            "ledger_rows": int(total[0])}


def scaling(gen: Path, root: Path) -> dict:
    """OPS-2 on a typed generation (needs U100 focus for the subset slope)."""
    req = pd.read_parquet(gen / "requests.parquet", columns=["t", "symbol", "program", "cached", "usd"])
    req["t"] = pd.to_datetime(req["t"])
    req["paid_usd"] = np.where(req["cached"].astype(bool), 0.0, req["usd"].astype(float))
    stock = req[req["symbol"] != "MARKET"]
    market = req[req["symbol"] == "MARKET"].groupby("t").agg(n=("program", "size"), usd=("paid_usd", "sum"))
    sw = stock.groupby(["t", "symbol"]).agg(n=("program", "size"), usd=("paid_usd", "sum"))
    names_by_t = stock.groupby("t")["symbol"].apply(lambda s: sorted(set(s)))
    pts = []
    for t, names in names_by_t.items():
        uni = list(names)
        sets = {"all": uni, **{s: list(C.subset(uni, s, root)) for s in SUBSETS if len(uni) > int(s[1:])}}
        mk = market.loc[t] if t in market.index else pd.Series({"n": 0, "usd": 0.0})
        for label, keep in sets.items():
            g = sw.loc[t].loc[keep] if len(keep) else sw.loc[t].iloc[0:0]
            pts.append({"t": t, "set": label, "names": len(keep), "requests": float(g["n"].sum() + mk["n"]),
                        "usd": float(g["usd"].sum() + mk["usd"])})
    P = pd.DataFrame(pts)
    out = {"by_set": P.groupby("set").agg(names=("names", "mean"), requests_per_week=("requests", "mean"),
                                          usd_per_week=("usd", "mean")).to_dict("index")}
    if P["names"].nunique() >= 2:
        for col in ("requests", "usd"):
            b = np.polyfit(P["names"].to_numpy(float), P[col].to_numpy(float), 1)
            out[f"{col}_per_week_per_10_names"] = float(10 * b[0])
            out[f"{col}_per_week_intercept"] = float(b[1])
    n0 = stock[stock["program"] == "N0"].groupby(["t", "symbol"]).size().rename("items")
    d = sw.join(n0, how="left").fillna({"items": 0})
    d["decile"] = pd.qcut(d["items"].rank(method="first"), DECILES, labels=False) + 1
    dec = d.groupby("decile").agg(items=("items", "mean"), requests=("n", "mean"), usd=("usd", "mean"))
    out["news_volume_deciles"] = {int(k): v for k, v in dec.to_dict("index").items()}
    if d["items"].nunique() > 1:
        out["usd_per_stock_week_per_item"] = float(np.polyfit(d["items"], d["usd"], 1)[0])
        out["requests_per_stock_week_per_item"] = float(np.polyfit(d["items"], d["n"], 1)[0])
    return out


def screening(gen: Path) -> dict:
    """Prereg §12 side costs from the M-work item rows and the risk table (Layer-A rates)."""
    items, n_sent, n_kept = [], 0, 0
    tag = '"key": "N01.items.'
    with open(gen / "memory" / "work.jsonl") as f:
        for line in f:
            if tag not in line:
                continue
            for r in json.loads(line)["record"]["value"]:
                st = r.get("status") or {}
                items.append((bool(r.get("n0_quarantined")), bool(st.get("quarantined")), bool(st.get("promotional")),
                              bool(st.get("kept")), bool(st.get("about"))))
                n_sent += int(r.get("n_sentences") or 0)
                n_kept += len(r.get("kept_sentences") or [])
    if not items:
        return {"status": "no item rows"}
    a = np.array(items, dtype=bool)
    n0q, q, promo, kept, about = a.T
    asked = ~n0q
    guard = q & ~n0q
    nonq = ~q
    out = {"items": len(a), "n0_quarantined": int(n0q.sum()), "n0_quarantine_rate": float(n0q.mean()),
           "asked": int(asked.sum()), "guard_quarantined": int(guard.sum()),
           "guard_quarantine_rate_of_asked": float(guard.sum() / asked.sum()) if asked.any() else None,
           "promotional_rate_of_non_quarantined": float(promo[nonq].mean()) if nonq.any() else None,
           "about_focal_rate_of_asked": float(about[asked].mean()) if asked.any() else None,
           "kept": int(kept.sum()), "sentences": n_sent,
           "sentence_removal_rate": float(1 - n_kept / n_sent) if n_sent else None}
    if (gen / "risk.parquet").exists():
        r = pd.read_parquet(gen / "risk.parquet")
        h = pd.to_numeric(r["h"], errors="coerce")
        out["sentinel_escalation_rate"] = float((h >= SENTINEL_ESCALATION).mean())
        out["sentinel_threshold"] = SENTINEL_ESCALATION
        out["risk_stock_weeks"] = len(r)
    return out


def deployment(root: Path) -> dict:
    files = sorted((Path(root) / "results/live").glob("*/deployment.json"))
    return {"n": len(files), "decisions": {f.parent.name: json.loads(f.read_text()) for f in files}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--typed", default="main_emu_wclean_u100", help="typed generation id or dir")
    ap.add_argument("--run-tag", default="MAIN", help="ledger tag prefix of the typed generation")
    ap.add_argument("--reader", action="append", default=None,
                    help="LABEL=generation dir of a comparator reader (repeatable; '' = none); default: TA-lite")
    ap.add_argument("--reader-tag", action="append", default=[], help="LABEL=ledger tag prefix")
    ap.add_argument("--window", default=C.REGISTERED_WINDOW, help="label of the run's window (output location)")
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    gen = Path(a.typed) if Path(a.typed).is_dir() else root / "results/judgments" / a.typed
    readers = DEFAULT_READERS if a.reader is None else dict(x.split("=", 1) for x in a.reader if x)
    tags = {"TA-lite": "TAL/", "H": "H/", "H-guard": "HGUARD/", "H-spot": "HSPOT/", "Mono-CoT": "MCOT/",
            **dict(x.split("=", 1) for x in a.reader_tag)}
    typed = typed_profile(gen)
    led = ledger_tag(root, a.run_tag)
    if "usd" in led:
        typed["ledger"] = {**led, "usd_per_week": led["usd"] / typed["weeks"],
                           "usd_per_stock_week": led["usd"] / typed["stock_weeks"],
                           "tables_minus_ledger_usd": typed["total"]["usd_paid"] - led["usd"]}
    else:
        typed["ledger"] = led
    comps, inputs = {}, {"typed requests": gen / "requests.parquet"}
    for label, d in readers.items():
        d = Path(d) if Path(d).is_absolute() else root / d
        if not (d / "requests.parquet").exists():
            comps[label] = {"status": "not run"}
            continue
        import pyarrow.parquet as pq

        generative = "role" in pq.read_schema(d / "requests.parquet").names
        prof = generative_profile(d) if generative else typed_profile(d)
        lt = ledger_tag(root, tags.get(label, label + "/"))
        if "usd" in lt:
            prof["ledger"] = {**lt, "usd_per_week": lt["usd"] / prof["weeks"],
                              "usd_per_stock_week": lt["usd"] / prof["stock_weeks"]}
        comps[label] = prof
        inputs[f"{label} requests"] = d / "requests.parquet"
    payload = {"typed": typed, "comparators": comps, "screening": screening(gen),
               "replay_determinism": C.read_json(root / "results/ops/replay_check.json")
               if (root / "results/ops/replay_check.json").exists() else {"status": "not run"},
               "deployment": deployment(root),
               "settings": {"wall_gap_min": WALL_GAP.total_seconds() / 60, "subsets": SUBSETS, "deciles": DECILES,
                            "sentinel_escalation": SENTINEL_ESCALATION}}
    if not a.no_scaling:
        payload["scaling"] = scaling(gen, root)
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, payload, script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv)
    t = typed["total"]
    print(f"PD0 {gen.name}: {typed['stock_weeks']} stock-weeks, {t['requests_per_stock_week']:.1f} req/sw, "
          f"${t['usd_per_stock_week']:.5f}/sw (tables), p95 {t['latency_p95_s']:.2f}s -> {C.rel(path, root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
