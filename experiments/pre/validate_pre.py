"""Operational validation of a PRE generation (no outcome is read: no price after any cutoff, no label, no return).

    .venv/bin/python experiments/pre/validate_pre.py <gen_dir> [<gen_dir> ...] [--out results/pre/validation.json]

Checks, per generation (each is PASS or FAIL; the run is valid when all pass):
- `frozen`: MANIFEST bank = FROZEN_SHA256; every request row carries it; the pinned model answered every paid row;
  one data freeze; the run tag is PRE/B or PRE/U; no decision week touches W_clean.
- `weeks`: the 60 decisions of W_pre_slice (2023-04-14..2024-05-31), every (program, week) shard DONE.
- `focus`: the stock names asked each week are exactly S30 of U_t (`subset_names`, salt S30-www27).
- `rows`: every (week, symbol, program) that should exist does exist:
    M one request per week; P1 >= 1 request per week and a `P1.week` blackboard entry; T and FB one request per focus
    stock-week; LM 1 feature, PX 2 features (s_pol, s_ind), N01 2 features (h, news_count) and N23 12 features per
    focus stock-week; the blackboard entries N01.items / N01.sentinel / N23.card per focus stock-week; FA: every focus
    stock-week whose latest usable 10-K exists has an M-sem profile of that filing usable before the cutoff.
- `items`: every news item N01 judged has its N0 request; every item N0 did not quarantine has its three stage-1 N1
  requests; the stage-2 requests (new_development, attributes) exist iff the stage-2 gate passed (P(about) >= .5 and a
  non-abstaining P(instructs_reader) < .5); every request key an item or card lists is in requests.parquet; an N3
  request exists iff the stock-week has a kept cluster; N2 requests match the cards' N2 provenance.
- `validity`: share of non-degraded answers >= 98% for every program (decision and diagnostic rows alike).
- `retries`: every request with a degraded answer made 1 + degraded_retries attempts (configs/backends.yaml).
Reported (not gated): requests, cached, spend, tokens, latency p50/p95, degraded answers by question, N01 errors,
availability notes, and, for a pair of arms, the identity of item sets and request structure across arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.programs.bank_v1 import FROZEN_SHA256  # noqa: E402
from jevtrader.programs.context import SUBSETS, TenKStore, subset_names, universe_asof  # noqa: E402
from jevtrader.data.news import decision_cutoff  # noqa: E402

W_PRE = ("2023-04-07", "2024-05-31")
W_CLEAN = (pd.Timestamp("2024-07-05"), pd.Timestamp("2026-09-11"))
N23_FEATURES = {"f_imp", "f_cont", "f_rev", "f_sur", "f_net", "f_und", "f_story", "f_cov", "no_news", "news_view",
                "f_med", "f_vote"}
MIN_VALIDITY = 0.98


def _bb(gen: Path) -> dict[tuple[str, pd.Timestamp], dict]:
    out = {}
    for line in (gen / "memory" / "work.jsonl").read_text().splitlines():
        r = json.loads(line)["record"]
        out[(r["key"], pd.Timestamp(r["t"]).tz_localize(None).normalize())] = r["value"]
    return out


def _sem(gen: Path) -> list[dict]:
    p = gen / "memory" / "sem.jsonl"
    return [json.loads(x)["record"] for x in p.read_text().splitlines()] if p.exists() else []


def validate(gen: Path, partial: bool = False) -> dict:
    """`partial`: a rehearsal on the first weeks of the window (the week check accepts a prefix of W_pre_slice)."""
    gen = Path(gen)
    res: dict = {"gen_dir": str(gen), "checks": {}, "report": {}}
    chk, rep = res["checks"], res["report"]
    man = json.loads((gen / "MANIFEST.json").read_text())
    req = pd.read_parquet(gen / "requests.parquet")
    ans = pd.read_parquet(gen / "answers.parquet")
    feats = pd.read_parquet(gen / "features.parquet")
    for df in (req, ans, feats):
        t = pd.to_datetime(df["t"])
        df["t"] = (t.dt.tz_convert(None) if t.dt.tz is not None else t).astype("datetime64[ns]")
    cfg = yaml.safe_load((ROOT / "configs/backends.yaml").read_text())
    emu = cfg["backends"]["llm-logprob"]
    paid = req[~req["cached"].astype(bool)]
    weeks = sorted(pd.to_datetime(man["spec"]["weeks"]))
    tag = man["spec"]["run_tag"]

    # ---- frozen identity
    bad = []
    if man["bank"]["sha256"] != FROZEN_SHA256:
        bad.append(f"MANIFEST bank {man['bank']['sha256'][:12]} != frozen")
    if set(req["bank_hash"]) != {FROZEN_SHA256}:
        bad.append(f"bank_hash values {sorted(set(req['bank_hash']))[:3]}")
    if tag not in ("PRE/B", "PRE/U"):
        bad.append(f"run tag {tag}")
    if not req["tag"].astype(str).str.startswith(tag + "/").all():
        bad.append("a request tag outside the run tag")
    if len(paid) and not ((paid["response_model"] == emu["model"]) & (paid["model_pinned"] == emu["model"])).all():
        bad.append("a paid row answered by another model")
    if req["data_freeze"].nunique() != 1:
        bad.append(f"{req['data_freeze'].nunique()} data freezes")
    if any(W_CLEAN[0] <= w <= W_CLEAN[1] for w in weeks) or req["t"].max() >= W_CLEAN[0]:
        bad.append("a W_clean week")
    chk["frozen"] = {"pass": not bad, "problems": bad, "bank": man["bank"], "run_tag": tag,
                     "data_freeze": sorted(set(req["data_freeze"]))}

    # ---- weeks and shards
    from jevtrader.programs.context import decision_dates

    bars = pd.read_parquet(ROOT / "data/raw/bars/sp500_daily_2018_2026.parquet", columns=["date"])
    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(bars["date"]).unique()))
    want_weeks = decision_dates(sessions, *W_PRE)
    shards = gen / "shards"
    missing_units = [f"{p}/{w.date()}" for p in man["spec"]["programs"] for w in weeks
                     if not (shards / p / str(w.date()) / "DONE").exists()]
    ok_weeks = weeks == want_weeks[: len(weeks)] if partial else weeks == want_weeks
    chk["weeks"] = {"pass": ok_weeks and not missing_units, "partial": partial, "n_weeks": len(weeks),
                    "first": str(weeks[0].date()), "last": str(weeks[-1].date()), "expected": len(want_weeks),
                    "units": len(man["spec"]["programs"]) * len(weeks), "missing_units": missing_units[:20]}

    # ---- focus
    uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
    uni["date"] = pd.to_datetime(uni["date"])
    salt, n = SUBSETS["S30"]
    focus = {w: set(subset_names(universe_asof(uni, w), n, salt)) for w in weeks}
    asked = req[req["program"] == "T"].groupby("t")["symbol"].apply(set).to_dict()
    bad_focus = [str(w.date()) for w in weeks if asked.get(w, set()) != focus[w]]
    chk["focus"] = {"pass": not bad_focus and all(len(v) == n for v in focus.values()), "weeks_mismatch": bad_focus,
                    "stock_weeks": int(sum(len(v) for v in focus.values()))}

    # ---- rows
    bb = _bb(gen)
    probs = []
    sw = [(w, s) for w in weeks for s in sorted(focus[w])]

    def per_sw(df: pd.DataFrame, program: str) -> Counter:
        d = df[df["program"] == program]
        return Counter(zip(d["t"], d["symbol"]))

    for prog in ("T", "FB"):
        c = per_sw(req, prog)
        miss = [(str(w.date()), s) for w, s in sw if c.get((w, s), 0) != 1]
        extra = sum(1 for k in c if (k[0], k[1]) not in set(sw))
        if miss or extra:
            probs.append(f"{prog}: {len(miss)} focus stock-weeks without exactly one request {miss[:3]}, {extra} extra")
    m = req[req["program"] == "M"].groupby("t").size()
    if not all(m.get(w, 0) == 1 for w in weeks):
        probs.append("M: not one request per week")
    p1 = req[req["program"] == "P1"].groupby("t").size()
    if not all(p1.get(w, 0) >= 1 for w in weeks) or not all(("P1.week", w) in bb for w in weeks):
        probs.append("P1: a week without P1 requests or without its P1.week entry")
    fcount = {prog: feats[feats["program"] == prog].groupby(["t", "symbol"])["feature"].apply(set).to_dict()
              for prog in ("LM", "PX", "N01", "N23")}
    want_f = {"LM": {"lm_neg"}, "PX": {"s_pol", "s_ind"}, "N01": {"h", "news_count"}, "N23": N23_FEATURES}
    for prog, want in want_f.items():
        miss = [(str(w.date()), s) for w, s in sw if fcount[prog].get((w, s)) != want]
        if miss:
            probs.append(f"{prog}: {len(miss)} focus stock-weeks without exactly {sorted(want)} {miss[:3]}")
    for key in ("N01.items.", "N01.sentinel.", "N23.card."):
        miss = [(str(w.date()), s) for w, s in sw if (key + s, w) not in bb]
        if miss:
            probs.append(f"blackboard {key}*: {len(miss)} focus stock-weeks missing {miss[:3]}")
    # F-A: the latest usable 10-K of every focus stock-week is profiled in M-sem before the cutoff
    tenk = TenKStore()
    sem = defaultdict(list)
    for r in _sem(gen):
        sem[r["symbol"]].append(r)
    fa_miss, fa_have, no10k = [], 0, 0
    for w, s in sw:
        f = tenk.latest(s, w)
        if f is None:
            no10k += 1
            continue
        c = decision_cutoff(w)
        ok = any(r.get("accession") == f.accession and pd.Timestamp(r["usable_from"]) < c for r in sem[s])
        fa_have += ok
        if not ok:
            fa_miss.append((str(w.date()), s, f.accession))
    if fa_miss:
        probs.append(f"FA: {len(fa_miss)} focus stock-weeks whose latest 10-K has no M-sem profile {fa_miss[:3]}")
    chk["rows"] = {"pass": not probs, "problems": probs, "fa_profiled_stock_weeks": fa_have,
                   "stock_weeks_without_10k": no10k}

    # ---- items, N2, N3
    keys = set(req["request_key"])
    iprobs, n_items, n_stage2, n_n0q, n_err = [], 0, 0, 0, 0
    item_sets = {}
    n3 = per_sw(req, "N3")
    n2 = per_sw(req, "N2")
    for w, s in sw:
        items = bb.get(("N01.items." + s, w)) or []
        item_sets[(w, s)] = tuple(sorted(it["eid"] for it in items))
        n_err += len(bb.get(("N01.errors." + s, w)) or [])
        for it in items:
            n_items += 1
            rk = it.get("request_keys") or {}
            if "N0" not in rk:
                iprobs.append(f"{w.date()} {s} {it['eid']}: no N0 request")
            if it.get("n0_quarantined"):
                n_n0q += 1
            else:
                for g in ("N1.about_focal", "N1.instructs_reader", "N1.promotional"):
                    if g not in rk:
                        iprobs.append(f"{w.date()} {s} {it['eid']}: no {g}")
                p = it.get("p") or {}
                a, g = p.get("about"), p.get("instructs")
                gate = a is not None and a >= 0.5 and g is not None and g < 0.5
                has2 = "N1.new_development" in rk and "N1.attributes" in rk
                n_stage2 += has2
                if gate != has2:
                    iprobs.append(f"{w.date()} {s} {it['eid']}: stage-2 gate {gate} but stage-2 requests {has2}")
            lost = [k for k in rk.values() if k not in keys]
            if lost:
                iprobs.append(f"{w.date()} {s} {it['eid']}: {len(lost)} listed request keys not in requests.parquet")
        card = bb.get(("N23.card." + s, w)) or {}
        clusters = card.get("clusters") or []
        prov = card.get("provenance") or {}
        if bool(clusters) != (n3.get((w, s), 0) == 1) or n3.get((w, s), 0) > 1:
            iprobs.append(f"{w.date()} {s}: {len(clusters)} clusters but {n3.get((w, s), 0)} N3 requests")
        if prov.get("N3") and prov["N3"] not in keys:
            iprobs.append(f"{w.date()} {s}: card N3 key not in requests.parquet")
        n2_keys = prov.get("N2") or []
        if len(n2_keys) != n2.get((w, s), 0) or any(k not in keys for k in n2_keys):
            iprobs.append(f"{w.date()} {s}: card lists {len(n2_keys)} N2 keys, {n2.get((w, s), 0)} N2 requests")
    chk["items"] = {"pass": not iprobs, "problems": iprobs[:30], "n_problems": len(iprobs), "items": n_items,
                    "n0_quarantined": n_n0q, "stage2_items": n_stage2, "n01_item_errors": n_err,
                    "n2_requests": int(sum(n2.values())), "n3_requests": int(sum(n3.values()))}
    res["_item_sets"] = item_sets

    # ---- validity per program
    ans["valid"] = ans["probs_json"].notna()
    by = ans.groupby("program")["valid"].agg(["size", "sum"])
    by["validity"] = by["sum"] / by["size"]
    low = by[by["validity"] < MIN_VALIDITY]
    chk["validity"] = {"pass": low.empty, "min_required": MIN_VALIDITY,
                       "per_program": {p: {"answers": int(r["size"]), "degraded": int(r["size"] - r["sum"]),
                                           "validity": round(float(r["validity"]), 6)} for p, r in by.iterrows()},
                       "overall": round(float(ans["valid"].mean()), 6)}
    deg_q = ans[~ans["valid"]].groupby(["program", "qid"]).size()
    rep["degraded_by_question"] = {f"{p}:{q}": int(v) for (p, q), v in deg_q.items()}

    # ---- retries of degraded answers
    need = 1 + int(emu["degraded_retries"])
    deg = req[req["degraded"] > 0]
    short = deg[deg["attempts"] < need]
    chk["retries"] = {"pass": short.empty, "required_attempts": need, "requests_with_degraded": int(len(deg)),
                      "of_which_paid": int((~deg["cached"].astype(bool)).sum()),
                      "short_of_retries": int(len(short)),
                      "attempts": {int(k): int(v) for k, v in deg["attempts"].value_counts().items()}}

    # ---- report
    rep["requests"] = int(len(req))
    rep["requests_paid"] = int(len(paid))
    rep["requests_cached_or_coalesced"] = int(req["cached"].astype(bool).sum())
    rep["spend_usd"] = round(float(paid["usd"].sum()), 4)
    rep["tokens"] = {k: int(req[k].sum()) for k in ("input_tokens", "cached_tokens", "output_tokens")}
    rep["latency_s_paid"] = {"p50": round(float(paid["latency_s"].median()), 3) if len(paid) else None,
                             "p95": round(float(paid["latency_s"].quantile(0.95)), 3) if len(paid) else None,
                             "max": round(float(paid["latency_s"].max()), 3) if len(paid) else None}
    rep["per_program"] = {p: {"requests": int(len(g)), "paid": int((~g["cached"].astype(bool)).sum()),
                              "usd": round(float(g.loc[~g["cached"].astype(bool), "usd"].sum()), 4),
                              "in_tok_per_req": round(float(g["input_tokens"].mean()), 1),
                              "degraded_requests": int((g["degraded"] > 0).sum())}
                          for p, g in req.groupby("program")}
    rep["usage_estimated"] = int(req["usage_estimated"].astype(bool).sum())
    rep["manifest"] = {k: man[k] for k in ("counts", "validity_rate", "spend_usd", "tokens", "wall_time_s")}
    av = gen / "availability.json"
    if av.exists():
        a = json.loads(av.read_text())
        rep["availability"] = {"weeks": len(a), "hard": sum(bool(x["hard"]) for x in a),
                               "stale": sum(bool(x["stale"]) for x in a),
                               "soft_notes": Counter(m for x in a for m in x["soft"]).most_common(5)}
    res["pass"] = all(v["pass"] for v in chk.values())
    return res


def compare_arms(b: dict, u: dict) -> dict:
    """The arms share everything upstream of the model: weeks, focus names and the item set of every stock-week."""
    ib, iu = b["_item_sets"], u["_item_sets"]
    same_keys = set(ib) == set(iu)
    diff = [k for k in ib if ib.get(k) != iu.get(k)]
    return {"pass": same_keys and not diff, "stock_weeks": len(ib), "item_sets_differ": len(diff),
            "examples": [f"{k[0].date()} {k[1]}" for k in diff[:5]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gens", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--partial", action="store_true", help="rehearsal: a prefix of the window's weeks")
    a = ap.parse_args(argv)
    out = {}
    res = [validate(Path(g), a.partial) for g in a.gens]
    for r in res:
        out[Path(r["gen_dir"]).name] = {k: v for k, v in r.items() if not k.startswith("_")}
    if len(res) == 2:
        out["arms"] = compare_arms(*res)
    txt = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    for name, r in out.items():
        if name == "arms":
            print("arms: item sets identical" if r["pass"] else f"arms: FAIL {r}")
            continue
        print(name, "PASS" if r["pass"] else "FAIL", {k: v["pass"] for k, v in r["checks"].items()})
    return 0 if all(r.get("pass", True) for r in out.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
