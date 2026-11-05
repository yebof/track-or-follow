"""Completeness audit of the Layer-A generations before the pre-registration commit (2026-09-28).

    nice -n 15 .venv/bin/python experiments/prereg/audit_generations.py --out results/prereg/completeness_audit.json

Scope: MAIN (`main_emu_wclean_u100`), PRE (`pre_blinded_emu_wpre_s30`, `pre_unblinded_emu_wpre_s30`), the reduced
generations of the 2026-09-28 re-plan (TAL `tal_ta-lite_wclean-last60_s30`), the NG item sample
(`ng_typed-local_wclean_items500`, a $0 local generation; audited as far as it has progressed if still running), the
pre-commit paid generations of the reduced list (docs/impl/precommit_paid.md: PROBE `probe_emu_d1d2`, MCOT
`mcot_mono-cot_wclean-last60_s30`, VERB `verb_llm-verbalized_wclean_items2000`, FID `fid_emu_wclean_v1-150_v2-100x3`,
MEM `mem_emu_mepi-off_wclean_s30` and `mem_emu_mwork-off_wclean-last98_s30`) and the re-ask of the 60 scheduled D1
probes (`probe_emu_d1_reask`, precommit_report finding 3). Plus the spend ledger by run tag against the sub-caps.

Pre-commit paid generations (added for the commit-time re-run, prereg §17 step 2). Shared checks: `plan` (jobs =
requests, key for key), `answers` (one answer per question of every request row), `validity`, `provenance` (frozen
bank, pinned model on every paid row, backend identity and cfg_hash, data freeze = MAIN's, tag prefix, window,
manifest counts and table hashes), `spend` (the ledger rows of the generation cover its paid rows, equal to the cent).
Design checks rebuilt from primary inputs: PROBE, the registered D1 schedule (prereg §7.2) from the session calendar
and the membership table, the union of the two PROBE generations against it, and D2's base states by their hash rule
among the stock-weeks (items) MAIN/PRE-B asked; MCOT, S30 x the last 60 W_clean decisions = TAL's stock-weeks; VERB,
the aligned sample (NG's 500 + Hamilton top-up by the smallest rank, from MAIN's N01 item rows) and MAIN's N1 states;
FID, the V1 frame and sample and the V2 composition; MEM, arm 1 = MAIN's S30 N3 stock-weeks with no N2, arm 2 = one T
request per S30 stock-week of the last 98 W_clean decisions.

Independent of the operators' validators: the expected rows are rebuilt from primary inputs (the session calendar, the
PIT membership table and the S30 hash rule), not read from the generations. The operators' stored validation results
are only cross-referenced.

W_CLEAN RULE. Nothing here reads an outcome: no price, return, label, IC, Sharpe, ASR or learned coefficient. It reads
Layer-A tables (requests, answers, evidence, features, risk rows, firm info, manifests), the membership table, the
`date` column of the ETF bar file (the session calendar only), the ledger (read-only sqlite) and configuration files.
Answer probabilities are parsed only to check that they are valid distributions (validity), never compared across
implementations. `pipeline/lock.py` is not imported.

Checks per generation (PASS/FAIL; `notes` explain legitimate multiplicities):
- `weeks`: the decision dates present equal the expected ones;
- `stock_weeks`: per-stock-week programs (T, FB requests; risk, firm_info rows; 14 code features; 17 features split
  LM 1 / PX 2 / N01 2 / N23 12) cover exactly the expected (t, symbol) pairs, each once;
- `per_week`: M exactly 1 request per week, P1 >= 1 per week, P2 exactly 1 (MAIN);
- `news`: every news request has evidence; per stock-week the N1 items are a subset of the N0 items;
- `answers`: every answer's request key exists; every request key that answered has one answer per question
  (n_questions) or is degraded;
- `duplicates`: exact duplicate rows and natural-key duplicates per table, with the legitimate cases classified;
- `validity`: non-degraded answer share >= 98% per program; every non-null answer a distribution (sum 1 +- 1e-6);
  degraded requests made 1 + degraded_retries attempts;
- `provenance`: one bank hash = FROZEN_SHA256 = FREEZE.json; model pinned on every row and answering every paid row;
  prompt version; backend; one data freeze; MANIFEST cfg_hash = FREEZE.json; blinder aliases identity = FREEZE.json;
  tag prefix; window rule (MAIN inside W_clean + the pseudo-forward week; PRE outside W_clean);
- `spend`: paid USD in the tables vs the ledger for the generation's tags.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.programs.bank_v1 import FROZEN_RUN_TAGS, FROZEN_SHA256  # noqa: E402
from jevtrader.programs.context import decision_dates, subset_names, universe_asof  # noqa: E402

W_CLEAN = ("2024-07-05", "2026-09-11")
W_PRE = ("2023-04-07", "2024-05-31")
PSEUDO_FORWARD = "2026-09-18"
REPLAN_UTC = pd.Timestamp("2026-09-28T00:41:09Z")  # MAIN crossed $76.19 at 00:41:09 UTC (main_report §1)
MIN_VALIDITY = 0.98
FEATURES_PER_SW = {"LM": 1, "PX": 2, "N01": 2, "N23": 12}
N_CODE_FEATURES = 14
GEN = {
    "MAIN": ROOT / "results/judgments/main_emu_wclean_u100",
    "PRE/B": ROOT / "results/judgments/pre_blinded_emu_wpre_s30",
    "PRE/U": ROOT / "results/judgments/pre_unblinded_emu_wpre_s30",
    "TAL": ROOT / "results/gens/tal_ta-lite_wclean-last60_s30",
    "NG": ROOT / "results/judgments/ng_typed-local_wclean_items500",
}
NOT_GENERATED = {  # cut runs whose generation does not exist (prereg §10); MCOT/VERB/PROBE/FID/MEM ran (precommit_paid)
    "PRE W_clean S30 unblinded arm (D3 Delta_clean)": "not run (pre_report R-1); cut, D3 has Delta_pre only (prereg §10)",
}
PAID = {  # the pre-commit paid generations of the reduced list (precommit_paid) and the PROBE re-ask (precommit_report)
    "PROBE": ROOT / "results/judgments/probe_emu_d1d2",
    "PROBE re-ask": ROOT / "results/judgments/probe_emu_d1_reask",
    "MCOT": ROOT / "results/judgments/mcot_mono-cot_wclean-last60_s30",
    "VERB": ROOT / "results/judgments/verb_llm-verbalized_wclean_items2000",
    "FID": ROOT / "results/judgments/fid_emu_wclean_v1-150_v2-100x3",
    "MEM arm 1": ROOT / "results/judgments/mem_emu_mepi-off_wclean_s30",
    "MEM arm 2": ROOT / "results/judgments/mem_emu_mwork-off_wclean-last98_s30",
}
VERB_IDENTITY = {"backend": "llm-verbalized", "prompt_version": "verb-v1", "cfg_hash": "3439578683939242"}  # prereg §1
REPLAN_RUNS = {"MAIN", "TAL", "PROBE", "MCOT", "VERB", "FID", "MEM"}  # the runs of the re-plan's paid list
D1_SEED, D2_SEED, FID_SEED, NG_SEED = "D1-www27", "D2-www27", "FID-www27", "NG-C2-www27"
D1_RULE_MONTHS, D1_WPRE_MONTHS = ("2024-01", "2025-06"), ("2023-04", "2023-12")  # prereg §7.2
D2_BASE = {"N1": 34, "T": 33, "FB": 33}  # prereg §8
ITEMS_KEY = "N01.items."
EXCLUDED_TYPE = "stock_activity_or_opinion"


# ------------------------------------------------------------------------------------------------ helpers
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sessions() -> pd.DatetimeIndex:
    """The session calendar: the `date` column of the ETF bar file only (no price is read)."""
    d = pd.read_parquet(ROOT / "data/raw/bars/etf_daily_2018_2026.parquet", columns=["date"])["date"]
    return pd.DatetimeIndex(sorted(pd.to_datetime(d.unique())))


def membership() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet", columns=["date", "symbol", "rank"])


def expected_pairs(weeks, focus: str, mem: pd.DataFrame, salt: str = "S30-www27") -> set[tuple[pd.Timestamp, str]]:
    out = set()
    for t in weeks:
        u = universe_asof(mem, t, 100)
        names = u if focus == "U100" else subset_names(u, 30, salt)
        out |= {(pd.Timestamp(t), s) for s in names}
    return out


def pairs(df: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    return list(zip(pd.to_datetime(df["t"]).dt.tz_localize(None).dt.normalize(), df["symbol"]))


def coverage(present: list, expected: set) -> dict:
    s = pd.Series(present, dtype=object)
    counts = s.value_counts()
    ps = set(counts.index)
    return {"expected": len(expected), "present_distinct": len(ps), "rows": int(len(s)),
            "missing": len(expected - ps), "unexpected": len(ps - expected), "more_than_once": int((counts > 1).sum()),
            "missing_examples": [f"{t.date()} {s_}" for t, s_ in sorted(expected - ps)[:5]],
            "ok": bool(expected == ps and (counts == 1).all())}


def check(ok: bool, **kw) -> dict:
    return {"status": "PASS" if ok else "FAIL", **kw}


def is_distribution(js: str) -> bool:
    try:
        d = json.loads(js)
        v = np.array(list(d.values()), dtype=float)
        return bool(len(v) >= 2 and np.all(v >= -1e-12) and abs(v.sum() - 1.0) <= 1e-6)
    except Exception:
        return False


def ledger() -> pd.DataFrame:
    con = sqlite3.connect(f"file:{ROOT / 'data/cache/spend.sqlite'}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query("select ts, backend, model, tag, input_tokens, output_tokens, usd from spend", con)
    finally:
        con.close()
    df["run"] = df["tag"].str.split("/").str[0]
    df["ts_utc"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def freeze_tree(root: Path) -> dict:
    """The freeze manifest's tree (scripts/write_freeze_manifest.py: tree), recomputed on the current files."""
    spec = importlib.util.spec_from_file_location("wfm", ROOT / "scripts/write_freeze_manifest.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.tree(root)


def code_since_freeze(freeze: dict, backends: dict) -> dict:
    """What changed in src/ and configs/ since FREEZE.json. The 2026-09-28 re-plan lowered the ledger's hard-cap
    default from 950.0 to 245.0 in code; a changed file whose freeze hash is restored by undoing exactly that
    substitution is classified as `hard_cap_default_only`."""
    now = freeze_tree(ROOT / "src")
    old = freeze["code"]["src"]
    changed = sorted(k for k in set(now["files"]) & set(old) if now["files"][k] != old[k])
    detail = {}
    for f in changed:
        b = (ROOT / f).read_bytes()
        detail[f] = ("hard_cap_default_only" if hashlib.sha256(b.replace(b"245.0", b"950.0")).hexdigest() == old[f]
                     else "other change")
    cfg = {f: ("same" if sha256_file(ROOT / "configs" / f) == h else "changed")
           for f, h in freeze["code"]["configs"].items()}
    emu_block_same = backends["backends"]["llm-logprob"] == freeze["emulator"]["config_block"]
    return {"src_tree_sha256_now": now["sha256"], "src_tree_sha256_freeze": freeze["code"]["src_tree_sha256"],
            "src_unchanged": now["sha256"] == freeze["code"]["src_tree_sha256"],
            "src_added": sorted(set(now["files"]) - set(old)), "src_removed": sorted(set(old) - set(now["files"])),
            "src_changed": detail, "configs": cfg, "llm_logprob_block_equal_to_freeze": emu_block_same,
            "only_hard_cap_and_sub_caps_changed": (set(detail.values()) <= {"hard_cap_default_only"}
                                                   and not (set(now["files"]) ^ set(old)) and emu_block_same
                                                   and all(v == "same" for k, v in cfg.items() if k != "backends.yaml"))}


# ------------------------------------------------------------------------------------------------ emu generations
def audit_emu(name: str, gen: Path, weeks: list, focus: str, mem: pd.DataFrame, freeze: dict, led: pd.DataFrame,
              backends: dict) -> dict:
    res: dict = {"gen_dir": str(gen.relative_to(ROOT))}
    man = json.loads((gen / "MANIFEST.json").read_text())
    rq = pd.read_parquet(gen / "requests.parquet", columns=[
        "request_key", "t", "symbol", "program", "stage", "backend", "model_pinned", "response_model", "prompt_version",
        "bank_hash", "data_freeze", "usd", "cached", "degraded", "attempts", "tag", "n_questions", "state_sha256"])
    rq["t"] = pd.to_datetime(rq["t"]).dt.normalize()
    exp_weeks = sorted(pd.Timestamp(w) for w in weeks)
    exp_pairs = expected_pairs(exp_weeks, focus, mem)
    # Layer B's cross-section is all of U_t in every run (buckets, code features, firm info), whatever the focus
    u_pairs = exp_pairs if focus == "U100" else expected_pairs(exp_weeks, "U100", mem)

    # weeks
    have_weeks = sorted(rq["t"].unique())
    res["weeks"] = check([pd.Timestamp(x) for x in have_weeks] == exp_weeks, expected=len(exp_weeks),
                         present=len(have_weeks), first=str(exp_weeks[0].date()), last=str(exp_weeks[-1].date()),
                         missing=[str(w.date()) for w in sorted(set(exp_weeks) - set(map(pd.Timestamp, have_weeks)))],
                         unexpected=[str(pd.Timestamp(w).date()) for w in
                                     sorted(set(map(pd.Timestamp, have_weeks)) - set(exp_weeks))])

    # stock-week coverage
    sw: dict = {}
    for prog in ("T", "FB"):
        sw[f"requests.{prog}"] = coverage(pairs(rq[rq["program"] == prog]), exp_pairs)
    sw["risk"] = coverage(pairs(pd.read_parquet(gen / "risk.parquet", columns=["t", "symbol"])), exp_pairs)
    sw["firm_info (all of U_t)"] = coverage(pairs(pd.read_parquet(gen / "firm_info.parquet", columns=["t", "symbol"])),
                                            u_pairs)
    cf = pd.read_parquet(gen / "code_features.parquet", columns=["t", "symbol", "feature"])
    cfn = cf.groupby([pd.to_datetime(cf["t"]).dt.normalize(), "symbol"])["feature"].nunique()
    sw["code_features (all of U_t)"] = {
        **coverage([k for k in cfn.index], u_pairs), "rows": int(len(cf)),
        "features_per_sw": sorted(set(cfn.values.tolist())),
        "ok": bool(set(cfn.index) == u_pairs and set(cfn.values.tolist()) == {N_CODE_FEATURES}
                   and len(cf) == N_CODE_FEATURES * len(u_pairs))}
    ft = pd.read_parquet(gen / "features.parquet", columns=["t", "symbol", "feature", "program"])
    ft["t"] = pd.to_datetime(ft["t"]).dt.normalize()
    for prog, k in FEATURES_PER_SW.items():
        sub = ft[ft["program"] == prog]
        n = sub.groupby(["t", "symbol"])["feature"].nunique()
        sw[f"features.{prog}"] = {**coverage(list(n.index), exp_pairs), "rows": int(len(sub)),
                                  "features_per_sw": sorted(set(n.values.tolist())),
                                  "ok": bool(set(n.index) == exp_pairs and set(n.values.tolist()) == {k}
                                             and len(sub) == k * len(exp_pairs))}
    fa_pairs = set(pairs(rq[rq["program"] == "FA"]))
    sw["FA (reported, not gated)"] = {"stock_weeks_with_FA_request": len(fa_pairs),
                                      "note": "FA asks when a new 10-K becomes usable; profiles live in memory/sem.jsonl"}
    res["stock_weeks"] = check(all(v.get("ok", True) for v in sw.values()), expected_stock_weeks=len(exp_pairs),
                               detail=sw)

    # per-week programs
    pw = {}
    m = rq[rq["program"] == "M"].groupby("t").size().reindex(exp_weeks, fill_value=0)
    pw["M_per_week"] = {"min": int(m.min()), "max": int(m.max()), "ok": bool((m == 1).all())}
    p1 = rq[rq["program"] == "P1"].groupby("t").size().reindex(exp_weeks, fill_value=0)
    pw["P1_per_week"] = {"min": int(p1.min()), "median": float(p1.median()), "max": int(p1.max()), "ok": bool((p1 >= 1).all())}
    if name == "MAIN":
        p2 = rq[(rq["program"] == "P2") & (rq["stage"] == "P2")].groupby("t").size().reindex(exp_weeks, fill_value=0)
        hist = rq[(rq["program"] == "P2") & (rq["stage"] != "P2")]
        pw["P2_per_week"] = {"min": int(p2.min()), "max": int(p2.max()), "ok": bool((p2 == 1).all()),
                             "p1_history_rows": int(len(hist)), "p1_history_cached_share": round(float(hist["cached"].mean()), 6)}
    res["per_week"] = check(all(v["ok"] for v in pw.values()), detail=pw)

    # news structure
    ev = pd.read_parquet(gen / "evidence.parquet", columns=["request_key", "eid", "quarantined"])
    news = rq[rq["program"].isin(["N0", "N1", "N2", "N3"])]
    ev_keys = set(ev["request_key"])
    no_ev = news[~news["request_key"].isin(ev_keys)]
    # Evidence carries no (t, symbol): a key shared by several stock-weeks (identical text) lists every item that sent
    # it. The rule is therefore: each N1 request row shares at least one item with its own stock-week's N0 items.
    kev = ev[["request_key", "eid"]].drop_duplicates()
    n0 = news[news["program"] == "N0"][["t", "symbol", "request_key"]]
    n1 = news[news["program"] == "N1"][["t", "symbol", "request_key"]].reset_index(drop=True)
    n0e = n0.merge(kev, on="request_key")[["t", "symbol", "eid"]].drop_duplicates()
    n1e = n1.reset_index().merge(kev, on="request_key")
    hit = n1e.merge(n0e.assign(ok=True), on=["t", "symbol", "eid"], how="left")
    row_ok = hit.groupby("index")["ok"].any().reindex(n1.index, fill_value=False)
    n1_bad = int((~row_ok).sum())
    code_ev = ev[ev["request_key"].str.startswith("code:")]
    orphan_ev = ev[~ev["request_key"].isin(set(rq["request_key"])) & ~ev["request_key"].str.startswith("code:")]
    res["news"] = check(len(no_ev) == 0 and n1_bad == 0 and len(orphan_ev) == 0,
                        requests={p: int((news["program"] == p).sum()) for p in ("N0", "N1", "N2", "N3")},
                        items_N0=int(len(n0)), news_requests_without_evidence=int(len(no_ev)),
                        stock_weeks_with_items=int(n0[["t", "symbol"]].drop_duplicates().shape[0]),
                        n1_rows_without_an_n0_item_in_their_stock_week=n1_bad,
                        evidence_rows=int(len(ev)), evidence_code_rows=int(len(code_ev)),
                        evidence_rows_without_request=int(len(orphan_ev)),
                        note="`code:` evidence keys link code-computed features (PX) to blackboard records; they have "
                             "no model request by design")

    # answers <-> requests
    an = pd.read_parquet(gen / "answers.parquet", columns=["t", "symbol", "program", "stage", "qid", "probs_json",
                                                            "request_key"])
    keys = set(rq["request_key"])
    orphan_an = int((~an["request_key"].isin(keys)).sum())
    per_key_q = an.groupby("request_key")["qid"].nunique()
    kq = rq.drop_duplicates("request_key").set_index("request_key")[["n_questions", "degraded", "program"]]
    j = kq.join(per_key_q.rename("answered_qids"), how="left")
    j["answered_qids"] = j["answered_qids"].fillna(0).astype(int)
    short = j[(j["answered_qids"] < j["n_questions"]) & (j["degraded"] == 0)]
    over = j[j["answered_qids"] > j["n_questions"]]
    # P2 recomputes 56 weeks of P1 history; history weeks before the generation's window are answered by PRE's P1
    # units or, for the W_buffer weeks, only in the response cache (P2 stores its own two answers per week).
    hist_keys = set(rq.loc[(rq["program"] == "P2") & rq["stage"].str.startswith("P1-history:"), "request_key"])
    short_hist = set(short.index) & hist_keys
    elsewhere = {"in_PRE_answers": 0, "in_response_cache_only": 0, "not_found": 0}
    if short_hist:
        pre_keys = set()
        for arm in ("PRE/B", "PRE/U"):
            if GEN[arm] != gen:
                pre_keys |= set(pd.read_parquet(GEN[arm] / "answers.parquet", columns=["request_key"])["request_key"])
        con = sqlite3.connect(f"file:{ROOT / 'data/cache/s1_v1.sqlite'}?mode=ro", uri=True)
        try:
            for k in short_hist:
                if k in pre_keys:
                    elsewhere["in_PRE_answers"] += 1
                elif con.execute("select 1 from cache where key = ?", (k,)).fetchone() is not None:
                    elsewhere["in_response_cache_only"] += 1
                else:
                    elsewhere["not_found"] += 1
        finally:
            con.close()
    unexplained = len(short) - len(short_hist) + elsewhere["not_found"]
    res["answers"] = check(orphan_an == 0 and unexplained == 0 and len(over) == 0, answers=int(len(an)),
                           answers_without_request=orphan_an, request_keys=int(len(j)),
                           keys_with_fewer_answers_than_questions_not_degraded=int(len(short)),
                           of_which_P2_P1_history_keys=len(short_hist), history_answers_found=elsewhere,
                           unexplained=int(unexplained), keys_with_more_qids_than_questions=int(len(over)),
                           by_program_missing=short["program"].value_counts().to_dict())

    # duplicates
    dup: dict = {}
    ident = ["request_key", "t", "symbol", "program", "stage"]
    d = rq[rq.duplicated(ident, keep=False)]
    dkeys = set(d["request_key"])
    ev_all = ev[ev["request_key"].isin(dkeys)][["request_key", "eid"]]
    if name == "MAIN":  # P1-history keys of pre-window weeks carry their evidence in PRE's generations
        for arm in ("PRE/B", "PRE/U"):
            e2 = pd.read_parquet(GEN[arm] / "evidence.parquet", columns=["request_key", "eid"])
            ev_all = pd.concat([ev_all, e2[e2["request_key"].isin(dkeys)]])
    n_items = ev_all.groupby("request_key")["eid"].nunique()
    g = d.groupby(ident).agg(rows=("cached", "size"), paid=("cached", lambda s: int((~s).sum())))
    g["items"] = [int(n_items.get(k[0], 0)) for k in g.index]
    explained = (g["items"] >= g["rows"]) & (g["paid"] <= 1)
    dup["requests.exact_rows_on_audited_columns"] = int(rq.duplicated().sum())
    dup["requests.same_key_same_stock_week"] = {
        "groups": int(len(g)), "rows": int(len(d)), "by_program": d.groupby("program").size().to_dict(),
        "groups_explained": int(explained.sum()), "groups_unexplained": int((~explained).sum()),
        "rule": "explained iff the key's evidence lists at least as many distinct items as the group has rows (distinct "
                "items with identical text) and at most one row of the group was paid (the others are cache hits)"}
    # answers of those duplicated request rows repeat with them
    adup = an[an.duplicated(keep=False)]
    in_groups = adup.set_index(["request_key", "t", "symbol"]).index.isin(
        d.set_index(["request_key", "t", "symbol"]).index) if len(adup) else np.array([], bool)
    dup["answers.exact_rows"] = {"rows": int(an.duplicated().sum()),
                                 "outside_duplicated_request_groups": int((~np.asarray(in_groups)).sum())}
    for tab, key in (("features", ["t", "symbol", "feature"]), ("code_features", ["t", "symbol", "feature"]),
                     ("risk", ["t", "symbol"]), ("firm_info", ["t", "symbol"])):
        df = pd.read_parquet(gen / f"{tab}.parquet", columns=key)
        dup[f"{tab}.natural_key"] = int(df.duplicated().sum())
    dup["evidence.(request_key,eid)"] = {
        "duplicate_rows": int(ev.duplicated(["request_key", "eid"]).sum()),
        "rows_exceeding_request_rows": int(_ev_excess(ev, rq)),
        "note": "evidence carries no (t, symbol); a request key used by k request rows repeats its evidence k times"}
    res["duplicates"] = check(dup["requests.same_key_same_stock_week"]["groups_unexplained"] == 0
                              and dup["answers.exact_rows"]["outside_duplicated_request_groups"] == 0
                              and all(dup[f"{t}.natural_key"] == 0 for t in ("features", "code_features", "risk", "firm_info"))
                              and dup["evidence.(request_key,eid)"]["rows_exceeding_request_rows"] == 0, detail=dup)

    # validity
    an["null"] = an["probs_json"].isna()
    val = an.groupby("program")["null"].agg(["size", "sum"])
    val["validity"] = 1 - val["sum"] / val["size"]
    nonnull = an.loc[~an["null"], "probs_json"]
    bad_dist = int((~nonnull.map(is_distribution)).sum())
    n3w = an[an["program"] == "N3"].groupby("t")["null"].mean()
    retries_cfg = int(backends["backends"]["llm-logprob"]["degraded_retries"])
    degr = rq[rq["degraded"] > 0]
    res["validity"] = check(bool((val["validity"] >= MIN_VALIDITY).all()) and bad_dist == 0
                            and bool((degr["attempts"] == 1 + retries_cfg).all()),
                            overall=round(float(1 - an["null"].mean()), 6), degraded_answers=int(an["null"].sum()),
                            by_program={k: round(float(v), 6) for k, v in val["validity"].items()},
                            min_program=str(val["validity"].idxmin()),
                            lowest_program_week_N3=round(float(1 - n3w.max()), 5) if len(n3w) else None,
                            non_distribution_answers=bad_dist, degraded_requests=int(len(degr)),
                            degraded_attempts=sorted(set(degr["attempts"].tolist())), expected_attempts=1 + retries_cfg)

    # provenance
    paid = rq[~rq["cached"]]
    emu = freeze["emulator"]
    ctx = man.get("context", {}).get("data", {})
    tag_ok = rq["tag"].str.split("/").str[0].eq(name.split("/")[0]).all() and (
        name == "MAIN" or rq["tag"].str.startswith(name + "/").all())
    wmin, wmax = min(exp_weeks), max(exp_weeks)
    in_clean = [w for w in exp_weeks if pd.Timestamp(W_CLEAN[0]) <= w <= pd.Timestamp(W_CLEAN[1])]
    window_ok = (len(in_clean) == 0) if name.startswith("PRE") else (
        len(in_clean) == len(exp_weeks) - 1 and wmax == pd.Timestamp(PSEUDO_FORWARD))
    prov = {
        "bank_hash_values": sorted(rq["bank_hash"].unique().tolist()),
        "manifest_bank": man["bank"], "frozen_sha256": FROZEN_SHA256,
        "freeze_json_bank": freeze["bank"]["canonical_sha256"],
        "run_tag_in_FROZEN_RUN_TAGS": name.split("/")[0] in FROZEN_RUN_TAGS,
        "model_pinned_values": sorted(rq["model_pinned"].unique().tolist()),
        "response_model_values_paid_rows": sorted(paid["response_model"].dropna().unique().tolist()),
        "paid_rows": int(len(paid)), "paid_rows_response_model_mismatch": int((paid["response_model"] != emu["model_pinned"]).sum()),
        "prompt_version_values": sorted(rq["prompt_version"].unique().tolist()),
        "backend_values": sorted(rq["backend"].unique().tolist()),
        "data_freeze_values": sorted(rq["data_freeze"].dropna().unique().tolist()),
        "manifest_cfg_hash": man["backend"].get("cfg_hash"), "freeze_cfg_hash": emu["cfg_hash"],
        "manifest_aliases": ctx.get("blinder", {}).get("aliases"), "freeze_aliases": freeze["blinder"]["aliases_sha256"],
        "manifest_archive": ctx.get("archive"),
        "tags": sorted(rq["tag"].unique().tolist()), "tag_prefix_ok": bool(tag_ok),
        "window": [str(wmin.date()), str(wmax.date())], "weeks_in_W_clean": len(in_clean), "window_rule_ok": bool(window_ok),
        "manifest_counts_match": {k: bool(man["counts"].get(k) == v) for k, v in
                                  (("requests", len(rq)), ("answers", len(an)), ("evidence", len(ev)),
                                   ("features", len(ft)))},
    }
    prov_ok = (prov["bank_hash_values"] == [FROZEN_SHA256] and man["bank"]["sha256"] == FROZEN_SHA256
               and prov["freeze_json_bank"] == FROZEN_SHA256 and prov["run_tag_in_FROZEN_RUN_TAGS"]
               and prov["model_pinned_values"] == [emu["model_pinned"]] and prov["paid_rows_response_model_mismatch"] == 0
               and prov["prompt_version_values"] == [emu["prompt_version"]] and prov["backend_values"] == [emu["backend"]]
               and len(prov["data_freeze_values"]) == 1 and prov["manifest_cfg_hash"] == emu["cfg_hash"]
               and prov["manifest_aliases"] == prov["freeze_aliases"] and prov["tag_prefix_ok"] and prov["window_rule_ok"]
               and all(prov["manifest_counts_match"].values()))
    res["provenance"] = check(bool(prov_ok), detail=prov)

    # spend
    tags = rq["tag"].unique().tolist()
    led_g = led[led["tag"].isin(tags)]
    by_prog_tab = paid.groupby("program")["usd"].sum()
    by_tag_led = led_g.groupby("tag")["usd"].sum()
    by_prog_led = {t.split("/")[-1]: v for t, v in by_tag_led.items()}
    res["spend"] = {
        "requests_paid_usd": round(float(paid["usd"].sum()), 4), "ledger_usd": round(float(led_g["usd"].sum()), 4),
        "ledger_rows": int(len(led_g)), "paid_request_rows": int(len(paid)),
        "difference_ledger_minus_tables": round(float(led_g["usd"].sum() - paid["usd"].sum()), 4),
        "difference_by_program": {p: round(float(by_prog_led.get(p, 0.0) - by_prog_tab.get(p, 0.0)), 4)
                                  for p in sorted(set(by_prog_led) | set(by_prog_tab.index))},
        "manifest_spend_usd": round(float(man.get("spend_usd", float("nan"))), 4),
    }
    res["table_sha256_16"] = {p.name: sha256_file(p)[:16] for p in sorted(gen.glob("*.parquet"))}
    res["status"] = "PASS" if all(res[k]["status"] == "PASS" for k in
                                  ("weeks", "stock_weeks", "per_week", "news", "answers", "duplicates", "validity",
                                   "provenance")) else "FAIL"
    return res


def _ev_excess(ev: pd.DataFrame, rq: pd.DataFrame) -> int:
    """Evidence (key, eid) pairs repeated more often than the key's request rows (a real double write)."""
    mult = ev.groupby(["request_key", "eid"]).size().rename("m").reset_index()
    rows = rq.groupby("request_key").size().rename("r")
    mult = mult.join(rows, on="request_key")
    mult = mult[mult["r"].notna()]
    return int((mult["m"] > mult["r"]).sum())


# ------------------------------------------------------------------------------------------------ TAL
def audit_tal(gen: Path, weeks: list, mem: pd.DataFrame, led: pd.DataFrame) -> dict:
    res: dict = {"gen_dir": str(gen.relative_to(ROOT))}
    man = json.loads((gen / "MANIFEST.json").read_text())
    rq = pd.read_parquet(gen / "requests.parquet")
    an = pd.read_parquet(gen / "answers.parquet")
    out = pd.read_parquet(gen / "outputs.parquet", columns=["request_key", "role"])
    rq["t"] = pd.to_datetime(rq["t"]).dt.normalize()
    an["t"] = pd.to_datetime(an["t"]).dt.normalize()
    exp_weeks = sorted(pd.Timestamp(w) for w in weeks)
    exp_pairs = expected_pairs(exp_weeks, "S30", mem)
    roles = ["fundamentals", "technical", "news", "macro_policy", "bull", "bear", "trader", "risk"]
    res["weeks"] = check(sorted(map(pd.Timestamp, rq["t"].unique())) == exp_weeks,
                         expected=len(exp_weeks), present=int(rq["t"].nunique()),
                         first=str(exp_weeks[0].date()), last=str(exp_weeks[-1].date()))
    cov = {"answers": coverage(pairs(an), exp_pairs)}
    per_role = {}
    for r in roles:
        per_role[r] = coverage(pairs(rq[rq["role"] == r]), exp_pairs)
    cov["requests_by_role"] = {r: {k: v for k, v in c.items() if k != "missing_examples"} for r, c in per_role.items()}
    res["stock_weeks"] = check(cov["answers"]["ok"] and all(c["ok"] for c in per_role.values())
                               and set(rq["role"]) == set(roles), expected_stock_weeks=len(exp_pairs), detail=cov)
    dup = {"requests.(t,symbol,role)": int(rq.duplicated(["t", "symbol", "role"]).sum()),
           "answers.(t,symbol)": int(an.duplicated(["t", "symbol"]).sum()),
           "requests.exact_rows": int(rq.duplicated().sum()),
           "request_key_reused_rows": int(rq["request_key"].duplicated().sum()),
           "request_key_reused_cached_share": round(float(rq.loc[rq["request_key"].duplicated(), "cached"].mean()), 4)
           if rq["request_key"].duplicated().any() else None,
           "outputs.request_key_missing": int((~rq["request_key"].isin(set(out["request_key"]))).sum()),
           "note": "a reused request key is a byte-identical request (the week's macro-and-policy analyst; unchanged "
                   "F-B or news blocks): the stated response-cache deviation (gens_report §2.2)"}
    res["duplicates"] = check(dup["requests.(t,symbol,role)"] == 0 and dup["answers.(t,symbol)"] == 0
                              and dup["outputs.request_key_missing"] == 0, detail=dup)
    paid = rq[~rq["cached"]]
    prov = {"model_pinned_values": sorted(rq["model_pinned"].unique().tolist()),
            "response_model_values_paid_rows": sorted(paid["response_model"].dropna().unique().tolist()),
            "prompt_version_values": sorted(rq["prompt_version"].unique().tolist()),
            "parser": sorted(set(zip(rq["parser_version"], rq["parser_sha256"].str[:16]))),
            "bank_hash_values": sorted(rq["bank_hash"].unique().tolist()),
            "manifest": {k: man[k] for k in ("bank_sha256", "parser_version", "parser_sha256", "model", "tag")},
            "tags": sorted(rq["tag"].unique().tolist()), "finish_reasons": rq["finish_reason"].value_counts().to_dict(),
            "weeks_in_W_clean": sum(pd.Timestamp(W_CLEAN[0]) <= w <= pd.Timestamp(W_CLEAN[1]) for w in exp_weeks),
            "is_last_60_of_W_clean": exp_weeks == sorted(decision_dates(sessions_cache(), *W_CLEAN))[-60:]}
    ok = (prov["model_pinned_values"] == ["gpt-4.1-mini-2025-04-14"]
          and prov["response_model_values_paid_rows"] == ["gpt-4.1-mini-2025-04-14"]
          and prov["bank_hash_values"] == [FROZEN_SHA256] and man["bank_sha256"] == FROZEN_SHA256
          and len(prov["parser"]) == 1 and len(prov["prompt_version_values"]) == 1 and prov["tags"] == ["TAL/ta-lite"]
          and prov["is_last_60_of_W_clean"])
    res["provenance"] = check(bool(ok), detail=prov)
    res["validity"] = check(int(rq["degraded"].sum()) == 0, degraded_calls=int(rq["degraded"].sum()),
                            invalid_final=int(an["invalid_final"].sum()) if "invalid_final" in an else None,
                            invalid_analyst_stance=int(an["invalid_analyst_stance"].sum()),
                            invalid_trader=int(an["invalid_trader"].sum()), invalid_risk=int(an["invalid_risk"].sum()),
                            note="an invalid output abstains and is counted (RP-1); decisions/p are not read here")
    led_g = led[led["tag"] == "TAL/ta-lite"]
    res["spend"] = {"requests_paid_usd": round(float(paid["usd"].sum()), 4), "ledger_usd": round(float(led_g["usd"].sum()), 4),
                    "ledger_rows": int(len(led_g)), "paid_request_rows": int(len(paid)),
                    "difference_ledger_minus_tables": round(float(led_g["usd"].sum() - paid["usd"].sum()), 6)}
    res["table_sha256_16"] = {p.name: sha256_file(p)[:16] for p in sorted(gen.glob("*.parquet"))}
    res["manifest_sha256_match"] = {k: sha256_file(gen / f"{k}.parquet") == v for k, v in man["sha256"].items()}
    res["status"] = "PASS" if all(res[k]["status"] == "PASS" for k in
                                  ("weeks", "stock_weeks", "duplicates", "provenance", "validity")) else "FAIL"
    return res


_SESS: pd.DatetimeIndex | None = None


def sessions_cache() -> pd.DatetimeIndex:
    global _SESS
    if _SESS is None:
        _SESS = sessions()
    return _SESS


# ------------------------------------------------------------------------------------------------ NG (partial)
def audit_ng(gen: Path, backends: dict) -> dict:
    res: dict = {"gen_dir": str(gen.relative_to(ROOT))}
    smp = pd.read_parquet(gen / "sample.parquet")
    meta = json.loads((gen / "sample.json").read_text())
    ucol = "unit" if "unit" in smp.columns else smp.columns[0]
    units = sorted(smp[ucol].astype(str))
    sample_sha = hashlib.sha256("\n".join(units).encode()).hexdigest()
    res["sample"] = check(len(units) == 500 and len(set(units)) == 500 and meta.get("n") == 500, n=len(units),
                          unique=len(set(units)), sha256=sample_sha, seed=meta.get("seed"),
                          frame_eligible=meta.get("frame", {}).get("eligible"),
                          sha256_matches_ng_report=sample_sha.startswith("0c6c3da275ca629d"))
    rows = []
    with open(gen / "checkpoint.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                res["partial_last_line"] = True
    done = [r["item"]["unit"] for r in rows]
    tl = backends["backends"]["typed-local"]
    pinned = tl["model"]
    main_keys = set(pd.read_parquet(GEN["MAIN"] / "requests.parquet", columns=["request_key"])["request_key"])
    n_req, bad_model, bad_bank, bad_prompt, nonzero_usd, abst, bad_dist, emu_missing, n_ans = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for r in rows:
        emu_missing += sum(k not in main_keys for k in r["item"]["emu_keys"].values())
        for stage, blk in r["asked"].items():
            q = blk["request"]
            n_req += 1
            bad_model += int(not (q["model_pinned"] == pinned and q["response_model"] == pinned))
            bad_bank += int(q["bank_hash"] != FROZEN_SHA256)
            bad_prompt += int(q["prompt_version"] != "ng-v1" or q["backend"] != "typed-local")
            nonzero_usd += int(float(q["usd"]) != 0.0)
            abst += int(q["degraded"] > 0)
            for a in blk["answers"]:
                n_ans += 1
                if a.get("probs_json") is not None and not is_distribution(a["probs_json"]):
                    bad_dist += 1
    prog = json.loads((gen / "progress.json").read_text()) if (gen / "progress.json").exists() else {}
    res["progress"] = {"done": len(done), "total": 500, "complete": (gen / "MANIFEST.json").exists(),
                       "eta_utc": prog.get("eta_utc"), "updated_utc": prog.get("updated_utc"),
                       "backend_failures": prog.get("backend_failures")}
    res["items_done"] = check(len(set(done)) == len(done) and set(done) <= set(units)
                              and all(len(r["asked"]) == 6 for r in rows) and n_req == 6 * len(rows)
                              and bad_model == 0 and bad_bank == 0 and bad_prompt == 0 and nonzero_usd == 0
                              and bad_dist == 0 and emu_missing == 0,
                              items=len(done), duplicate_items=len(done) - len(set(done)),
                              items_outside_sample=len(set(done) - set(units)), requests=n_req, answers=n_ans,
                              model_not_pinned=bad_model, bank_not_frozen=bad_bank, prompt_or_backend_wrong=bad_prompt,
                              nonzero_usd=nonzero_usd, abstaining_requests=abst, non_distribution_answers=bad_dist,
                              emu_keys_not_in_MAIN=emu_missing, pinned=pinned,
                              note="structure only; NG-vs-emu agreement is not computed before the unlock")
    res["status"] = "PASS (partial: generation running)" if res["items_done"]["status"] == "PASS" and \
        res["sample"]["status"] == "PASS" and not res["progress"]["complete"] else (
        "PASS" if res["items_done"]["status"] == "PASS" and res["sample"]["status"] == "PASS" else "FAIL")
    return res


# ------------------------------------------------------------------------------------------------ pre-commit paid
def h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def iso_week_ends(ses: pd.DatetimeIndex, lo: str, hi: str) -> list[pd.Timestamp]:
    """Weekly decisions on the FULL session calendar (through `hi`): the last session of each ISO week, then cut to
    [lo, hi]. A week that straddles a month edge gives one decision, in the month of its last session."""
    s = pd.Series(ses[ses <= pd.Timestamp(hi)])
    iso = s.dt.isocalendar()
    ends = s.groupby([iso["year"].values, iso["week"].values]).max()
    return sorted(pd.Timestamp(x) for x in ends if pd.Timestamp(lo) <= x <= pd.Timestamp(hi))


def work_items(gen: Path, weeks: set[str]) -> list[dict]:
    """The N01 item rows of a generation's M-work blackboard (`N01.items.<symbol>`) for the given decision dates:
    t, symbol, eid, status, event_type answers, n1_sha256, request_keys (Layer-A only)."""
    out = []
    with open(gen / "memory" / "work.jsonl") as f:
        for line in f:
            if ITEMS_KEY not in line:
                continue
            rec = json.loads(line)["record"]
            if not rec["key"].startswith(ITEMS_KEY) or rec["t"][:10] not in weeks:
                continue
            sym = rec["key"][len(ITEMS_KEY):]
            out += [{"t": rec["t"][:10], "symbol": sym, **r} for r in rec["value"]]
    return out


def top_label(probs) -> str | None:
    """argmax of an answer distribution; ties to the first label in bank order (dict order)."""
    if isinstance(probs, str):
        probs = json.loads(probs)
    return None if not probs else max(probs, key=probs.get)


def hamilton(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Largest-remainder allocation of n over the cells; equal remainders go to the smaller cell key."""
    tot = sum(sizes.values())
    quota = {k: n * v / tot for k, v in sizes.items()}
    alloc = {k: int(np.floor(q)) for k, q in quota.items()}
    for k in sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k))[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def metas(df: pd.DataFrame) -> pd.DataFrame:
    """jobs.parquet rows with their meta_json expanded (prefixed m_)."""
    m = pd.DataFrame([json.loads(x) if isinstance(x, str) else {} for x in df["meta_json"]], index=df.index)
    return pd.concat([df.drop(columns=["meta_json"]), m.add_prefix("m_")], axis=1)


def audit_s1(gen: Path, run: str, ident: dict, backends: dict, led_run: pd.DataFrame, window: tuple[str, str],
             archive: str, ledger_tags: list[str] | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Checks shared by the S1 generations of experiments/run_gen_*.py (PROBE, its re-ask, VERB, FID, MEM arms).
    `ident` = the expected backend identity; `led_run` = the ledger rows that can belong to this generation (the
    generation's own tags are taken from its table unless `ledger_tags` is given)."""
    res: dict = {"gen_dir": str(gen.relative_to(ROOT))}
    man = json.loads((gen / "MANIFEST.json").read_text())
    rq = pd.read_parquet(gen / "requests.parquet", columns=[
        "request_key", "t", "symbol", "program", "stage", "backend", "model_pinned", "response_model", "prompt_version",
        "bank_hash", "data_freeze", "usd", "cached", "degraded", "attempts", "tag", "n_questions", "state_sha256"])
    rq["t"] = pd.to_datetime(rq["t"]).dt.normalize()
    an = pd.read_parquet(gen / "answers.parquet", columns=["t", "symbol", "program", "stage", "qid", "probs_json",
                                                            "request_key"])
    parts = ["answers", "validity", "provenance", "spend"]
    if (gen / "jobs.parquet").exists():
        jobs = pd.read_parquet(gen / "jobs.parquet", columns=["uid", "request_key"])
        same = Counter(jobs["request_key"]) == Counter(rq["request_key"])
        res["plan"] = check(bool(jobs["uid"].is_unique and same and len(jobs) == len(rq)), jobs=int(len(jobs)),
                            requests=int(len(rq)), uid_unique=bool(jobs["uid"].is_unique),
                            request_key_multiset_equal=bool(same))
        parts.insert(0, "plan")

    keys = set(rq["request_key"])
    orphan = int((~an["request_key"].isin(keys)).sum())
    per_key_q = an.groupby("request_key")["qid"].nunique()
    j = rq.drop_duplicates("request_key").set_index("request_key")[["n_questions", "degraded"]]
    j = j.join(per_key_q.rename("answered"), how="left")
    j["answered"] = j["answered"].fillna(0).astype(int)
    short = j[(j["answered"] < j["n_questions"]) & (j["degraded"] == 0)]
    over = j[j["answered"] > j["n_questions"]]
    res["answers"] = check(orphan == 0 and len(short) == 0 and len(over) == 0
                           and len(an) == int(rq["n_questions"].sum()),
                           answers=int(len(an)), sum_n_questions_over_request_rows=int(rq["n_questions"].sum()),
                           answers_without_request=orphan, keys_short_not_degraded=int(len(short)),
                           keys_with_more_qids_than_questions=int(len(over)))

    an["null"] = an["probs_json"].isna()
    val = an.groupby("program")["null"].agg(["size", "sum"])
    val["validity"] = 1 - val["sum"] / val["size"]
    bad_dist = int((~an.loc[~an["null"], "probs_json"].map(is_distribution)).sum())
    retries = int(backends["backends"][ident["backend"]]["degraded_retries"])
    degr = rq[rq["degraded"] > 0]
    res["validity"] = check(bool((val["validity"] >= MIN_VALIDITY).all()) and bad_dist == 0
                            and bool((degr["attempts"] == 1 + retries).all()),
                            overall=round(float(1 - an["null"].mean()), 6), null_answers=int(an["null"].sum()),
                            by_program={k: round(float(v), 6) for k, v in val["validity"].items()},
                            non_distribution_answers=bad_dist, degraded_requests=int(len(degr)),
                            degraded_attempts=sorted(set(degr["attempts"].tolist())), expected_attempts=1 + retries)

    paid = rq[~rq["cached"].astype(bool)]
    mb = man.get("backend", {})
    mbank = man.get("bank_sha256") or (man.get("bank") or {}).get("sha256")
    if "rows" in man:
        counts = {k: bool(man["rows"][k] == pq.read_metadata(gen / f"{k}.parquet").num_rows) for k in man["rows"]}
    else:
        counts = {k: bool(man["counts"].get(k) == v) for k, v in (("requests", len(rq)), ("answers", len(an)))}
    shas = {k: sha256_file(gen / f"{k}.parquet") == v for k, v in man.get("sha256", {}).items()}
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    prov = {
        "bank_hash_values": sorted(rq["bank_hash"].unique().tolist()), "manifest_bank": mbank,
        "run_tag_in_FROZEN_RUN_TAGS": run in FROZEN_RUN_TAGS,
        "model_pinned_values": sorted(rq["model_pinned"].unique().tolist()),
        "paid_rows": int(len(paid)),
        "paid_rows_response_model_mismatch": int((paid["response_model"] != ident["model_pinned"]).sum()),
        "prompt_version_values": sorted(rq["prompt_version"].unique().tolist()),
        "backend_values": sorted(rq["backend"].unique().tolist()),
        "data_freeze_values": sorted(rq["data_freeze"].dropna().unique().tolist()), "data_freeze_expected": archive,
        "manifest_backend": {k: mb.get(k) for k in ("backend", "model_pinned", "prompt_version", "cfg_hash")},
        "expected_backend": ident,
        "tags": sorted(rq["tag"].unique().tolist()), "tag_prefix_ok": bool(rq["tag"].str.startswith(run + "/").all()),
        "dates": [str(rq["t"].min().date()), str(rq["t"].max().date())], "window": list(window),
        "inside_window": bool(((rq["t"] >= lo) & (rq["t"] <= hi)).all()),
        "manifest_counts_match": counts, "manifest_sha256_match": shas,
    }
    ok = (prov["bank_hash_values"] == [FROZEN_SHA256] and mbank == FROZEN_SHA256 and prov["run_tag_in_FROZEN_RUN_TAGS"]
          and prov["model_pinned_values"] == [ident["model_pinned"]] and prov["paid_rows_response_model_mismatch"] == 0
          and prov["prompt_version_values"] == [ident["prompt_version"]]
          and prov["backend_values"] == [ident["backend"]] and prov["data_freeze_values"] == [archive]
          and all(prov["manifest_backend"][k] == ident[k] for k in ident) and prov["tag_prefix_ok"]
          and prov["inside_window"] and all(counts.values()) and all(shas.values()))
    res["provenance"] = check(bool(ok), detail=prov)

    tags = ledger_tags or rq["tag"].unique().tolist()
    led_g = led_run[led_run["tag"].isin(tags)]
    diff = float(led_g["usd"].sum() - paid["usd"].sum())
    res["spend"] = check(len(led_g) >= len(paid) and abs(diff) < 0.005,
                         requests_paid_usd=round(float(paid["usd"].sum()), 6),
                         ledger_usd=round(float(led_g["usd"].sum()), 6), ledger_rows=int(len(led_g)),
                         paid_request_rows=int(len(paid)), difference_ledger_minus_tables=round(diff, 6),
                         ledger_tags=sorted(tags),
                         note="ledger rows >= paid rows (a degraded call's retries are ledger rows too); USD equal to "
                              "the cent")
    res["table_sha256_16"] = {p.name: sha256_file(p)[:16] for p in sorted(gen.glob("*.parquet"))}
    res["_parts"] = parts
    return res, rq, an


def finish(res: dict, design: dict) -> dict:
    res["design"] = design
    parts = res.pop("_parts") + ["design"]
    res["status"] = "PASS" if all(res[k]["status"] == "PASS" for k in parts) else "FAIL"
    return res


def emu_identity(freeze: dict) -> dict:
    e = freeze["emulator"]
    return {"backend": e["backend"], "model_pinned": e["model_pinned"], "prompt_version": e["prompt_version"],
            "cfg_hash": e["cfg_hash"]}


def d1_schedule(ses: pd.DatetimeIndex, mem: pd.DataFrame, w_pre: list) -> pd.DataFrame:
    """The registered D1 schedule (prereg §7.2), rebuilt: per month of 2024-01..2025-06 (rule) and 2023-04..2023-12
    (W_pre slice; decisions inside the slice), the 2 weekly decisions with the smallest sha256("D1-www27|" + ISO
    date), and at each the 10 names of U_t with the smallest sha256("D1-www27" + symbol)."""
    cal = iso_week_ends(ses, W_PRE[0], W_CLEAN[1])
    pre = {pd.Timestamp(x) for x in w_pre}
    rows = []
    for seg, (a, b) in (("rule", D1_RULE_MONTHS), ("wpre", D1_WPRE_MONTHS)):
        for m in [p.strftime("%Y-%m") for p in pd.period_range(a, b, freq="M")]:
            ts = [t for t in cal if t.strftime("%Y-%m") == m and (seg == "rule" or t in pre)]
            for t in sorted(sorted(ts, key=lambda d: h(f"{D1_SEED}|{d.date().isoformat()}"))[:2]):
                names = sorted(universe_asof(mem, t, 100), key=lambda s: h(f"{D1_SEED}{s}"))[:10]
                rows += [{"t": t, "symbol": s, "month": m, "segment": seg, "pos": k} for k, s in enumerate(names)]
    return pd.DataFrame(rows)


def audit_probe(ses: pd.DatetimeIndex, mem: pd.DataFrame, w_pre: list, w_clean: list, freeze: dict, backends: dict,
                led: pd.DataFrame, archive: str) -> dict:
    """PROBE (`probe_emu_d1d2`: D1 + D2) and its re-ask of the scheduled D1 probes (`probe_emu_d1_reask`)."""
    g1, g2 = PAID["PROBE"], PAID["PROBE re-ask"]
    launch = float(json.loads((g2 / "MANIFEST.json").read_text())["launch_ts"])
    lp = led[led["run"] == "PROBE"]
    ts = lp["ts"].astype(float)
    ident = emu_identity(freeze)
    window = (W_PRE[0], W_CLEAN[1])
    r1, rq1, an1 = audit_s1(g1, "PROBE", ident, backends, lp[ts < launch], window, archive)
    r2, rq2, an2 = audit_s1(g2, "PROBE", ident, backends, lp[ts >= launch], window, archive)

    # D1: the registered schedule against the union of the two generations
    sch = d1_schedule(ses, mem, w_pre)
    S = set(zip(sch["t"], sch["symbol"]))
    d1a = {(t, s) for t, s in zip(an1.loc[an1["stage"] == "D1", "t"], an1.loc[an1["stage"] == "D1", "symbol"])}
    d1a = {(pd.Timestamp(t).normalize(), s) for t, s in d1a}
    d1b = {(pd.Timestamp(t).normalize(), s) for t, s in zip(an2.loc[an2["stage"] == "D1", "t"],
                                                           an2.loc[an2["stage"] == "D1", "symbol"])}
    j2 = metas(pd.read_parquet(g2 / "jobs.parquet"))
    pos = {(t, s): p for t, s, p in zip(sch["t"], sch["symbol"], sch["pos"])}
    pos_ok = all(pos.get((pd.Timestamp(t), s)) == p for t, s, p in zip(j2["t"], j2["symbol"], j2["m_pos"]))
    ev_keys = set(pd.read_parquet(g1 / "evidence.parquet", columns=["request_key"])["request_key"]) | set(
        pd.read_parquet(g2 / "evidence.parquet", columns=["request_key"])["request_key"])
    d1_keys = set(rq1.loc[rq1["stage"] == "D1", "request_key"]) | set(rq2.loc[rq2["stage"] == "D1", "request_key"])
    off = sorted({str(t.date()) for t, _ in d1a - S})
    d1 = {"scheduled": len(S), "scheduled_rule_period": int((sch["segment"] == "rule").sum()),
          "scheduled_W_pre": int((sch["segment"] == "wpre").sum()),
          "d1d2_answered_on_schedule": len(d1a & S), "d1d2_answered_off_schedule": len(d1a - S),
          "d1d2_off_schedule_dates": off,
          "reask_answered": len(d1b), "reask_equals_the_scheduled_probes_missing_from_d1d2": d1b == S - d1a,
          "reask_dates": sorted({str(t.date()) for t, _ in d1b}), "answered_in_both": len(d1a & d1b),
          "union_missing_scheduled": len(S - (d1a | d1b)), "reask_meta_reask_of": sorted(set(j2["m_reask_of"])),
          "reask_pos_equals_hash_order": bool(pos_ok), "d1_requests_with_evidence": len(d1_keys & ev_keys),
          "note": "answers are only located (t, symbol), never read; the rule's statistic waits for the unlock"}
    d1_ok = (len(S) == 540 and d1["union_missing_scheduled"] == 0 and d1["reask_equals_the_scheduled_probes_missing_from_d1d2"]
             and d1["answered_in_both"] == 0 and d1["reask_meta_reask_of"] == ["probe_emu_d1d2"] and pos_ok
             and d1["d1_requests_with_evidence"] == 0)

    # D2: 600 base-state x level states and their questions; base states by their hash rule
    j1 = metas(pd.read_parquet(g1 / "jobs.parquet"))
    d2 = j1[j1["stage"].str.startswith("D2")].copy()
    d2["t"] = pd.to_datetime(d2["t"]).dt.strftime("%Y-%m-%d")
    d2["question"] = np.where(d2["stage"].str.startswith("D2y"), "reid_year", "reid_company")
    d2["eid"] = [e if isinstance(e, str) else "-" for e in d2["m_eid"]]  # T and F-B states have no item
    base = d2.drop_duplicates(["m_period", "m_source", "t", "symbol", "eid"])
    rank_ok = all(h(f"{D2_SEED}|{p}|{src}|{t}|{s}" + (f"|{e}" if e != "-" else "")) == r for p, src, t, s, e, r in
                  zip(base["m_period"], base["m_source"], base["t"], base["symbol"], base["eid"], base["m_rank"]))
    per = base.groupby(["m_period", "m_source"]).size().to_dict()
    lv = d2.groupby(["m_period", "m_source", "t", "symbol", "eid", "question"], dropna=False)["m_level"].apply(
        lambda x: tuple(sorted(x)))
    levels_ok = bool(lv.map(lambda x: x == ("full", "names", "none")).all())
    q_ok = bool(len(lv) == len(base) + int((base["m_source"] == "N1").sum())
                and set(lv.reset_index().query("question == 'reid_year'")["m_source"]) == {"N1"})
    minimal: dict = {}
    for period, gen, weeks in (("W_pre", GEN["PRE/B"], w_pre), ("W_clean", GEN["MAIN"], w_clean)):
        wk = {str(pd.Timestamp(t).date()): pd.Timestamp(t) for t in weeks}
        s30 = {d: set(subset_names(universe_asof(mem, t, 100), 30, "S30-www27")) for d, t in wk.items()}
        rq = pd.read_parquet(gen / "requests.parquet", columns=["t", "symbol", "program"])
        rq = rq[rq["program"].isin(["T", "FB"])]
        rq["d"] = pd.to_datetime(rq["t"]).dt.strftime("%Y-%m-%d")
        for src in ("T", "FB"):
            c = rq[(rq["program"] == src) & rq["d"].isin(wk)]
            cand = {(d, s) for d, s in zip(c["d"], c["symbol"]) if s in s30[d]}
            want = set(sorted(cand, key=lambda k: h(f"{D2_SEED}|{period}|{src}|{k[0]}|{k[1]}"))[:D2_BASE[src]])
            b = base[(base["m_period"] == period) & (base["m_source"] == src)]
            minimal[f"{period}|{src}"] = {"candidates": len(cand), "chosen_equal_smallest_ranks":
                                          set(zip(b["t"], b["symbol"])) == want}
        items = [r for r in work_items(gen, set(wk)) if r.get("n1_sha256") and r["symbol"] in s30[r["t"]]]
        want = set(sorted(((r["t"], r["symbol"], r["eid"]) for r in items),
                          key=lambda k: h(f"{D2_SEED}|{period}|N1|{k[0]}|{k[1]}|{k[2]}"))[:D2_BASE["N1"]])
        b = base[(base["m_period"] == period) & (base["m_source"] == "N1")]
        minimal[f"{period}|N1"] = {"candidates": len(items),
                                   "chosen_equal_smallest_ranks": set(zip(b["t"], b["symbol"], b["eid"])) == want}
    d2r = {"base_states_by_period_source": {f"{k[0]}|{k[1]}": int(v) for k, v in per.items()},
           "states": int((d2["question"] == "reid_company").sum()),
           "reid_year_requests": int((d2["question"] == "reid_year").sum()), "requests": int(len(d2)),
           "rank_strings_ok": bool(rank_ok), "three_levels_per_base_state_and_question": levels_ok,
           "reid_company_on_every_state_reid_year_on_every_N1_state": q_ok, "selection": minimal,
           "weeks_drawn": {p: int(base.loc[base["m_period"] == p, "t"].nunique()) for p in ("W_pre", "W_clean")}}
    d2_ok = (all(per.get((p, s)) == n for p in ("W_pre", "W_clean") for s, n in D2_BASE.items()) and len(per) == 6
             and d2r["states"] == 600 and d2r["reid_year_requests"] == 204 and rank_ok and levels_ok and q_ok
             and all(v["chosen_equal_smallest_ranks"] for v in minimal.values()))
    both = {"D1": check(bool(d1_ok), **d1), "D2": check(bool(d2_ok), **d2r)}
    r1 = finish(r1, check(bool(d1_ok and d2_ok), **both))
    r2 = finish(r2, check(bool(d1_ok), **{"D1": "see PROBE.design.D1 (the union of both generations)",
                                          "reask_answered": len(d1b)}))
    return {"PROBE": r1, "PROBE re-ask": r2}


def audit_mcot(gen: Path, weeks: list, mem: pd.DataFrame, led: pd.DataFrame) -> dict:
    """MCOT: Mono-CoT (`mono-cot-v1`, `gen-parse-v1`) on S30 x the last 60 W_clean decisions, TAL's stock-weeks."""
    res: dict = {"gen_dir": str(gen.relative_to(ROOT))}
    man = json.loads((gen / "MANIFEST.json").read_text())
    rq = pd.read_parquet(gen / "requests.parquet")
    an = pd.read_parquet(gen / "answers.parquet", columns=["t", "symbol", "reader", "invalid", "degraded_calls"])
    out = pd.read_parquet(gen / "outputs.parquet", columns=["request_key", "role"])
    rq["t"] = pd.to_datetime(rq["t"]).dt.normalize()
    an["t"] = pd.to_datetime(an["t"]).dt.normalize()
    exp_weeks = sorted(pd.Timestamp(w) for w in weeks)
    exp_pairs = expected_pairs(exp_weeks, "S30", mem)
    tal = pd.read_parquet(GEN["TAL"] / "answers.parquet", columns=["t", "symbol"])
    tal_pairs = set(pairs(tal))
    res["weeks"] = check(sorted(map(pd.Timestamp, rq["t"].unique())) == exp_weeks, expected=len(exp_weeks),
                         present=int(rq["t"].nunique()), first=str(exp_weeks[0].date()), last=str(exp_weeks[-1].date()))
    cov = {"requests": coverage(pairs(rq), exp_pairs), "answers": coverage(pairs(an), exp_pairs)}
    res["stock_weeks"] = check(cov["requests"]["ok"] and cov["answers"]["ok"] and set(pairs(an)) == tal_pairs,
                               expected_stock_weeks=len(exp_pairs), same_stock_weeks_as_TAL=set(pairs(an)) == tal_pairs,
                               roles=rq["role"].value_counts().to_dict(), detail=cov)
    dup = {"requests.(t,symbol)": int(rq.duplicated(["t", "symbol"]).sum()),
           "answers.(t,symbol)": int(an.duplicated(["t", "symbol"]).sum()),
           "outputs.request_key_missing": int((~rq["request_key"].isin(set(out["request_key"]))).sum())}
    res["duplicates"] = check(all(v == 0 for v in dup.values()), detail=dup)
    paid = rq[~rq["cached"]]
    prov = {"model_pinned_values": sorted(rq["model_pinned"].unique().tolist()),
            "response_model_values_paid_rows": sorted(paid["response_model"].dropna().unique().tolist()),
            "prompt_version_values": sorted(rq["prompt_version"].unique().tolist()),
            "parser": sorted(set(zip(rq["parser_version"], rq["parser_sha256"].str[:16]))),
            "bank_hash_values": sorted(rq["bank_hash"].unique().tolist()),
            "manifest": {k: man[k] for k in ("bank_sha256", "parser_version", "parser_sha256", "model", "tag")},
            "run_tag_in_FROZEN_RUN_TAGS": "MCOT" in FROZEN_RUN_TAGS,
            "tags": sorted(rq["tag"].unique().tolist()), "finish_reasons": rq["finish_reason"].value_counts().to_dict(),
            "is_last_60_of_W_clean": exp_weeks == sorted(decision_dates(sessions_cache(), *W_CLEAN))[-60:]}
    ok = (prov["model_pinned_values"] == ["gpt-4.1-mini-2025-04-14"]
          and prov["response_model_values_paid_rows"] == ["gpt-4.1-mini-2025-04-14"]
          and prov["bank_hash_values"] == [FROZEN_SHA256] and man["bank_sha256"] == FROZEN_SHA256
          and prov["parser"] == [("gen-parse-v1", "45a3fe949f47ea4d")] and man["parser_version"] == "gen-parse-v1"
          and prov["prompt_version_values"] == ["mono-cot-v1"] and prov["tags"] == ["MCOT/mono-cot"]
          and prov["run_tag_in_FROZEN_RUN_TAGS"] and prov["is_last_60_of_W_clean"])
    res["provenance"] = check(bool(ok), detail=prov)
    res["validity"] = check(int(rq["degraded"].sum()) == 0, degraded_calls=int(rq["degraded"].sum()),
                            invalid_outputs=int(an["invalid"].sum()),
                            note="an invalid output abstains and is counted (RP-1); decisions/p are not read here")
    led_g = led[led["tag"] == "MCOT/mono-cot"]
    diff = float(led_g["usd"].sum() - paid["usd"].sum())
    res["spend"] = check(len(led_g) >= len(paid) and abs(diff) < 0.005,
                         requests_paid_usd=round(float(paid["usd"].sum()), 4),
                         ledger_usd=round(float(led_g["usd"].sum()), 4), ledger_rows=int(len(led_g)),
                         paid_request_rows=int(len(paid)), difference_ledger_minus_tables=round(diff, 6))
    res["table_sha256_16"] = {p.name: sha256_file(p)[:16] for p in sorted(gen.glob("*.parquet"))}
    res["manifest_sha256_match"] = {k: sha256_file(gen / f"{k}.parquet") == v for k, v in man["sha256"].items()}
    res["status"] = "PASS" if all(res[k]["status"] == "PASS" for k in
                                  ("weeks", "stock_weeks", "duplicates", "provenance", "validity", "spend")) \
        and all(res["manifest_sha256_match"].values()) else "FAIL"
    return res


def audit_verb(items_clean: list[dict], main_n1: pd.DataFrame, freeze: dict, backends: dict, led: pd.DataFrame,
               archive: str) -> dict:
    """VERB: the aligned 2,000-item sample (prereg §3.2) x the frozen bank's five N1 requests on MAIN's N1 states."""
    gen = PAID["VERB"]
    ident = {"model_pinned": freeze["emulator"]["model_pinned"], **VERB_IDENTITY}
    res, rq, an = audit_s1(gen, "VERB", ident, backends, led[led["run"] == "VERB"], W_CLEAN, archive)
    smp = pd.read_parquet(gen / "sample.parquet")
    meta = json.loads((gen / "sample.json").read_text())
    man = json.loads((gen / "MANIFEST.json").read_text())
    ng = pd.read_parquet(GEN["NG"] / "sample.parquet")
    ng_units = set(ng["unit"].astype(str))
    # the C2 frame from MAIN's N01 item rows (status.kept, top event type != stock_activity_or_opinion)
    fr = []
    for r in items_clean:
        top = top_label(r.get("event_type"))
        if r["status"]["kept"] and top is not None and top != EXCLUDED_TYPE:
            u = f"{r['t']}|{r['symbol']}|{r['eid']}"
            fr.append({"unit": u, "stratum": f"{r['t'][:4]}|{top}", "rank": h(f"{NG_SEED}|{u}")})
    fr = pd.DataFrame(fr)
    sizes = fr.groupby("stratum").size().to_dict()
    alloc = hamilton(sizes, 2000)
    ngc = fr[fr["unit"].isin(ng_units)].groupby("stratum").size().to_dict()
    extra = {s: max(0, alloc[s] - ngc.get(s, 0)) for s in sizes}
    want = set(ng_units) | set().union(*[set(g[~g["unit"].isin(ng_units)].sort_values("rank").head(extra[s])["unit"])
                                         for s, g in fr.groupby("stratum")])
    units = sorted(smp["unit"].astype(str))
    sha = h("\n".join(units))
    cells_ok = all(meta["cells"].get(s, {}).get("frame") == sizes[s] and meta["cells"][s]["hamilton_2000"] == alloc[s]
                   for s in sizes) and set(meta["cells"]) == set(sizes)
    # requests: five per unit on MAIN's N1 state of the same stock-week
    jb = metas(pd.read_parquet(gen / "jobs.parquet"))
    per_unit = jb.groupby("m_unit")["m_group"].apply(lambda x: tuple(sorted(x)))
    groups = ("about_focal", "attributes", "instructs_reader", "new_development", "promotional")
    main_states = set(zip(main_n1["t"], main_n1["symbol"], main_n1["state_sha256"]))
    on_main = [(pd.Timestamp(t).normalize(), s, x) in main_states for t, s, x in zip(rq["t"], rq["symbol"],
                                                                                        rq["state_sha256"])]
    nq = rq.groupby("stage")["n_questions"].agg(lambda x: sorted(set(x))).to_dict()
    design = {"frame_units": int(len(fr)), "frame_equals_ng_frame": int(len(fr)) == json.loads(
                  (GEN["NG"] / "sample.json").read_text())["frame"]["eligible"],
              "sample_units": len(units), "unique": len(set(units)), "sample_sha256": sha,
              "sample_sha256_equals_manifest": sha == man.get("sample_sha256") == meta.get("sample_sha256"),
              "sample_sha256_matches_prereg": sha.startswith("708e34519fb9957a"),
              "ng_units_in_sample": len(ng_units & set(units)), "in_ng_flag_ok": bool(
                  (smp["in_ng"].astype(bool) == smp["unit"].astype(str).isin(ng_units)).all()),
              "sample_equals_rebuilt_rule": set(units) == want, "cells_ok": bool(cells_ok),
              "cells_where_ng_exceeds_hamilton": sorted(s for s in sizes if ngc.get(s, 0) > alloc[s]),
              "weeks": int(smp["t"].nunique()), "units_with_the_five_requests": int((per_unit == groups).sum()),
              "jobs_units_equal_sample": set(per_unit.index) == set(units),
              "request_states_equal_MAIN_N1_states": float(np.mean(on_main)),
              "n_questions_by_stage": {k: [int(v) for v in vs] for k, vs in nq.items()},
              "keys_shared_with_MAIN": int(rq["request_key"].isin(set(main_n1["request_key"])).sum())}
    ok = (design["frame_equals_ng_frame"] and len(units) == 2000 and design["unique"] == 2000
          and design["sample_sha256_equals_manifest"] and design["sample_sha256_matches_prereg"]
          and design["ng_units_in_sample"] == 500 and design["in_ng_flag_ok"] and design["sample_equals_rebuilt_rule"]
          and cells_ok and design["weeks"] == 115 and design["units_with_the_five_requests"] == 2000
          and design["jobs_units_equal_sample"] and design["request_states_equal_MAIN_N1_states"] == 1.0
          and len(rq) == 10000 and nq.get("N1.attributes") == [13]
          and all(nq.get(f"N1.{g}") == [1] for g in groups if g != "attributes"))
    return finish(res, check(bool(ok), **design))


def audit_fid(items_clean: list[dict], main_rq: pd.DataFrame, w_clean: list, mem: pd.DataFrame, freeze: dict,
              backends: dict, led: pd.DataFrame, archive: str) -> dict:
    """FID: V1 joint vs single on 150 stock-weeks, V2 re-ask floor 100 x 3 (prereg §9)."""
    gen = PAID["FID"]
    res, rq, an = audit_s1(gen, "FID", emu_identity(freeze), backends, led[led["run"] == "FID"], W_CLEAN, archive)
    meta = json.loads((gen / "sample.json").read_text())
    jb = metas(pd.read_parquet(gen / "jobs.parquet"))
    jb["t"] = pd.to_datetime(jb["t"]).dt.strftime("%Y-%m-%d")
    by_sw: dict = {}
    for r in items_clean:
        if "N1.attributes" in (r.get("request_keys") or {}):
            by_sw.setdefault((r["t"], r["symbol"]), []).append(r)
    frame = sorted(by_sw, key=lambda k: h(f"{FID_SEED}|{k[0]}|{k[1]}"))
    chosen = frame[:150]
    item = {k: min(by_sw[k], key=lambda r: h(f"{FID_SEED}|{k[0]}|{k[1]}|{r['eid']}")) for k in chosen}
    sample_sha = h("\n".join(sorted(f"{t}|{s}|{item[(t, s)]['eid']}" for t, s in chosen)))
    v1 = jb[jb["m_arm"] == "V1"]
    v1t, v1n = v1[v1["program"] == "T"], v1[v1["program"] == "N1"]
    main_keys = set(main_rq["request_key"])
    tkey = {(str(t.date()), s): k for t, s, k in zip(main_rq.loc[main_rq["program"] == "T", "t"],
                                                     main_rq.loc[main_rq["program"] == "T", "symbol"],
                                                     main_rq.loc[main_rq["program"] == "T", "request_key"])}
    prod = v1n[v1n["m_production_single"].astype(bool)]
    rq_cached = rq.drop_duplicates("request_key").set_index("request_key")["cached"]
    prod_ok = all(k in main_keys and bool(rq_cached.get(k, False)) and
                  item[(t, s)]["request_keys"].get(f"N1.{q}") == k
                  for t, s, q, k in zip(prod["t"], prod["symbol"], prod["m_q"], prod["request_key"]))
    joint_ok = all(item[(t, s)]["request_keys"]["N1.attributes"] == k and item[(t, s)]["eid"] == e
                   for t, s, k, e in zip(v1n["t"], v1n["symbol"], v1n["m_joint_key"], v1n["m_eid"])) and all(
        tkey.get((t, s)) == k for t, s, k in zip(v1t["t"], v1t["symbol"], v1t["m_joint_key"]))
    v2 = jb[jb["m_arm"] == "V2"]
    uid_ok = all(h(f"FID-V2|{k}|{int(r)}")[:12] == u for k, r, u in zip(v2["m_main_key"], v2["m_repeat"],
                                                                       v2["m_uid"]))
    first = sorted(chosen, key=lambda k: h(f"{FID_SEED}|{k[0]}|{k[1]}"))
    v2t, v2n, v2n3 = (v2[v2["program"] == p] for p in ("T", "N1", "N3"))
    s30 = {str(t.date()): set(subset_names(universe_asof(mem, t, 100), 30, "S30-www27")) for t in w_clean}
    n3 = main_rq[main_rq["program"] == "N3"]
    n3c = [(str(t.date()), s, k) for t, s, k in zip(n3["t"], n3["symbol"], n3["request_key"])
           if str(t.date()) in s30 and s in s30[str(t.date())]]
    n3_per_sw = Counter((t, s) for t, s, _ in n3c)
    n3_want = {k for _, _, k in sorted(n3c, key=lambda x: h(f"{FID_SEED}|N3|{x[0]}|{x[1]}"))[:33]}
    design = {"frame_stock_weeks": len(frame), "frame_equals_sample_json": len(frame) == meta["frame_stock_weeks"],
              "v1_stock_weeks_equal_rebuilt": set(zip(v1["t"], v1["symbol"])) == set(chosen),
              "sample_sha256": sample_sha, "sample_sha256_equals_sample_json": sample_sha == meta["sample_sha256"],
              "sample_sha256_matches_prereg": sample_sha.startswith("215c9503"),
              "v1_T_requests": int(len(v1t)), "v1_N1_requests": int(len(v1n)),
              "v1_T_per_stock_week": sorted(set(v1t.groupby(["t", "symbol"]).size().tolist())),
              "v1_N1_per_stock_week": sorted(set(v1n.groupby(["t", "symbol"]).size().tolist())),
              "production_singles": int(len(prod)), "production_singles_are_MAIN_keys_and_cache_hits": bool(prod_ok),
              "joint_keys_and_items_equal_MAIN": bool(joint_ok),
              "v2_requests_by_program": {p: int(len(x)) for p, x in (("T", v2t), ("N1", v2n), ("N3", v2n3))},
              "v2_keys_distinct": bool(v2["request_key"].is_unique),
              "v2_keys_not_MAIN_keys": bool((~v2["request_key"].isin(main_keys)).all()),
              "v2_main_keys_are_MAIN_keys": bool(v2["m_main_key"].isin(main_keys).all()), "v2_uid_rule_ok": bool(uid_ok),
              "v2_T_stock_weeks_first_34_by_rank": set(zip(v2t["t"], v2t["symbol"])) == set(first[:34]),
              "v2_N1_first_33_by_rank_attributes_keys": set(zip(v2n["t"], v2n["symbol"])) == set(first[:33]) and all(
                  item[(t, s)]["request_keys"]["N1.attributes"] == k
                  for t, s, k in zip(v2n["t"], v2n["symbol"], v2n["m_main_key"])),
              "v2_N3_main_keys_equal_33_smallest_S30_ranks": set(v2n3["m_main_key"]) == n3_want,
              "main_N3_requests_per_S30_stock_week_max": max(n3_per_sw.values()) if n3_per_sw else 0}
    ok = (design["frame_equals_sample_json"] and design["v1_stock_weeks_equal_rebuilt"]
          and design["sample_sha256_equals_sample_json"] and design["sample_sha256_matches_prereg"]
          and design["v1_T_requests"] == 900 and design["v1_N1_requests"] == 1950 and design["production_singles"] == 600
          and prod_ok and joint_ok and design["v2_requests_by_program"] == {"T": 102, "N1": 99, "N3": 99}
          and design["v2_keys_distinct"] and design["v2_keys_not_MAIN_keys"] and design["v2_main_keys_are_MAIN_keys"]
          and uid_ok and design["v2_T_stock_weeks_first_34_by_rank"] and design["v2_N1_first_33_by_rank_attributes_keys"]
          and design["v2_N3_main_keys_equal_33_smallest_S30_ranks"] and len(rq) == 3150)
    return finish(res, check(bool(ok), **design))


def audit_mem(main_rq: pd.DataFrame, w_clean: list, mem: pd.DataFrame, freeze: dict, backends: dict,
              led: pd.DataFrame, archive: str) -> dict:
    """MEM arm 1 (M-epi off: N3 re-ask, no N2, MAIN's N01 replayed) and arm 2 (M-work off: T on the last 98 weeks)."""
    ident = emu_identity(freeze)
    lm = led[led["run"] == "MEM"]
    out = {}
    wc = sorted(pd.Timestamp(w) for w in w_clean)
    s30_pairs = expected_pairs(wc, "S30", mem)
    main_keys = set(main_rq["request_key"])
    # arm 1
    g = PAID["MEM arm 1"]
    res, rq, an = audit_s1(g, "MEM", ident, backends, lm, W_CLEAN, archive, ledger_tags=["MEM/MEM/N3", "MEM/N3"])
    man = json.loads((g / "MANIFEST.json").read_text())
    n3 = main_rq[main_rq["program"] == "N3"]
    main_n3 = Counter(p for p in pairs(n3) if p in s30_pairs)
    arm = Counter(pairs(rq))
    shared = rq[rq["request_key"].isin(main_keys)]
    ft = pd.read_parquet(g / "features.parquet", columns=["t", "symbol", "feature", "program"])
    fn = ft.groupby([pd.to_datetime(ft["t"]).dt.normalize(), "symbol"])["feature"].nunique()
    rc = json.loads((g / "replay_check.json").read_text()) if (g / "replay_check.json").exists() else {}
    design = {"programs": sorted(rq["program"].unique().tolist()), "no_N2": bool((rq["program"] != "N2").all()),
              "manifest_weeks_equal_W_clean": [pd.Timestamp(w) for w in man["spec"]["weeks"]] == wc,
              "variant": man["spec"].get("variant"),
              "requests": int(len(rq)), "main_N3_requests_on_S30": int(sum(main_n3.values())),
              "stock_week_multiset_equals_MAIN_N3_on_S30": arm == main_n3,
              "keys_shared_with_MAIN": int(len(shared)), "shared_keys_all_cache_hits": bool(shared["cached"].all()),
              "features_N23_per_S30_stock_week": sorted(set(fn.values.tolist())),
              "features_cover_S30": set(fn.index) == s30_pairs and set(ft["program"]) == {"N23"},
              "replay_check": {k: rc.get(k) for k in list(rc)[:8]}}
    ok = (design["programs"] == ["N3"] and design["manifest_weeks_equal_W_clean"] and design["variant"] == ["mepi_off"]
          and design["stock_week_multiset_equals_MAIN_N3_on_S30"] and design["shared_keys_all_cache_hits"]
          and design["features_cover_S30"] and design["features_N23_per_S30_stock_week"] == [12])
    out["MEM arm 1"] = finish(res, check(bool(ok), **design))
    # arm 2
    g = PAID["MEM arm 2"]
    last = wc[-98:]
    res, rq, an = audit_s1(g, "MEM", ident, backends, lm, W_CLEAN, archive, ledger_tags=["MEM/T"])
    man = json.loads((g / "MANIFEST.json").read_text())
    exp = expected_pairs(last, "S30", mem)
    design = {"programs": sorted(rq["program"].unique().tolist()),
              "manifest_weeks_equal_last_98_of_W_clean": [pd.Timestamp(w) for w in man["spec"]["weeks"]] == last,
              "first_week": str(last[0].date()), "variant": man["spec"].get("variant"),
              "coverage": coverage(pairs(rq), exp), "n_questions": sorted(set(rq["n_questions"].tolist())),
              "keys_shared_with_MAIN": int(rq["request_key"].isin(main_keys).sum()),
              "missing_dependencies": man.get("missing_dependencies"),
              "note": "M is not run (M-work off), so T's M dependency is missing by design"}
    ok = (design["programs"] == ["T"] and design["manifest_weeks_equal_last_98_of_W_clean"]
          and design["variant"] == ["mwork_off"] and design["coverage"]["ok"] and design["n_questions"] == [6]
          and len(rq) == 2940)
    out["MEM arm 2"] = finish(res, check(bool(ok), **design))
    return out


# ------------------------------------------------------------------------------------------------ ledger
def audit_ledger(led: pd.DataFrame, backends: dict) -> dict:
    caps = backends["ledger"]["sub_caps_usd"]
    hard = float(backends["ledger"]["hard_cap_usd"])
    by_run = led.groupby("run").agg(rows=("usd", "size"), usd=("usd", "sum"), first=("ts_utc", "min"),
                                    last=("ts_utc", "max")).sort_values("usd", ascending=False)
    runs = {}
    for run, r in by_run.iterrows():
        cap = caps.get(run)
        runs[run] = {"usd": round(float(r["usd"]), 4), "rows": int(r["rows"]),
                     "first_utc": r["first"].isoformat(timespec="seconds"), "last_utc": r["last"].isoformat(timespec="seconds"),
                     "sub_cap": cap, "within_sub_cap": None if cap is None else bool(r["usd"] <= cap + 1e-9)
                     if cap > 0 else "sub-cap now 0 (disabled after the run)"}
    post = led[led["ts_utc"] >= REPLAN_UTC]
    post_by = post.groupby("tag")["usd"].agg(["size", "sum"])
    total = float(led["usd"].sum())
    models = led.groupby(["backend", "model"])["usd"].agg(["size", "sum"]).reset_index()
    return {
        "total_usd": round(total, 4), "rows": int(len(led)), "last_row_utc": led["ts_utc"].max().isoformat(timespec="seconds"),
        "hard_cap_usd": hard, "user_budget_usd": 250.0, "remaining_below_hard_cap": round(hard - total, 4),
        "remaining_of_budget": round(250.0 - total, 4), "by_run": runs,
        "within_hard_cap": total <= hard,
        "post_replan_spend": {"since_utc": REPLAN_UTC.isoformat(), "usd": round(float(post["usd"].sum()), 4),
                              "by_tag_prefix": post.assign(p=post["tag"].str.split("/").str[0]).groupby("p")["usd"].sum()
                              .round(4).to_dict(),
                              "runs": sorted(set(post["run"])),
                              "only_runs_of_the_replan_list": bool(set(post["run"]) <= REPLAN_RUNS)},
        "models": [{"backend": r.backend, "model": r.model, "rows": int(r["size"]), "usd": round(float(r["sum"]), 4)}
                   for _, r in models.iterrows()],
        "enabled_sub_caps_sum_usd": round(sum(v for v in caps.values() if v > 0), 2),
        "unspent_headroom_of_enabled_caps_usd": {k: round(v - runs.get(k, {}).get("usd", 0.0), 4)
                                                 for k, v in caps.items() if v > 0},
        "ledger_if_every_unrun_run_spends_its_full_sub_cap": round(
            total + sum(v for k, v in caps.items() if v > 0 and k not in runs)
            + sum(max(v - runs[k]["usd"], 0.0) for k, v in caps.items() if v > 0 and k in runs and k == "SMOKE"), 4),
        "runs_with_cap_but_no_spend": sorted(k for k, v in caps.items() if v > 0 and k not in runs),
        "note": "a tag counts against the run before its first '/' (configs/backends.yaml). Rows before 09-25 13:50 "
                "(pilot_news, smoke, design_smoke_*) predate the pinned snapshot and the tag scheme",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/prereg/completeness_audit.json"))
    a = ap.parse_args()
    t0 = dt.datetime.now(dt.timezone.utc)
    backends = yaml.safe_load((ROOT / "configs/backends.yaml").read_text())
    freeze = json.loads((ROOT / "results/freeze/FREEZE.json").read_text())
    led = ledger()
    mem = membership()
    ses = sessions_cache()
    w_clean = decision_dates(ses, *W_CLEAN)
    w_main = w_clean + decision_dates(ses, PSEUDO_FORWARD, PSEUDO_FORWARD)
    w_pre = decision_dates(ses, *W_PRE)
    out = {"created_utc": t0.isoformat(timespec="seconds"), "script": "experiments/prereg/audit_generations.py",
           "w_clean_rule": "no outcome read (see module docstring)",
           "calendar": {"W_clean_decisions": len(w_clean), "MAIN_decisions": len(w_main), "W_pre_slice_decisions": len(w_pre)}}
    print(f"calendar: W_clean {len(w_clean)}, MAIN {len(w_main)}, W_pre slice {len(w_pre)}", flush=True)
    gens = {}
    for name, weeks, focus in (("MAIN", w_main, "U100"), ("PRE/B", w_pre, "S30"), ("PRE/U", w_pre, "S30")):
        print(f"auditing {name} ...", flush=True)
        gens[name] = audit_emu(name, GEN[name], weeks, focus, mem, freeze, led, backends)
        print(f"  {name}: {gens[name]['status']}", flush=True)
    print("auditing TAL ...", flush=True)
    gens["TAL"] = audit_tal(GEN["TAL"], w_clean[-60:], mem, led)
    print(f"  TAL: {gens['TAL']['status']}", flush=True)
    print("auditing NG ...", flush=True)
    gens["NG"] = audit_ng(GEN["NG"], backends)
    print(f"  NG: {gens['NG']['status']}", flush=True)
    # pre-commit paid generations (precommit_paid) and the PROBE re-ask
    archive = json.loads((GEN["MAIN"] / "MANIFEST.json").read_text())["context"]["data"]["archive"]
    main_rq = pd.read_parquet(GEN["MAIN"] / "requests.parquet",
                              columns=["request_key", "t", "symbol", "program", "n_questions", "state_sha256"])
    main_rq["t"] = pd.to_datetime(main_rq["t"]).dt.normalize()
    items_clean = work_items(GEN["MAIN"], {str(w.date()) for w in w_clean})
    print("auditing PROBE and the PROBE re-ask ...", flush=True)
    gens.update(audit_probe(ses, mem, w_pre, w_clean, freeze, backends, led, archive))
    print("auditing MCOT ...", flush=True)
    gens["MCOT"] = audit_mcot(PAID["MCOT"], w_clean[-60:], mem, led)
    print("auditing VERB ...", flush=True)
    gens["VERB"] = audit_verb(items_clean, main_rq[main_rq["program"] == "N1"], freeze, backends, led, archive)
    print("auditing FID ...", flush=True)
    gens["FID"] = audit_fid(items_clean, main_rq, w_clean, mem, freeze, backends, led, archive)
    print("auditing MEM ...", flush=True)
    gens.update(audit_mem(main_rq, w_clean, mem, freeze, backends, led, archive))
    for k in PAID:
        print(f"  {k}: {gens[k]['status']}", flush=True)
    out["generations"] = gens
    out["not_generated"] = NOT_GENERATED
    out["ledger"] = audit_ledger(led, backends)
    # cross-reference of the operators' own validators (read, not re-run)
    xref = {}
    for k, p in (("MAIN", "results/main/main_validation.json"), ("PRE", "results/pre/validation.json"),
                 ("TAL", "results/gens/tal_validation.json")):
        f = ROOT / p
        if f.exists():
            d = json.loads(f.read_text())
            xref[k] = {"file": p, "sha256_16": sha256_file(f)[:16], "top_level_keys": sorted(d)[:12]}
    for k, g in PAID.items():
        f = g / "validation.json"
        if f.exists():
            d = json.loads(f.read_text())
            xref[k] = {"file": str(f.relative_to(ROOT)), "sha256_16": sha256_file(f)[:16],
                       "pass": d.get("pass", d.get("ok")), "checks": d.get("checks")}
        else:
            xref[k] = {"file": None, "note": "no validation.json in this generation (MEM arm 2 is validated in arm 1's "
                                             "validation.json, arms.work)" if k == "MEM arm 2" else "missing"}
    out["operator_validators"] = xref
    out["code_since_freeze"] = code_since_freeze(freeze, backends)
    out["overall"] = "PASS" if all(g["status"].startswith("PASS") for g in gens.values()) else "FAIL"
    out["wall_s"] = round((dt.datetime.now(dt.timezone.utc) - t0).total_seconds(), 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(f"overall {out['overall']}; ledger ${out['ledger']['total_usd']:.4f}; wrote {a.out} ({out['wall_s']} s)")


if __name__ == "__main__":
    main()
