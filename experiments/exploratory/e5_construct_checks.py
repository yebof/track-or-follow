#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E5: construct checks of the registered H2 association (positioning_v3 §1.5,
§2.1, D-16; outline_v2 §3 Fig. 2b and Table 2, §5.3). The specification is experiments/exploratory/e5_spec.yaml;
its SHA-256 is logged before the first run and carried in every output.

    .venv/bin/python experiments/exploratory/e5_construct_checks.py log-spec    # 1. log the spec hash (no data read)
    .venv/bin/python experiments/exploratory/e5_construct_checks.py audit       # 2. code the 200 audit items ($0, local)
    .venv/bin/python experiments/exploratory/e5_construct_checks.py code-frame  # (v1.1) code every frame text (~8 h CPU)
    .venv/bin/python experiments/exploratory/e5_construct_checks.py run         # 3. statistics -> results/exploratory/
                                                                                 #    e5_construct_checks.json

What it answers. H2 correlates an item's expected effect level E[effect] with the focal firm's announcement-window
abnormal return AR[0,+1]. Three construct questions follow, all descriptive:
1. Web news is often written after, and about, the price move. How many frame items report the move in their own
   text (a fixed regex, audited on 200 items by an independent coder), and what is the statistic without them?
2. How large is the association against a yardstick: a Loughran-McDonald dictionary on the same items, the item AUC
   against the sign of the reaction and the directional hit rate?
3. Where does it live: the statistic within each top event type of the reader (the paper's error analysis), and on
   firm-days instead of items.

Rules. The frame is the registered one, rebuilt read-only (experiments/analysis/h2_validity.py) and checked to
reproduce h2.json's statistic to 1e-12 before anything else is computed. The item text (licensed) is read from
data/processed/news_table.parquet and never written to results/: the audit file holds eids and codes only. No
p-value is computed or printed; every interval is a two-sided 95% interval (Newey-West(4) for weekly means, a
week-block bootstrap for AUCs and hit rates, Wilson or a stratified bootstrap for the audit). Outputs carry the
label "exploratory (not pre-registered)" (experiments/exploratory/_xcommon.py). $0: the audit coder is the open typed
model on the local CPU (typed-local; no ledger, no key, no network).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402  (registered, read-only)
import h2_validity as H2  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402  (registered, read-only)

NAME = "e5_construct_checks"
HERE = Path(__file__).resolve().parent
SPEC = HERE / "e5_spec.yaml"
SPEC_LOG = X.OUT_DIR / "e5_spec_log.json"
AUDIT_CODES = X.OUT_DIR / "e5_audit_codes.csv"
HUMAN_CODES = X.OUT_DIR / "e5_audit_human_codes.csv"
FRAME_CODES = X.OUT_DIR / "e5_frame_codes.csv"  # spec v1.1 full_frame_coding (sensitivity only)
NEWS = C.ROOT / "data" / "processed" / "news_table.parquet"
GEN = "main_emu_wclean_u100"
N_BOOT, SEED = 2000, 0
TOL = 1e-12


# ------------------------------------------------------------------------------------------------ spec
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_spec() -> tuple[dict, str]:
    return yaml.safe_load(SPEC.read_text()), sha256_file(SPEC)


def log_spec(amend: bool = False) -> dict:
    """Record the spec's SHA-256 with a UTC time (history kept). Reads no data."""
    _, sha = load_spec()
    now = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    log = json.loads(SPEC_LOG.read_text()) if SPEC_LOG.is_file() else {"label": X.LABEL, "history": []}
    cur = log.get("current")
    if cur and cur["sha256"] == sha:
        return log
    if cur and not amend:
        raise SystemExit(f"{SPEC.name} changed since it was logged ({cur['sha256'][:12]}... at {cur['logged_utc']}); "
                         "rerun with --amend to log the new version (the history is kept)")
    entry = {"sha256": sha, "logged_utc": now, "file": C.rel(SPEC), "before_outputs": not X.out_path(NAME).exists()
             and not AUDIT_CODES.exists()}
    log["history"].append(entry)
    log["current"] = entry
    SPEC_LOG.parent.mkdir(parents=True, exist_ok=True)
    SPEC_LOG.write_text(json.dumps(log, indent=1) + "\n")
    return log


def require_logged_spec() -> tuple[dict, str, dict]:
    spec, sha = load_spec()
    log = json.loads(SPEC_LOG.read_text()) if SPEC_LOG.is_file() else {}
    if (log.get("current") or {}).get("sha256") != sha:
        raise SystemExit(f"{SPEC.name} is not logged in its current form: run `{Path(__file__).name} log-spec` first")
    return spec, sha, log


# ------------------------------------------------------------------------------------------------ frame and text
def frame(root: Path, guard) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, dict]:
    """The registered H2 frame (Layer-A answers only; no outcome)."""
    start, end = guard.check_window("W_clean", "E5 construct checks")
    d1 = C.d1_start(root)
    f = H2.c2_frame(H2.item_rows(root / "results" / "judgments" / GEN, start, end))
    f = f[f["t"] >= pd.Timestamp(d1["start"])].reset_index(drop=True)
    return f, start, end, d1


def attach_text(f: pd.DataFrame) -> pd.DataFrame:
    nt = pd.read_parquet(NEWS, columns=["id", "headline", "summary"])
    nt["eid"] = ["news:" + hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in nt["id"]]
    nt = nt.drop_duplicates("eid")[["eid", "headline", "summary"]]
    g = f.merge(nt, on="eid", how="left")
    g["has_text"] = g["headline"].notna() | g["summary"].notna()
    return g


def classify(g: pd.DataFrame, spec: dict) -> pd.DataFrame:
    pat = re.compile(spec["classifier"]["pattern"])
    text = (g["headline"].fillna("") + " " + g["summary"].fillna("")).str.lower()
    g = g.copy()
    g["price"] = text.map(lambda s: bool(pat.search(s)))
    g["price_hl"] = g["headline"].fillna("").str.lower().map(lambda s: bool(pat.search(s)))
    return g


def audit_sample(g: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """The audit draw (spec audit.draw): distinct eids per classifier stratum, smallest hash first. No outcome."""
    t = g[g["has_text"]].drop_duplicates("eid")[["eid", "headline", "summary", "price"]]
    parts, sizes = [], {}
    for stratum, sub in (("match", t[t["price"]]), ("nonmatch", t[~t["price"]])):
        n = int(spec["audit"]["strata"][stratum])
        sizes[stratum] = int(len(sub))
        rank = [hashlib.sha256(f"E5-AUDIT|{stratum}|{e}".encode()).hexdigest() for e in sub["eid"]]
        parts.append(sub.assign(rank=rank, stratum=stratum).sort_values("rank").head(n))
    return pd.concat(parts, ignore_index=True), sizes


# ------------------------------------------------------------------------------------------------ audit coder
async def _code_audit(sample: pd.DataFrame, spec: dict) -> pd.DataFrame:
    from jevtrader.s1.typed_local import NG_MODEL, TypedDecisionLocal
    from jevtrader.s1.types import Noul

    c = spec["audit"]["coder"]
    if c["model"] != NG_MODEL:
        raise SystemExit(f"spec coder model {c['model']} differs from the pinned typed-local model {NG_MODEL}")
    name = c["question_name"]
    q = {name: Noul(instructions=c["instructions"], true=c["true"], false=c["false"])}
    backend = TypedDecisionLocal(threads=int(c["threads"]))
    rows = []
    try:
        for i, r in enumerate(sample.itertuples(index=False)):
            state = {"item": {"headline": r.headline if isinstance(r.headline, str) else "",
                              "summary": r.summary if isinstance(r.summary, str) else ""}}
            a = await backend.ask(state, q)
            p = a.answers[name].p_yes
            rows.append({"eid": r.eid, "stratum": r.stratum, "p_yes": p,
                         "code": None if p is None else int(p >= 0.5), "served_model": a.response_model})
            if (i + 1) % 25 == 0:
                print(f"  audit: {i + 1}/{len(sample)} coded", flush=True)
    finally:
        backend.close()
    return pd.DataFrame(rows)


async def _code_frame(texts: pd.DataFrame, spec: dict) -> int:
    """Spec v1.1: code every distinct frame text with the audit's coder; append one line per text (resumable)."""
    import os

    from jevtrader.s1.typed_local import NG_MODEL, TypedDecisionLocal
    from jevtrader.s1.types import Noul

    c = spec["audit"]["coder"]
    if c["model"] != NG_MODEL:
        raise SystemExit(f"spec coder model {c['model']} differs from the pinned typed-local model {NG_MODEL}")
    name = c["question_name"]
    q = {name: Noul(instructions=c["instructions"], true=c["true"], false=c["false"])}
    backend = TypedDecisionLocal(threads=int(c["threads"]))
    new = not FRAME_CODES.exists()
    n = 0
    try:
        with open(FRAME_CODES, "a") as fh:
            if new:
                fh.write(f"# {X.LABEL}; E5 v1.1 full-frame codes; coder {c['model']}; eids and codes only\n")
                fh.write("eid,p_yes,code\n")
            for r in texts.itertuples(index=False):
                state = {"item": {"headline": r.headline if isinstance(r.headline, str) else "",
                                  "summary": r.summary if isinstance(r.summary, str) else ""}}
                a = await backend.ask(state, q)
                if a.response_model != c["model"]:
                    raise SystemExit(f"served model {a.response_model} is not the pinned coder")
                p = a.answers[name].p_yes
                fh.write(f"{r.eid},{'' if p is None else repr(float(p))},{'' if p is None else int(p >= 0.5)}\n")
                fh.flush()
                os.fsync(fh.fileno())
                n += 1
                if n % 500 == 0:
                    print(f"  frame coding: {n}/{len(texts)} this session", flush=True)
    finally:
        backend.close()
    return n


def frame_codes_complete(g: pd.DataFrame) -> pd.DataFrame | None:
    """The v1.1 frame codes merged on eid, or None unless every frame text is coded."""
    if not FRAME_CODES.exists():
        return None
    fc = pd.read_csv(FRAME_CODES, comment="#").drop_duplicates("eid")
    need = set(g.loc[g["has_text"], "eid"])
    if not need <= set(fc["eid"]):
        return None
    return fc


def recall_from(prec: float, miss: float, n_m: int, n_nm: int) -> float:
    tp, fn = prec * n_m, miss * n_nm
    return float(tp / (tp + fn)) if tp + fn > 0 else float("nan")


def audit_stats(codes: pd.DataFrame, sizes: dict, coder: str) -> dict:
    m = codes.loc[codes["stratum"] == "match", "code"].dropna().astype(int).to_numpy()
    nm = codes.loc[codes["stratum"] == "nonmatch", "code"].dropna().astype(int).to_numpy()
    if not len(m) or not len(nm):
        return {"coder": coder, "status": "incomplete"}
    prec, miss = float(m.mean()), float(nm.mean())
    n_m, n_nm = sizes["match"], sizes["nonmatch"]
    rng = np.random.default_rng(SEED)
    boot = [recall_from(rng.choice(m, len(m)).mean(), rng.choice(nm, len(nm)).mean(), n_m, n_nm)
            for _ in range(N_BOOT)]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"coder": coder, "status": "ok", "n_match_coded": len(m), "n_nonmatch_coded": len(nm),
            "abstained": int(codes["code"].isna().sum()),
            "precision": prec, "precision_ci": S.wilson(int(m.sum()), len(m)),
            "miss_rate": miss, "miss_rate_ci": S.wilson(int(nm.sum()), len(nm)),
            "recall": recall_from(prec, miss, n_m, n_nm), "recall_ci": [float(lo), float(hi)],
            "coder_prevalence": (prec * n_m + miss * n_nm) / (n_m + n_nm),
            "stratum_sizes_distinct_eids": sizes, "n_boot": N_BOOT, "seed": SEED}


# ------------------------------------------------------------------------------------------------ statistics
def wk(f: pd.DataFrame, x: str = "E_effect", y: str = "ar") -> dict:
    """The registered weekly statistic with its two-sided NW(4) 95% interval; the p-value and t are dropped."""
    r = H2.weekly_test(f.dropna(subset=[x]), x, y)
    return {k: r.get(k) for k in ("mean", "ci", "se", "n_weeks", "n_weeks_undefined", "n_units",
                                  "distinct_focal_d0")}


def _hit(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y) & (y != 0) & ((x > 2.5) | (x < 1.5))
    return float(((x[ok] > 2) == (y[ok] > 0)).mean()) if ok.any() else float("nan")


def auc_block(f: pd.DataFrame, x: str) -> dict:
    from jevtrader.eval.metrics import auc

    s, a = f[x].to_numpy(dtype=float), f["ar"].to_numpy(dtype=float)
    ok = np.isfinite(s) & np.isfinite(a) & (a != 0)
    g = pd.DataFrame({"t": f["t"].to_numpy()[ok], "s": s[ok], "y": (a[ok] > 0).astype(float)})
    bs = S.week_block_bootstrap_xy(g, "t", "s", "y", auc, n_boot=N_BOOT, seed=SEED)
    return {"auc": bs["estimate"], "ci": bs["ci"], "n": int(len(g)), "positives": int(g["y"].sum()),
            "n_boot": N_BOOT, "seed": SEED}


def hit_block(f: pd.DataFrame) -> dict:
    x, a = f["E_effect"].to_numpy(dtype=float), f["ar"].to_numpy(dtype=float)
    g = pd.DataFrame({"t": f["t"].to_numpy(), "x": x, "y": a})
    bs = S.week_block_bootstrap_xy(g, "t", "x", "y", _hit, n_boot=N_BOOT, seed=SEED)
    n = int((np.isfinite(x) & np.isfinite(a) & (a != 0) & ((x > 2.5) | (x < 1.5))).sum())
    return {"hit_rate": bs["estimate"], "ci": bs["ci"], "n": n, "rule": "E[effect] outside [1.5, 2.5], AR != 0",
            "n_boot": N_BOOT, "seed": SEED}


def lm_tone(g: pd.DataFrame) -> np.ndarray:
    from jevtrader.baselines.text import LMDictionary

    lm = LMDictionary()
    text = g["headline"].fillna("") + " " + g["summary"].fillna("")
    out = np.full(len(g), np.nan)
    for i, (txt, has, t) in enumerate(zip(text, g["has_text"], g["t"])):
        if has:
            v = lm.net_negative([txt], int(pd.Timestamp(t).year))[0]
            out[i] = -v if np.isfinite(v) else np.nan
    return out


def statistics(g: pd.DataFrame, h2: dict) -> dict:
    local = g["created_at"].dt.tz_convert(C.ET)
    pre_open = (local.dt.strftime("%H:%M") < H2.PRE_OPEN).to_numpy()
    no, yes = g[~g["price"]], g[g["price"]]
    out: dict = {}
    allw = wk(g)
    diff = abs(allw["mean"] - h2["confirmatory"]["mean"])
    if diff > TOL:
        raise SystemExit(f"E5 frame does not reproduce h2.json: |diff| = {diff:.3g}")
    out["consistency"] = {"h2_all_recomputed_minus_registered": float(allw["mean"] - h2["confirmatory"]["mean"]),
                          "pre_open_recomputed_minus_registered": float(
                              wk(g[pre_open])["mean"] - h2["controls"]["pre_open_only"]["mean"]),
                          "placebo_recomputed_minus_registered": float(
                              wk(g, y="ar_placebo")["mean"] - h2["placebo_ar_m5_m1"]["mean"])}
    out["price_report"] = {
        "share_units": float(g["price"].mean()), "n_units": int(g["price"].sum()), "units": int(len(g)),
        "share_units_headline": float(g["price_hl"].mean()), "n_units_headline": int(g["price_hl"].sum()),
        "share_distinct_texts": float(g[g["has_text"]].drop_duplicates("eid")["price"].mean()),
        "units_without_text": int((~g["has_text"]).sum()),
        "share_pre_open_units": float(g.loc[pre_open, "price"].mean())}
    out["weekly"] = {"all": allw, "no_price": wk(no), "price_only": wk(yes),
                     "no_price_headline": wk(g[~g["price_hl"]]),
                     "pre_open_all": wk(g[pre_open]), "pre_open_no_price": wk(g[pre_open & ~g["price"].to_numpy()]),
                     "placebo_all": wk(g, y="ar_placebo"), "placebo_no_price": wk(no, y="ar_placebo")}
    out["dictionary_reference"] = {"reader": "Loughran-McDonald tone (minus net-negative share), raw text",
                                   "all": wk(g, x="lm_pos"), "no_price": wk(no, x="lm_pos"),
                                   "coverage_units": int(np.isfinite(g["lm_pos"]).sum())}
    out["item_auc"] = {"E_effect_all": auc_block(g, "E_effect"), "E_effect_no_price": auc_block(no, "E_effect"),
                       "lm_all": auc_block(g, "lm_pos"), "lm_no_price": auc_block(no, "lm_pos")}
    out["hit_rate"] = {"all": hit_block(g), "no_price": hit_block(no)}
    shares = g["top_type"].value_counts(normalize=True)
    out["by_concept"] = {tt: {"share": float(shares[tt]), "n_units": int((g["top_type"] == tt).sum()),
                              "n_price": int(g.loc[g["top_type"] == tt, "price"].sum()),
                              "all": wk(g[g["top_type"] == tt]),
                              "no_price": wk(no[no["top_type"] == tt])}
                         for tt in shares.index}
    if "coder" in g:
        either = g["price"] | (g["coder"] == 1)
        pre = pre_open & ~either.to_numpy()
        out["either_classifier_sensitivity"] = {
            "rule": "units flagged by the regex or by the model coder (spec v1.1 full_frame_coding)",
            "share_units_coder": float((g["coder"] == 1).mean()), "share_units_either": float(either.mean()),
            "coder_abstained_units": int(g["coder"].isna().sum()),
            "no_price_either": wk(g[~either]), "price_either": wk(g[either]),
            "pre_open_no_price_either": wk(g[pre]), "placebo_no_price_either": wk(g[~either], y="ar_placebo")}
    else:
        out["either_classifier_sensitivity"] = {"status": "not run (full-frame coding incomplete)"}
    fd = g.groupby(["t", "symbol", "d0"], as_index=False).agg(E_effect=("E_effect", "mean"), ar=("ar", "first"))
    out["firm_day"] = {**wk(fd), "firm_days": int(len(fd))}
    return out


# ------------------------------------------------------------------------------------------------ entry points
def build_classified(root: Path, guard, spec: dict) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, dict]:
    f, start, end, d1 = frame(root, guard)
    return classify(attach_text(f), spec), start, end, d1


def cmd_audit(root: Path) -> int:
    spec, sha, _ = require_logged_spec()
    guard = C.Guard(root)
    g, *_ = build_classified(root, guard, spec)
    sample, sizes = audit_sample(g, spec)
    if AUDIT_CODES.exists():
        prev = pd.read_csv(AUDIT_CODES, comment="#")
        if sorted(prev["eid"]) == sorted(sample["eid"]) and prev["code"].notna().all():
            print(f"audit already coded: {C.rel(AUDIT_CODES)}")
            return 0
    print(f"audit: {len(sample)} items ({sizes})", flush=True)
    codes = asyncio.run(_code_audit(sample, spec))
    AUDIT_CODES.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_CODES, "w") as fh:
        fh.write(f"# {X.LABEL}; E5 audit codes (spec sha256 {sha}); coder {spec['audit']['coder']['model']}; "
                 "eids and codes only (no licensed text)\n")
        codes.to_csv(fh, index=False)
    print(f"wrote {C.rel(AUDIT_CODES)}")
    return 0


def cmd_code_frame(root: Path) -> int:
    spec, sha, _ = require_logged_spec()
    if "full_frame_coding" not in spec:
        raise SystemExit("the logged spec has no full_frame_coding section")
    guard = C.Guard(root)
    g, *_ = build_classified(root, guard, spec)
    t = g[g["has_text"]].drop_duplicates("eid")[["eid", "headline", "summary"]]
    t = t.assign(rank=[hashlib.sha256(f"E5-FRAME|{e}".encode()).hexdigest() for e in t["eid"]]).sort_values("rank")
    done = set(pd.read_csv(FRAME_CODES, comment="#")["eid"]) if FRAME_CODES.exists() else set()
    todo = t[~t["eid"].isin(done)]
    print(f"frame coding: {len(done)} done, {len(todo)} to code", flush=True)
    n = asyncio.run(_code_frame(todo, spec)) if len(todo) else 0
    print(f"frame coding: {n} coded this session -> {C.rel(FRAME_CODES)}")
    return 0


def cmd_run(root: Path, argv) -> int:
    spec, sha, log = require_logged_spec()
    guard = C.Guard(root)
    g, start, end, d1 = build_classified(root, guard, spec)
    sample, sizes = audit_sample(g, spec)  # the draw is a function of the frame and the regex only
    audit = {"sample_eids_sha256": C.sha256_text("\n".join(sorted(sample["eid"])))}
    if AUDIT_CODES.exists():
        codes = pd.read_csv(AUDIT_CODES, comment="#")
        if sorted(codes["eid"]) != sorted(sample["eid"]):
            raise SystemExit(f"{C.rel(AUDIT_CODES)} does not hold the spec's audit draw; rerun `audit`")
        served = sorted(set(codes["served_model"].dropna()))
        audit["model_coder"] = {**audit_stats(codes, sizes, spec["audit"]["coder"]["model"]),
                                "served_models": served}
    else:
        audit["model_coder"] = {"status": "not run"}
    if HUMAN_CODES.exists():
        hc = pd.read_csv(HUMAN_CODES, comment="#").merge(sample[["eid", "stratum"]], on="eid", how="inner")
        audit["human_coder"] = audit_stats(hc, sizes, "human (one coder, blind to outcomes and strata)")
    else:
        audit["human_coder"] = {"status": "not run"}
    audit["paper_coder"] = "human_coder" if audit["human_coder"].get("status") == "ok" else "model_coder"
    # outcomes from here on
    guard.check(g["t"].unique(), "E5 labels (AR)")
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    g = H2.add_abnormal_returns(g, px.book)
    g["lm_pos"] = lm_tone(g)
    fc = frame_codes_complete(g)
    if fc is not None:
        g = g.merge(fc[["eid", "code"]].rename(columns={"code": "coder"}), on="eid", how="left")
    h2 = C.read_json(root / "results" / "analysis" / "h2.json")
    payload = {"question": "construct checks of the H2 association: items that report the price move, a dictionary "
                           "yardstick, item AUC and hit rate, validity by event type, firm-day collapse",
               "spec": {"file": C.rel(SPEC), "sha256": sha, "logged_utc": log["current"]["logged_utc"],
                        "logged_before_outputs": log["current"].get("before_outputs")},
               "frame": {"n_units": int(len(g)), "n_weeks": int(g["t"].nunique()), "start": str(d1["start"])[:10],
                         "end": str(end.date()), "text_matched_share": float(g["has_text"].mean()),
                         "rule": "the registered H2 frame (h2_validity.c2_frame) from the D1 start"},
               "audit": audit, **statistics(g, h2),
               "intervals": "two-sided 95%: NW(4) for weekly means; week-block bootstrap (2,000, seed 0) for AUC and "
                            "hit rate; Wilson and a stratified bootstrap for the audit. No p-values."}
    inputs = {"work.jsonl": root / "results/judgments" / GEN / "memory" / "work.jsonl",
              "news_table": NEWS, "h2": root / "results/analysis/h2.json", "d1": root / "results/analysis/d1.json",
              "spec": SPEC, "audit codes": AUDIT_CODES if AUDIT_CODES.exists() else None,
              "human codes": HUMAN_CODES if HUMAN_CODES.exists() else None,
              "frame codes": FRAME_CODES if fc is not None else None}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs, argv=argv,
                why=spec["why_exploratory"])
    w = payload["weekly"]
    print(f"E5: price-report share {payload['price_report']['share_units']:.4f}; all {w['all']['mean']:.4f}; "
          f"no price {w['no_price']['mean']:.4f}; price only {w['price_only']['mean']:.4f} -> {C.rel(p)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("log-spec", "audit", "code-frame", "run"))
    ap.add_argument("--amend", action="store_true", help="log-spec: log a changed spec (history kept)")
    ap.add_argument("--root", default=str(C.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "log-spec":
        log = log_spec(a.amend)
        print(json.dumps(log["current"], indent=1))
        return 0
    if a.cmd == "audit":
        return cmd_audit(root)
    if a.cmd == "code-frame":
        return cmd_code_frame(root)
    return cmd_run(root, argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
