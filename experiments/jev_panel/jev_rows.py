#!/usr/bin/env python3
"""Jev rows: the DESCRIPTIVE analysis of the completed Jev panel (jev-1.13.0 on Cloudflare Workers AI).

    .venv/bin/python experiments/jev_panel/jev_rows.py            # -> results/analysis/jev_rows.json

Specification: experiments/jev_panel/jev_rows_spec.yaml, committed before any Jev outcome was read; the script refuses
to run from an uncommitted or edited spec. Label: "descriptive (Jev panel; not part of H1-H3)". Nothing here enters a
Holm family, forms a Jev-vs-emu difference or computes a p-value for one; no model is called.

(a) forward post-release validity: pooled Spearman of Jev's E[effect] with AR[0,+1] and with placebo AR[-5,-1] on the
    forward generation's units (emu's C2-eligible items created after the release date), labels from the live data
    root with the registered definition (h2_validity.add_abnormal_returns), units whose AR window is not realized in
    the data dropped and counted; emu on the identical units as a reference.
(b) aligned2000 agreement of E[effect] (Spearman) and top event type (Cohen's kappa) with emu, VERB and NG (no
    market outcome read).
(c) h2frame 8-K concept agreement (h2_validity.sec_agreement) next to emu's registered values.
(d) cost and latency per component (requests.parquet; the JEV-CF ledger read-only).
(e) D1/D2 probes as lower bounds only.
No Jev statistic of E[effect] vs AR on W_clean items is computed (contamination unverified).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "experiments" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import common as C  # noqa: E402
import d1_window_start as D1  # noqa: E402
import d2_reid as D2  # noqa: E402
import h2_validity as H  # noqa: E402
import stats as S  # noqa: E402

LABEL = "descriptive (Jev panel; not part of H1-H3)"
SPEC = Path(__file__).resolve().parent / "jev_rows_spec.yaml"
OUT = ROOT / "results" / "analysis" / "jev_rows.json"
JUDG = ROOT / "results" / "judgments"
MODEL = "jev-1.13.0"
GENS = {"forward": f"jevcf_{MODEL}_fwd", "aligned2000": f"jevcf_{MODEL}_wclean_items2000",
        "h2frame": f"jevcf_{MODEL}_wclean_h2frame", "probes": f"jevcf_{MODEL}_d1d2"}
CANARY_DIR = JUDG / f"jevcf_{MODEL}_canary"
MAIN_DIR = JUDG / "main_emu_wclean_u100"
VERB_DIR = JUDG / H.DEFAULT_VERB
NG_DIR = JUDG / H.DEFAULT_NG
LIVE_ROOT = ROOT / "data" / "live" / "root"
LEDGER = ROOT / "data" / "cache" / "spend_jev_cf.sqlite"
RELEASE = pd.Timestamp("2026-09-15")  # docs/jev (release_date_archived in every MANIFEST; checked below)
N_BOOT, SEED = H.N_BOOT, H.SEED
FILES = ("answers", "requests", "evidence", "jobs")


# ------------------------------------------------------------------------------------------------ provenance
def spec_provenance(root: Path = ROOT) -> dict:
    rel = C.rel(SPEC, root)
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)  # noqa: E731
    commit = run("log", "-1", "--format=%H", "--", rel).stdout.strip()
    if not commit or run("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} is not committed in its current form; jev_rows runs only from the committed spec")
    first = run("log", "--diff-filter=A", "--format=%H %cI", "--", rel).stdout.strip().split()
    spec = yaml.safe_load(SPEC.read_text())
    if spec["label"] != LABEL:
        raise SystemExit("spec label differs from the script's")
    return {"file": rel, "sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(), "commit": commit,
            "added_in": first[0] if first else None, "added_utc": first[1] if len(first) > 1 else None}


def generation(gen_dir: Path, validated: bool = True) -> dict:
    """MANIFEST hashes checked against the files; validation must pass (the canary has no validation file: it is
    an identity check only and no statistic is computed on it)."""
    man = json.loads((gen_dir / "MANIFEST.json").read_text())
    got = {f: C.sha256_file(gen_dir / f"{f}.parquet") for f in FILES}
    if got != man["sha256"]:
        raise SystemExit(f"{gen_dir.name}: file hashes differ from its MANIFEST")
    if validated and not json.loads((gen_dir / "validation.json").read_text()).get("pass"):
        raise SystemExit(f"{gen_dir.name}: validation did not pass")
    b = man["backend"]
    if b["model_pinned"] != MODEL or b["identity"]["release_date_archived"] != str(RELEASE.date()):
        raise SystemExit(f"{gen_dir.name}: not {MODEL} with release {RELEASE.date()}")
    return {"gen_id": man["gen_id"], "component": man.get("component"), "sha256": man["sha256"],
            "manifest_sha256": C.sha256_file(gen_dir / "MANIFEST.json"), "labels": man.get("labels"),
            "validity_rate": man.get("validity_rate"), "validation_pass": True if validated else None,
            "response_model": b["identity"].get("response_model")}


def latest_canary() -> Path:
    runs = sorted(d for d in CANARY_DIR.iterdir() if (d / "MANIFEST.json").exists())
    if not runs:
        raise SystemExit("no canary generation")
    return runs[-1]


# ------------------------------------------------------------------------------------------------ Jev items
def jev_items(gen_dir: Path) -> pd.DataFrame:
    """Jev's item-level answers per unit, exactly as h2_validity.verb_items (which needs a sample.parquet that a
    panel generation does not have: the units here come from jobs.parquet's meta_json instead)."""
    from jevtrader.programs.news import triage
    from jevtrader.s1.types import Answer

    jobs = pd.read_parquet(gen_dir / "jobs.parquet", columns=["uid", "request_key", "meta_json"])
    meta = [json.loads(x) for x in jobs["meta_json"]]
    jobs = jobs.assign(unit=[m["unit"] for m in meta], group=[m["group"] for m in meta],
                       use=[tuple(m["use"]) for m in meta], label=[m["label"] for m in meta])
    ans = pd.read_parquet(gen_dir / "answers.parquet", columns=["request_key", "qid", "kind", "probs_json"])
    ans["qid"] = ans["qid"].astype(str).str.replace("N1.", "", regex=False)
    ans = ans.drop_duplicates(["request_key", "qid"])
    j = jobs[["unit", "group", "use", "request_key"]].merge(ans, on="request_key", how="left")
    j = j[[q in u for q, u in zip(j["qid"], j["use"])]]
    per: dict[str, dict] = {}
    groups: dict[str, set] = {}
    for u, g, q, k, pj in j[["unit", "group", "qid", "kind", "probs_json"]].itertuples(index=False):
        probs = json.loads(pj) if isinstance(pj, str) else None
        per.setdefault(u, {})[q] = Answer(str(k), {str(a): float(b) for a, b in probs.items()} if probs else None)
        groups.setdefault(u, set()).add(g)
    labels = jobs.groupby("unit")["label"].agg(lambda s: "|".join(sorted(set(s))))
    rows = []
    for u in sorted(jobs["unit"].unique()):
        t, sym, eid = u.split("|", 2)
        a = per.get(u, {})
        dist = {q: (a[q].probs if q in a else None) for q in ("event_type", "effect", "vs_expectations")}
        tri = triage(a, n0_quarantined=False)
        top = H.top_label(dist["event_type"])
        complete = groups.get(u, set()) == set(H.VERB_GROUPS)
        rows.append({"unit": u, "t": pd.Timestamp(t), "symbol": sym, "eid": eid, "label": labels[u],
                     "E_effect": H.expected_effect(dist["effect"]), "top_type": top,
                     "p_type": dist["event_type"] or {}, "kept": bool(tri.kept), "complete": complete,
                     "c2_eligible": bool(complete and tri.kept and top is not None and top != H.EXCLUDED_TYPE)})
    return pd.DataFrame(rows)


def emu_units(gen_dir: Path, start, end) -> pd.DataFrame:
    return H.c2_frame(H.item_rows(gen_dir, start, end))


# ------------------------------------------------------------------------------------------------ (a) forward
def forward_sources() -> list[tuple[str, Path]]:
    out = [(w, MAIN_DIR) for w in ("2026-09-18",)]
    for d in sorted(JUDG.glob("live_*")):
        if (d / "MANIFEST.json").exists():
            out.append((d.name.split("_")[1], d))
    return out


def bars_written_utc(droot: Path) -> pd.Timestamp:
    return min(pd.Timestamp((droot / f).stat().st_mtime, unit="s", tz="UTC") for f in C.BARS_FILES)


def forward(guard: C.Guard) -> tuple[dict, dict]:
    from jevtrader.learn.labels import _event_session, close_time_utc

    jev = jev_items(JUDG / GENS["forward"])
    if set(jev["label"]) != {"post-release"}:
        raise SystemExit("forward: a unit is not labelled post-release")
    parts = []
    for week, gdir in forward_sources():
        f = emu_units(gdir, week, week)
        f = f[f["created_at"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None) > RELEASE]
        parts.append(f.assign(source=gdir.name))
    emu = pd.concat(parts, ignore_index=True)
    if set(emu["unit"]) != set(jev["unit"]) or emu["unit"].duplicated().any():
        raise SystemExit(f"forward: rebuilt units ({len(emu)}) differ from the Jev generation's ({len(jev)})")
    # labels (identical definition to H2), bars of the live data root
    px = C.load_prices(guard, emu["t"].min(), emu["t"].max() + pd.Timedelta(days=7), droot=LIVE_ROOT)
    emu = H.add_abnormal_returns(emu, px.book)
    sess = px.book.sessions
    last, written = pd.Timestamp(sess[-1]), bars_written_utc(px.root)
    if not close_time_utc(last) < written:
        raise SystemExit("forward: the live bars' last session closed after the files were written")
    reason = []
    for ts, ar, pl in zip(emu["created_at"], emu["ar"], emu["ar_placebo"]):
        k0 = _event_session(ts, sess)
        if k0 is None or k0 + H.AR_WINDOW[1] >= len(sess):
            reason.append("AR window not realized by the last session in the data")
        elif not (np.isfinite(ar) and np.isfinite(pl)):
            reason.append("AR or placebo not computable")
        else:
            reason.append(None)
    emu["drop"] = reason
    j = jev.set_index("unit")
    emu["E_jev"] = j.loc[emu["unit"], "E_effect"].to_numpy(dtype=float)
    emu["jev_c2_eligible"] = j.loc[emu["unit"], "c2_eligible"].to_numpy(dtype=bool)
    keep = emu[emu["drop"].isna()].copy()
    per_week = {}
    for t, g in emu.groupby("t"):
        per_week[str(pd.Timestamp(t).date())] = {
            "source": g["source"].iloc[0], "units": len(g), "kept": int(g["drop"].isna().sum()),
            "dropped": {k: int(v) for k, v in g["drop"].value_counts().items()}}
    res = {
        "admissible_use": "post-release item validity, E[effect] vs later reactions (descriptive)",
        "units_total": len(emu), "units_used": len(keep),
        "units_dropped": {k: int(v) for k, v in emu["drop"].value_counts().items()},
        "per_week": per_week,
        "jev": {"spearman_ar": H.pooled_spearman(keep, "E_jev"),
                "spearman_ar_placebo": H.pooled_spearman(keep, "E_jev", "ar_placebo")},
        "emu_reference_same_units": {"spearman_ar": H.pooled_spearman(keep, "E_effect"),
                                     "spearman_ar_placebo": H.pooled_spearman(keep, "E_effect", "ar_placebo"),
                                     "note": "reference, not a test: no p-value, no difference"},
        "jev_emu_agreement_spearman_E_effect": S.spearman(keep["E_jev"], keep["E_effect"]),
        "jev_c2_eligible_own_rate": float(keep["jev_c2_eligible"].mean()) if len(keep) else None,
        "jev_c2_eligible_own_ci": S.wilson(int(keep["jev_c2_eligible"].sum()), len(keep)),
        "point_in_time": {"data_root": C.rel(px.root), "last_session": str(last.date()),
                          "bars_written_utc": written.isoformat(timespec="seconds"),
                          "rule": "AR[0,+1] window's last session <= the last session of the live bars, whose close "
                                  "precedes the bar files' write time"},
        "settings": {"ar_window": H.AR_WINDOW, "placebo_window": H.PLACEBO_WINDOW, "n_boot": N_BOOT, "seed": SEED,
                     "release_date": str(RELEASE.date()), "post_release": "created on a UTC day after the release"},
    }
    inputs = {f"forward source {g.name}": g / "memory" / "work.jsonl" for _, g in forward_sources()}
    inputs.update({f"live bars {Path(f).name}": px.root / f for f in C.BARS_FILES})
    return res, inputs


# ------------------------------------------------------------------------------------------------ (b) aligned
def aligned() -> dict:
    jev = jev_items(JUDG / GENS["aligned2000"]).set_index("unit")
    verb = H.verb_items(VERB_DIR).set_index("unit")
    ng = H.implementation_items(NG_DIR).set_index("unit")
    emu = emu_units(MAIN_DIR, *C.windows()["W_clean"])
    emu_all = H.item_rows(MAIN_DIR, *C.windows()["W_clean"])
    emu_all["unit"] = [f"{t.date()}|{s}|{e}" for t, s, e in zip(emu_all["t"], emu_all["symbol"], emu_all["eid"])]
    emu_all = emu_all.set_index("unit")
    if set(jev.index) != set(verb.index):
        raise SystemExit("aligned2000: Jev's units differ from VERB's sample")
    if not set(ng.index) <= set(jev.index):
        raise SystemExit("aligned2000: NG's units are not nested in the aligned sample")

    def pair(a: pd.DataFrame, b: pd.DataFrame, units) -> dict:
        g = pd.DataFrame({"t": a.loc[units, "t"].to_numpy(), "x": a.loc[units, "E_effect"].to_numpy(dtype=float),
                          "y": b.loc[units, "E_effect"].to_numpy(dtype=float)})
        g = g[np.isfinite(g["x"]) & np.isfinite(g["y"])]
        bs = S.week_block_bootstrap_xy(g, "t", "x", "y", S.spearman, n_boot=N_BOOT, seed=SEED)
        return {"n_units": len(units), "spearman_E_effect": {"rho": S.spearman(g["x"], g["y"]), "n": len(g),
                                                             "weeks": int(g["t"].nunique()),
                                                             "ci_week_block": bs["ci"]},
                "top_event_type": S.cohen_kappa(a.loc[units, "top_type"], b.loc[units, "top_type"])}

    u2000, u500 = sorted(jev.index), sorted(ng.index)
    return {"admissible_use": "agreement with emu, VERB and NG on identical requests (descriptive)",
            "units": len(u2000), "units_ng": len(u500),
            "aligned_units_in_emu_c2_frame": int(pd.Index(u2000).isin(emu["unit"]).sum()),
            "jev_complete_units": int(jev["complete"].sum()),
            "jev_vs_emu": pair(jev, emu_all, u2000), "jev_vs_verb": pair(jev, verb, u2000),
            "jev_vs_ng": pair(jev, ng, u500),
            "reference": {"emu_vs_verb": pair(emu_all, verb, u2000), "emu_vs_ng": pair(emu_all, ng, u500)},
            "outcome_read": "none"}


# ------------------------------------------------------------------------------------------------ (c) 8-K
def h2frame() -> dict:
    reg = json.loads((ROOT / "results/analysis/h2.json").read_text())
    w0, w1 = C.windows()["W_clean"]
    start = pd.Timestamp(C.d1_start(ROOT)["start"])
    f = emu_units(MAIN_DIR, w0, w1)
    f = f[f["t"] >= start].reset_index(drop=True)
    jev = jev_items(JUDG / GENS["h2frame"]).set_index("unit")
    frame_n = json.loads((ROOT / "results/analysis/h2_frame.json").read_text())["frame"]["units"]
    if len(f) != frame_n:
        raise SystemExit(f"h2frame: rebuilt frame {len(f)} != h2_frame.json {frame_n}")
    missing = int((~f["unit"].isin(jev.index)).sum())
    f = H.sec_labels(f, ROOT / C.SUBMISSIONS)
    emu_re = H.sec_agreement(f)
    jf = f[f["unit"].isin(jev.index)].copy()
    jf["p_type"] = [jev.at[u, "p_type"] for u in jf["unit"]]
    jev_sec = H.sec_agreement(jf)
    reproduced = {k: abs(emu_re[k]["auc"] - reg["sec_8k"][k]["auc"]) < 1e-9 for k in emu_re}
    if missing == 0 and not all(reproduced.values()):
        raise SystemExit(f"h2frame: emu's 8-K AUCs not reproduced on the identical units: {reproduced}")
    return {"admissible_use": "8-K concept agreement on the registered frame (C2 secondary); never a confirmatory "
                              "or comparative H2",
            "frame_units": len(f), "jev_units": len(jf), "frame_units_without_jev_answer": missing,
            "frame_start": str(start.date()), "jev_sec_8k": jev_sec,
            "emu_registered_sec_8k": reg["sec_8k"], "emu_reproduced": reproduced,
            "sec_symbols_without_archive": f.attrs.get("sec_symbols_without_archive", [])}


# ------------------------------------------------------------------------------------------------ (d) cost
def cost_latency(dirs: dict[str, Path]) -> dict:
    out, tot = {}, {"requests": 0, "paid": 0, "input_tokens": 0, "usd_rows": 0.0, "usd_paid": 0.0}
    for comp, d in dirs.items():
        r = pd.read_parquet(d / "requests.parquet", columns=["usd", "latency_s", "cached", "input_tokens"])
        paid = r[~r["cached"].astype(bool)]
        lat = paid["latency_s"].to_numpy(dtype=float)
        lat = lat[np.isfinite(lat)]
        val = json.loads((d / "validation.json").read_text()) if (d / "validation.json").exists() else {}
        row = {"requests": len(r), "paid": len(paid), "input_tokens": int(r["input_tokens"].sum()),
               "usd_rows": float(r["usd"].sum()), "usd_paid": float(paid["usd"].sum()),
               "latency_paid_s": {"median": float(np.median(lat)), "p90": float(np.quantile(lat, .9)),
                                  "p99": float(np.quantile(lat, .99)), "n": int(lat.size)} if lat.size else None,
               "jev_ledger_usd_at_validation": (val.get("summary") or {}).get("jev_ledger_usd")}
        if comp in ("aligned2000", "forward", "h2frame"):
            items = len(r) / 5
            row["usd_per_1000_items_paid"] = float(paid["usd"].sum() / items * 1000) if items else None
            row["items"] = int(items)
        out[comp] = row
        for k in tot:
            tot[k] += row[k]
    con = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    n, usd, t0, t1 = con.execute("select count(*), sum(usd), min(ts), max(ts) from spend").fetchone()
    con.close()
    ts = lambda x: pd.Timestamp(x, unit="s", tz="UTC").isoformat(timespec="seconds") if x else None  # noqa: E731
    return {"admissible_use": "cost and latency", "by_component": out, "totals_requests_parquet": tot,
            "ledger": {"path": C.rel(LEDGER), "rows": int(n), "usd": float(usd or 0.0), "first_utc": ts(t0),
                       "last_utc": ts(t1), "opened": "read-only"},
            "price_usd_per_m_input": json.loads((dirs["forward"] / "MANIFEST.json").read_text())["backend"]
            ["price_usd_per_m"]}


# ------------------------------------------------------------------------------------------------ (e) probes
def auc_cell(g: pd.DataFrame) -> dict:
    from jevtrader.eval.metrics import auc

    g = g.dropna(subset=["p", "y"])
    out = {"n": len(g), "weeks": int(g["t"].nunique()), "positives": int(g["y"].sum()), "auc": float("nan"),
           "ci_week_block": [float("nan")] * 2}
    if len(g) and 0 < g["y"].sum() < len(g):
        out["auc"] = float(auc(g["p"], g["y"]))
        out["ci_week_block"] = S.week_block_bootstrap_xy(g, "t", "p", "y", auc, n_boot=N_BOOT, seed=SEED)["ci"]
    return out


def probes(guard: C.Guard) -> dict:
    m0, m1 = D1.PROBE_MONTHS
    months = D1.months_between(m0, m1)
    lo, hi = pd.Timestamp(f"{m0}-01"), pd.Period(m1, "M").end_time.normalize()
    cal = D1.rule_calendar(ROOT, lo)
    uni = D1._universe(ROOT, cal)
    sched = D1.schedule(cal, uni, months)
    jev = D1.probe_answers(JUDG / GENS["probes"]).rename(columns={"p": "p_jev"})
    emu = D1.union_answers([JUDG / x for x in D1.REGISTERED_PROBES]).rename(columns={"p": "p_emu"})
    fr = sched.merge(jev[["t", "symbol", "p_jev"]], on=["t", "symbol"], how="left") \
        .merge(emu[["t", "symbol", "p_emu"]], on=["t", "symbol"], how="left")
    guard.check(fr["t"].unique(), "Jev D1 labels")
    px = C.load_prices(guard, lo, hi)
    ts = sorted(fr["t"].unique())
    labels = C.load_labels(px, ts, {t: uni[t] for t in ts})
    fr = fr.merge(labels.table[["t", "symbol", "y"]], on=["t", "symbol"], how="left")
    centre = D1.months_between(*(p.strftime("%Y-%m") for p in (pd.Period(D1.CENTER, "M") - 1,
                                                                    pd.Period(D1.CENTER, "M") + 1)))
    d1 = {}
    for who, col in (("jev", "p_jev"), ("emu_reference", "p_emu")):
        g = fr.rename(columns={col: "p"})
        d1[who] = {"answered": int(g["p"].notna().sum()), "scheduled": len(g),
                   "centre_window": {"months": centre, **auc_cell(g[g["month"].isin(centre)])},
                   "probe_period": {"months": [m0, m1], **auc_cell(g)}}
    d2rows = D2.load_rows(JUDG / GENS["probes"])
    guard.check(d2rows["t"].unique(), "Jev D2 rates")
    s = D2.score_rows(d2rows)
    d2 = {"jev": {"pooled_by_level": D2.summarize(s)["pooled_by_level"], "n_requests": len(s),
                  "n_unanswered": int((~s["answered"]).sum())}}
    reg_d2 = json.loads((ROOT / "results/analysis/d2.json").read_text())
    d2["emu_reference_registered"] = {"pooled_by_level": reg_d2.get("pooled_by_level")}
    return {"admissible_use": "D1 recall and D2 re-identification as lower bounds only",
            "reading": "a probe that detects recall bounds Jev's recall from below; one that does not is not "
                       "evidence of no recall", "d1": d1, "d2": d2,
            "d1_definition": "P(beat) of D.recall_beats_median vs the realized T1 label on the registered D1 "
                             "schedule; AUC with a descriptive week-block 95% CI"}


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    spec = spec_provenance()
    guard = C.Guard(ROOT)
    if not guard.open:
        raise SystemExit("refused: the W_clean lock is closed")
    gens = {k: generation(JUDG / v) for k, v in GENS.items()}
    can = latest_canary()
    gens["canary"] = generation(can, validated=False)
    fwd, fwd_inputs = forward(guard)
    payload = {"label": LABEL, "model": MODEL,
               "forward": fwd, "aligned2000": aligned(), "h2frame": h2frame(),
               "cost_latency": cost_latency({"canary": can, **{k: JUDG / v for k, v in GENS.items()}}),
               "probes": probes(guard),
               "not_computed": yaml.safe_load(SPEC.read_text())["not_computed"]}
    inputs = {"spec": SPEC, "h2.json": ROOT / "results/analysis/h2.json",
              "h2_frame.json": ROOT / "results/analysis/h2_frame.json", "d2.json": ROOT / "results/analysis/d2.json",
              "main work.jsonl": MAIN_DIR / "memory" / "work.jsonl", **fwd_inputs,
              **H.implementation_inputs("VERB", VERB_DIR), **H.implementation_inputs("NG", NG_DIR)}
    path = C.write_result(Path(a.out), payload, script=__file__, window="jev_panel", guard=guard, inputs=inputs,
                          argv=argv, extra_meta={"label": LABEL, "descriptive": True, "spec": spec,
                                                 "generations": gens})
    f = fwd["jev"]["spearman_ar"]
    print(f"Jev rows: forward Spearman {f['rho']:.3f} {f['ci_week_block']} (n {f['n']}, {f['weeks']} weeks) -> "
          f"{C.rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
