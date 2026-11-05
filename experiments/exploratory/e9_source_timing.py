#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E9: does the information survive at the source? ($0; no model call)

    .venv/bin/python experiments/exploratory/e9_source_timing.py fetch  # missing minute bars (network, read-only)
    .venv/bin/python experiments/exploratory/e9_source_timing.py run    # -> results/exploratory/e9_source_timing.json

Specification: experiments/exploratory/e9_spec.yaml, committed before any E9 outcome existed; the script refuses to
run if the file differs from its committed version and writes its SHA-256 and commit into the output's meta.

For frame units whose top event type maps to an 8-K concept (Item 2.02 earnings; 1.01/2.01 deals; 5.02 management),
the focal firm's matching 8-K accepted nearest to the item's created_at within the registered label window is
located. The registered label AR[0,+1] is decomposed into the part realized before the first trade after the 8-K's
SEC acceptance, the part between that trade and the first trade after the news item's publication, and the part after
it (E6's publication-aligned criterion, reused unchanged with the acceptance time in place of created_at). The
reader's existing E[effect] (emu, registered H2 frame) is then correlated, week by week, with the return from the
first trade after acceptance and from the first trade after publication on the same units. Every interval is a
two-sided 95% interval (NW(4) for weekly statistics; week-block bootstrap, 2,000 draws, seed 0, for shares); no
p-value is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402  (registered, read-only)
import e5_construct_checks as E5  # noqa: E402  (exploratory frame; read-only)
import e6_publication_timing as E6  # noqa: E402  (exploratory frame, minute bars, publication-aligned AR; read-only)
import h2_validity as H2  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402  (registered, read-only)

NAME = "e9_source_timing"
GEN = E6.GEN
SPEC = Path(__file__).resolve().parent / "e9_spec.yaml"
WINDOW_BEFORE, WINDOW_AFTER = pd.Timedelta(days=2), pd.Timedelta(days=1)  # registered 8-K label window
GROUPS = {"earnings": ("earnings_results",), "other": ("deal_or_investment", "management_or_labor"),
          "all": ("earnings_results", "deal_or_investment", "management_or_labor")}
COMPONENTS = ("pre", "between", "after")
N_BOOT, SEED = 2000, 0


# ------------------------------------------------------------------------------------------------ spec
def spec_provenance(root: Path) -> dict:
    """SHA-256 of the spec and the commit that holds it; refuses an uncommitted or edited spec."""
    rel = C.rel(SPEC, root)
    run = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)  # noqa: E731
    commit = run("log", "-1", "--format=%H", "--", rel).stdout.strip()
    if not commit or run("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SystemExit(f"{rel} is not committed in its current form; E9 runs only from the committed spec")
    first = run("log", "--diff-filter=A", "--format=%H %cI", "--", rel).stdout.strip().split()
    return {"file": rel, "sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(), "commit": commit,
            "added_in": first[0] if first else None, "added_utc": first[1] if len(first) > 1 else None,
            "id": yaml.safe_load(SPEC.read_text())["id"]}


# ------------------------------------------------------------------------------------------------ filings
def match_filings(root: Path, f: pd.DataFrame) -> pd.DataFrame:
    """Per unit of a mapped type: the nearest concept 8-K accepted in [created - 2d, created + 1d] (no outcome)."""
    from jevtrader.data.earnings import load_8k_items

    sub = root / C.SUBMISSIONS
    out = pd.DataFrame({"unit": f["unit"], "accepted_utc": pd.NaT, "accession": None, "items": None})
    out["accepted_utc"] = pd.to_datetime(out["accepted_utc"], utc=True)
    mapped = f["top_type"].isin(H2.SEC_CONCEPTS)
    for sym, hs in f[mapped].groupby("symbol"):
        k = load_8k_items(sub, sym)
        if not len(k):
            continue
        for tt, hh in hs.groupby("top_type"):
            codes = H2.SEC_CONCEPTS[tt]
            kk = k[k["items"].map(lambda xs, c=codes: any(x in c for x in xs))].reset_index(drop=True)
            if not len(kk):
                continue
            ts = kk["accepted_utc"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")
            for i, c in zip(hh.index, hh["created_at"].dt.tz_convert("UTC").dt.tz_localize(None)
                            .to_numpy(dtype="datetime64[ns]")):
                ok = (ts >= c - WINDOW_BEFORE.to_timedelta64()) & (ts <= c + WINDOW_AFTER.to_timedelta64())
                if not ok.any():
                    continue
                j = np.flatnonzero(ok)[np.argmin(np.abs(ts[ok] - c))]
                out.at[i, "accepted_utc"] = kk.at[j, "accepted_utc"]
                out.at[i, "accession"] = kk.at[j, "accession"]
                out.at[i, "items"] = ",".join(kk.at[j, "items"])
    out["group"] = np.where(f["top_type"] == "earnings_results", "earnings",
                            np.where(mapped, "other", None))
    return out


def cmd_fetch(root: Path) -> int:
    from dotenv import load_dotenv

    from jevtrader.data.alpaca_data import AlpacaData

    load_dotenv(root / ".env")
    guard = C.Guard(root)
    f, start, end, _ = E5.frame(root, guard)  # Layer-A answers only; no outcome is read
    m = match_filings(root, f.reset_index(drop=True))
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    ok = m["accepted_utc"].notna().to_numpy()
    cands = E6.entry_dates(m.loc[ok, "accepted_utc"], px.book.sessions)
    need: dict[pd.Timestamp, set] = {}
    for sym, cs in zip(f.loc[ok, "symbol"], cands):
        for d in cs:
            need.setdefault(d, set()).update((sym, E6.MARKET))
    missing = {}
    for d, syms in need.items():
        side = E6._day_path(d).with_suffix(".json")
        have = set(C.read_json(side)["symbols"]) if side.exists() else set()
        if syms - have:
            missing[d] = syms
    print(f"E9 entries need {len(need)} session dates; {len(missing)} lack some symbol in the cache")
    client = AlpacaData()
    for i, d in enumerate(sorted(missing), 1):
        E6.fetch_day(client, d, sorted(missing[d]))
        if i % 25 == 0 or i == len(missing):
            print(f"  minute bars: {i}/{len(missing)} session dates", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ statistics
def aligned(g: pd.DataFrame, book, when: pd.Series, tag: str) -> pd.DataFrame:
    """E6.publication_aligned with `when` in place of created_at (same d0, beta, exit, bars)."""
    h = g[["unit", "symbol", "d0"]].copy()
    h["created_at"] = when.to_numpy()
    h["created_at"] = pd.to_datetime(h["created_at"], utc=True)
    r = E6.publication_aligned(h.reset_index(drop=True), book)
    return r.rename(columns={"ar_pub": f"ar_{tag}", "ar_pub_regular": f"ar_{tag}_regular", "lag_min": f"lag_{tag}",
                             "lag_min_regular": f"lag_{tag}_regular", "status": f"status_{tag}"})


def _week_sums(h: pd.DataFrame, comp: str):
    a, c = h["ar"].to_numpy(dtype=float), h[comp].to_numpy(dtype=float)
    d = pd.DataFrame({"t": h["t"].to_numpy(), "sc": np.sign(a) * c, "abs": np.abs(a), "n": 1.0, "a": a, "c": c,
                      "aa": a * a, "ac": a * c})
    return d.groupby("t", sort=True).sum().to_numpy()


def _shares(w: np.ndarray) -> tuple[float, float]:
    sc, ab, n, a, c, aa, ac = w
    slope = (ac - a * c / n) / (aa - a * a / n)
    return float(sc / ab), float(slope)


def shares(h: pd.DataFrame) -> dict:
    """Sign-aligned and slope shares of each component with week-block bootstrap intervals (spec decomposition)."""
    out = {"n_units": int(len(h)), "n_weeks": int(h["t"].nunique()), "mean_abs_ar_0_1": float(h["ar"].abs().mean())}
    if len(h) < 3:
        return out
    rng = np.random.default_rng(SEED)
    W = {k: _week_sums(h, k) for k in COMPONENTS}
    G = len(next(iter(W.values())))
    idx = rng.integers(0, G, size=(N_BOOT, G))
    for k in COMPONENTS:
        est = _shares(W[k].sum(axis=0))
        bs = np.array([_shares(W[k][ii].sum(axis=0)) for ii in idx])
        out[k] = {"sign_aligned": est[0], "sign_aligned_ci": np.nanpercentile(bs[:, 0], [2.5, 97.5]).tolist(),
                  "slope": est[1], "slope_ci": np.nanpercentile(bs[:, 1], [2.5, 97.5]).tolist(),
                  "mean": float(h[k].mean()), "median_signed_by_ar": float(np.median(np.sign(h["ar"]) * h[k]))}
    return out


def wk(f: pd.DataFrame, y: str) -> dict:
    return E6.wk(f, y=y)


def paired(f: pd.DataFrame, y1: str, y2: str) -> dict:
    h = f.dropna(subset=["E_effect", y1, y2])
    a = S.weekly_spearman(h, "E_effect", y1, "t", H2.MIN_UNITS).set_index("t")["rho"]
    b = S.weekly_spearman(h, "E_effect", y2, "t", H2.MIN_UNITS).set_index("t")["rho"]
    d = (a - b).dropna()
    r = S.nw_test(d.to_numpy(), S.NW_LAGS, "two-sided")
    return {"mean": r["mean"], "ci": r["ci"], "se": r["se"], "n_weeks": int(len(d)), "n_units": int(len(h)),
            "definition": f"mean over weeks of rho(E[effect], {y1}) - rho(E[effect], {y2}), same units; NW(4)"}


def association(h: pd.DataFrame) -> dict:
    return {"ar_0_1": wk(h, "ar"), "ar_acc": wk(h, "ar_acc"), "ar_pub": wk(h, "ar_pub"),
            "ar_acc_regular": wk(h, "ar_acc_regular"), "acc_minus_pub": paired(h, "ar_acc", "ar_pub")}


def clock_group(ts: pd.Series, sessions) -> np.ndarray:
    et = ts.dt.tz_convert(C.ET)
    ses = et.dt.tz_localize(None).dt.normalize().isin(pd.DatetimeIndex(sessions)).to_numpy()
    hm = et.dt.strftime("%H:%M").to_numpy()
    return np.where(~ses | (hm >= E6.EXT_CLOSE), "other",
                    np.where(hm < E6.REG_OPEN, "pre_open", np.where(hm < E6.REG_CLOSE, "intraday", "after_close")))


def event_level(h: pd.DataFrame) -> pd.DataFrame:
    return h.sort_values(["created_at", "eid"]).drop_duplicates(["symbol", "accession"]).sort_values("t")


def cache_coverage(root: Path) -> dict:
    js = sorted(E6.MIN_DIR.glob("*.json"))
    win = {tuple(C.read_json(p).get("window_et") or []) for p in js}
    return {"dir": C.rel(E6.MIN_DIR, root), "session_dates": len(js),
            "first": js[0].stem if js else None, "last": js[-1].stem if js else None,
            "window_et": [list(w) for w in win], "feed": "sip", "adjustment": "raw"}


def cmd_run(root: Path, argv) -> int:
    spec = spec_provenance(root)
    guard = C.Guard(root)
    g, px, reg, start, end, d1 = E6.build(root, guard)
    book = px.book
    m = match_filings(root, g)
    g = g.merge(m, on="unit", how="left")
    g = g.merge(E6.publication_aligned(g, book), on="unit", how="left")
    mt = g[g["accepted_utc"].notna()].copy()
    ses = pd.DatetimeIndex(book.sessions)
    k0 = ses.searchsorted(pd.to_datetime(mt["d0"]))
    from jevtrader.learn.labels import close_time_utc

    mt["window_start"] = [close_time_utc(ses[k - 1]) if k >= 1 else pd.NaT for k in k0]
    mt["window_start"] = pd.to_datetime(mt["window_start"], utc=True)
    mt["filing_first"] = mt["accepted_utc"] <= mt["created_at"]
    mt["gap_min"] = (mt["created_at"] - mt["accepted_utc"]).dt.total_seconds() / 60
    src = mt[["accepted_utc", "created_at"]].min(axis=1)
    mt = mt.merge(aligned(mt, book, mt["accepted_utc"], "acc"), on="unit", how="left")
    mt = mt.merge(aligned(mt, book, src, "src"), on="unit", how="left")
    mt["acc_clock"] = clock_group(mt["accepted_utc"], ses)
    reason = np.where(~mt["filing_first"], "news_first",
                      np.where(mt["accepted_utc"] < mt["window_start"], "filing_before_window",
                               np.where(mt[["ar", "ar_acc", "ar_pub"]].notna().all(axis=1), "decomposition",
                                        "no_return")))
    mt["reason"] = reason
    dec = mt[mt["reason"] == "decomposition"].copy()
    dec["pre"], dec["between"], dec["after"] = dec["ar"] - dec["ar_acc"], dec["ar_acc"] - dec["ar_pub"], dec["ar_pub"]

    out_groups = {}
    for gname, types in GROUPS.items():
        sel = lambda d: d[d["top_type"].isin(types)]  # noqa: E731
        md, dd = sel(mt), sel(dec)
        ev = event_level(dd)
        cl = {}
        for k in ("pre_open", "intraday", "after_close", "other"):
            hk = dd[dd["acc_clock"] == k]
            cl[k] = {"share": float(len(hk) / len(dd)) if len(dd) else None,
                     "shares": shares(hk) if len(hk) >= 20 else {"n_units": int(len(hk))}}
        out_groups[gname] = {
            "frame_units_of_type": int(g["top_type"].isin(types).sum()),
            "counts": {"matched": int(len(md)), **{str(k): int(v) for k, v in md["reason"].value_counts().items()},
                       "decomposition_events": int(len(ev))},
            "share_filing_first_among_matched": float(md["filing_first"].mean()) if len(md) else None,
            "minutes_acceptance_to_publication": {
                "median": float(dd["gap_min"].median()) if len(dd) else None,
                "p25": float(dd["gap_min"].quantile(.25)) if len(dd) else None,
                "p75": float(dd["gap_min"].quantile(.75)) if len(dd) else None},
            "minutes_acceptance_to_entry_bar": {
                "median": float(dd["lag_acc"].median()) if len(dd) else None,
                "share_le_5": float((dd["lag_acc"] <= 5).mean()) if len(dd) else None},
            "acceptance_clock": cl,
            "decomposition": {"unit_level": shares(dd), "event_level": shares(ev)},
            "association": {"unit_level": association(dd), "event_level": association(ev)},
            "earliest_source_on_matched": {"ar_src": wk(md, "ar_src"), "ar_pub": wk(md, "ar_pub"),
                                           "ar_0_1": wk(md, "ar"), "src_minus_pub": paired(md, "ar_src", "ar_pub")},
        }
    cov = cache_coverage(root)
    stat = {k: {str(a): int(b) for a, b in mt[f"status_{k}"].value_counts().items()} for k in ("acc", "src")}
    payload = {"question": "how much of AR[0,+1] is still ahead at the first trade after the issuer's 8-K "
                           "acceptance, and does the reader's E[effect] track the return from that trade",
               "spec": spec,
               "frame": {"n_units": int(len(g)), "n_weeks": int(g["t"].nunique()),
                         "h2_mean_recomputed": H2.weekly_test(g)["mean"],
                         "h2_mean_registered": reg["confirmatory"]["mean"],
                         "units_of_mapped_types": int(g["top_type"].isin(H2.SEC_CONCEPTS).sum())},
               "minute_bars": {**cov, "status_acc": stat["acc"], "status_src": stat["src"]},
               "groups": out_groups,
               "settings": {"nw_lags": S.NW_LAGS, "min_units_per_week": H2.MIN_UNITS, "n_boot": N_BOOT,
                            "seed": SEED, "label_window_days": [-2, 1], "concepts": H2.SEC_CONCEPTS,
                            "groups": {k: list(v) for k, v in GROUPS.items()}}}
    inputs = {"work.jsonl": root / "results/judgments" / GEN / "memory" / "work.jsonl",
              "h2": root / "results/analysis/h2.json", "spec": SPEC, "submissions": root / C.SUBMISSIONS,
              "minute cache": E6.MIN_DIR}
    meta = {"label": X.LABEL, "exploratory": True,
            "why_exploratory": "source-timing decomposition specified after the H2 outcome and E6/E8 "
                               "(experiments/exploratory/e9_spec.yaml, committed before any E9 outcome)",
            "spec": {"file": spec["file"], "sha256": spec["sha256"], "commit": spec["commit"],
                     "added_in": spec["added_in"]}}
    p = C.write_result(X.out_path(NAME), {"label": X.LABEL, **payload}, script=__file__, window="W_clean",
                       guard=guard, inputs=inputs, argv=argv, extra_meta=meta)
    for gname, v in out_groups.items():
        d, a = v["decomposition"]["unit_level"], v["association"]["unit_level"]
        print(f"[{gname}] counts {v['counts']}")
        if "pre" in d:
            print("   shares (sign-aligned) " + "  ".join(
                f"{k} {d[k]['sign_aligned']:.3f} {[round(x, 3) for x in d[k]['sign_aligned_ci']]}" for k in COMPONENTS))
            print("   shares (slope)        " + "  ".join(
                f"{k} {d[k]['slope']:.3f} {[round(x, 3) for x in d[k]['slope_ci']]}" for k in COMPONENTS))
        print("   rho " + "  ".join(f"{k} {a[k]['mean']:+.4f} {[round(x, 3) for x in a[k]['ci']]}"
                                    for k in ("ar_0_1", "ar_acc", "ar_pub", "acc_minus_pub")))
    print(f"-> {C.rel(p, root)} ({X.LABEL})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("fetch", "run"))
    ap.add_argument("--root", default=str(X.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    return cmd_fetch(root) if a.cmd == "fetch" else cmd_run(root, argv)


if __name__ == "__main__":
    sys.exit(main())
