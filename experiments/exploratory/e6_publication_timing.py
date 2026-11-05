#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E6: when the registered H2 criterion is realized relative to the item's
publication, and three further qualifiers of the registered H2 result ($0; no model call).

    .venv/bin/python experiments/exploratory/e6_publication_timing.py fetch  # vendor minute bars (network, read-only)
    .venv/bin/python experiments/exploratory/e6_publication_timing.py run    # -> results/exploratory/e6_publication_timing.json

Why exploratory. Specified on 2026-09-29 after the unlock, after the registered H2 outcome and after the round-1 paper
reviews (docs/reviews/paper_v2_r1_pc.md B1, M1 and minor 2; paper_v2_r1_stats.md M5 and minor 5) had printed
read-only diagnostics for several of these cells (docs/reviews/paper_v2_r1_pc_checks.py). Nothing here decides a
registered claim: every output is a descriptive estimate with a 95% interval; no p-value is computed; no
configuration, threshold or reader is chosen from it.

Frame. The registered H2 frame, rebuilt read-only exactly as E5 does (e5_construct_checks.frame: h2_validity.item_rows
and c2_frame from results/judgments/main_emu_wclean_u100 on W_clean, weeks from the D1 start), with the registered
labels (h2_validity.add_abnormal_returns); its weekly statistic must equal results/analysis/h2.json's confirmatory mean
to 1e-12 or the script stops. The price-report flag is E5's classifier of record (the regex, e5_spec.yaml).

1. Timing groups (item clock time in ET relative to its event session d0, the registered label's rule: d0 is the
   session of created_at if before 16:00 ET, else the next session):
   pre-open  = created on d0 before 09:30 ET (the registered pre-open control's rule);
   intraday  = created on d0 between 09:30 and 16:00 ET, so AR[0] is partly realized before publication;
   after-close or non-session = every other item (d0 is the next session, so AR[0] is almost entirely after it).
   Statistic per group vs AR[0,+1], AR[0] and AR[+1] (learn.labels.abnormal_returns windows (0,1), (0,0), (1,1)).
2. A publication-aligned criterion from the vendor's minute bars (SIP, extended hours 04:00-20:00 ET, unadjusted):
   AR_pub = r_i - beta_i r_SPY with r = the return from the OPEN of the first one-minute bar that starts at or after
   created_at rounded up to the next full minute (on the item's day if it is a session and the minute is before 20:00
   ET, else from 04:00 ET of the next session) to the CLOSE of session d0+1; SPY's return runs from its first bar at or
   after the stock's entry minute to its close of d0+1; beta_i is the registered label's trailing beta (252 sessions,
   at least 60, ending the session before d0); closes are the unadjusted daily closes, and both legs are put on the
   adjusted scale by the daily factor adjusted close / unadjusted close of their own date (dividends and splits). An
   item whose first bar after publication is after the close of d0+1, or without a bar, is dropped (counted). The
   window starts after publication by construction, so AR_pub drops every return realized before the item existed
   (and, for items published before a sparse pre-market, a few minutes after it). Sensitivity AR_pub_regular: the
   entry is the first regular-session bar (09:30-16:00 ET) at or after the rounded publication minute.
   Statistic on all units, on units without a price report, per timing group, on W_release, and per top event type
   (every type with at least 5% of the frame).
3. Controls of the registered statistic (stats review M5, minor 5): the weekly statistic on W_release units without a
   price report, and on W_release pre-open units without one; the paired weekly difference of the registered
   statistic and its placebo (rho(AR[0,+1]) - rho(AR[-5,-1]) per week, same 115 weeks), NW(4) interval; the weekly
   partial Spearman correlation of E[effect] with AR[0,+1] given AR[-5,-1] (Pearson partial correlation of the
   within-week ranks), NW(4) interval.
4. How peaked the confirmatory reader's effect distributions are (PC review minor 2): the share of E[effect] within
   0.05 of an integer level; the shares of rounded levels (harmful: below 1.5; neutral or mixed: 1.5 to 2.5; helpful:
   2.5 or more); the mean normalized entropy of the five-level distribution (entropy / ln 5); the registered statistic
   with the rounded level in place of E[effect].
5. The approximate residual after the price-report classifier's measured misses (PC review M1): if the share m of
   unflagged units that report the move (the E5 audit's miss rate, with its Wilson interval) carried the flagged
   units' association rho_price, the units that report no move would give about (rho_no_price - m rho_price) / (1 - m).
   A linear mixture approximation of weekly correlations, printed with the words "about" and "approximation"; the
   either-classifier rows of E5 replace it when the full-frame coding completes.

Every weekly statistic is h2_validity.weekly_test (mean over weeks of the Spearman correlation, at least 4 units per
week) with the two-sided Newey-West(4) 95% interval of stats.nw_test; the one-sided p-value is dropped. Minute bars are
cached under data/cache/minute_sip/ (a cache, outside the archive: data/raw/bars is part of the archive id) (one parquet per session date, with the requested symbols in a sidecar JSON);
the fetch reads no outcome (only item symbols and publication times) and uses the read-only market-data client.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

import common as C  # noqa: E402
import e5_construct_checks as E5  # noqa: E402  (exploratory frame, text and classifier, read-only)
import h2_validity as H2  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402

NAME = "e6_publication_timing"
GEN = "main_emu_wclean_u100"
MIN_DIR = C.ROOT / "data" / "cache" / "minute_sip"
RAW_DAILY = C.ROOT / "data" / "raw" / "bars" / "all_daily_raw_2018_2026.parquet"
ADJ_DAILY = (C.ROOT / "data" / "raw" / "bars" / "sp500_daily_2018_2026.parquet",
             C.ROOT / "data" / "raw" / "bars" / "etf_daily_2018_2026.parquet")
MARKET = "SPY"
EXT_OPEN, REG_OPEN, REG_CLOSE, EXT_CLOSE = "04:00", "09:30", "16:00", "20:00"
MIN_TYPE_SHARE = 0.05
TOL = 1e-12


# ------------------------------------------------------------------------------------------------ frame
def build(root: Path, guard):
    """The registered H2 frame with E5's text flags and the registered labels (checked against h2.json)."""
    spec, _ = E5.load_spec()
    f, start, end, d1 = E5.frame(root, guard)
    g = E5.classify(E5.attach_text(f), spec)
    guard.check(g["t"].unique(), "E6 labels")
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    g = H2.add_abnormal_returns(g, px.book)
    from jevtrader.learn.labels import abnormal_returns

    ev = g[["symbol", "created_at"]]
    g["ar0"] = abnormal_returns(px.book, ev, window=(0, 0))["ar"].to_numpy(dtype=float)
    g["ar1"] = abnormal_returns(px.book, ev, window=(1, 1))["ar"].to_numpy(dtype=float)
    reg = C.read_json(root / "results/analysis/h2.json")
    mine = H2.weekly_test(g)["mean"]
    if abs(mine - reg["confirmatory"]["mean"]) > TOL:
        raise SystemExit(f"E6: frame does not reproduce h2.json ({mine} vs {reg['confirmatory']['mean']})")
    et = g["created_at"].dt.tz_convert(C.ET)
    same = et.dt.tz_localize(None).dt.normalize() == g["d0"]
    hm = et.dt.strftime("%H:%M")
    g["timing"] = np.where(same & (hm < REG_OPEN), "pre_open",
                           np.where(same & (hm < REG_CLOSE), "intraday", "after_close_or_nonsession"))
    return g, px, reg, start, end, d1


def effect_distributions(root: Path, start, end) -> pd.DataFrame:
    """The five-level effect distribution of every item row (the registered item_rows keeps only E[effect])."""
    rows = []
    tag = f'"key": "{H2.ITEMS_KEY}'
    with open(root / "results/judgments" / GEN / "memory" / "work.jsonl") as fh:
        for line in fh:
            if tag not in line:
                continue
            rec = json.loads(line)["record"]
            if not rec["key"].startswith(H2.ITEMS_KEY):
                continue
            t = pd.Timestamp(rec["t"][:10])
            if not (pd.Timestamp(start) <= t <= pd.Timestamp(end)):
                continue
            sym = rec["key"][len(H2.ITEMS_KEY):]
            for r in rec["value"]:
                p = r.get("effect") or {}
                rows.append({"unit": f"{t.date()}|{sym}|{r['eid']}",
                             **{f"p{k}": float(p.get(str(k), np.nan)) for k in range(5)}})
    return pd.DataFrame(rows).drop_duplicates("unit")


# ------------------------------------------------------------------------------------------------ minute bars
def entry_dates(created_utc: pd.Series, sessions: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, ...]]:
    """Candidate session dates for each item's first bar after publication (module doc, item 2)."""
    out = []
    ses = pd.DatetimeIndex(sessions)
    for ts in created_utc:
        tm = ts.ceil("min").tz_convert(C.ET)
        day = tm.tz_localize(None).normalize()
        k = ses.searchsorted(day)
        cands = []
        if k < len(ses) and ses[k] == day and tm.strftime("%H:%M") < EXT_CLOSE:
            cands.append(ses[k])
            if k + 1 < len(ses):
                cands.append(ses[k + 1])
        else:
            k2 = ses.searchsorted(day, side="right") if (k < len(ses) and ses[k] == day) else k
            if k2 < len(ses):
                cands.append(ses[k2])
        out.append(tuple(cands))
    return out


def _day_path(d: pd.Timestamp) -> Path:
    return MIN_DIR / f"{pd.Timestamp(d).date()}.parquet"


def fetch_day(client, d: pd.Timestamp, symbols: list[str]) -> None:
    """Minute bars 04:00-20:00 ET of session d for `symbols` (plus what the cache already holds)."""
    p, side = _day_path(d), _day_path(d).with_suffix(".json")
    have = set(json.loads(side.read_text())["symbols"]) if side.exists() else set()
    need = sorted(set(symbols) - have)
    if not need:
        return
    lo = pd.Timestamp(f"{pd.Timestamp(d).date()} {EXT_OPEN}").tz_localize(C.ET).tz_convert("UTC")
    hi = pd.Timestamp(f"{pd.Timestamp(d).date()} {EXT_CLOSE}").tz_localize(C.ET).tz_convert("UTC")
    rows = []
    for i in range(0, len(need), 100):
        chunk, token = need[i:i + 100], None
        while True:
            params = {"symbols": ",".join(chunk), "timeframe": "1Min", "start": lo.isoformat(), "end": hi.isoformat(),
                      "adjustment": "raw", "feed": "sip", "limit": 10000}
            if token:
                params["page_token"] = token
            j = client._get("/v2/stocks/bars", params)
            for sym, bars in (j.get("bars") or {}).items():
                for b in bars:
                    rows.append((sym, b["t"], float(b["o"]), float(b["c"]), float(b.get("v") or 0.0)))
            token = j.get("next_page_token")
            if not token:
                break
    new = pd.DataFrame(rows, columns=["symbol", "t", "o", "c", "v"])
    new["t"] = pd.to_datetime(new["t"], utc=True)
    old = pd.read_parquet(p) if p.exists() else new.iloc[:0]
    out = pd.concat([old, new], ignore_index=True).drop_duplicates(["symbol", "t"]).sort_values(["symbol", "t"])
    MIN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(p)
    side.write_text(json.dumps({"date": str(pd.Timestamp(d).date()), "symbols": sorted(have | set(need)),
                                "window_et": [EXT_OPEN, EXT_CLOSE], "feed": "sip", "adjustment": "raw",
                                "timeframe": "1Min"}) + "\n")


def cmd_fetch(root: Path) -> int:
    from dotenv import load_dotenv

    from jevtrader.data.alpaca_data import AlpacaData

    load_dotenv(root / ".env")
    guard = C.Guard(root)
    f, start, end, _ = E5.frame(root, guard)  # Layer-A answers only; no outcome is read
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    cands = entry_dates(f["created_at"], px.book.sessions)
    need: dict[pd.Timestamp, set] = {}
    for sym, cs in zip(f["symbol"], cands):
        for d in cs:
            need.setdefault(d, set()).update((sym, MARKET))
    client = AlpacaData()
    days = sorted(need)
    for i, d in enumerate(days, 1):
        fetch_day(client, d, sorted(need[d]))
        if i % 25 == 0 or i == len(days):
            print(f"  minute bars: {i}/{len(days)} session dates", flush=True)
    print(f"cached under {C.rel(MIN_DIR, root)}")
    return 0


def daily_factors(symbols: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unadjusted daily closes and the factor adjusted / unadjusted close, sessions x symbols."""
    raw = pd.read_parquet(RAW_DAILY, columns=["symbol", "date", "close"])
    raw = raw[raw["symbol"].isin(symbols)]
    adj = pd.concat([pd.read_parquet(p, columns=["symbol", "date", "close"]) for p in ADJ_DAILY], ignore_index=True)
    adj = adj[adj["symbol"].isin(symbols)].drop_duplicates(["symbol", "date"])
    rc = raw.pivot_table(index="date", columns="symbol", values="close")
    ac = adj.pivot_table(index="date", columns="symbol", values="close")
    rc.index, ac.index = pd.to_datetime(rc.index), pd.to_datetime(ac.index)
    return rc, (ac / rc).reindex_like(rc)


def publication_aligned(g: pd.DataFrame, book) -> pd.DataFrame:
    """AR_pub and AR_pub_regular per unit (module doc, item 2), with the entry minute and its lag."""
    from jevtrader.learn.labels import rolling_beta

    ses = pd.DatetimeIndex(book.sessions)
    ret = book.close.pct_change(fill_method=None)
    beta = rolling_beta(ret, MARKET, 252, 60)
    bcol = {s: j for j, s in enumerate(beta.columns)}
    bv = beta.to_numpy()
    rc, fac = daily_factors(set(g["symbol"]) | {MARKET})
    cands = entry_dates(g["created_at"], ses)
    cache: dict = {}

    def bars(d):
        if d not in cache:
            p = _day_path(d)
            if p.exists():
                b = pd.read_parquet(p)
                b["t"] = pd.to_datetime(b["t"], utc=True).dt.tz_localize(None)
                cache[d] = {s: (x["t"].to_numpy("datetime64[ns]"), x["o"].to_numpy(dtype=float))
                            for s, x in b.sort_values(["symbol", "t"]).groupby("symbol")}
            else:
                cache[d] = {}
            if len(cache) > 40:  # keep memory flat: items are processed in time order
                cache.pop(next(iter(cache)))
        return cache[d]

    def first_bar(sym, tmin, dates, regular):
        for d in dates:
            b = bars(d).get(sym)
            if b is None:
                continue
            ts, op = b
            lo = tmin
            if regular:
                ro = pd.Timestamp(f"{pd.Timestamp(d).date()} {REG_OPEN}").tz_localize(C.ET).tz_convert("UTC")
                lo = max(tmin, ro)
                rcl = pd.Timestamp(f"{pd.Timestamp(d).date()} {REG_CLOSE}").tz_localize(C.ET).tz_convert("UTC")
            k = np.searchsorted(ts, np.datetime64(lo.tz_convert("UTC").tz_localize(None), "ns"), side="left")
            if k < len(ts):
                if regular and pd.Timestamp(ts[k]).tz_localize("UTC") >= rcl:
                    continue
                return d, pd.Timestamp(ts[k]).tz_localize("UTC"), float(op[k])
        return None

    order = np.argsort(g["created_at"].to_numpy())
    out = {k: np.full(len(g), np.nan) for k in ("ar_pub", "ar_pub_regular", "lag_min", "lag_min_regular")}
    status = np.array(["" for _ in range(len(g))], dtype=object)
    for i in order:
        sym, ts, d0 = g["symbol"].iat[i], g["created_at"].iat[i], g["d0"].iat[i]
        if pd.isna(d0) or sym not in rc.columns:
            status[i] = "no_label"
            continue
        k0 = ses.searchsorted(pd.Timestamp(d0))
        if k0 < 1 or k0 + 1 >= len(ses):
            status[i] = "edge"
            continue
        dx = ses[k0 + 1]
        j = bcol.get(sym)
        be = bv[k0 - 1, j] if j is not None else np.nan
        tmin = ts.ceil("min")
        for regular, key, lag in ((False, "ar_pub", "lag_min"), (True, "ar_pub_regular", "lag_min_regular")):
            hit = first_bar(sym, tmin, cands[i], regular)
            if hit is None:
                status[i] = status[i] or "no_bar"
                continue
            de, te, pe = hit
            if de > dx:
                status[i] = status[i] or "entry_after_exit"
                continue
            m = first_bar(MARKET, te, [de], False)
            if m is None:
                status[i] = status[i] or "no_market_bar"
                continue
            try:
                r_i = rc.at[dx, sym] * fac.at[dx, sym] / (pe * fac.at[de, sym]) - 1.0
                r_m = rc.at[dx, MARKET] * fac.at[dx, MARKET] / (m[2] * fac.at[de, MARKET]) - 1.0
            except KeyError:
                status[i] = status[i] or "no_daily_close"
                continue
            v = r_i - be * r_m
            if np.isfinite(v):
                out[key][i] = v
                out[lag][i] = (te - ts).total_seconds() / 60.0
        status[i] = status[i] or "ok"
    res = g[["unit"]].copy()
    for k, v in out.items():
        res[k] = v
    res["status"] = status
    return res


# ------------------------------------------------------------------------------------------------ statistics
def wk(f: pd.DataFrame, x: str = "E_effect", y: str = "ar") -> dict:
    return E5.wk(f.dropna(subset=[y]), x, y)


def paired_weekly(f: pd.DataFrame) -> dict:
    a = S.weekly_spearman(f, "E_effect", "ar", "t", H2.MIN_UNITS).set_index("t")["rho"]
    b = S.weekly_spearman(f, "E_effect", "ar_placebo", "t", H2.MIN_UNITS).set_index("t")["rho"]
    d = (a - b).dropna()
    r = S.nw_test(d.to_numpy(), S.NW_LAGS, "two-sided")
    return {"mean": r["mean"], "ci": r["ci"], "se": r["se"], "n_weeks": int(len(d)),
            "definition": "mean over weeks of rho(E[effect], AR[0,+1]) - rho(E[effect], AR[-5,-1]); NW(4) two-sided"}


def partial_weekly(f: pd.DataFrame) -> dict:
    vals = []
    for _, h in f.dropna(subset=["E_effect", "ar", "ar_placebo"]).groupby("t"):
        if len(h) < H2.MIN_UNITS + 1:
            continue
        rx, ry, rz = (h[c].rank().to_numpy(dtype=float) for c in ("E_effect", "ar", "ar_placebo"))
        c = np.corrcoef(np.vstack([rx, ry, rz]))
        rxy, rxz, ryz = c[0, 1], c[0, 2], c[1, 2]
        den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        if np.isfinite(den) and den > 0:
            vals.append((rxy - rxz * ryz) / den)
    r = S.nw_test(np.asarray(vals), S.NW_LAGS, "two-sided")
    return {"mean": r["mean"], "ci": r["ci"], "se": r["se"], "n_weeks": len(vals),
            "definition": "mean over weeks of the partial Spearman correlation of E[effect] with AR[0,+1] given "
                          "AR[-5,-1] (Pearson partial correlation of within-week ranks); NW(4) two-sided"}


def peakedness(g: pd.DataFrame, dist: pd.DataFrame) -> dict:
    e = g["E_effect"].to_numpy(dtype=float)
    d = g[["unit"]].merge(dist, on="unit", how="left")
    p = d[[f"p{k}" for k in range(5)]].to_numpy(dtype=float)
    ok = np.isfinite(p).all(axis=1)
    q = np.clip(p[ok], 1e-12, 1.0)
    q = q / q.sum(axis=1, keepdims=True)
    ent = -(q * np.log(q)).sum(axis=1) / np.log(5)
    h = g.assign(level=np.round(e))
    return {"share_within_0.05_of_integer": float((np.abs(e - np.round(e)) < 0.05).mean()),
            "share_harmful_below_1.5": float((e < 1.5).mean()),
            "share_neutral_1.5_to_2.5": float(((e >= 1.5) & (e < 2.5)).mean()),
            "share_helpful_2.5_plus": float((e >= 2.5).mean()),
            "mean_normalized_entropy": float(ent.mean()), "median_normalized_entropy": float(np.median(ent)),
            "n_with_distribution": int(ok.sum()), "n_units": int(len(g)),
            "rounded_level_vs_ar": wk(h, x="level"), "E_effect_vs_ar": wk(g)}


def cmd_run(root: Path, argv) -> int:
    guard = C.Guard(root)
    g, px, reg, start, end, d1 = build(root, guard)
    pa = publication_aligned(g, px.book)
    g = g.merge(pa, on="unit", how="left")
    release = C.release_start(root)
    no = g[~g["price"]]
    rel = g[g["t"] >= release]
    rel_no = rel[~rel["price"]]
    local = rel_no["created_at"].dt.tz_convert(C.ET).dt.strftime("%H:%M")
    timing = {}
    for tm in ("pre_open", "intraday", "after_close_or_nonsession"):
        h = g[g["timing"] == tm]
        timing[tm] = {"share": float(len(h) / len(g)), "n_units": int(len(h)),
                      "ar_0_1": wk(h), "ar_0": wk(h, y="ar0"), "ar_1": wk(h, y="ar1"), "ar_pub": wk(h, y="ar_pub"),
                      "ar_pub_regular": wk(h, y="ar_pub_regular"),
                      "median_lag_min": float(np.nanmedian(h["lag_min"])) if h["lag_min"].notna().any() else None}
    shares = g["top_type"].value_counts(normalize=True)
    by_type = {tt: {"share": float(sh), "n_units": int((g["top_type"] == tt).sum()),
                    "ar_0_1": wk(g[g["top_type"] == tt]), "ar_pub": wk(g[g["top_type"] == tt], y="ar_pub")}
               for tt, sh in shares.items() if sh >= MIN_TYPE_SHARE}
    ok = g["ar_pub"].notna()
    lag = g.loc[ok, "lag_min"]
    pub = {"all": wk(g, y="ar_pub"), "no_price": wk(no, y="ar_pub"), "W_release": wk(rel, y="ar_pub"),
           "W_release_no_price": wk(rel_no, y="ar_pub"), "regular_entry_all": wk(g, y="ar_pub_regular"),
           "coverage": {"units": int(len(g)), "with_ar_pub": int(ok.sum()),
                        "with_ar_pub_regular": int(g["ar_pub_regular"].notna().sum()),
                        "status_counts": {str(k): int(v) for k, v in g["status"].value_counts().items()}},
           "entry_lag_minutes": {"median": float(lag.median()), "p75": float(lag.quantile(0.75)),
                                 "p90": float(lag.quantile(0.9)), "share_le_5": float((lag <= 5).mean()),
                                 "share_gt_60": float((lag > 60).mean())},
           "by_timing": {k: v["ar_pub"] for k, v in timing.items()}, "by_type": by_type}
    controls = {"W_release_all": wk(rel), "W_release_no_price": wk(rel_no),
                "W_release_pre_open_no_price": wk(rel_no[(local < H2.PRE_OPEN).to_numpy()]),
                "h2_minus_placebo_weekly": paired_weekly(g), "partial_given_placebo": partial_weekly(g)}
    dist = effect_distributions(root, start, end)
    peak = peakedness(g, dist)
    e5 = X.OUT_DIR / "e5_construct_checks.json"
    e5d = C.read_json(e5)
    mc = e5d["audit"]["model_coder"]
    rn, rp = e5d["weekly"]["no_price"]["mean"], e5d["weekly"]["price_only"]["mean"]

    def mix(m):
        return float((rn - m * rp) / (1 - m))

    residual = {"rho_no_price": rn, "rho_price_only": rp, "miss_rate": mc["miss_rate"],
                "miss_rate_ci": mc["miss_rate_ci"], "approx_residual": mix(mc["miss_rate"]),
                "approx_residual_at_miss_ci": [mix(mc["miss_rate_ci"][1]), mix(mc["miss_rate_ci"][0])],
                "coder_prevalence": mc["coder_prevalence"],
                "definition": "(rho_no_price - m rho_price) / (1 - m): a linear mixture approximation of weekly "
                              "correlations (module doc, item 5); not an estimate"}
    payload = {"question": "when the registered H2 criterion is realized relative to publication; controls and "
                           "qualifiers of the registered H2 result",
               "frame": {"n_units": int(len(g)), "n_weeks": int(g["t"].nunique()), "start": str(g["t"].min().date()),
                         "end": str(g["t"].max().date()), "rule": "the registered H2 frame (as E5)",
                         "h2_mean_recomputed": H2.weekly_test(g)["mean"],
                         "h2_mean_registered": reg["confirmatory"]["mean"]},
               "timing": timing, "publication_aligned": pub, "controls": controls, "peakedness": peak,
               "price_report_residual_approximation": residual,
               "settings": {"nw_lags": S.NW_LAGS, "ci": "two-sided NW 95%", "min_units_per_week": H2.MIN_UNITS,
                            "minute_bars": {"feed": "sip", "adjustment": "raw", "window_et": [EXT_OPEN, EXT_CLOSE],
                                            "cache": C.rel(MIN_DIR, root)},
                            "release_start": str(release.date()), "d1_start": str(pd.Timestamp(d1["start"]).date()),
                            "min_type_share": MIN_TYPE_SHARE}}
    inputs = {"work.jsonl": root / "results/judgments" / GEN / "memory" / "work.jsonl",
              "h2": root / "results/analysis/h2.json", "e5": e5, "raw daily": RAW_DAILY,
              "news table": E5.NEWS}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard, inputs=inputs, argv=argv,
                why="timing and publication-aligned checks specified after the H2 outcome and the round-1 reviews")
    print(f"all AR[0,+1] {payload['frame']['h2_mean_recomputed']:.4f}; AR_pub {pub['all']['mean']:.4f} "
          f"{[round(x, 3) for x in pub['all']['ci']]}; regular {pub['regular_entry_all']['mean']:.4f}")
    for k, v in timing.items():
        print(f"  {k:28s} share {v['share']:.3f} AR01 {v['ar_0_1']['mean']:.3f} AR0 {v['ar_0']['mean']:.3f} "
              f"AR1 {v['ar_1']['mean']:.3f} pub {v['ar_pub']['mean']:.3f}")
    print(f"coverage {pub['coverage']}; lag {pub['entry_lag_minutes']}")
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
