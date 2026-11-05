#!/usr/bin/env python3
"""EXPLORATORY v2, item 4 (docs/design/v2_exploratory_spec.md): event-driven overlay of typed news judgments on a
base book, traded at the next open after each item's cutoff, plus the item-level open-to-open diagnostic. $0.

    .venv/bin/python experiments/v2/v2_event.py --window W_pre_slice   # plumbing only (S30 news, recall risk)
    .venv/bin/python experiments/v2/v2_event.py --window W_clean       # the one evaluation pass (+ W_release rows)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import v2_portfolio as P  # noqa: E402

import h2_validity as H2  # noqa: E402  (registered, read-only)
import stats as S  # noqa: E402

HOLDS = (1, 2, 5)
CUTOFFS = ("A", "B")
PRIMARY = "EV-A-H2-EW"
RELEASE_START = pd.Timestamp("2025-04-18")


def items(d: C.Data) -> pd.DataFrame:
    s, e = C.resolve_window(d.window)
    f = H2.c2_frame(H2.item_rows(C.ROOT / "results/judgments" / C.GENS[d.window], s, e))
    f["e"] = f["E_effect"] - 2.0
    return f[["symbol", "created_at", "e", "t"]].dropna()


def entry_session(created: pd.Series, sessions: pd.DatetimeIndex, cutoff: str) -> pd.Series:
    """Index of the session whose open trades the item. A: first session k with created < 09:25 ET on k.
    B: first session k with created < 16:00 ET on session k-1."""
    loc = pd.DatetimeIndex(sessions).tz_localize(C.ET)
    cut = (loc + pd.Timedelta(hours=9, minutes=25)) if cutoff == "A" else (loc + pd.Timedelta(hours=16))
    cut = cut.tz_convert("UTC")
    j = cut.searchsorted(created.to_numpy(), side="right")  # first cutoff >= created (strict: created < cut)
    return pd.Series(j if cutoff == "A" else j + 1, index=created.index)


def multipliers(ev: pd.DataFrame, n_sessions: int, hold: int, positive: bool = True) -> list[dict]:
    """Per session index: {symbol: m} for active events (latest event wins). positive=False: the negative-only
    risk filter of addendum v2.1 (no boost)."""
    g = ev.groupby(["symbol", "k"])["e"].sum().reset_index()
    g = g[((g["e"] >= 1) & positive) | (g["e"] <= -1)].sort_values("k")
    out = [dict() for _ in range(n_sessions)]
    for r in g.itertuples():
        m = 2.0 if r.e >= 1 else 0.0
        for k in range(r.k, min(r.k + hold, n_sessions)):
            out[k][r.symbol] = m
    return out


def overlay(d: C.Data, base: dict, mult: list[dict], sessions: pd.DatetimeIndex) -> C.SimResult:
    p = d.inputs.prices
    fills = {p.next_session(t): w for t, w in base.items()}
    idx = {k: i for i, k in enumerate(sessions)}
    st = {"b": None, "m": {}, "prev": None}

    def decide(k, w):
        prev, st["prev"] = st["prev"], k
        mk = mult[idx[k]] if k in idx else {}
        if k in fills:
            b = dict(fills[k])
            m = {s: mk.get(s, 1.0) for s in b}
            x = {s: b[s] * m[s] for s in b}
            tot = sum(x.values())
            st["b"], st["m"] = b, m
            return {s: v / tot for s, v in x.items()} if tot > 0 else b
        b = st["b"]
        if b is None:
            return None
        o0, o1 = p.open.loc[prev], p.open.loc[k]
        b = {s: v * o1.get(s, np.nan) / o0.get(s, np.nan) for s, v in b.items()}
        b = {s: v for s, v in b.items() if np.isfinite(v)}
        tb = sum(b.values())
        st["b"] = b = {s: v / tb for s, v in b.items()}
        m_new = {s: mk.get(s, 1.0) for s in b}
        changed = [s for s in b if m_new[s] != st["m"].get(s, 1.0)]
        if not changed:
            return None
        st["m"] = m_new
        denom = sum(b[s] * m_new[s] for s in b)
        if denom <= 0:
            return None
        x = {s: b[s] * m_new[s] / denom for s in changed}
        rest = {s: v for s, v in w.items() if s not in x}
        sr = sum(rest.values())
        scale = (1 - sum(x.values())) / sr if sr > 0 else 0.0
        tgt = {**{s: v * scale for s, v in rest.items()}, **x}
        return tgt

    last = p.next_session(d.dates[-1])
    end = p.sessions[min(p.sessions.searchsorted(last) + 4, len(p.sessions) - 1)]
    return C.simulate(p, min(fills), end, decide, d.inputs.cash_rate_daily, d.inputs.delisting_returns,
                      d.inputs.ma_exits)


def item_diagnostic(ev: pd.DataFrame, d: C.Data, sessions: pd.DatetimeIndex) -> dict:
    op = d.inputs.prices.open.reindex(sessions)
    out = {}
    for h in (1, 2, 5):
        fwd = op.shift(-h) / op - 1
        uni = {t: list(d.inputs.universe[t]) for t in d.dates}
        mkt = pd.Series({k: fwd.loc[k, uni[max([t for t in d.dates if t < k] or [d.dates[0]])]].mean()
                         for k in sessions})
        r = [fwd.iat[k, fwd.columns.get_loc(s)] - mkt.iloc[k] if s in fwd.columns and k < len(sessions) else np.nan
             for s, k in zip(ev["symbol"], ev["k"])]
        f = ev.assign(y=r).dropna(subset=["y"])
        wk = S.weekly_spearman(f, "e", "y", week="t")
        out[f"h{h}"] = S.weekly_mean_test(wk, alternative="two-sided")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True, choices=list(C.GENS))
    ap.add_argument("--boot", type=int, default=999)
    a = ap.parse_args(argv)
    d = C.load(a.window)
    sessions = d.inputs.prices.sessions
    it = items(d)
    ew_tg = {t: {s: 1 / len(d.inputs.universe[t]) for s in d.inputs.universe[t]} for t in d.dates}
    X = P.ranked_panel(d.code)
    prim_tg = P.targets(d, X, "UAW", "BUF", 4, None)
    bases = {"EW": ew_tg, "UAWBUF4": prim_tg}
    ew = P.run_targets(d, ew_tg)
    base_res = {"EW": ew, "UAWBUF4": P.run_targets(d, prim_tg)}
    rows, diag, series = {}, {}, {}
    for cut in CUTOFFS:
        ev = it.assign(k=entry_session(it["created_at"], sessions, cut))
        ev = ev[ev["k"] < len(sessions)]
        diag[cut] = item_diagnostic(ev, d, sessions)
        print(cut, {h: round(v["mean"], 4) for h, v in diag[cut].items()}, flush=True)
        for h in HOLDS:
            mult = multipliers(ev, len(sessions), h)
            for bn, tg in bases.items():
                name = f"EV-{cut}-H{h}-{bn}"
                r = overlay(d, tg, mult, sessions)
                rows[name] = {"full": C.metrics(r, ew, a.boot), "vs_base": C.metrics(r, base_res[bn], a.boot)}
                series[name] = r.returns
                m = rows[name]["vs_base"]
                print(f"{name:18s} ex {m['ann_excess']:+.3f} SR {m['sharpe']['sharpe']:+.2f} "
                      f"vs base {m['active_ann']:+.4f} IR {m['ir']:+.2f} TO/yr {m['turnover_per_year']:.2f}",
                      flush=True)
    series.update({f"base-{k}": v.returns for k, v in base_res.items()})
    df = pd.DataFrame(series)
    release = None
    if a.window == "W_clean":  # W_release rows: active return vs base on the post-release part only
        sub = df[df.index >= RELEASE_START]
        release = {n: {"active_ann_vs_base": float((sub[n] - sub[f"base-{n.rsplit('-', 1)[1]}"]).mean() * 252),
                       "n_days": len(sub)} for n in rows}
    df.to_parquet(C.OUT / f"v2_event_{a.window}_returns.parquet")
    p = C.write(f"v2_event_{a.window}", {"window": a.window, "primary": PRIMARY, "n_variants": len(rows),
                                         "n_items": len(it), "rows": rows, "item_diagnostic": diag,
                                         "w_release": release,
                                         "bases": {k: C.metrics(v, ew, a.boot) for k, v in base_res.items()}})
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
