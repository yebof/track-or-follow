#!/usr/bin/env python3
"""EXPLORATORY v2, addendum v2.1 (docs/design/v2_exploratory_spec.md): pick the deployable book on W_hist by the
mechanical rule, report it on W_hist / W_pre / W_clean, and decide the negative-only news risk filter on W_release.
Needs results/v2/v2_portfolio_{W_hist,W_pre_slice,W_clean}.json. $0.

    .venv/bin/python experiments/v2/v2_select.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import v2_event as E  # noqa: E402
import v2_portfolio as P  # noqa: E402

MAX_TO, TIE = 4.0, 0.02


def pick(rows: dict) -> str:
    ok = {k: v for k, v in rows.items() if k != "EW-U100" and v["turnover_per_year"] <= MAX_TO}
    best = max(v["sharpe"]["sharpe"] for v in ok.values())
    near = [k for k, v in ok.items() if v["sharpe"]["sharpe"] >= best - TIE]
    return min(near, key=lambda k: ok[k]["turnover_per_year"])


def main() -> int:
    res = {w: json.loads((C.OUT / f"v2_portfolio_{w}.json").read_text())["rows"]
           for w in ("W_hist", "W_pre_slice", "W_clean")}
    chosen = pick(res["W_hist"])
    sc, bk, rn = chosen.split("-")
    print("chosen on W_hist:", chosen)
    table = {w: {"chosen": r[chosen], "EW-U100": r["EW-U100"]} for w, r in res.items()}
    # news risk filter on W_clean (decided on its W_release part)
    d = C.load("W_clean")
    sessions = d.inputs.prices.sessions
    X = P.ranked_panel(d.code)
    Xl = pd.concat([P.ranked_panel(C.load(w).code) for w in ("W_hist", "W_pre_slice")] + [X])
    L = pd.concat([C.load(w).labels for w in ("W_hist", "W_pre_slice")] + [d.labels])
    Xl = Xl[~Xl.index.duplicated(keep="last")]
    slopes = {"FM4": P.fm_slopes(Xl, L, 4), "FM1": P.fm_slopes(Xl, L, 1)}
    tg = P.targets(d, X, sc, bk, 4 if rn == "R4" else 1, slopes)
    base = P.run_targets(d, tg)
    it = E.items(d)
    ev = it.assign(k=E.entry_session(it["created_at"], sessions, "A"))
    ev = ev[ev["k"] < len(sessions)]
    filt = E.overlay(d, tg, E.multipliers(ev, len(sessions), 5, positive=False), sessions)
    ew = P.run_targets(d, {t: {s: 1 / len(d.inputs.universe[t]) for s in d.inputs.universe[t]} for t in d.dates})
    rel = (filt.returns - base.returns)
    rel_rel = float(rel[rel.index >= E.RELEASE_START].mean() * 252)
    keep = rel_rel >= 0
    out = {"chosen": chosen, "rule": {"max_turnover": MAX_TO, "tie": TIE}, "table": table,
           "risk_filter": {"active_vs_book_W_release": rel_rel, "active_vs_book_W_clean": float(rel.mean() * 252),
                           "kept": keep, "book_W_clean": C.metrics(base, ew), "filtered_W_clean": C.metrics(filt, ew)}}
    for w, r in table.items():
        c, e = r["chosen"], r["EW-U100"]
        print(f"{w:12s} chosen ex {c['ann_excess']:+.3f} SR {c['sharpe']['sharpe']:+.2f} {[round(x, 2) for x in c['sharpe']['ci']]}"
              f" active {c.get('active_ann', float('nan')):+.3f} | EW ex {e['ann_excess']:+.3f} SR {e['sharpe']['sharpe']:+.2f}")
    print("risk filter: active vs book W_release", round(rel_rel, 4), "W_clean", round(float(rel.mean() * 252), 4),
          "kept" if keep else "dropped")
    print("wrote", C.write("v2_select", out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
