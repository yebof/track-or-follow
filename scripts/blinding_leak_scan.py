"""Blinding leak scan (audit finding "blinding leaks", 2026-09-27): per study symbol, the share of its headlines
that still carry one of its own feed short names or its SEC first word after D0 and blinding (`data.leak_probe`).

Usage: .venv/bin/python scripts/blinding_leak_scan.py [--from 2024-06-28] [--to 2026-09-11] [--top 30] [--out PATH]
Default window: W_clean, items created in [c(2024-06-28), c(2026-09-11)), <= 3 tickers, the symbol a universe member
at the item's date. Also lists every frequent short name (>= 10 uses) that does not blind to exactly the code name.
Diagnostic only (never a paper number); no network, no model call.
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevtrader.data.blinding import load_blinder  # noqa: E402
from jevtrader.data.leak_probe import feed_short_names, leak_report, membership_mask, probes  # noqa: E402
from jevtrader.data.news import decision_cutoff  # noqa: E402
from jevtrader.data.universe import SymbolResolver, load_membership, load_renames  # noqa: E402


def study_blinder(symbols):
    m = load_membership(ROOT / "data/raw/universe/sp500_hist_components.csv", "2018-01-01")
    res = SymbolResolver(load_renames(ROOT / "data/raw/universe/alpaca_name_changes.json", m))
    return load_blinder(ROOT / "data/raw/sec/submissions", symbols, resolver=res,
                        history=(date(2018, 1, 1), date(2026, 12, 31)))


def sec_names(symbols):
    out = {}
    for s in symbols:
        f = ROOT / f"data/raw/sec/submissions/{s}.json"
        if f.exists():
            out[s] = json.loads(f.read_text()).get("name", "")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2024-06-28")
    ap.add_argument("--to", dest="end", default="2026-09-11")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--max-per-symbol", type=int, default=0, help="sample at most N headlines per symbol (0: all)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from jevtrader.programs.render import HEADLINE_MAX, d0

    t0 = time.time()
    uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
    syms = sorted(set(uni.symbol))
    news = pd.read_parquet(ROOT / "data/processed/news_table.parquet",
                           columns=["id", "symbol", "created_at", "n_tags", "headline"])
    news = news[news.symbol.isin(syms)]
    b = study_blinder(syms)
    short = feed_short_names(news, uni, clean=lambda h: str(d0(h, HEADLINE_MAX)))
    unmasked = []
    for s, names in sorted(short.items()):
        for n, k in names:
            if k >= 10 and b(n, cutoff=None, tagged=(s,)).strip() != b.code_name(s):
                unmasked.append({"symbol": s, "name": n, "count": k, "blinded": b(n, tagged=(s,))})
    c0, c1 = decision_cutoff(pd.Timestamp(a.start)), decision_cutoff(pd.Timestamp(a.end))
    w = news[(news.created_at >= c0) & (news.created_at < c1) & (news.n_tags <= 3)]
    w = w[membership_mask(w, uni)].drop_duplicates(["symbol", "headline"])
    if a.max_per_symbol:
        w = w.groupby("symbol", group_keys=False).apply(lambda g: g.sample(min(len(g), a.max_per_symbol),
                                                                           random_state=0))
    rep = leak_report(lambda h, s, t: b(str(d0(h, HEADLINE_MAX)), cutoff=t, tagged=(s,)), w,
                      probes(short, sec_names(syms)))
    print(f"{len(w)} headlines, {rep.n.sum()} checked, {rep.leaks.sum()} with a residual name; "
          f"max per-symbol rate {rep.rate.max():.4f}; {len(unmasked)} frequent short names not fully masked; "
          f"{time.time() - t0:.0f} s")
    print(rep.head(a.top).to_string())
    for u in unmasked:
        print("NOT MASKED", u)
    if a.out:
        Path(a.out).write_text(json.dumps({"window": [str(c0), str(c1)], "report": rep.to_dict("records"),
                                           "unmasked_short_names": unmasked}, indent=1, default=str))
    return rep, unmasked


if __name__ == "__main__":
    main()
