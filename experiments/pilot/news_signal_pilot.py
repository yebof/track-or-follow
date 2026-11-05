"""Pilot (design input, not a paper result): do System-One semantic news judgments carry next-week
cross-sectional signal? Uses the LLM-logprob emulator on post-knowledge-cutoff weeks (2024-07+)."""

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")
from jevtrader.backtest.engine import PriceBook, weekly_decision_dates  # noqa: E402
from jevtrader.data.news import NewsStore, build_news_table, decision_cutoff  # noqa: E402
from jevtrader.data.universe import SymbolResolver, load_membership, load_renames  # noqa: E402
from jevtrader.s1.backends import LogprobEmulator, VerbalizedLLM  # noqa: E402
from jevtrader.s1.budget import SpendLedger  # noqa: E402
from jevtrader.s1.cache import ResponseCache  # noqa: E402
from jevtrader.s1.types import Noul, Score  # noqa: E402

SYMS = ["AAPL", "ABBV", "ABNB", "ABT", "ADBE", "AMD", "AMZN", "AVGO", "BA", "BAC", "AXP", "APP", "CAT", "COST",
        "CRM", "CSCO", "CVX", "COP", "CVS", "BKNG"]
rng = np.random.default_rng(7)

bars = pd.read_parquet(ROOT / "data/raw/bars/sp500_daily_2018_2026.parquet")
pb = PriceBook(bars)
weeks = weekly_decision_dates(pb.sessions, "2024-07-01", "2026-08-28")
weeks = sorted(rng.choice(weeks, 30, replace=False))
m = load_membership(ROOT / "data/raw/universe/sp500_hist_components.csv", "2018-01-01")
res = SymbolResolver(load_renames(ROOT / "data/raw/universe/alpaca_name_changes.json", m))
table = build_news_table(ROOT / "data/raw/news/batches", res, SYMS, date(2019, 6, 1), date(2026, 9, 24))
store = NewsStore(table)

QS = {
    "direction": Score("Taken together, how does this week's news change the fundamental outlook of the company?",
                       ("clearly worse", "somewhat worse", "no real change", "somewhat better", "clearly better")),
    "material": Noul("Does this week's news contain information that materially changes the company's expected earnings?"),
}


def state_for(sym, d):
    end = decision_cutoff(d)
    w = store.window(sym, end - pd.Timedelta(days=7), end)
    w = w[w.n_tags <= 3].tail(30)  # drop broad market wraps tagged with many tickers
    items = [f"{h}. {s}".strip() for h, s in zip(w.headline, w.summary.fillna(""))]
    return {"company": "the company (name withheld)", "news_this_week": items or ["(no company news this week)"]}, len(items)


def fwd_rel(d):
    e = pb.next_session(d)
    e2 = pb.next_session(e + pd.Timedelta(days=6)) if e is not None else None
    op = pb.open
    r = op.loc[e2] / op.loc[e] - 1
    return r


async def main():
    led = SpendLedger(ROOT / "data/cache/spend.sqlite")
    cache = ResponseCache(ROOT / "data/cache/s1_pilot.sqlite")
    backends = {"logprob": LogprobEmulator("gpt-4.1-mini", cache=cache, ledger=led, max_concurrency=16),
                "verbal": VerbalizedLLM("gpt-4.1-mini", cache=cache, ledger=led, max_concurrency=16)}
    rows = []
    for d in weeks:
        r = fwd_rel(d)
        med = r[[s for s in SYMS if s in r]].median()
        tasks, meta = [], []
        for s in SYMS:
            st, n = state_for(s, d)
            for bname, b in backends.items():
                tasks.append(b.ask(st, QS, tag="pilot_news"))
                meta.append((s, bname, n))
        outs = await asyncio.gather(*tasks)
        for (s, bname, n), o in zip(meta, outs):
            rows.append({"date": d, "symbol": s, "backend": bname, "n_news": n,
                         "exp_dir": o.answers["direction"].expected_score, "p_material": o.answers["material"].p_yes,
                         "fwd_rel": float(r.get(s, np.nan) - med)})
    df = pd.DataFrame(rows)
    out = ROOT / "results/pilot"
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "news_signal_pilot.parquet", index=False)
    for bname, g in df.groupby("backend"):
        ic = g.groupby("date").apply(lambda x: x.exp_dir.corr(x.fwd_rel, method="spearman"))
        print(bname, "mean rank-IC", round(ic.mean(), 4), "t", round(ic.mean() / ic.std() * np.sqrt(len(ic)), 2),
              "| exp_dir distinct values", g.exp_dir.round(3).nunique(), "| spend", round(led.total(), 3))
    print(json.dumps(df.groupby("backend").exp_dir.describe().round(3).to_dict()))


asyncio.run(main())
