"""Write the trusted code features (design_v1 section 6.1: 6 technical + 6 fundamental + FCF yield and accruals) as a
long table `t, symbol, feature, value` of RAW values under the Layer-B catalog names, for every weekly decision in
[start, end] and every member of U_t. Rank-Gaussian standardization is Layer B's (`learn/features.py`). Each value
uses data available at c_t only (bars up to d_t, filings with filed < d_t; tests/test_future_perturbation.py).

Also writes, next to it, the per-(t, symbol) firm facts the context builder needs: `firm_info.parquet` with
`industry` (FF12 from the current SIC; declared non-PIT), `size_rank` (1 = largest raw market value in U_t),
`sector_etf` (trailing-correlation map, V-m6) and `sessions_to_earnings` (earnings rule; the column name that
`decide.risk` and `pipeline.layer_b.RISK_COLUMNS` read).

Run: .venv/bin/python scripts/build_code_features.py --out <dir> [--start 2019-06-07] [--end 2026-09-25]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.backtest.engine import weekly_decision_dates  # noqa: E402
from jevtrader.baselines.ml import code_feature_panel  # noqa: E402
from jevtrader.data import fundamentals as F  # noqa: E402
from jevtrader.data.earnings import EarningsCalendar, sessions_to_next_panel  # noqa: E402
from jevtrader.data.prices import load_price_panels  # noqa: E402
from jevtrader.data.universe import ff12_map  # noqa: E402
from jevtrader.features import technical as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="2019-06-07")
    ap.add_argument("--end", default="2026-09-25")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = load_price_panels()
    uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
    uni["date"] = pd.to_datetime(uni.date)
    recon = sorted(uni.date.unique())
    by_recon = {d: g.symbol.tolist() for d, g in uni.groupby("date")}
    cal = weekly_decision_dates(panels.sessions, panels.sessions[0], panels.sessions[-1])
    dates = [t for t in cal if pd.Timestamp(a.start) <= t <= pd.Timestamp(a.end) and t >= recon[0]]
    members = {t: by_recon[max(r for r in recon if r <= t)] for t in dates}
    syms = sorted(set(uni.symbol))
    fund = F.FundamentalsPanel(ROOT / "data/raw/sec/facts", syms, panels)
    sector = T.SectorMap(panels.adjusted)
    feats = code_feature_panel(panels, fund, sector, dates, members)
    long = pd.concat([df.rename_axis("symbol").reset_index().melt(id_vars="symbol", var_name="feature")
                      .assign(t=t) for t, df in feats.items()], ignore_index=True)
    long = long[long.feature.isin(T.CODE_FEATURES + F.CODE_FEATURES)][["t", "symbol", "feature", "value"]]
    long.to_parquet(out / "code_features.parquet", index=False)
    ind = ff12_map(syms, ROOT / "data/raw/sec/submissions")
    earn = sessions_to_next_panel(EarningsCalendar(ROOT / "data/raw/sec/submissions", syms), dates, members,
                                  panels.sessions).set_index(["t", "symbol"]).sessions_to_earnings
    rows = []
    for t in dates:
        m = members[t]
        ranks = F.size_ranks(feats[t]["market_value"], m)
        smap = sector.asof(t, m)
        rows += [(t, s, ind.get(s), ranks[s], smap.get(s), earn.get((t, s))) for s in m]
    cols = ["t", "symbol", "industry", "size_rank", "sector_etf", "sessions_to_earnings"]
    pd.DataFrame(rows, columns=cols).to_parquet(out / "firm_info.parquet", index=False)
    print(f"{len(dates)} decisions, {len(long)} feature rows -> {out}")


if __name__ == "__main__":
    main()
