"""Prior-scale check rule of design_v1 section 6.5 (validity M4), code only, no LLM.

The allocator's prior sd sigma = 0.04 puts a single standardized feature's implied weekly rank-IC within about
+-0.046 (95% prior mass). Before the freeze this script computes, on W_pre (2020-01-03..2024-05-31, the pre-W_clean
window of results/plan/plan_v1.json) and the point-in-time U100, the mean weekly Spearman IC of every trusted code
feature with the T1 relative return r - m_t, and the 95th percentile of |mean IC| across the features. If it exceeds
0.05, sigma is scaled proportionally: sigma = 0.04 * p95 / 0.05; otherwise it stays 0.04. The pilot IC is not used,
and no W_clean label is read (the last W_pre label is realized at the 2024-06-10 open).

Run: .venv/bin/python experiments/plan/prior_check.py [--out results/plan/prior_check.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.backtest.engine import PriceBook, weekly_decision_dates  # noqa: E402
from jevtrader.baselines.ml import code_feature_panel  # noqa: E402
from jevtrader.data import fundamentals as F  # noqa: E402
from jevtrader.data.prices import build_panels, cut_at, load_bars  # noqa: E402
from jevtrader.data.universe import ma_exits  # noqa: E402
from jevtrader.eval.metrics import information_coefficient, t_stat_newey_west  # noqa: E402
from jevtrader.features import technical as T  # noqa: E402
from jevtrader.learn.labels import LabelStore  # noqa: E402

W_PRE = ("2020-01-03", "2024-05-31")
SIGMA0, IC_BOUND = 0.04, 0.05
FEATURES = T.CODE_FEATURES + F.CODE_FEATURES


def sigma_rule(p95: float, sigma0: float = SIGMA0, bound: float = IC_BOUND) -> float:
    """Section 6.5: scale sigma proportionally when the 95th percentile of |mean IC| exceeds the bound."""
    return sigma0 * p95 / bound if p95 > bound else sigma0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/plan/prior_check.json"))
    args = ap.parse_args()
    raw, adj = load_bars("raw"), load_bars("adjusted")
    panels = build_panels(raw, adj)
    pb = PriceBook(cut_at(adj, panels.life_end))
    uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
    uni["date"] = pd.to_datetime(uni.date)
    recon = sorted(uni.date.unique())
    by_recon = {d: g.symbol.tolist() for d, g in uni.groupby("date")}
    cal = weekly_decision_dates(pb.sessions, pb.sessions[0], pb.sessions[-1])
    dates = [t for t in cal if pd.Timestamp(W_PRE[0]) <= t <= pd.Timestamp(W_PRE[1])]
    members = {t: by_recon[max(r for r in recon if r <= t)] for t in dates}
    ma = {s for s, k in ma_exits(sorted(set(uni.symbol)), raw, ROOT / "data/raw/sec/submissions").items() if k == "mna"}
    labels = LabelStore.build(pb, dates, members, ma_exits=ma)
    assert labels.table.available_at.max() < pd.Timestamp("2024-07-05", tz="UTC"), "a W_clean label leaked in"
    fund = F.FundamentalsPanel(ROOT / "data/raw/sec/facts", sorted(set(uni.symbol)), panels)
    feats = code_feature_panel(panels, fund, T.SectorMap(panels.adjusted), dates, members)
    lab = labels.table.assign(rel=labels.table.r - labels.table.m)
    rel = lab.pivot(index="t", columns="symbol", values="rel")
    per = {}
    for f in FEATURES:
        sc = pd.DataFrame({t: df[f] for t, df in feats.items() if f in df}).T
        ic = information_coefficient(sc, rel)
        per[f] = {"mean_ic": float(ic.mean()), "sd_ic": float(ic.std(ddof=1)), "t_nw4": t_stat_newey_west(ic, 4),
                  "n_weeks": int(ic.notna().sum()), "coverage": float(sc.notna().mean().mean())}
    abs_ic = np.array([abs(v["mean_ic"]) for v in per.values()])
    p95 = float(np.percentile(abs_ic, 95))
    out = {"window": list(W_PRE), "universe": "PIT top-100 (data/processed/universe_monthly_top100.parquet)",
           "n_decisions": len(dates), "features": list(FEATURES), "per_feature": per,
           "p95_abs_mean_ic": p95, "bound": IC_BOUND, "sigma0": SIGMA0, "sigma": sigma_rule(p95),
           "rule": "sigma = 0.04 * p95 / 0.05 if p95 > 0.05 else 0.04 (design_v1 section 6.5)"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for f, v in per.items():
        print(f"{f:22s} mean IC {v['mean_ic']:+.4f}  t {v['t_nw4']:+.2f}  weeks {v['n_weeks']}")
    print(f"p95 |mean IC| = {p95:.4f} -> sigma = {out['sigma']:.4f}  (written to {args.out})")


if __name__ == "__main__":
    main()
