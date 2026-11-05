#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): preview figures for E2 (horizon profile) and E3 (look-ahead by month), drawn
only from the figure-ready tables in results/exploratory/figdata/ (written by e2_priced_horizon.py and
e3_lookahead_by_month.py). Numbers are never typed by hand.

    .venv/bin/python experiments/exploratory/fig_exploratory.py
        # -> results/exploratory/figures/figX_{e2_horizon,e3_lookahead}.{pdf,png}

The figures use the paper's style module (experiments/report/style.py, read-only: Tol "muted" palette, Biolinum,
single axis per panel) and carry the label "Exploratory (not pre-registered)" inside the figure. They are written to
results/exploratory/figures/, never to paper/figures/ (the registered figures' directory).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402

sys.path.insert(0, str(X.ROOT / "experiments" / "report"))
import style as ST  # noqa: E402  (registered style, read-only)

FIGDATA = X.OUT_DIR / "figdata"
OUT = X.OUT_DIR / "figures"
TAG = "Exploratory (not pre-registered)"


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(FIGDATA / name, comment="#")


def errbars(ax, x, df, color, marker, mfc, label, dx=0.0, ls="-"):
    y = df["mean"].to_numpy()
    lo, hi = df["ci_low"].to_numpy(), df["ci_high"].to_numpy()
    ax.errorbar(np.asarray(x) + dx, y, yerr=[y - lo, hi - y], color=color, marker=marker, mfc=mfc, ls=ls,
                lw=ST.LW_DATA, elinewidth=ST.LW_ERR, ms=ST.MARKER_SIZE, label=label)


def fig_e2() -> None:
    import matplotlib.pyplot as plt

    prof, win = read("e2_profile.csv"), read("e2_windows.csv")
    fig, axes = plt.subplots(1, 3, figsize=(ST.FULL_WIDTH, 2.0), gridspec_kw={"width_ratios": [1.0, 1.0, 1.15]})
    ind, rose = ST.COLORS["indigo"], ST.COLORS["rose"]
    # (a) event time, items
    ax = axes[0]
    e = prof[(prof["panel"] == "event_time")].sort_values("day")
    ax.axhline(0, color=ST.COLORS["grey"], lw=ST.LW_AXES)
    ax.axvline(0, color=ST.COLORS["pale_grey"], lw=ST.LW_GRID * 2, zorder=0)
    errbars(ax, e["day"], e, ind, "o", ind, "item E[effect]")
    ax.set_xlabel("trading day relative to the item's event session $d_0$")
    ax.set_ylabel("mean weekly Spearman with that day's AR")
    ax.set_title("(a) event time (items)", loc="left", fontweight="bold")
    # (b) decision time, items and N3 f_net; the fill happens between day 0's close and day +1's open
    ax = axes[1]
    ax.axhline(0, color=ST.COLORS["grey"], lw=ST.LW_AXES)
    ax.axvspan(0.5, 0.62, color=ST.COLORS["pale_grey"], lw=0, zorder=0)
    for series, col, mk, mfc, dx, ls in (("item E[effect]", ind, "o", ind, -0.12, "-"),
                                         ("N3 f_net", rose, "s", "none", 0.12, "--")):
        d = prof[(prof["panel"] == "decision_time") & (prof["series"] == series)].sort_values("day")
        errbars(ax, d["day"], d, col, mk, mfc, series, dx, ls)
    ax.text(0.7, 0.04, "fill at open($e_t$)", fontsize=ST.FS_TICK, color=ST.COLORS["dark_grey"], va="bottom",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("trading day relative to the decision session $d_t$")
    ax.set_ylabel("mean weekly Spearman with that day's AR")
    ax.set_title("(b) decision time", loc="left", fontweight="bold")
    ax.legend(loc="upper right", handlelength=1.6)
    # (c) windows: the registered reaction vs the traded week
    ax = axes[2]
    order = [("item E[effect]", "AR[0,+1]", "AR[0,+1]"), ("item E[effect]", "AR[d0,d_t] (before fill)", "AR[$d_0$,$d_t$]"),
             ("item E[effect]", "AR[+2,+5]", "AR[+2,+5]"), ("item E[effect]", "AR[+6,+10]", "AR[+6,+10]"),
             ("item E[effect]", "traded week r-m (open e_t->open e_t+1)", "traded week"),
             ("N3 f_net", "AR[d_t-4,d_t] (news week)", "news week"),
             ("N3 f_net", "traded week r-m (open e_t->open e_t+1)", "traded week"),
             ("N3 f_net", "next week r-m", "next week")]
    ys = np.arange(len(order))[::-1]
    for y, (series, key, lab) in zip(ys, order):
        r = win[(win["series"] == series) & (win["window"] == key)].iloc[0]
        col = ind if series.startswith("item") else rose
        ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]], color=col,
                    marker="o" if series.startswith("item") else "s", mfc=col if series.startswith("item") else "none",
                    lw=0, elinewidth=ST.LW_ERR, ms=ST.MARKER_SIZE)
        ax.text(-0.02, y, lab, ha="right", va="center", fontsize=ST.FS_TICK, transform=ax.get_yaxis_transform())
    ax.axvline(0, color=ST.COLORS["grey"], lw=ST.LW_AXES)
    ax.axhline(ys[4] - 0.5, color=ST.COLORS["pale_grey"], lw=ST.LW_GRID * 2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("mean weekly Spearman [NW 95% CI]")
    ax.set_title("(c) windows: items (top), N3 f_net (bottom)", loc="left", fontweight="bold")
    fig.text(0.995, -0.02, TAG, ha="right", va="top", fontsize=ST.FS_TICK, style="italic",
             color=ST.COLORS["dark_grey"])
    fig.tight_layout(w_pad=1.2)
    ST.save(fig, "figX_e2_horizon", out_dir=OUT)
    plt.close(fig)


def fig_e3() -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    w, mo, ev = read("e3_d1_windows.csv"), read("e3_h2h3_monthly.csv"), read("e3_events.csv")
    # the development run's months that the registered probe period also covers are drawn once (registered)
    w = w[w["period"].str.startswith("registered") | (w["center"] < "2024-01")].copy()
    w["x"] = pd.to_datetime(w["center"] + "-15")
    mo["x"] = pd.to_datetime(mo["month"] + "-15")
    fig, axes = plt.subplots(3, 1, figsize=(ST.FULL_WIDTH, 4.1), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1.0, 1.0]})
    ind, rose, grey = ST.COLORS["indigo"], ST.COLORS["rose"], ST.COLORS["grey"]
    ax = axes[0]
    for per, g in w.groupby("period", sort=False):
        g = g.sort_values("x")
        ax.fill_between(g["x"], g["null95_low"], g["null95_high"], step="mid", color=ST.COLORS["pale_grey"], lw=0,
                        zorder=0, label="name-permutation null, 95%" if per.startswith("registered") else None)
        ax.errorbar(g["x"], g["auc"], yerr=[g["auc"] - g["ci_desc_low"], g["ci_desc_high"] - g["auc"]],
                    color=ind if per.startswith("registered") else grey, lw=0, elinewidth=ST.LW_ERR, marker="none")
        dif = g["differs_from_0_5"].astype(bool)
        ax.plot(g.loc[~dif, "x"], g.loc[~dif, "auc"], "o", mfc="white",
                color=ind if per.startswith("registered") else grey,
                label="registered probe months" if per.startswith("registered") else "W_pre slice (development run)")
        ax.plot(g.loc[dif, "x"], g.loc[dif, "auc"], "o", color=ind, mfc=ind,
                label="differs from 0.5 (perm. p < .05)" if dif.any() else None)
    ax.axhline(0.5, color=grey, lw=ST.LW_AXES, ls=":")
    ax.set_ylabel("D1 recall AUC\n(3-month window)")
    ax.set_title("(a) D1 look-ahead probe by month (point: AUC; bar: descriptive week-block CI)", loc="left",
                 fontweight="bold")
    ax.legend(loc="center right", ncol=1, handlelength=1.2)
    ax = axes[1]
    ax.axhline(0, color=grey, lw=ST.LW_AXES)
    ax.bar(mo["x"], mo["h2_mean_weekly_spearman"], width=22, color=ind, alpha=0.85, lw=0)
    ax.set_ylabel("H2: weekly Spearman\n(monthly mean)")
    ax.set_title("(b) item E[effect] vs AR[0,+1] by month (H2's weekly statistic)", loc="left", fontweight="bold")
    ax = axes[2]
    ax.axhline(0, color=grey, lw=ST.LW_AXES)
    ax.bar(mo["x"], mo["h3_mean_delta_ic"], width=22, color=rose, alpha=0.9, lw=0)
    ax.set_ylabel("H3: $\\Delta$IC\n(monthly mean)")
    ax.set_title("(c) rank-IC of the system minus code-only by month (H3's weekly statistic)", loc="left",
                 fontweight="bold")
    for ax in axes:
        for _, e in ev.iterrows():
            ax.axvline(pd.Timestamp(e["date"]), color=ST.COLORS["dark_grey"], lw=ST.LW_AXES, ls="--", zorder=0)
    # event labels in a strip above panel (a); events closer than 30 days share one label
    groups: list[list] = []
    for _, e in ev.sort_values("date").iterrows():
        d = pd.Timestamp(e["date"])
        if groups and (d - groups[-1][-1][0]).days < 30:
            groups[-1].append((d, e["label"]))
        else:
            groups.append([(d, e["label"])])
    for k, g in enumerate(groups):
        axes[0].text(g[0][0], 1.13 + 0.11 * (k % 2), " / ".join(x[1] for x in g), ha="center", va="bottom",
                     fontsize=ST.FS_TICK - 0.5, color=ST.COLORS["dark_grey"], transform=axes[0].get_xaxis_transform())
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.text(0.995, -0.01, TAG, ha="right", va="top", fontsize=ST.FS_TICK, style="italic",
             color=ST.COLORS["dark_grey"])
    fig.tight_layout(h_pad=0.6)
    ST.save(fig, "figX_e3_lookahead", out_dir=OUT)
    plt.close(fig)


def main() -> int:
    ST.apply()
    fig_e2()
    fig_e3()
    for n in ("figX_e2_horizon", "figX_e3_lookahead"):
        print(f"wrote {(OUT / n).relative_to(X.ROOT)}.{{pdf,png}} ({X.LABEL})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
