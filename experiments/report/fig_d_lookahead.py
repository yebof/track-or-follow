#!/usr/bin/env python3
"""Appendix figure (App. D; outline_v2 §6): the look-ahead screen and blinding (D1, E3, D2).

    .venv/bin/python experiments/report/fig_d_lookahead.py   # -> paper/figures/figD_lookahead.{pdf,png} + _gray.png

(a) Registered, D1: the date-only probe's AUC against realized labels per 3-month window of the registered probe
    schedule, filled where the registered name-permutation test gives p < .05 (the descriptive week-block intervals
    are not drawn: prereg §7.2 found them too narrow; round-1 stats review m7); the window on which the registered start rule was decided is
    ringed; the buffer after the documented knowledge cutoff (W_buffer), the start of W_clean and the start of W_release
    are marked from configs/system_v1.yaml.
(b) Exploratory, not pre-registered (grey panel), E3: the same probe on the W_pre slice (a development run), windows
    wholly before the documented cutoff, with the 2.5-97.5% band of the name-permuted AUCs (the null band; not a
    confidence interval). No p-value is printed.
(c) Registered, D2: share of re-identification probes whose top answer is the true firm (of five same-industry
    options plus cannot_tell) or the true year, by blinding level and period, with Wilson 95% intervals; chance marked
    per question; the share of cannot_tell answers under W_clean printed at the right.

Inputs: results/analysis/d1.json and d2.json (registered, lock open; refused otherwise); results/exploratory/
e3_lookahead_by_month.json (exploratory loader). Numbers are never typed by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _validity as V  # noqa: E402
import style  # noqa: E402
from fig2_validity import label_panels  # noqa: E402

NAME = "figD_lookahead"
C = style.COLORS
IND, GREY = C["indigo"], C["grey"]


def month_mid(m: str) -> pd.Timestamp:
    return pd.Timestamp(f"{m}-15")


def windows_cfg() -> dict:
    return yaml.safe_load((V.ROOT / "configs" / "system_v1.yaml").read_text())["windows"]


def panel_a(ax, d1: dict, win: dict):
    ws = d1["monthly_windows"]
    x = [month_mid(w["center"]) for w in ws]
    auc = np.array([w["auc"] for w in ws])
    diff = np.array([w.get("p_perm", 1.0) < 0.05 for w in ws])  # the registered permutation test (stats review m7)
    b0, c0 = pd.Timestamp(win["W_buffer"][0]), pd.Timestamp(win["W_clean"][0])
    r0 = pd.Timestamp(win["W_release"][0])
    ax.axvspan(b0, c0, color=C["pale_grey"], lw=0, zorder=0)
    ax.axhline(0.5, color=GREY, lw=style.LW_AXES, ls=(0, (3, 2)), zorder=1)
    xa = np.array(x)
    ax.plot(xa[~diff], auc[~diff], ls="none", marker="o", ms=style.MARKER_SIZE, mfc="white", mec=IND, mew=0.8,
            zorder=3, label="permutation $p \\geq$ 0.05")
    ax.plot(xa[diff], auc[diff], ls="none", marker="o", ms=style.MARKER_SIZE, mfc=IND, mec=IND, zorder=3,
            label="permutation $p <$ 0.05")
    ax.legend(loc="lower right", fontsize=style.FS_TICK, handletextpad=0.2, borderaxespad=0.2, labelspacing=0.2,
              frameon=False)
    centre = d1["rule"]["center_window"]["center"]
    xc = month_mid(centre)
    ax.plot([xc], [d1["rule"]["center_window"]["auc"]], marker="o", ms=style.MARKER_SIZE * 2.3, mfc="none",
            mec="black", mew=0.8, zorder=4)
    ax.annotate("start-rule window", xy=(xc, d1["rule"]["center_window"]["auc"]), xytext=(xc, 0.235),
                textcoords="data", fontsize=style.FS_TICK, ha="center", va="center", color="black",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black", shrinkA=0, shrinkB=4))
    for xx, lab in ((b0, "buffer"), (r0, "post-release")):
        ax.axvline(xx, color=C["dark_grey"], lw=0.5, ls=(0, (1, 1.5)), zorder=1) if lab != "buffer" else None
        ax.text(xx + pd.Timedelta(days=4), 0.985, lab, transform=ax.get_xaxis_transform(), fontsize=style.FS_TICK,
                color=C["dark_grey"], ha="left", va="top")
    ax.set_ylim(0.2, 0.9)
    ax.set_ylabel("AUC of the date-only probe")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(month_mid(ws[0]["center"]) - pd.Timedelta(days=25), month_mid(ws[-1]["center"]) + pd.Timedelta(days=25))
    ax.set_xlabel("centre month of the 3-month window")


def panel_b(ax, e3: dict):
    ax.set_facecolor(V.PANEL_X_BG)
    cut = next(x["date"] for x in e3["dates"] if "cutoff" in x["label"])
    ws = [w for w in e3["d1"]["w_pre_slice"]["pooled_windows"] if all(f"{m}-01" < cut for m in w["months"])]
    x = [month_mid(w["center"]) for w in ws]
    band = np.array([w["null_band_95"] for w in ws])
    ax.fill_between(x, band[:, 0], band[:, 1], color=C["pale_grey"], lw=0, zorder=1, step=None)
    ax.axhline(0.5, color=GREY, lw=style.LW_AXES, ls=(0, (3, 2)), zorder=2)
    ax.plot(x, [w["auc"] for w in ws], ls="none", marker="o", ms=style.MARKER_SIZE, mfc="white", mec=IND, mew=0.8,
            zorder=3)
    ax.text(0.03, 0.03, "grey band: permutation null (2.5\N{EN DASH}97.5%)", transform=ax.transAxes,
            fontsize=style.FS_TICK, color=C["dark_grey"], ha="left", va="bottom")
    ax.set_ylim(0.2, 0.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(x[0] - pd.Timedelta(days=25), x[-1] + pd.Timedelta(days=25))
    ax.set_xlabel("centre month (development slice)")
    ax.tick_params(axis="y", labelleft=True)


def panel_c(ax, d2: dict):
    rows = d2["by_period_level"]
    chance = d2["definitions"]["chance"]
    levels = (("full", "full blinding"), ("names", "names masked"), ("none", "unblinded"))
    qs = (("reid_company", "firm"), ("reid_year", "year"))
    y, ys, labels = 0.0, [], []
    for q, qlab in qs:
        y -= 0.5 if not ys else 1.1
        ytop = y + 0.45
        for lev, llab in levels:
            y -= 1.0
            ys.append(y)
            labels.append(f"{qlab}, {llab}")
            # round-2 figure review A-1: the development slice is a grey square (open markers mean placebo elsewhere)
            for per, dy, col, mk in (("W_clean", 0.17, IND, "o"), ("W_pre", -0.17, GREY, "s")):
                r = next(r for r in rows if r["period"] == per and r["level"] == lev and r["question"] == q)
                ax.plot([r["ci"][0], r["ci"][1]], [y + dy, y + dy], color=col, lw=style.LW_ERR * 1.4, zorder=2)
                ax.plot([r["accuracy"]], [y + dy], marker=mk, ms=style.MARKER_SIZE - (0.4 if mk == "s" else 0),
                        mfc=col, mec=col, mew=0.8, ls="none", zorder=3)
                if per == "W_clean":
                    ax.text(1.0, y, f"{100 * r['cannot_tell_share']:.0f}%", transform=ax.get_yaxis_transform(),
                            fontsize=style.FS_TICK, color=C["dark_grey"], ha="right", va="center")
        ax.plot([chance[q]] * 2, [y - 0.5, ytop], color=GREY, lw=style.LW_AXES, ls=(0, (3, 2)), zorder=1)
        ax.text(chance[q] + 0.02, ytop - 0.2, "chance", fontsize=style.FS_TICK, color=C["dark_grey"], ha="left",
                va="center")
    ax.text(1.0, 0.25, "cannot tell\n(evaluation window)", transform=ax.get_yaxis_transform(), fontsize=style.FS_TICK,
            color=C["dark_grey"], ha="right", va="bottom", linespacing=0.95)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(min(ys) - 0.7, 0.6)
    ax.set_xlim(-0.02, 1.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25", "50", "75", "100"])  # A-2: percent, like the cannot-tell column
    ax.set_xlabel("share whose top answer is true (%)")
    ax.plot([], [], marker="o", ls="none", mfc=IND, mec=IND, ms=style.MARKER_SIZE, label="evaluation window")
    ax.plot([], [], marker="s", ls="none", mfc=GREY, mec=GREY, ms=style.MARKER_SIZE - 0.4, label="development slice")
    ax.legend(loc="lower right", bbox_to_anchor=(0.86, 0.0), fontsize=style.FS_TICK, handletextpad=0.2,
              borderaxespad=0.1, labelspacing=0.2)


def main(argv=None) -> int:
    style.apply()
    d1, d2 = V.require_registered("d1"), V.require_registered("d2")
    e3 = V.exploratory("e3_lookahead_by_month")
    if e3 is None:
        raise SystemExit("results/exploratory/e3_lookahead_by_month.json missing")
    fig = plt.figure(figsize=(style.FULL_WIDTH, 2.0))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.35, 0.95, 1.0], wspace=0.55, left=0.065, right=0.995, top=0.82,
                  bottom=0.2)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))
    panel_a(axa, d1, windows_cfg())
    panel_b(axb, e3)
    panel_c(axc, d2)
    for ax in (axa, axb):
        for t in ax.get_xticklabels():
            t.set_fontsize(style.FS_TICK)
    label_panels(fig, ((axa, "a", "Date-only probe D1 (registered)", None),
                       (axb, "b", "D1 before the documented cutoff", "exploratory, not pre-registered"),
                       (axc, "c", "Re-identification D2 (registered)", None)), italic_below=True)
    style.save(fig, NAME, tight=False)  # exact text width (round-2 figure review F3-1)
    style.check_grayscale(style.FIG_DIR / f"{NAME}.png")
    print(f"{NAME} -> paper/figures/{NAME}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
