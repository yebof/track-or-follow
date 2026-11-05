#!/usr/bin/env python3
"""Body figure (§5.3; round-1 PC review M2 moved it from the appendix) and formerly App. F: E2 in event time.
Exploratory (not pre-registered). Drawn at the column width it is placed at (editor review G4: 6 pt ticks).

    .venv/bin/python experiments/report/fig_f_horizon.py    # -> paper/figures/figF_horizon.{pdf,png} + _gray.png

(a) A schematic timeline of one item (round-2 AC review W5/F7: the paper's key idea in one picture; no data, no
number, not to scale): the prior close where the registered criterion AR[0,+1] starts, the reported event, the item's
publication, the close of day +1 where both criteria end, the decision cutoff and the next-open fill that starts the
holding week; brackets mark the registered window and the exploratory publication-aligned window (E6).
(b) One grey panel: for each trading day d from -5 to +10 relative to the item's event session (day 0 marked), the
mean over weeks of the Spearman correlation of E[effect] with the focal firm's abnormal return on day d, with its
two-sided Newey-West(4) 95% interval (results/exploratory/e2_priced_horizon.json item_level.event_day_profile, read
through the exploratory loader). Event time only: never "IC", never an IC-by-forward-horizon chart (R3-2). No p-value.
Saved at exactly the column width (tight=False; round-2 figure review F3-1), so LaTeX does not rescale the type.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _validity as V  # noqa: E402
import style  # noqa: E402
from fig2_validity import label_panels  # noqa: E402

NAME = "figF_horizon"
C = style.COLORS


def timeline(ax) -> None:
    """Schematic strip (panel a): points in time of one item, with the two criterion windows as brackets."""
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.25, 1.0)
    ax.axis("off")
    k = C["black"]
    ax.plot([0.25, 7.05], [0, 0], color=k, lw=0.7, zorder=2)
    ax.plot([7.35, 9.55], [0, 0], color=k, lw=0.7, zorder=2)
    for xb in (7.1, 7.25):  # axis break: days may pass before the cutoff
        ax.plot([xb - 0.06, xb + 0.06], [-0.13, 0.13], color=k, lw=0.6)
    ax.annotate("", xy=(9.8, 0), xytext=(9.5, 0), arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15",
                                                                  lw=0.7, color=k))
    pts = [(0.45, "prior\nclose", k), (1.55, "reported\nevent", k), (3.85, "item\npublished", C["indigo"]),
           (6.3, "close of\nday +1", k), (8.05, "cutoff", k), (9.2, "fill", k)]
    for xp, lab, col in pts:
        ax.plot([xp, xp], [-0.14, 0.14], color=col, lw=0.9 if col != k else 0.7, zorder=3)
        ax.text(xp, 0.24, lab, fontsize=style.FS_TICK, ha="center", va="bottom", color=col, linespacing=0.9)
    ax.plot([2.1, 2.1 + 0.8], [0.0, 0.0], color=C["rose"], lw=2.2, alpha=0.9, zorder=2.5, solid_capstyle="butt")
    ax.text(2.5, 0.24, "reaction", fontsize=style.FS_TICK, ha="center", va="bottom", color=C["wine"])

    def bracket(x0, x1, y, text, col, ls="-"):
        ax.plot([x0, x0, x1, x1], [y + 0.12, y, y, y + 0.12], color=col, lw=0.6, ls=ls)
        ax.text(x1 + 0.08, y, text, fontsize=style.FS_GLOSS - 0.5, ha="left", va="center", color=col)

    bracket(0.45, 6.3, -0.72, "AR[0, +1] (registered)", k)
    bracket(3.85, 6.3, -1.08, "from publication (exploratory)", C["indigo"], ls=(0, (2, 1)))


def main(argv=None) -> int:
    style.apply()
    e2 = V.exploratory("e2_priced_horizon")
    if e2 is None:
        raise SystemExit("results/exploratory/e2_priced_horizon.json missing")
    prof = e2["item_level"]["event_day_profile"]
    days = sorted(prof, key=int)
    x = np.array([int(d) for d in days])
    m = np.array([prof[d]["mean"] for d in days])
    lo = np.array([prof[d]["ci"][0] for d in days])
    hi = np.array([prof[d]["ci"][1] for d in days])
    W, H = style.COLUMN_WIDTH, 2.2
    fig = plt.figure(figsize=(W, H))
    timeline(fig.add_axes([0.0, 1.42 / H, 1.0, 0.56 / H]))
    ax = fig.add_axes([0.17, 0.36 / H, 0.8, 0.84 / H])
    ax.set_facecolor(V.PANEL_X_BG)
    ax.axhline(0, color=C["grey"], lw=style.LW_AXES, zorder=1)
    ax.axvline(0, color=C["dark_grey"], lw=0.5, ls=(0, (1, 1.5)), zorder=1)
    ax.fill_between(x, lo, hi, color=C["indigo"], alpha=style.CI_ALPHA, lw=0, zorder=2)
    ax.plot(x, m, color=C["indigo"], lw=style.LW_DATA, marker="o", ms=style.MARKER_SIZE - 0.5, zorder=3)
    ax.text(0.25, 0.985, "day 0: the item's event session", transform=ax.get_xaxis_transform(), fontsize=style.FS_TICK,
            color=C["dark_grey"], ha="left", va="top")
    ax.set_xticks([d for d in x if d % 5 == 0])
    ax.set_xticks(x, minor=True)
    ax.tick_params(axis="x", which="minor", length=1.5)
    ax.set_xticklabels([f"{d:+d}".replace("-", "\N{MINUS SIGN}") if d else "0" for d in ax.get_xticks().astype(int)])
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_xlabel("trading day $d$ relative to the event session")
    ax.set_ylabel("Spearman with\nAR on day $d$", linespacing=1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}".replace("-", "\N{MINUS SIGN}")))
    label_panels(fig, ((ax, "b", "Event-time profile", "exploratory, not pre-registered"),), y_pad=0.035)
    fig.text(0.012, 2.18 / H, "a", fontsize=style.FS_PANEL, fontweight="bold", ha="left", va="top")
    fig.text(0.045, 2.18 / H, "One item's timeline", fontsize=style.FS_LABEL, ha="left", va="top")
    fig.text(0.305, 2.18 / H, "(schematic, not to scale)", fontsize=style.FS_GLOSS, fontstyle="italic",
             color=C["dark_grey"], ha="left", va="top")
    style.save(fig, NAME, tight=False)
    style.check_grayscale(style.FIG_DIR / f"{NAME}.png")
    print(f"{NAME} -> paper/figures/{NAME}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
