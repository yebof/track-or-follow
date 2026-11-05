#!/usr/bin/env python3
"""Fig. 2: the price of the bound (outline_v1 §3 Fig. 2 as amended by the draft-0 review; results.tex caption).

    .venv/bin/python experiments/report/fig2_frontier.py          # results/analysis/frontier.json -> paper/figures/
    .venv/bin/python experiments/report/fig2_frontier.py --in-dir results/analysis/W_pre_slice --out-dir X

Single column (3.33 x 2.6 in), two stacked panels sharing the axes: x = in-band attack success (%, week-clustered
95% CI), y = the paired clean selection return of the signal book over the same reader's J = 0 book (bps per week,
NW 95% CI). (a) J in {0, 5, 10, 20, inf} at the registered cap; (b) the cap multiplier mu in {1, 2, 4, inf} at J = 10.
Typed programs: indigo filled circles joined in parameter order by a solid line; holistic reader: rose open circles,
dashed (experiments/report/style.py READER_STYLE). Each point is labelled with its parameter value; the operating point
(J = 10, mu = 2) is ringed; "J = 0: trusted only" and "J = inf: no band" are direct labels; capless 0.6 pt error bars in
both directions; x starts at 0; no legend box (series names at the line ends). A point without an ASR (no attacked
batch scored) is not drawn, and a panel without any says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style

NAME = "fig2_frontier"
ANALYSIS = style.ROOT / "results" / "analysis"
SERIES = (("typed", "Typed programs"), ("H", "Holistic"))
ORDER = {"J": ("0", "5", "10", "20", "inf"), "mu": ("1", "2", "4", "inf")}
OPERATING = {"J": "10", "mu": "2"}


def _pt(p: dict):
    asr = (p.get("asr") or {})
    y = (p.get("paired_vs_J0") or {})
    if asr.get("asr") is None or y.get("mean_bps") is None:
        return None
    xl, xh = asr.get("ci") or (None, None)
    yl, yh = y.get("ci") or (None, None)
    return (100 * asr["asr"], y["mean_bps"], None if xl is None else 100 * xl, None if xh is None else 100 * xh, yl, yh)


def label_for(keys: list[str], sweep: str, direct: bool) -> str:
    """The point label of one location: the parameter values that sit there; for the typed series the direct labels
    "J = 0: trusted only" and "J = inf: no band" replace the plain ones."""
    sym = "J" if sweep == "J" else "\\mu"
    vals = ", ".join("\\infty" if k == "inf" else k for k in keys)
    text = f"${sym}={vals}$"
    if direct and sweep == "J":
        if "0" in keys:
            text += ": trusted only"
        elif "inf" in keys:
            text += ": no band"
    return text


def panel(ax, fr: dict, sweep: str, letter: str) -> int:
    drawn = 0
    for key, name in SERIES:
        pts = (fr.get("readers", {}).get(key) or {}).get(sweep) or {}
        st = style.READER_STYLE[name]
        seq = [(k, _pt(pts[k])) for k in ORDER[sweep] if k in pts]
        seq = [(k, p) for k, p in seq if p is not None]
        if not seq:
            continue
        xs, ys = [p[0] for _, p in seq], [p[1] for _, p in seq]
        ax.plot(xs, ys, color=st["color"], linestyle=st["linestyle"], linewidth=style.LW_DATA, zorder=2)
        for k, (x, y, xl, xh, yl, yh) in seq:
            xerr = None if xl is None else [[max(x - xl, 0)], [max(xh - x, 0)]]
            yerr = None if yl is None else [[max(y - yl, 0)], [max(yh - y, 0)]]
            ax.errorbar([x], [y], xerr=xerr, yerr=yerr, fmt="none", ecolor=st["color"], elinewidth=style.LW_ERR,
                        capsize=0, zorder=1, alpha=0.8)
            ax.plot([x], [y], marker=st["marker"], markerfacecolor=st["mfc"], markeredgecolor=st["color"],
                    markersize=style.MARKER_SIZE, linestyle="none", zorder=3)
            if k == OPERATING[sweep]:
                ax.plot([x], [y], marker="o", markersize=style.MARKER_SIZE + 3.5, markerfacecolor="none",
                        markeredgecolor=style.COLORS["black"], markeredgewidth=1.0, linestyle="none", zorder=4)
        # one label per location: coinciding points (e.g. caps that never bind) share it
        groups: dict[tuple, list[str]] = {}
        for k, p in seq:
            groups.setdefault((round(p[0], 6), round(p[1], 6)), []).append(k)
        for j, ((x, y), ks) in enumerate(sorted(groups.items())):
            ax.annotate(label_for(ks, sweep, key == "typed"), (x, y), xytext=(4, 3 if j % 2 == 0 else -8),
                        textcoords="offset points", fontsize=style.FS_TICK - 0.5, color=style.COLORS["dark_grey"])
        xe, ye = seq[-1][1][0], seq[-1][1][1]
        ax.annotate(name, (xe, ye), xytext=(4, -15), textcoords="offset points", fontsize=style.FS_TICK,
                    color=st["color"])
        drawn += len(seq)
    ax.axhline(0.0, color=style.COLORS["grey"], linewidth=style.LW_GRID * 2, linestyle=":", zorder=0)
    ax.set_xlim(left=0)
    ax.yaxis.grid(True, color=style.COLORS["pale_grey"], linewidth=style.LW_GRID)
    ax.text(-0.14, 1.0, letter, transform=ax.transAxes, fontsize=style.FS_PANEL, fontweight="bold", va="top")
    if not drawn:
        ax.text(0.5, 0.5, "no attacked units scored", transform=ax.transAxes, ha="center", va="center",
                fontsize=style.FS_TICK, color=style.COLORS["grey"])
    return drawn


def build(fr: dict):
    import matplotlib.pyplot as plt

    style.apply()
    fig, axes = plt.subplots(2, 1, figsize=(style.COLUMN_WIDTH, 2.6), sharex=False)
    n = 0
    for ax, sweep, letter in zip(axes, ("J", "mu"), "ab"):
        n += panel(ax, fr, sweep, letter)
        ax.set_ylabel("Clean sel. return\nover $J=0$ (bps/wk)")
    axes[1].set_xlabel("In-band attack success (%)")
    fig.tight_layout(h_pad=0.6)
    return fig, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    p = (Path(a.in_dir) if a.in_dir else ANALYSIS) / "frontier.json"
    if not p.exists():
        sys.exit("fig2: frontier.json missing (run experiments/analysis/frontier.py)")
    fr = json.loads(p.read_text())
    if a.out_dir is None and not fr["meta"].get("registered_window"):
        sys.exit(f"refused: frontier.json is for window {fr['meta']['window']}; pass --out-dir for a development figure")
    fig, n = build(fr)
    pdf, png = style.save(fig, NAME, out_dir=Path(a.out_dir) if a.out_dir else None)
    style.check_grayscale(png)
    print(f"fig2 ({n} points) -> {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
