#!/usr/bin/env python3
"""Figure 1 of the validity-first paper: the multi-agent testbed, by trust lane (single column).

    .venv/bin/python experiments/report/fig1_validity_architecture.py
        # -> paper/figures/fig1_validity.{pdf,png} + fig1_validity_gray.png

Revision of 2026-10-08 (user: centre every label, sentence case, finer finish; the agent row now matches §3: news and
policy agents read untrusted Web text, a profile agent reads annual-report text for exposure flags, and technical,
fundamentals and macro agents read trusted data). Layout, top to bottom: three trust lanes (untrusted rose,
semi-trusted sand, trusted indigo) hold the point-in-time sources, the blackboard (single-writer point-in-time stores
and the blinder, code), the agents (question programs; teal border: model calls) and one typed interface that follows
the Jev schema, whichever reader is plugged in. Below the lanes: the decision layer (code; black tag: the influence
bound holds for any reader) and the validity harness (evaluation only; nothing flows back). The trusted-data path into
the decision layer is a double line under the interface bar.

Every text block is centred in its box (both axes) unless the box is a left-aligned list, and `check_fit` fails the
build if any text extends beyond its box. The only numbers are constants read from the configs: K and J
(configs/system_v1.yaml selection) and the text-cap multiplier (configs/learning_v1.yaml text_cap.multiplier).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style  # noqa: E402

ROOT = style.ROOT
NAME = "fig1_validity"
W_IN, H_IN = style.COLUMN_WIDTH, 3.16
C, CH = style.COLORS, style.CHANNEL_COLORS
FS, FS_G, FS_S = 6.8, 6.0, 5.8  # box titles, glosses, tag (pt)
LINE = 1.12  # line height as a multiple of the font size
PT = 1 / 72  # inches per point
PAD = 0.025  # minimum clearance between text and a box edge (in)
ARROW = "-|>,head_length=2.0,head_width=1.0"
HARNESS_EDGE = "#7A7A7A"
S_DARK = "#A8963E"  # sand darkened for arrows (legible in grayscale)
ROUND = "round,pad=0,rounding_size=0.022"
TEAL_FILL = "#EAF5F2"  # agents and the interface: a light tint of the model-call teal
BOARD_FILL = "#F6F6F6"  # the blackboard (code)
HEAD_DL, HEAD_HV = "#1F1F1F", "#6B6B6B"  # title strips of the decision layer (code) and the harness (evaluation)
LANE_TEXT = {"W": "#8E3A4A", "S": "#7A6A22", "T": "#332288"}  # lane headers, dark tones of the lane colours

FIT: list[tuple] = []  # (text artist, (x0, y0, x1, y1) box in data inches) checked after drawing
GROUPS: list[tuple] = []  # (artists, box, centre_x): shifted after a draw so the block is centred on its box


def constants() -> dict[str, str]:
    sysc = yaml.safe_load((ROOT / "configs" / "system_v1.yaml").read_text())
    learn = yaml.safe_load((ROOT / "configs" / "learning_v1.yaml").read_text())
    return {"K": str(int(sysc["selection"]["K"])), "J": str(int(sysc["selection"]["J"])),
            "m": f"{float(learn['text_cap']['multiplier']):g}"}


def rect(ax, x0, y0, w, h, *, edge="black", lw=0.6, fill="white", z=3):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=ROUND, fc=fill, ec=edge, lw=lw, zorder=z))


def centred_box(ax, x0, y0, w, h, title, gloss=None, *, edge="black", lw=0.6, fill="white", title_weight="bold",
                title_fs=FS, gloss_fs=FS_G):
    """A box whose title and gloss form one block, centred horizontally and vertically."""
    rect(ax, x0, y0, w, h, edge=edge, lw=lw, fill=fill)
    g_lines = gloss.split("\n") if gloss else []
    gap = 0.3 * gloss_fs * PT if g_lines else 0
    block = title_fs * PT * LINE + gap + len(g_lines) * gloss_fs * PT * LINE
    top = y0 + h / 2 + block / 2
    cx = x0 + w / 2
    t = ax.text(cx, top, title, fontsize=title_fs, fontweight=title_weight, ha="center", va="top", zorder=4,
                linespacing=LINE)
    FIT.append((t, (x0, y0, x0 + w, y0 + h)))
    group = [t]
    if g_lines:
        g = ax.text(cx, top - title_fs * PT * LINE - gap, gloss, fontsize=gloss_fs, ha="center", va="top", zorder=4,
                    color=C["dark_grey"], linespacing=LINE)
        FIT.append((g, (x0, y0, x0 + w, y0 + h)))
        group.append(g)
    GROUPS.append((group, (x0, y0, x0 + w, y0 + h), True))


def list_box(ax, x0, y0, w, h, title, rows, *, edge="black", lw=0.6, subtitle=None, foot=None, head="black"):
    """A box with a dark title strip (white bold title) and a left-aligned list below it: optional italic subtitle,
    rows, optional grey footnote; the list is centred vertically in the body."""
    hh = 0.17
    rect(ax, x0, y0, w, h, edge=edge, lw=lw)
    ax.add_patch(FancyBboxPatch((x0, y0 + h - hh), w, hh, boxstyle=ROUND, fc=head, ec=head, lw=lw, zorder=3.2))
    ax.add_patch(Rectangle((x0, y0 + h - hh), w, hh / 2, fc=head, ec="none", lw=0, zorder=3.3))
    tt = ax.text(x0 + 0.06, y0 + h - hh / 2, title, fontsize=FS, fontweight="bold", color="white", ha="left",
                 va="center", zorder=4)
    FIT.append((tt, (x0, y0 + h - hh - PAD, x0 + w, y0 + h + PAD)))
    step = FS_G * PT * 1.38
    body = (x0, y0, x0 + w, y0 + h - hh)
    y = y0 + h - hh - 0.05
    xl = x0 + 0.06
    n0 = len(FIT)
    if subtitle:
        FIT.append((ax.text(xl, y, subtitle, fontsize=FS_G, fontstyle="italic", color=C["dark_grey"], ha="left",
                            va="top", zorder=4), body))
        y -= step
    for r in rows:
        FIT.append((ax.text(xl, y, r, fontsize=FS_G, ha="left", va="top", zorder=4), body))
        y -= step
    if foot:
        FIT.append((ax.text(xl, y, foot, fontsize=FS_G, color=C["dark_grey"], ha="left", va="top", zorder=4,
                            linespacing=LINE), body))
    GROUPS.append(([f[0] for f in FIT[n0:]], body, False))


def arrow(ax, p, q, color, ls="-", lw=0.7, both=False):
    style_ = ARROW.replace("-|>", "<|-|>") if both else ARROW
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style_, color=color, lw=lw, ls=ls, shrinkA=0.6 if both else 0,
                                 shrinkB=0.6, mutation_scale=1.0, zorder=2))


def centre_groups(fig, ax) -> None:
    """Shift each group so the union of its rendered text extents is centred on its box (vertically; horizontally
    too for centred boxes)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for arts, (x0, y0, x1, y1), cx in GROUPS:
        ext = [inv.transform(a.get_window_extent(r).get_points()) for a in arts]
        lo_x = min(e[0][0] for e in ext); hi_x = max(e[1][0] for e in ext)
        lo_y = min(e[0][1] for e in ext); hi_y = max(e[1][1] for e in ext)
        dy = (y0 + y1) / 2 - (lo_y + hi_y) / 2
        dx = ((x0 + x1) / 2 - (lo_x + hi_x) / 2) if cx else 0.0
        for a in arts:
            px, py = a.get_position()
            a.set_position((px + dx, py + dy))


def check_fit(fig, ax) -> None:
    """Fail if any text extends beyond its box (with PAD clearance)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    bad = []
    for t, (x0, y0, x1, y1) in FIT:
        (a, b), (c, d) = inv.transform(t.get_window_extent(r).get_points())
        if a < x0 + PAD - 1e-3 or c > x1 - PAD + 1e-3 or b < y0 + PAD - 1e-3 or d > y1 - PAD + 1e-3:
            bad.append(f"{t.get_text()!r}: text [{a:.3f}, {c:.3f}] x [{b:.3f}, {d:.3f}] vs box "
                       f"[{x0:.3f}, {x1:.3f}] x [{y0:.3f}, {y1:.3f}]")
    if bad:
        raise SystemExit("text outside its box:\n  " + "\n  ".join(bad))


def main(argv=None) -> int:
    style.apply()
    k = constants()
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_IN)
    ax.set_ylim(0, H_IN)
    ax.axis("off")

    # --- trust lanes and their headers
    lanes = {"W": (0.02, 1.50), "S": (1.52, 2.27), "T": (2.29, 3.31)}
    lane_lo, lane_hi = 1.05, H_IN - 0.02
    names = {"W": "Untrusted", "S": "Semi-trusted", "T": "Trusted"}
    head_h = 0.17
    for key, (a, b) in lanes.items():
        col, alpha = style.LANE_FILL[key]
        ax.add_patch(Rectangle((a, lane_lo), b - a, lane_hi - lane_lo, fc=col, alpha=alpha, lw=0, zorder=0))
        ax.text((a + b) / 2, lane_hi - head_h / 2, names[key], fontsize=FS, fontweight="bold", ha="center",
                va="center", color=LANE_TEXT[key], zorder=4)
        ax.plot([a + 0.10, b - 0.10], [lane_hi - head_h + 0.005] * 2, color=LANE_TEXT[key], lw=0.6, alpha=0.55,
                solid_capstyle="round", zorder=1)
    inset = 0.035

    def lane_box(key, y, h, title, gloss, **kw):
        a, b = lanes[key]
        centred_box(ax, a + inset, y, b - a - 2 * inset, h, title, gloss, **kw)

    # --- row 1: point-in-time sources (outside sources: grey outline, regular titles)
    y1, h1 = 2.57, 0.40
    lane_box("W", y1, h1, "Web news", "one newswire:\nheadlines, summaries", edge=C["grey"], lw=0.5,
             title_weight="normal")
    lane_box("S", y1, h1, "Annual reports", "business\nsections", edge=C["grey"], lw=0.5, title_weight="normal")
    lane_box("T", y1, h1, "Trusted records", "prices, XBRL facts,\n8-K items, macro vintages", edge=C["grey"],
             lw=0.5, title_weight="normal")

    # --- row 2: the blackboard (point-in-time stores) and the blinder: the agents' only shared medium
    y2, h2 = 2.05, 0.40
    x_span0, x_span1 = lanes["W"][0] + inset, lanes["T"][1] - inset
    centred_box(ax, x_span0, y2, x_span1 - x_span0, h2, "Blackboard: point-in-time stores and blinder (code)",
                "One writer per store; a record is readable only if available before the cutoff;\n"
                "firm names become pseudonyms and dates become relative time", fill=BOARD_FILL)

    # --- row 3: the agents (question programs; model calls: teal border)
    y3, h3 = 1.50, 0.42
    a, b = lanes["W"]
    wn = (b - a - 2 * inset - 0.04) * 0.55
    centred_box(ax, a + inset, y3, wn, h3, "News agents", "per-item questions,\nsynthesis, sentinel",
                edge=CH["program"], lw=0.9, fill=TEAL_FILL, title_fs=6.5)
    centred_box(ax, a + inset + wn + 0.04, y3, b - a - 2 * inset - wn - 0.04, h3, "Policy agent", "policy\nthemes",
                edge=CH["program"], lw=0.9, fill=TEAL_FILL, title_fs=6.5)
    lane_box("S", y3, h3, "Profile agent", "exposure\nflags", edge=CH["program"], lw=0.9, fill=TEAL_FILL, title_fs=6.5)
    lane_box("T", y3, h3, "Trusted-data agents", "technical, fundamentals,\nmacro", edge=CH["program"], lw=0.9, fill=TEAL_FILL,
             title_fs=6.5)

    # --- row 4: the typed interface (the reader under test)
    y4, h4 = 1.10, 0.27
    rect(ax, x_span0, y4, x_span1 - x_span0, h4, edge=CH["program"], lw=0.9, fill=TEAL_FILL)
    x_left_end = lanes["T"][0] - 0.06  # the labels sit left of the trusted path, which crosses the bar on top
    lab = ax.text((x_span0 + x_left_end) / 2, y4 + h4 / 2, "Typed interface (Jev schema)", fontsize=FS,
                  fontweight="bold", ha="center", va="bottom", zorder=4)
    rd = ax.text((x_span0 + x_left_end) / 2, y4 + h4 / 2, "Readers: Jev, emu, VERB, NG", fontsize=FS_G, ha="center",
                 va="top", zorder=4, color=C["dark_grey"])
    ibox = (x_span0, y4, x_left_end, y4 + h4)
    FIT.extend([(lab, ibox), (rd, ibox)])
    GROUPS.append(([lab, rd], ibox, True))

    # lane arrows: sources -> stores; agents <-> stores (read and write); agents -> interface
    centres = {"W": (a + inset + wn / 2, a + inset + wn + 0.04 + (b - a - 2 * inset - wn - 0.04) / 2),
               "S": ((lanes["S"][0] + lanes["S"][1]) / 2,), "T": ((lanes["T"][0] + lanes["T"][1]) / 2,)}
    colours = {"W": C["black"], "S": S_DARK, "T": CH["T"]}
    for key, xs in centres.items():
        col = colours[key]
        src_x = (lanes[key][0] + lanes[key][1]) / 2
        arrow(ax, (src_x, y1), (src_x, y2 + h2), col)
        for x in xs:
            arrow(ax, (x, y3 + h3), (x, y2), col, both=True)
            arrow(ax, (x, y3), (x, y4 + h4), col)

    # --- consumers (white, below the lanes)
    yc0, yc1 = 0.02, 0.96
    dl0, dl1 = 0.02, 2.00
    hv0, hv1 = 2.38, 3.31  # leaves the gap 2.00-2.38 for the trusted path
    list_box(ax, dl0, yc0, dl1 - dl0, yc1 - yc0, "Decision layer (code)", [
        "Reads judgment tables and trusted data",
        "Trust split: $L = L^T +$ capped text logit $L^W$",
        f"Cap: $|L^W| \\leq \\kappa_t$, with $\\kappa_t$ = {k['m']} $s^T_t$",
        f"Band: $K$ = {k['K']} names, $J$ = {k['J']}; sizing; risk rules",
        "Decision card: exact logit and weight ledgers",
    ], foot="Backtest engine; pre-trade checks", head=HEAD_DL)
    # the influence bound as a white pill inside the decision layer's title strip (right-aligned)
    tag_w, tag_h, hh = 0.98, 0.115, 0.17
    tx0, ty = dl1 - tag_w - 0.035, yc1 - hh / 2
    ax.add_patch(FancyBboxPatch((tx0, ty - tag_h / 2), tag_w, tag_h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="white", ec="white", lw=0, zorder=5))
    tg = ax.text(tx0 + tag_w / 2, ty, "Bound for any reader", fontsize=FS_S, color=HEAD_DL, fontweight="bold",
                 ha="center", va="center", zorder=6)
    FIT.append((tg, (tx0 - PAD, ty - tag_h / 2 - PAD, tx0 + tag_w + PAD, ty + tag_h / 2 + PAD)))
    list_box(ax, hv0, yc0, hv1 - hv0, yc1 - yc0, "Validity harness", [
        "$E$[effect] vs AR[0, +1]",
        "Event type vs 8-K item",
        "Re-identification and",
        "date probes",
    ], edge=HARNESS_EDGE, foot="Outcomes from\ntrusted records", head=HEAD_HV)

    # interface -> both consumers; trusted data -> decision layer (double line under the interface bar);
    # trusted records -> harness (dashed: evaluation only)
    xn = centres["W"][0]  # aligned with the news-agent column above
    arrow(ax, (xn, y4), (xn, yc1), CH["mixed"])
    xh = (lanes["T"][0] + lanes["T"][1]) / 2  # aligned with the trusted lane's arrows above
    arrow(ax, (xh, y4), (xh, yc1), HARNESS_EDGE, ls=(0, (2.5, 1.5)))
    xg, yg = lanes["T"][0] - 0.01, yc1 - 0.42
    for lw, col in ((2.1, CH["T"]), (0.7, "white")):
        ax.plot([xg, xg, dl1 + 0.06], [y2, yg, yg], color=col, lw=lw, zorder=3.6, solid_capstyle="butt",
                solid_joinstyle="miter")
    arrow(ax, (dl1 + 0.07, yg), (dl1, yg), CH["T"], lw=0.9)

    centre_groups(fig, ax)
    check_fit(fig, ax)
    style.save(fig, NAME, tight=False)
    style.check_grayscale(style.FIG_DIR / f"{NAME}.png")
    print(f"{NAME} -> paper/figures/{NAME}.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
