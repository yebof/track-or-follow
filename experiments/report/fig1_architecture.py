"""Figure 1: the system by trust lane (outline_v1 §3 "Fig. 1"; style: outline_v1 §4).

    .venv/bin/python experiments/report/fig1_architecture.py

Writes paper/figures/fig1_architecture.pdf (vector, fonts embedded), a 300-dpi PNG preview and a grayscale preview, and
prints every layout problem the checker finds (none when the figure is clean). The only data are constants: K and J
(configs/system_v1.yaml, selection) and the text-cap multiplier (configs/learning_v1.yaml, text_cap.multiplier).

Layout (inches, figure 7.0 x 2.2): three horizontal trust lanes (untrusted W, semi-trusted S, trusted T) separated by
white gutters and crossed by five columns (sources -> reading layer -> typed judgments -> decision layer -> execution).
Question programs are white boxes with a teal border; code is white with a black outline; outside sources and the
broker are white with a grey outline; typed judgments and point-in-time stores are pale grey. Arrows are rose
(untrusted-tainted values), indigo (trusted values) or dark grey (mixed); a dot marks each branch point. The typed
judgments are drawn per lane (W and T; no S judgment enters a learner), so no table spans two lanes. The S lane ends in
the reading layer and the W and T lanes end after the risk rules, so the decision layer's middle row, where trusted and
untrusted values meet, and the execution column are white. Black tags mark where each part of the Proposition
(design_v2_delta BG-1) is enforced; every tag sits on the edge of the box it marks, except P0 (under a bracket over the
trusted decision boxes) and P5's store clause (in front of the read predicate).

Routes drawn (design_v1 §4.1, §5, §6; design_v2_delta BG-1..BG-3):
- W: news -> N0 -> N1 (the sentinel runs inside N1: its box shares N1's bottom edge) -> cluster -> N2 -> N3 -> W
  judgments; policy items (P1) -> theme exposure (code: weekly theme aggregates x F-A profile flags) -> W judgments, the
  only policy route into L^W. The weekly policy-regime program (P2) branches off P1's output and has no arrow onward:
  it feeds only the overlay variant X7 (design_v1 §5.5, §9.3), outside the primary books (BG-1 "does not say" 5).
- S: 10-K Item 1 -> F-A -> profile flags -> theme exposure. F-A's profile is not a learner feature (design_v1 §6.1), so
  S values reach the decision layer only through the theme exposure (W).
- T: trusted sources -> code features (buckets for the T programs; factors, sigma, earnings dates and 8-K items for the
  decision layer) -> technical, F-B, macro -> T judgments; the indigo bus carries the code values to the trusted
  learner, the reference book and the anchor without passing through the judgments.
- Decision layer: W judgments -> untrusted learner (fit on the residual of the logged L^T) -> L^W -> cap -> fusion;
  T judgments -> trusted learner -> L^T -> fusion and rank R^T -> band selection; reference book -> sizing; sentinel h
  and anchor A -> risk rules. BG-2 applies the sentinel's x1/2 inside sizing and anchored exclusions as vetoes in
  selection; the note on the sentinel route says so, and the box order follows outline_v1 §3.4-§3.5.

Type (Linux Biolinum O): box names 7 pt; glosses and parameter lines inside boxes and free annotations 6.5 pt (so
scripts stay >= 4.5 pt); column headers and lane labels 7 pt bold; P-tags 6 pt bold. Labels are placed on baselines
from the font's em metrics (baseline-aligned across neighbouring boxes) and checked against their true ink (TextPath).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import style

ROOT = style.ROOT
NAME = "fig1_architecture"
FIG_W, FIG_H = style.FULL_WIDTH, 2.2

C = style.COLORS
CH = style.CHANNEL_COLORS
FS = style.FS_BOX  # 7 pt box names
FS_G = style.FS_GLOSS  # 6.5 pt glosses, parameter lines and free annotations
FS_HEAD = style.FS_LABEL  # 7 pt bold column headers and lane labels
LW_PROGRAM = 1.0  # pt, teal border of question programs
LW_CODE = 0.6  # pt, black outline of code boxes
LW_EXTERNAL = 0.5  # pt, grey outline of sources and the broker
LW_ARROW = 0.8  # pt
ARROW_HEAD = "-|>,head_length=2.4,head_width=1.2"  # pt
DOT_R = 1.0 / 72  # in, radius of branch-point dots (2 pt diameter)
ROUND = 0.018  # in, corner radius of boxes
GAP = 0.10  # in, gap between boxes joined by an arrow (news chain, decision chain, execution)
GAP_T = 0.08  # in, sources -> reading layer and into / out of the typed judgments
GUTTER = 0.02  # in, white gap between neighbouring lane fills
H1, H2, HC = 0.13, 0.24, 0.35  # in: one-line boxes, two-line boxes, decision-layer boxes
TAG_W, TAG_H = 0.14, 0.08  # in, P-tag size
DATA_FILL = C["pale_grey"]  # typed judgments and point-in-time stores
# em-based optical extent of a line (Biolinum: x-height .46, cap height .66, ascender ~.70, descender ~.22)
EM_TOP, EM_BOT = 0.70, 0.16
INK_CLEAR = 0.015  # in, minimum clear space between a box label's ink and the box outline (top and bottom)
PAD_X = 0.025  # in, minimum horizontal clear space inside a box


def load_constants() -> dict[str, str]:
    """K, J and the text-cap multiplier, formatted from the configs (never typed by hand)."""
    system = yaml.safe_load((ROOT / "configs" / "system_v1.yaml").read_text())
    learning = yaml.safe_load((ROOT / "configs" / "learning_v1.yaml").read_text())
    mult = float(learning["text_cap"]["multiplier"])
    return {
        "K": f"{int(system['selection']['K'])}",
        "J": f"{int(system['selection']['J'])}",
        "cap_mult": f"{mult:g}",
    }


# ---------------------------------------------------------------------------------------------------------------------
# Drawing primitives (all coordinates in inches; the axes span the whole figure with 1 data unit = 1 inch)


@dataclass
class Box:
    x0: float
    y0: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x0 + self.w

    @property
    def y1(self) -> float:
        return self.y0 + self.h

    @property
    def cx(self) -> float:
        return self.x0 + self.w / 2

    @property
    def cy(self) -> float:
        return self.y0 + self.h / 2

    def left(self, dy: float = 0.0) -> tuple[float, float]:
        return (self.x0, self.cy + dy)

    def right(self, dy: float = 0.0) -> tuple[float, float]:
        return (self.x1, self.cy + dy)

    def top(self, dx: float = 0.0) -> tuple[float, float]:
        return (self.cx + dx, self.y1)

    def bottom(self, dx: float = 0.0) -> tuple[float, float]:
        return (self.cx + dx, self.y0)

    def rect(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def on_edge(self, p: tuple[float, float], tol: float = 0.002) -> bool:
        x, y = p
        in_x = self.x0 - tol <= x <= self.x1 + tol
        in_y = self.y0 - tol <= y <= self.y1 + tol
        return (in_x and (abs(y - self.y0) <= tol or abs(y - self.y1) <= tol)) or (
            in_y and (abs(x - self.x0) <= tol or abs(x - self.x1) <= tol))


@dataclass
class Label:
    t: object  # matplotlib Text
    owner: Box | None
    baseline: float | None  # None: rotated (checked by its extent)
    group: int  # lines of one label share a group


class Canvas:
    def __init__(self) -> None:
        self.fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)  # measure at the preview's resolution
        ax = self.fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, FIG_W)
        ax.set_ylim(0, FIG_H)
        ax.set_axis_off()
        self.ax = ax
        self.boxes: dict[str, Box] = {}
        self.labels: list[Label] = []
        self.tags: list[tuple[Box, Box | None]] = []  # (tag, the box it sits on)
        self.polylines: list[list[tuple[float, float]]] = []
        self.arrows: list[tuple[list[tuple[float, float]], bool]] = []  # (points, starts on another arrow)
        self._group = 0

    # ---- text
    @staticmethod
    def ink_y(s: str, fs: float, weight: str = "normal") -> tuple[float, float]:
        """True ink extent of s (inches, relative to its baseline)."""
        prop = FontProperties(family=plt.rcParams["font.sans-serif"][0], size=fs, weight=weight)
        e = TextPath((0, 0), s, size=fs, prop=prop).get_extents()
        return e.y0 / 72.0, e.y1 / 72.0

    def new_group(self) -> int:
        self._group += 1
        return self._group

    def text(self, x: float, y: float, s: str, *, fs: float = FS, ha: str = "center", va: str = "baseline",
             color: str = C["black"], weight: str = "normal", z: float = 6.0, owner: Box | None = None,
             group: int | None = None):
        """Text at (x, y); va='baseline' puts the baseline at y, va='ink' centres the true ink on y."""
        base = y
        if va == "ink":
            lo, hi = self.ink_y(s, fs, weight)
            base = y - (lo + hi) / 2
        t = self.ax.text(x, base, s, fontsize=fs, ha=ha, va="baseline", color=color, fontweight=weight, zorder=z)
        self.labels.append(Label(t, owner, base, group if group is not None else self.new_group()))
        return t

    def rotated(self, x: float, y: float, s: str, *, fs: float = FS_HEAD, weight: str = "bold", linespacing=1.0):
        t = self.ax.text(x, y, s, fontsize=fs, ha="center", va="center", rotation=90, fontweight=weight, zorder=6,
                         linespacing=linespacing, multialignment="center")
        self.labels.append(Label(t, None, None, self.new_group()))
        return t

    def block(self, x: float, cy: float, rows: list[tuple[str, float]], *, ha: str = "center", lead: float | None = None,
              owner: Box | None = None) -> list:
        """Lines centred as a block on cy by their em-based optical extent (baselines line up across boxes)."""
        lead = lead if lead is not None else (1.12 if len(rows) <= 2 else 1.08)
        base = [0.0]
        for _, fs in rows[1:]:
            base.append(base[-1] - lead * fs / 72.0)
        top = base[0] + EM_TOP * rows[0][1] / 72.0
        bot = base[-1] - EM_BOT * rows[-1][1] / 72.0
        off = cy - (top + bot) / 2
        g = self.new_group()
        return [self.text(x, b + off, s, fs=fs, ha=ha, owner=owner, group=g) for (s, fs), b in zip(rows, base)]

    # ---- boxes
    def box(self, key: str, x0: float, y0: float, w: float, h: float, kind: str,
            rows: list[tuple[str, float]] | None = None, *, lead: float | None = None) -> Box:
        b = Box(x0, y0, w, h)
        if kind == "program":
            ec, lw, fc = CH["program"], LW_PROGRAM, C["white"]
        elif kind == "code":
            ec, lw, fc = CH["code"], LW_CODE, C["white"]
        elif kind == "external":
            ec, lw, fc = C["grey"], LW_EXTERNAL, C["white"]
        elif kind == "data":  # typed judgments and point-in-time stores: pale grey, no outline
            ec, lw, fc = "none", 0.0, DATA_FILL
        else:
            raise ValueError(kind)
        patch = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={ROUND}",
                               linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3, joinstyle="round")
        self.ax.add_patch(patch)
        if rows:
            self.block(b.cx, b.cy, rows, lead=lead, owner=b)
        self.boxes[key] = b
        return b

    # ---- arrows
    def arrow(self, pts: list[tuple[float, float]], color: str, *, head: bool = True, lw: float = LW_ARROW,
              z: float = 4.0, branch: bool = False) -> None:
        """Polyline arrow through pts (first point on the source edge, or on another arrow if `branch`; last point on
        the target edge)."""
        codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
        a = FancyArrowPatch(path=MPath(pts, codes), arrowstyle=ARROW_HEAD if head else "-", mutation_scale=1.0,
                            color=color, linewidth=lw, shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter",
                            capstyle="butt")
        self.ax.add_patch(a)
        self.polylines.append(list(pts))
        if head:
            self.arrows.append((list(pts), branch))

    def dot(self, p: tuple[float, float], color: str) -> None:
        """Branch-point marker."""
        self.ax.add_patch(Circle(p, DOT_R, facecolor=color, edgecolor="none", zorder=5))

    # ---- P-tags
    def tag(self, x0: float, y0: float, s: str, *, on: Box | None = None) -> Box:
        """6-pt bold white text on a black rounded rectangle with its lower-left corner at (x0, y0)."""
        patch = FancyBboxPatch((x0, y0), TAG_W, TAG_H, boxstyle="round,pad=0,rounding_size=0.02",
                               linewidth=0, facecolor=C["black"], zorder=7)
        self.ax.add_patch(patch)
        b = Box(x0, y0, TAG_W, TAG_H)
        self.text(b.cx, b.cy, s, fs=style.FS_TAG, va="ink", color=C["white"], weight="bold", z=8, owner=b)
        self.tags.append((b, on))
        return b

    def tab(self, b: Box, s: str, *, nth: int = 0, below: bool = False) -> Box:
        """A tag standing on the top (or hanging from the bottom) edge of b, right-aligned 0.025 in from the corner."""
        x0 = b.x1 - 0.025 - TAG_W - nth * (TAG_W + 0.02)
        return self.tag(x0, b.y0 - TAG_H if below else b.y1, s, on=b)

    # ---- geometry of labels
    def extent(self, t) -> tuple[float, float, float, float]:
        self.fig.canvas.draw()
        e = t.get_window_extent(renderer=self.fig.canvas.get_renderer())
        d = self.fig.dpi
        return (e.x0 / d, e.y0 / d, e.x1 / d, e.y1 / d)

    def ink(self, lab: Label, r) -> tuple[float, float, float, float]:
        e = lab.t.get_window_extent(renderer=r)
        d = self.fig.dpi
        x0, y0, x1, y1 = e.x0 / d, e.y0 / d, e.x1 / d, e.y1 / d
        if lab.baseline is None:
            return (x0, y0, x1, y1)
        prop = FontProperties(family=plt.rcParams["font.sans-serif"][0], size=lab.t.get_fontsize(),
                              weight=lab.t.get_fontweight())
        p = TextPath((0, 0), lab.t.get_text(), size=lab.t.get_fontsize(), prop=prop).get_extents()
        return (x0 + p.x0 / 72.0, lab.baseline + p.y0 / 72.0, x0 + p.x1 / 72.0, lab.baseline + p.y1 / 72.0)

    # ---- layout check
    def issues(self, margin: float = 0.02) -> tuple[list[str], list[tuple[float, str]]]:
        """Problems: a box label closer than INK_CLEAR (vertical) or PAD_X (horizontal) to its outline; labels of
        different groups overlapping; a free label on a box, a tag or an arrow; ink within `margin` of the figure edge;
        an arrow that does not start on a box edge (or on another arrow, for a branch) and end on a box edge, or that
        crosses a box; overlapping boxes; a tag on a box other than its own. Also returns the vertical clear space of
        every multi-line box label (for review)."""
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()

        def inter(a, b, tol=0.0):
            return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol

        def seg_hits(bb, p, q, tol=0.004):
            (x0, y0, x1, y1) = (bb[0] - tol, bb[1] - tol, bb[2] + tol, bb[3] + tol)
            n = 80
            return any(x0 <= p[0] + (q[0] - p[0]) * i / n <= x1 and y0 <= p[1] + (q[1] - p[1]) * i / n <= y1
                       for i in range(n + 1))

        def on_poly(p, pts, tol=0.002):
            for a, b in pairwise(pts):
                if min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol and \
                        min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol:
                    return True
            return False

        out: list[str] = []
        clear: dict[int, list] = {}
        items = [(lab, self.ink(lab, r)) for lab in self.labels if lab.t.get_text()]
        tag_boxes = [t for t, _ in self.tags]
        rects = list(self.boxes.values()) + tag_boxes
        for lab, bb in items:
            name = lab.t.get_text().replace("\n", " ")
            if bb[0] < margin or bb[1] < margin or bb[2] > FIG_W - margin or bb[3] > FIG_H - margin:
                out.append(f"near edge: {name!r} {tuple(round(v, 3) for v in bb)}")
            o = lab.owner
            if o is not None:
                is_tag = o in tag_boxes
                px, py = (0.012, 0.008) if is_tag else (PAD_X, INK_CLEAR)
                if bb[0] < o.x0 + px or bb[2] > o.x1 - px or bb[1] < o.y0 + py or bb[3] > o.y1 - py:
                    out.append(f"tight in box: {name!r} ink {tuple(round(v, 3) for v in bb)} box "
                               f"{tuple(round(v, 3) for v in o.rect())}")
                if not is_tag:
                    c = clear.setdefault(lab.group, [o, 9.0, 9.0, name])
                    c[1] = min(c[1], bb[1] - o.y0)
                    c[2] = min(c[2], o.y1 - bb[3])
            else:
                for b in rects:
                    if inter(bb, b.rect(), tol=-0.008):
                        out.append(f"free label on a box: {name!r}")
                for pts in self.polylines:
                    if any(seg_hits(bb, p_, q_) for p_, q_ in pairwise(pts)):
                        out.append(f"free label on an arrow: {name!r}")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i][0].group != items[j][0].group and inter(items[i][1], items[j][1], tol=-0.01):
                    out.append(f"labels overlap: {items[i][0].t.get_text()!r} / {items[j][0].t.get_text()!r}")
        for i, (a, on) in enumerate(self.tags):
            for b in list(self.boxes.values()) + tag_boxes[i + 1:]:
                if b is not on and inter(a.rect(), b.rect(), tol=-0.006):
                    out.append(f"tag near or on another box at ({a.cx:.2f}, {a.cy:.2f})")
            for pts in self.polylines:
                if any(seg_hits(a.rect(), p_, q_, tol=0.006) for p_, q_ in pairwise(pts)):
                    out.append(f"tag on an arrow at ({a.cx:.2f}, {a.cy:.2f})")
        bxs = list(self.boxes.items())
        for i in range(len(bxs)):
            for j in range(i + 1, len(bxs)):
                if inter(bxs[i][1].rect(), bxs[j][1].rect(), tol=0.001):
                    out.append(f"boxes overlap: {bxs[i][0]} / {bxs[j][0]}")
        boxes = list(self.boxes.items())
        for pts, branch in self.arrows:
            start_ok = any(b.on_edge(pts[0]) for _, b in boxes) or (
                branch and any(on_poly(pts[0], q) for q in self.polylines if q is not pts))
            end_ok = any(b.on_edge(pts[-1]) for _, b in boxes)
            if not (start_ok and end_ok):
                out.append(f"arrow not on box edges: {pts[0]} -> {pts[-1]}")
        for pts in self.polylines:
            for key, b in boxes:
                inner = (b.x0 + 0.006, b.y0 + 0.006, b.x1 - 0.006, b.y1 - 0.006)
                if any(seg_hits(inner, p_, q_, tol=0.0) for p_, q_ in pairwise(pts)):
                    out.append(f"arrow crosses box {key}: {pts[0]} -> {pts[-1]}")
        report = sorted((min(c[1], c[2]), f"{c[3]!r}: bottom {c[1]:.3f} top {c[2]:.3f}") for c in clear.values())
        return out, report


# ---------------------------------------------------------------------------------------------------------------------


def draw() -> Canvas:
    style.apply()
    k = load_constants()
    cv = Canvas()
    ax = cv.ax
    ROSE, INDIGO, MIXED = CH["W"], CH["T"], CH["mixed"]

    # ---- columns (x) ---------------------------------------------------------------------------------------------
    x_lab0, x_lab1 = 0.02, 0.235  # lane labels
    x_src0, x_src1 = 0.245, 0.905  # sources
    x_rd0 = x_src1 + GAP_T  # reading layer starts
    d_w = [0.52, 0.52, 0.52, 0.52, 0.40]  # decision sub-columns: spine, selection, sizing, risk rules, card
    x_ex1 = FIG_W - 0.02
    x_ex0 = x_ex1 - 0.62  # execution
    x_dc1 = x_ex0 - GAP
    x_dc0 = x_dc1 - sum(d_w) - GAP * (len(d_w) - 1)  # decision layer
    d_x0 = [x_dc0]
    for w_ in d_w[:-1]:
        d_x0.append(d_x0[-1] + w_ + GAP)
    x_tb1 = x_dc0 - GAP_T
    x_tb0 = x_tb1 - 0.11  # typed judgments
    x_rd1 = x_tb0 - GAP_T  # reading layer ends
    x_col = x_rd1 - 1.19  # right program column (every program output leaves at x_rd1)
    x_lc1 = x_col - 2 * 0.065  # left column (policy items, code features) ends; the P2 branch rises at x_col - 0.065

    # ---- rows (y, from the top) --------------------------------------------------------------------------------------
    y_head = 2.095  # baseline of the column headers
    W_top = 2.05
    ya1 = W_top - 0.015  # news chain (two-line boxes)
    ya0 = ya1 - H2
    ysn0 = ya0 - H1  # sentinel, sharing N1's bottom edge
    yc0 = 1.425  # policy row (the cap shares it)
    yp0 = (ya0 + yc0 + H1) / 2 - H1 / 2  # policy regime (P2), midway between the news chain and the policy row
    W_bot = yc0 - 0.03
    S_top = W_bot
    ych1 = S_top - 0.095  # decision chain (room for the tabs above it)
    ych0 = ych1 - HC
    S_bot = ych0 - 0.02
    T_top = S_bot
    yt = ych0 - 0.12 - HC  # trusted decision row
    gt = 0.025  # gap in the T-program stack
    fb_cy = yt + HC / 2  # F-B is centred on the trusted decision row
    yt_top = fb_cy + H1 / 2 + gt + H1
    y_stack0 = fb_cy - H1 / 2 - gt - H1
    y_bus = y_stack0 - 0.045  # trusted code bus
    y_code0 = y_bus - 0.035
    T_bot = y_code0 - 0.02

    # ---- lanes -------------------------------------------------------------------------------------------------------
    risk_x1 = d_x0[3] + d_w[3]
    lane_right = {"W": risk_x1 + GAP / 2, "S": x_rd1 + 0.04, "T": risk_x1 + GAP / 2}
    lane_span = {"W": (W_bot + GUTTER / 2, W_top), "S": (S_bot + GUTTER / 2, S_top - GUTTER / 2),
                 "T": (T_bot, T_top - GUTTER / 2)}
    for lane, (y0, y1) in lane_span.items():
        col, alpha = style.LANE_FILL[lane]
        ax.add_patch(Rectangle((x_lab0, y0), lane_right[lane] - x_lab0, y1 - y0, facecolor=col, alpha=alpha,
                               linewidth=0, zorder=0))
    xl = (x_lab0 + x_lab1) / 2
    cv.rotated(xl, sum(lane_span["W"]) / 2, r"untrusted $\mathbf{W}$")
    cv.rotated(xl, sum(lane_span["S"]) / 2, "semi-\n" + r"trusted $\mathbf{S}$", linespacing=0.95)
    cv.rotated(xl, sum(lane_span["T"]) / 2, r"trusted $\mathbf{T}$")

    # ---- column headers ----------------------------------------------------------------------------------------------
    def head(x: float, s: str):
        return cv.text(x, y_head, s, fs=FS_HEAD, weight="bold")

    h_src = head((x_src0 + x_src1) / 2, "point-in-time sources")
    h_tb = head((x_tb0 + x_tb1) / 2, "typed judgments")
    h_rd = head(0.0, "reading layer (question programs)")  # centred in the space the two neighbours leave
    x_a, x_b = cv.extent(h_src)[2], cv.extent(h_tb)[0]
    h_rd.set_x((x_a + x_b) / 2)
    head((x_dc0 + x_dc1) / 2, "decision layer (code)")
    head((x_ex0 + x_ex1) / 2, "execution")

    # ---- W lane: reading layer ---------------------------------------------------------------------------------------
    src_w = cv.box("src_W", x_src0, yc0, x_src1 - x_src0, ya1 - yc0, "external", [("news items", FS)])

    news = [("N0", "screen", 0.293), ("N1", "per item", 0.367), ("cluster", None, 0.317), ("N2", "novelty", 0.332),
            ("N3", "synthesis", 0.407)]
    g = (x_rd1 - x_rd0 - sum(w_ for *_, w_ in news)) / (len(news) - 1)
    x = x_rd0
    chain = []
    for ident, gloss, w_ in news:
        if gloss is None:
            b = cv.box(ident, x, ya0, w_, H2, "code", [(ident, FS)])
        else:
            b = cv.box(ident, x, ya0, w_, H2, "program", [(ident, FS), (gloss, FS_G)])
        chain.append(b)
        x += w_ + g
    n1 = cv.boxes["N1"]
    cv.box("sentinel", n1.x0, ysn0, n1.w, H1, "program", [("sentinel", FS)])  # runs inside N1: shared edge

    p1 = cv.box("P1", x_rd0, yc0, x_lc1 - x_rd0, H1, "program", [("policy items", FS)])
    px = cv.box("PX", x_col, yc0, x_rd1 - x_col, H1, "code", [("theme exposure", FS)])
    p2 = cv.box("P2", x_col, yp0, x_rd1 - x_col, H1, "program", [("policy regime (overlay only)", FS)])

    cv.arrow([src_w.right(chain[0].cy - src_w.cy), chain[0].left()], ROSE)
    for a_, b_ in pairwise(chain):
        cv.arrow([a_.right(), b_.left()], ROSE)
    cv.arrow([src_w.right(p1.cy - src_w.cy), p1.left()], ROSE)
    cv.arrow([p1.right(), px.left()], ROSE)
    x_fork = (p1.x1 + px.x0) / 2
    cv.arrow([(x_fork, p1.cy), (x_fork, p2.cy), p2.left()], ROSE, branch=True)  # P2: a branch off the P1 output
    cv.dot((x_fork, p1.cy), ROSE)

    # ---- S lane: reading layer ---------------------------------------------------------------------------------------
    ys = sum(lane_span["S"]) / 2
    src_s = cv.box("src_S", x_src0, ys - H1 / 2, x_src1 - x_src0, H1, "external", [("10-K Item 1", FS)])
    fa = cv.box("F-A", x_col, ys - H1 / 2, x_rd1 - x_col, H1, "program", [("10-K profile (F-A)", FS)])
    cv.arrow([src_s.right(), fa.left()], ROSE)
    cv.arrow([fa.top(), px.bottom()], ROSE)  # profile flags enter the theme exposure

    # ---- T lane: reading layer ---------------------------------------------------------------------------------------
    src_t = cv.box("src_T", x_src0, y_code0, x_src1 - x_src0, yt_top - y_code0, "external",
                   [("prices", FS), ("XBRL facts", FS), ("8-K filings", FS), ("macro vintages", FS)])
    tcode = cv.box("tcode", x_rd0, y_code0, x_lc1 - x_rd0, yt_top - y_code0, "code",
                   [("code features", FS), (r"buckets, factors, $\sigma$", FS_G)])
    cv.arrow([src_t.right(), tcode.left()], INDIGO)
    tprogs = []
    for i, label in enumerate(["technical", "fundamentals (F-B)", "macro"]):
        y0 = yt_top - H1 - i * (H1 + gt)
        b = cv.box(label, x_col, y0, x_rd1 - x_col, H1, "program", [(label, FS)])
        tprogs.append(b)
        cv.arrow([tcode.right(b.cy - tcode.cy), b.left()], INDIGO)

    # ---- typed judgments: one pale-grey segment per lane that has judgments (W, T) -----------------------------------
    tbw = cv.box("tables_W", x_tb0, lane_span["W"][0], x_tb1 - x_tb0, ya1 - lane_span["W"][0], "data")
    tbt = cv.box("tables_T", x_tb0, y_stack0 - 0.01, x_tb1 - x_tb0, lane_span["T"][1] - (y_stack0 - 0.01), "data")
    for b in [chain[-1], px]:
        cv.arrow([b.right(), (tbw.x0, b.cy)], ROSE)
    for b in tprogs:
        cv.arrow([b.right(), (tbt.x0, b.cy)], INDIGO)

    # ---- decision layer --------------------------------------------------------------------------------------------
    d = list(zip(d_x0, d_w))
    y_track = ya1 - 0.02  # sentinel h -> risk rules, leaving the W judgments below their rounded top corner
    # W row: the spine untrusted learner -> cap (one column with the fusion box and the trusted learner)
    ul = cv.box("ul", d[0][0], y_track - 0.10 - H2, d[0][1], H2, "code", [("untrusted", FS), ("learner", FS)])
    cap = cv.box("cap", d[0][0], yc0, d[0][1], H1, "code", [(r"cap $\pm\kappa_t$", FS)])

    # middle row (white): fusion chain
    lbox = cv.box("L", d[0][0], ych0, d[0][1], HC, "code", [("fusion", FS), (r"$L = L^{T} + \overline{L}{}^{W}$", FS)])
    sel = cv.box("sel", d[1][0], ych0, d[1][1], HC, "code",
                 [("band", FS), ("selection", FS), (rf"$K = {k['K']}$, $J = {k['J']}$", FS_G)])
    siz = cv.box("siz", d[2][0], ych0, d[2][1], HC, "code", [("reference-", FS), ("anchored", FS), ("sizing", FS)])
    risk = cv.box("risk", d[3][0], ych0, d[3][1], HC, "code", [("risk", FS), ("rules", FS)])
    card = cv.box("card", d[4][0], ych0, d[4][1], HC, "code", [("decision", FS), ("card", FS)])
    for a_, b_ in zip([lbox, sel, siz, risk], [sel, siz, risk, card]):
        cv.arrow([a_.right(), b_.left()], MIXED)

    # T row: trusted learner -> rank; reference book; anchor
    tl = cv.box("tl", d[0][0], yt, d[0][1], HC, "code", [("trusted", FS), ("learner", FS)])
    rk = cv.box("rank", d[1][0], yt, d[1][1], HC, "code", [(r"rank $R^{T}$", FS)])
    rb = cv.box("ref", d[2][0], yt, d[2][1], HC, "code",
                [("reference", FS), ("book", FS), (r"$v^{\mathrm{ref}},\,\varphi,\,f_g$", FS_G)])
    an = cv.box("anchor", d[3][0], yt, d[3][1], HC, "code", [(r"anchor $A$", FS), ("(prices, 8-K)", FS_G)])

    # arrows into the decision layer
    cv.arrow([(tbw.x1, ul.cy), ul.left()], ROSE)
    cv.arrow([(tbt.x1, tl.cy), tl.left()], INDIGO)
    x_desc = risk.x1 - 0.08  # the sentinel's descent, near the right end of the W lane
    cv.arrow([(tbw.x1, y_track), (x_desc, y_track), (x_desc, risk.y1)], ROSE)  # sentinel h -> risk rules
    # the trusted code bus: under the T programs and the T judgments, into the trusted learner, the book and the anchor
    cv.arrow([tcode.right(y_bus - tcode.cy), (an.cx, y_bus), an.bottom()], INDIGO)
    for b in [tl, rb]:
        cv.arrow([(b.cx, y_bus), b.bottom()], INDIGO, branch=True)
        cv.dot((b.cx, y_bus), INDIGO)
    # the spine, with its value labels left of the arrows (the right side carries the tags)
    cv.arrow([ul.bottom(), cap.top()], ROSE)
    cv.arrow([cap.bottom(), lbox.top()], ROSE)
    cv.arrow([tl.top(), lbox.bottom()], INDIGO)
    xs = ul.cx - 0.035
    cv.text(xs, (ul.y0 + cap.y1) / 2, r"$L^{W}$", ha="right", va="ink")
    cv.text(xs, (cap.y0 + lbox.y1) / 2, r"$\overline{L}{}^{W}$", ha="right", va="ink")
    cv.text(xs, (tl.y1 + lbox.y0) / 2, r"$L^{T}$", ha="right", va="ink")

    # trusted row
    cv.arrow([tl.right(), rk.left()], INDIGO)
    cv.arrow([rk.right(), rb.left()], INDIGO)
    cv.arrow([rk.top(), sel.bottom()], INDIGO)
    cv.arrow([rb.top(), siz.bottom()], INDIGO)
    cv.arrow([an.top(), risk.bottom()], INDIGO)

    # ---- execution (white output column): backtest, or pre-trade checks -> broker ------------------------------------
    xw = x_ex1 - x_ex0
    pre = cv.box("pre", x_ex0, card.cy - H2 / 2, xw, H2, "code", [("pre-trade", FS), ("checks", FS)])
    bt = cv.box("bt", x_ex0, cap.y0, xw, H2, "code", [("backtest", FS), ("engine", FS)])
    api = cv.box("api", x_ex0, yt, xw, HC, "external", [("paper-trading", FS), ("broker API", FS)])
    cv.arrow([card.right(), pre.left()], MIXED)
    cv.arrow([card.top(), (card.cx, bt.cy), bt.left()], MIXED)
    cv.arrow([pre.bottom(), api.top()], MIXED)

    # ---- point-in-time stores (strip under the reading layer; it runs on under the judgments column) -----------------
    ys1 = T_bot - 0.025
    stores = [("weekly blackboard", 0.82), ("event store", 0.52), ("filing-profile store", 0.80)]
    sx0, sx1 = x_rd0, x_tb1
    sg = (sx1 - sx0 - sum(w_ for _, w_ in stores)) / (len(stores) - 1)
    x = sx0
    for i, (s, w_) in enumerate(stores):
        cv.box(f"store{i}", x, ys1 - H1, w_, H1, "data", [(s, FS)])
        x += w_ + sg
    cv.text(sx0 - 0.03, ys1 - H1 / 2, "point-in-time stores", ha="right", va="ink")
    # P5 (history) at the stores: records keep their origin taint; the read predicate follows the tag
    y_pred = ys1 - H1 - 0.075
    t5s = cv.tag(sx0, y_pred - TAG_H / 2, "P5")
    cv.text(t5s.x1 + 0.04, y_pred,
            r"read only if $\mathrm{available\_at}\,<\,c_t$, writer allowed, $\mathrm{taint}\,\leq\,\mathrm{clearance}$",
            fs=FS_G, ha="left", va="ink")

    # ---- Proposition tags ------------------------------------------------------------------------------------------
    cv.tab(sel, "P2")
    cv.tab(sel, "P1", nth=1)
    cv.tab(siz, "P3")
    cv.tab(risk, "P2", below=True)  # the anchor gate: where the sentinel and the anchor meet
    cv.tab(ul, "P5")  # history: past untrusted inputs reach week t through the untrusted learner's fit
    cv.tab(cap, "P4")  # the cap
    # glosses (black) right of the spine
    cv.text(ul.x1 + 0.04, ul.cy, r"fit on residual of $L^{T}$", fs=FS_G, ha="left", va="ink")
    cv.text(cap.x1 + 0.04, cap.cy, rf"$\kappa_t = {k['cap_mult']}\,s^{{T}}_t$", fs=FS_G, ha="left", va="ink")
    # what the sentinel may do: next to the lower end of its descent, right-aligned on it
    note = [(r"sentinel $h$: ×½ in sizing;", FS_G), ("exclusion only if anchored", FS_G), ("(a veto at selection)", FS_G)]
    cv.block(x_desc - 0.035, 1.565, note, ha="right", lead=1.15)
    # P0 bracket under the trusted decision boxes and the bus, below the T lane
    yb = T_bot - 0.035
    xb0, xb1 = tl.x0, an.x1
    cv.arrow([(xb0, yb + 0.03), (xb0, yb), (xb1, yb), (xb1, yb + 0.03)], C["black"], head=False, lw=0.6)
    xm = (xb0 + xb1) / 2
    cv.arrow([(xm, yb), (xm, yb - 0.025)], C["black"], head=False, lw=0.6)
    y_p0 = yb - 0.025 - 0.02 - TAG_H / 2
    p0_note = cv.text(0.0, y_p0, "invariant to untrusted inputs", fs=FS_G, ha="left", va="ink")
    w_note = cv.extent(p0_note)[2] - cv.extent(p0_note)[0]
    x_grp = xm - (TAG_W + 0.04 + w_note) / 2  # tag + note centred under the bracket
    tg = cv.tag(x_grp, y_p0 - TAG_H / 2, "P0")
    p0_note.set_x(tg.x1 + 0.04)
    return cv


def main() -> None:
    cv = draw()
    problems, clear = cv.issues()
    for msg in problems:
        print("layout:", msg)
    if "-v" in sys.argv:
        for v, msg in clear:
            print(f"clearance {v:.3f}  {msg}")
    pdf, png = style.save(cv.fig, NAME, tight=False)
    gray = style.check_grayscale(png)
    print(pdf, png, gray, sep="\n")


if __name__ == "__main__":
    main()
