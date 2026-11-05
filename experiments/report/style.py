"""Shared figure style for the paper (outline_v1 §4).

Every figure script under experiments/report/ calls `apply()` before drawing and `save(fig, name)` to write
paper/figures/<name>.pdf (vector, TrueType fonts embedded: pdf.fonttype 42) and a 300-dpi PNG preview.
`check_grayscale(path)` writes a grayscale copy of a PNG preview for review.

Palette: Paul Tol's "muted" qualitative scheme (colour-blind safe). Channels: trusted = indigo, semi-trusted = sand,
untrusted = rose; mixed trusted/untrusted values = dark grey; question programs (model calls) = teal border; code = black
outline; reference path = grey dashed.

Type: Linux Biolinum O (acmart's sans companion), else DejaVu Sans. The regular, bold and italic Biolinum OTFs live in
experiments/report/fonts/ (copied from the TeX Live tree by fonts/fetch_biolinum.sh; the Libertine fonts are GPL/OFL).
They are CFF-flavoured, and matplotlib embeds a CFF .otf under pdf.fonttype 42 as "CID Type 0C (OT)" with a font file
that poppler reports as mismatched, so `font_family()` converts each to TrueType outlines once (fontTools, cu2qu) and
registers only the .ttf copies, which embed as CID TrueType.
"""

from __future__ import annotations

import shutil
import subprocess
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper" / "figures"

PT = 1.0 / 72.0  # inches per point

# ---------------------------------------------------------------------------------------------------------------------
# Sizes (outline_v1 §4 "Sizes"): acmart sigconf column and text widths.
COLUMN_WIDTH = 3.33  # in (241 pt)
FULL_WIDTH = 7.0  # in (506 pt)
PAD_INCHES = 0.02  # tight bounding box padding

# Type sizes (pt)
FS_LABEL = 7.0  # labels
FS_TICK = 6.0  # ticks and point labels
FS_PANEL = 8.0  # bold lower-case panel letters
FS_BOX = 7.0  # box labels in diagrams (6.5-7 pt)
FS_GLOSS = 6.5  # secondary lines in diagram boxes (glosses, parameters) and free annotations
FS_TAG = 6.0  # P-tags

# Lines and marks (pt)
LW_AXES = 0.5
LW_DATA = 1.0
LW_ERR = 0.6
LW_GRID = 0.3
MARKER_SIZE = 3.5
CI_ALPHA = 0.2

# ---------------------------------------------------------------------------------------------------------------------
# Palette: Paul Tol "muted" + pale grey; neutrals for text, outlines and the code-only/reference series.
COLORS: dict[str, str] = {
    "indigo": "#332288",
    "cyan": "#88CCEE",
    "teal": "#44AA99",
    "green": "#117733",
    "olive": "#999933",
    "sand": "#DDCC77",
    "rose": "#CC6677",
    "wine": "#882255",
    "purple": "#AA4499",
    "pale_grey": "#DDDDDD",
    # neutrals (not colour-coded categories)
    "black": "#000000",
    "dark_grey": "#595959",
    "mid_grey": "#666666",
    "grey": "#808080",
    "white": "#FFFFFF",
}

# Channels and component kinds (outline_v1 §4 "Palette"; Fig. 1 visual spec).
CHANNEL_COLORS: dict[str, str] = {
    "T": COLORS["indigo"],  # trusted
    "S": COLORS["sand"],  # semi-trusted
    "W": COLORS["rose"],  # untrusted
    # values that combine trusted and untrusted inputs: dark grey #595959 (luma 89) sits 45 levels from rose (134) and
    # 38 from indigo (51) in grayscale
    "mixed": COLORS["dark_grey"],
    "program": COLORS["teal"],  # question programs (model calls): teal border
    "code": COLORS["black"],  # code: black outline
    "reference": COLORS["grey"],  # reference (text-free) path: grey dashed
}

# Lane fills for the trust-lane architecture figure: (colour, alpha).
LANE_FILL: dict[str, tuple[str, float]] = {
    "W": (COLORS["rose"], 0.10),
    "S": (COLORS["sand"], 0.20),
    "T": (COLORS["indigo"], 0.08),
}

# Reader encoding (NAME-1 labels). Every series is distinguishable in grayscale by marker and line style.
# `mfc` is the marker face colour ("none" = open marker).
READER_STYLE: dict[str, dict[str, object]] = {
    "Typed programs": dict(color=COLORS["indigo"], marker="o", mfc=COLORS["indigo"], linestyle="-"),
    "Holistic": dict(color=COLORS["rose"], marker="o", mfc="none", linestyle="--"),
    "Holistic + guards": dict(color=COLORS["purple"], marker="s", mfc=COLORS["purple"], linestyle="-."),
    "Holistic + marking": dict(color=COLORS["wine"], marker="^", mfc=COLORS["wine"], linestyle=(0, (5, 1.5))),
    "Headline tone": dict(color=COLORS["sand"], marker="D", mfc=COLORS["sand"], linestyle=(0, (3, 1, 1, 1, 1, 1))),
    "Typed, median aggregation": dict(color=COLORS["cyan"], marker="P", mfc=COLORS["cyan"], linestyle="-"),
    "TA-style pipeline": dict(color=COLORS["olive"], marker="X", mfc=COLORS["olive"], linestyle=(0, (1, 1))),
    "Mono-CoT": dict(color=COLORS["green"], marker="*", mfc=COLORS["green"], linestyle=(0, (4, 2))),
    "Code-only": dict(color=COLORS["grey"], marker="", mfc="none", linestyle=":"),
}

# ---------------------------------------------------------------------------------------------------------------------
# Fonts
_PREFERRED_FAMILY = "Linux Biolinum O"
_FALLBACK_FAMILY = "DejaVu Sans"
FONT_DIR = Path(__file__).resolve().parent / "fonts"
_BIOLINUM_STEMS = ("LinBiolinum_R", "LinBiolinum_RB", "LinBiolinum_RI")  # regular, bold, italic


def _kpsewhich(name: str) -> Path | None:
    if shutil.which("kpsewhich") is None:
        return None
    try:
        out = subprocess.run(["kpsewhich", name], capture_output=True, text=True, timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(out) if out and Path(out).is_file() else None


def otf_to_ttf(src: Path, dst: Path, max_err: float = 1.0) -> None:
    """Rewrite a CFF-flavoured OpenType font with TrueType (quadratic) outlines; `max_err` in font units."""
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    from fontTools.ttLib import TTFont, newTable

    font = TTFont(str(src))
    if font.sfntVersion != "OTTO" or "CFF " not in font:
        raise ValueError(f"{src} is not a CFF-flavoured OpenType font")
    order = font.getGlyphOrder()
    glyph_set = font.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(glyph_set)
        glyph_set[name].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, g in glyf.glyphs.items():  # left side bearing = xMin of the new outline
        if hasattr(g, "xMin"):
            hmtx[name] = (hmtx[name][0], g.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    maxp.maxZones = 1
    for field in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
                  "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, field, 0)
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = "\x00\x01\x00\x00"
    dst.parent.mkdir(parents=True, exist_ok=True)
    font.save(str(dst))


def _biolinum_ttfs() -> list[Path]:
    """TrueType copies of Biolinum R/RB/RI in FONT_DIR, converted on first use from the OTFs there (or in the TeX tree)."""
    out = []
    for stem in _BIOLINUM_STEMS:
        ttf = FONT_DIR / f"{stem}.ttf"
        if not ttf.is_file():
            otf = FONT_DIR / f"{stem}.otf"
            src = otf if otf.is_file() else _kpsewhich(f"{stem}.otf")
            if src is None:
                continue
            try:
                otf_to_ttf(src, ttf)
            except Exception as exc:  # noqa: BLE001 - any conversion failure falls back to DejaVu Sans
                warnings.warn(f"could not convert {src} to TrueType ({exc}); using {_FALLBACK_FAMILY}", stacklevel=2)
                continue
        out.append(ttf)
    return out


def font_family() -> str:
    """Linux Biolinum O (TrueType copies registered with the font manager) if all three styles are available, else
    DejaVu Sans. Any other Biolinum entry the font manager knows (for example a system .otf) is dropped, so a CFF file
    is never embedded."""
    ttfs = _biolinum_ttfs()
    if len(ttfs) < len(_BIOLINUM_STEMS):
        return _FALLBACK_FAMILY
    fm = font_manager.fontManager
    fm.ttflist = [f for f in fm.ttflist if f.name != _PREFERRED_FAMILY or Path(f.fname).suffix.lower() == ".ttf"]
    known = {Path(f.fname).resolve() for f in fm.ttflist}
    for ttf in ttfs:
        if ttf.resolve() not in known:
            fm.addfont(str(ttf))
    fm._findfont_cached.cache_clear()  # ttflist was filtered above
    return _PREFERRED_FAMILY


def apply() -> str:
    """Set the paper rcParams; returns the font family in use."""
    family = font_family()
    plt.rcdefaults()
    rc = {
        # type
        "font.family": "sans-serif",
        "font.sans-serif": [family, _FALLBACK_FAMILY],
        "font.size": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_TICK,
        "mathtext.default": "it",
        # output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": PAD_INCHES,
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        # axes and lines
        "axes.linewidth": LW_AXES,
        "axes.edgecolor": COLORS["black"],
        "axes.labelcolor": COLORS["black"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.titlepad": 2.0,
        "axes.labelpad": 2.0,
        "grid.color": COLORS["pale_grey"],
        "grid.linewidth": LW_GRID,
        "lines.linewidth": LW_DATA,
        "lines.markersize": MARKER_SIZE,
        "lines.markeredgewidth": 0.6,
        "errorbar.capsize": 0.0,
        "xtick.major.width": LW_AXES,
        "ytick.major.width": LW_AXES,
        "xtick.minor.width": LW_AXES * 0.8,
        "ytick.minor.width": LW_AXES * 0.8,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "xtick.color": COLORS["black"],
        "ytick.color": COLORS["black"],
        "legend.frameon": False,
        "patch.linewidth": 0.6,
        "hatch.linewidth": 0.5,
    }
    if family == _PREFERRED_FAMILY:
        rc.update({
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.bfit": f"{family}:bold",  # Biolinum has no bold italic
            "mathtext.cal": family,  # unused; avoids a search for a cursive family
            "mathtext.sf": family,
        })
    else:
        rc["mathtext.fontset"] = "dejavusans"
    plt.rcParams.update(rc)
    return family


def save(fig: plt.Figure, name: str, *, tight: bool = True, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write <out_dir>/<name>.pdf (vector, fonts embedded) and a 300-dpi <name>.png preview.

    `tight=False` keeps the figure's exact size (for diagrams laid out in figure inches)."""
    out = Path(out_dir) if out_dir is not None else FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    pdf, png = out / f"{name}.pdf", out / f"{name}.png"
    kw: dict[str, object] = {"facecolor": "white"}
    if tight:
        kw.update(bbox_inches="tight", pad_inches=PAD_INCHES)
    with plt.rc_context({"savefig.bbox": "tight" if tight else "standard"}):
        fig.savefig(pdf, metadata={"Creator": None, "Producer": None, "Author": None, "Title": None, "CreationDate": None}, **kw)
        fig.savefig(png, dpi=300, metadata={"Software": None}, **kw)
    return pdf, png


def check_grayscale(path: str | Path) -> Path:
    """Render a grayscale copy of a PNG preview next to it (<stem>_gray.png) for review; returns its path."""
    from PIL import Image

    src = Path(path)
    if src.suffix.lower() != ".png":
        src = src.with_suffix(".png")
    dst = src.with_name(f"{src.stem}_gray.png")
    with Image.open(src) as im:
        im.convert("L").save(dst, dpi=(300, 300))
    return dst
