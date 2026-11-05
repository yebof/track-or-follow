"""Code-side rendering of System-One states (design_v1 §5.0; question bank §A conventions, §B schemas).

Jev is not a calculator and cannot compare dates (docs/jev/pages/model-jaggedness__jev-1.13, failure modes
2-3), so code does the arithmetic and hands the model words: numbers as "<value> (<bucket>)", probabilities
as bins, times relative to the review. `envelope` assembles a state in the bank's schema order and rejects
any absolute date, (for trusted numeric states) any dollar amount, and (for the typed-only synthesis states N3
and P2, rule R3) any word that is not code-rendered label vocabulary.

D0 (design_v1 §3.3 as amended by design_v2_delta SAN-1..SAN-4) lives here too: `d0` sanitizes one text field and
returns the event flags that go to evidence.parquet, the cards and the FEED-1 table; `d0_blind` is the only way a
reader turns Web text into state text, so D0 always runs before blinding (SAN-2).
"""

from __future__ import annotations

import colorsys
import html
import inspect
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Any

import numpy as np
import pandas as pd

from jevtrader.data.blinding import absolute_time
from jevtrader.features.technical import _pct_bucket as universe_bucket  # noqa: F401  (universe percentile words)
from jevtrader.programs.bank_v1 import Bank
from jevtrader.s1.types import Answer

HEADLINE_MAX, SUMMARY_MAX = 300, 1200  # D0 caps (§3.3)
NUMERIC_PROGRAMS = ("T", "FB", "M")  # states of trusted buckets only: no dollar amounts or levels
TYPED_ONLY_PROGRAMS = ("N3", "P2")  # synthesis states: code-rendered typed labels only (R3, the taint barrier)
NOT_AVAILABLE = "not available"
DATAMARK = "\u02c6"  # conventions.datamark.char (U+02C6 MODIFIER LETTER CIRCUMFLEX ACCENT), H-spot evidence

_INVISIBLE = frozenset(("Cf", "Co", "Cs", "Cn"))  # format (zero-width, bidi, tag block), private, surrogate, unassigned
# Unicode Default_Ignorable_Code_Point (DerivedCoreProperties) outside Cf/Cn: combining grapheme joiner, Hangul
# fillers, Khmer inherent vowels, Mongolian variation selectors, the variation selectors U+FE00-FE0F and U+E0100-E01EF
# (one byte per selector smuggles a payload like the tag block), and the whole U+E0000-E0FFF plane block (SAN-3)
_IGNORABLE = frozenset(cp for lo, hi in (
    (0x00AD, 0x00AD), (0x034F, 0x034F), (0x061C, 0x061C), (0x115F, 0x1160), (0x17B4, 0x17B5), (0x180B, 0x180F),
    (0x200B, 0x200F), (0x202A, 0x202E), (0x2060, 0x206F), (0x3164, 0x3164), (0xFE00, 0xFE0F), (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0), (0xFFF0, 0xFFF8), (0x1BCA0, 0x1BCA3), (0x1D173, 0x1D17A), (0xE0000, 0xE0FFF),
) for cp in range(lo, hi + 1))
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_TAG = re.compile(r"<[^<>]{0,1000}>")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_DOLLAR = re.compile(r"\$\s?\d")
# Letters that look like Latin ones (a subset of Unicode TR39 confusables). NFKC does not fold them, so "\u0410pple"
# (Cyrillic A) or "Boreal\u0131s" (dotless i, which Python's case-insensitive `re` also matches to "i") would pass D0
# and fool or break the blinder; inside a word made only of ASCII letters and look-alikes they are folded to Latin.
_CONFUSABLE_MAP = {
    # Cyrillic upper and lower case
    **dict(zip("\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0425\u0423\u0405\u0406\u0408"
               "\u051a\u051c", "ABEKMHOPCTXYSIJQW")),
    **dict(zip("\u0430\u0435\u043e\u0440\u0441\u0443\u0445\u0455\u0456\u0458\u0501\u04bb\u051b\u051d\u04cf",
               "aeopcyxsijdhqwl")),
    # Greek upper and lower case
    **dict(zip("\u0391\u0392\u0395\u0396\u0397\u0399\u039a\u039c\u039d\u039f\u03a1\u03a4\u03a5\u03a7",
               "ABEZHIKMNOPTYX")),
    **dict(zip("\u03bf\u03c1\u03b9\u03bd\u03b1\u03c7\u03ba", "opivaxk")),
    # Latin-script look-alikes: dotless i/j, dotted capital I, alpha, script g, iota, dental click, small capitals
    **dict(zip("\u0131\u0130\u0237\u0251\u0261\u0269\u01c0\u028f\u029f\u026a\u0280\u0274\u0299\u0262\u029c"
               "\u1d00\u1d04\u1d05\u1d07\u1d0a\u1d0b\u1d0d\u1d0f\u1d18\u1d1b\u1d1c\u1d20\u1d21\u1d22\ua730\ua731",
               "iIjagilylirnbghacdejkmoptuvwzfs")),
    # Armenian
    **dict(zip("\u0585\u057d\u0570\u0578\u0581\u0566\u0575\u056c\u0555\u054d\u054f\u053c", "ouhngqjlOUSL")),
}
# Cherokee capitals (and their small forms, which look like small capitals)
_CHEROKEE = dict(zip("\u13a0\u13a1\u13a2\u13a5\u13a9\u13aa\u13ab\u13ac\u13b3\u13b7\u13bb\u13c0\u13c2\u13c3\u13cf"
                     "\u13d9\u13da\u13de\u13df\u13e2\u13e6\u13f4", "DRTiYAJEWMHGhZbVSLCPKB"))
_CONFUSABLE_MAP.update(_CHEROKEE)
_CONFUSABLE_MAP.update({c.lower(): v.lower() for c, v in _CHEROKEE.items() if len(c.lower()) == 1 and c.lower() != c})
_CONFUSABLE = str.maketrans(_CONFUSABLE_MAP)
_CONFUSABLE_CHARS = frozenset(_CONFUSABLE_MAP)
_LETTER_RUN = re.compile(r"[^\W\d_]+")


class RenderError(ValueError):
    pass


# ------------------------------------------------------------------------------------ D0 sanitation
@dataclass(frozen=True)
class D0Flags:
    """D0 event flags of one text (SAN-4): counts of what D0 removed or changed. They are logged per item to
    evidence.parquet, shown on cards when non-zero and aggregated by year in the FEED-1 table."""

    hidden_chars_removed: int = 0  # non-space characters D0 removed as not rendered (SAN-1; a whole field if dropped)
    invisible_removed: int = 0  # Cf/Co/Cs/Cn and default-ignorable code points dropped (SAN-3)
    confusables_folded: int = 0  # look-alike letters folded to Latin (SAN-2)
    mixed_script_residual: int = 0  # tokens still mixing Latin with another script after folding (SAN-2)
    truncated: bool = False  # the length cap cut the text

    def __add__(self, other: D0Flags) -> D0Flags:
        return D0Flags(self.hidden_chars_removed + other.hidden_chars_removed,
                       self.invisible_removed + other.invisible_removed,
                       self.confusables_folded + other.confusables_folded,
                       self.mixed_script_residual + other.mixed_script_residual, self.truncated or other.truncated)

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)

    def nonzero(self) -> dict[str, int | bool]:
        return {k: v for k, v in asdict(self).items() if v}

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @property
    def any(self) -> bool:
        return bool(self.nonzero())


class Clean(str):
    """A D0-sanitized text (not yet blinded). Blinding returns a plain `str`, so the type marks the stage:
    `d0_blind` refuses to blind anything that is not `Clean` (SAN-2: D0 runs before blinding for every reader)."""

    flags: D0Flags

    def __new__(cls, text: str, flags: D0Flags):
        s = super().__new__(cls, text)
        s.flags = flags
        return s


# ---------------------------------------------------------------- D0 markup: the fail-closed rendering model (SAN-1)
# D0 keeps text only where it can show that a browser renders it; where it cannot, it removes more, never less:
# - an element's own subtree is hidden (`_HIDE`) by a hiding attribute, a never-rendered element, or an inline style
#   that hides it in the computed-style model below (colour contrast against the effective background, cumulative
#   opacity, effective font size, clipping, visibility, display);
# - the whole field is dropped (`_DROP`) when rendering depends on something D0 does not model: a stylesheet (a
#   <style> element with content, <link rel=stylesheet>), attributes on <html>/<body>, CSS that can move or paint over
#   other text (positioning, negative margins, transforms, box shadows, outlines, visual reordering), or any CSS
#   property or value D0 cannot parse (calc(), var(), unknown properties, unbalanced quotes).
# The news archive holds no markup at all (0 of 857,416 headline and summary fields), so this costs nothing on clean
# text; every drop is counted in `hidden_chars_removed`.
_OK, _HIDE, _DROP = 0, 1, 2
# content never rendered (UA stylesheet display:none, fallback content, annotations, foreign content) and void elements
_NEVER_RENDERED = frozenset((
    "script", "style", "noscript", "template", "title", "iframe", "noembed", "noframes", "datalist", "select",
    "object", "applet", "video", "audio", "canvas", "meter", "progress", "rp", "rt", "rtc", "svg", "math",
    "frameset"))
_VOID = frozenset(("area", "base", "basefont", "bgsound", "br", "col", "embed", "frame", "hr", "img", "input",
                   "keygen", "link", "meta", "param", "source", "track", "wbr"))
_NOT_PUSHED = frozenset(("html", "body", "head"))  # HTML5 merges their attributes into the one root element
_NEVER_POPPED = frozenset(("plaintext",))  # everything after <plaintext> is text: its end tag never closes it
# HTML5 "special" elements (spec 13.2.4.2): an end tag never closes an element across one of these (and the scope
# boundaries of every "has an element in scope" test are among them)
_SPECIAL = frozenset((
    "address", "applet", "area", "article", "aside", "base", "basefont", "bgsound", "blockquote", "body", "br", "button",
    "caption", "center", "col", "colgroup", "dd", "details", "dir", "div", "dl", "dt", "embed", "fieldset",
    "figcaption", "figure", "footer", "form", "frame", "frameset", "h1", "h2", "h3", "h4", "h5", "h6", "head", "header",
    "hgroup", "hr", "html", "iframe", "img", "input", "keygen", "li", "link", "listing", "main", "marquee", "menu",
    "meta", "nav", "noembed", "noframes", "noscript", "object", "ol", "p", "param", "plaintext", "pre", "script",
    "search", "section", "select", "source", "style", "summary", "table", "tbody", "td", "template", "textarea",
    "tfoot", "th", "thead", "title", "tr", "track", "ul", "wbr", "xmp", "mi", "mo", "mn", "ms", "mtext",
    "annotation-xml", "foreignobject", "desc"))
# inline (and inline-block) elements render without a break, so "Ap<b>p</b>le" reads "Apple" (as a browser shows it)
_INLINE = frozenset(("a", "abbr", "acronym", "b", "bdi", "bdo", "big", "blink", "button", "cite", "code", "data",
                     "del", "dfn", "em", "font", "i", "ins", "kbd", "label", "mark", "nobr", "output", "picture",
                     "q", "rb", "ruby", "s", "samp", "small", "span", "strike", "strong", "sub", "sup", "textarea",
                     "time", "tt", "u", "var", "slot"))
_SMALLER = frozenset(("small", "sub", "sup"))
# elements whose own lines never wrap or whose box clips or scrolls its text: hidden (their tail can be off-screen)
_NOWRAP_ELEMENTS = frozenset(("pre", "listing", "xmp", "plaintext", "nobr", "textarea", "marquee"))
_FLOATING_ALIGN = frozenset(("table", "img", "object", "applet", "embed", "iframe", "input", "image"))
_SIZE_ATTRS = frozenset(("width", "height", "cellpadding", "cellspacing", "border", "hspace", "vspace"))
_ABRUPT_COMMENT = re.compile(r"<!--(?:>|->)")
_CSS_COMMENT = re.compile(r"/\*.*?(?:\*/|$)", re.DOTALL)  # an unterminated comment runs to the end
_CSS_ESCAPE = re.compile(r"\\(?:([0-9a-fA-F]{1,6})[ \t\r\n\f]?|(\n)|(.))", re.DOTALL)
_CSS_NAME = re.compile(r"-{0,2}[a-z_][a-z0-9_-]*")
_CSS_IMPORTANT = re.compile(r"!\s*important\s*$", re.IGNORECASE)
_CSS_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?"
_CSS_LEN = re.compile(rf"^({_CSS_NUM})([a-z%]*)$")
VIEWPORT_PX = (320.0, 568.0)  # a small screen: viewport units and percentages resolve small (fail-closed)
_UNIT_PX = {"": 1.0, "px": 1.0, "pt": 4 / 3, "pc": 16.0, "in": 96.0, "cm": 37.8, "mm": 3.78, "q": 0.945,
            "rem": 16.0, "rlh": 19.2}
_FONT_UNITS = {"em": 1.0, "ex": 0.5, "ch": 0.5, "cap": 0.7, "ic": 1.0, "lh": 1.2}
_VIEW_UNITS = {"vw": 0, "svw": 0, "lvw": 0, "dvw": 0, "vi": 0, "vmin": 0, "vh": 1, "svh": 1, "lvh": 1, "dvh": 1,
               "vb": 1, "vmax": 1}
_FONT_KEYWORDS = {"xx-small": 9.0, "x-small": 10.0, "small": 13.0, "medium": 16.0, "large": 18.0, "x-large": 24.0,
                  "xx-large": 32.0, "xxx-large": 48.0}
_FONT_ATTR_SIZES = (10.0, 13.0, 16.0, 18.0, 24.0, 32.0, 48.0)  # <font size=1..7>
_GLOBAL = frozenset(("inherit", "initial", "unset", "revert", "revert-layer"))
TINY_FONT_PX = 4.0  # an effective font size below this is unreadable (includes font-size:0)
OPACITY_MIN = 0.1  # a cumulative opacity at or below this is invisible
MIN_CONTRAST = 1.5  # WCAG contrast ratio below which text is invisible against its effective background
_NAMED_COLORS = dict(x.split(":") for x in """
aliceblue:f0f8ff antiquewhite:faebd7 aqua:00ffff aquamarine:7fffd4 azure:f0ffff beige:f5f5dc bisque:ffe4c4
black:000000 blanchedalmond:ffebcd blue:0000ff blueviolet:8a2be2 brown:a52a2a burlywood:deb887 cadetblue:5f9ea0
chartreuse:7fff00 chocolate:d2691e coral:ff7f50 cornflowerblue:6495ed cornsilk:fff8dc crimson:dc143c cyan:00ffff
darkblue:00008b darkcyan:008b8b darkgoldenrod:b8860b darkgray:a9a9a9 darkgreen:006400 darkgrey:a9a9a9
darkkhaki:bdb76b darkmagenta:8b008b darkolivegreen:556b2f darkorange:ff8c00 darkorchid:9932cc darkred:8b0000
darksalmon:e9967a darkseagreen:8fbc8f darkslateblue:483d8b darkslategray:2f4f4f darkslategrey:2f4f4f
darkturquoise:00ced1 darkviolet:9400d3 deeppink:ff1493 deepskyblue:00bfff dimgray:696969 dimgrey:696969
dodgerblue:1e90ff firebrick:b22222 floralwhite:fffaf0 forestgreen:228b22 fuchsia:ff00ff gainsboro:dcdcdc
ghostwhite:f8f8ff gold:ffd700 goldenrod:daa520 gray:808080 green:008000 greenyellow:adff2f grey:808080
honeydew:f0fff0 hotpink:ff69b4 indianred:cd5c5c indigo:4b0082 ivory:fffff0 khaki:f0e68c lavender:e6e6fa
lavenderblush:fff0f5 lawngreen:7cfc00 lemonchiffon:fffacd lightblue:add8e6 lightcoral:f08080 lightcyan:e0ffff
lightgoldenrodyellow:fafad2 lightgray:d3d3d3 lightgreen:90ee90 lightgrey:d3d3d3 lightpink:ffb6c1
lightsalmon:ffa07a lightseagreen:20b2aa lightskyblue:87cefa lightslategray:778899 lightslategrey:778899
lightsteelblue:b0c4de lightyellow:ffffe0 lime:00ff00 limegreen:32cd32 linen:faf0e6 magenta:ff00ff maroon:800000
mediumaquamarine:66cdaa mediumblue:0000cd mediumorchid:ba55d3 mediumpurple:9370db mediumseagreen:3cb371
mediumslateblue:7b68ee mediumspringgreen:00fa9a mediumturquoise:48d1cc mediumvioletred:c71585
midnightblue:191970 mintcream:f5fffa mistyrose:ffe4e1 moccasin:ffe4b5 navajowhite:ffdead navy:000080
oldlace:fdf5e6 olive:808000 olivedrab:6b8e23 orange:ffa500 orangered:ff4500 orchid:da70d6 palegoldenrod:eee8aa
palegreen:98fb98 paleturquoise:afeeee palevioletred:db7093 papayawhip:ffefd5 peachpuff:ffdab9 peru:cd853f
pink:ffc0cb plum:dda0dd powderblue:b0e0e6 purple:800080 rebeccapurple:663399 red:ff0000 rosybrown:bc8f8f
royalblue:4169e1 saddlebrown:8b4513 salmon:fa8072 sandybrown:f4a460 seagreen:2e8b57 seashell:fff5ee
sienna:a0522d silver:c0c0c0 skyblue:87ceeb slateblue:6a5acd slategray:708090 slategrey:708090 snow:fffafa
springgreen:00ff7f steelblue:4682b4 tan:d2b48c teal:008080 thistle:d8bfd8 tomato:ff6347 turquoise:40e0d0
violet:ee82ee wheat:f5deb3 white:ffffff whitesmoke:f5f5f5 yellow:ffff00 yellowgreen:9acd32
""".split())  # noqa: SIM905  (CSS Color 4 named colours, hex)
# CSS properties that affect neither whether the element's text is painted nor where any box sits (keywords, colours
# of decorations, and alignment that CSS start-aligns on overflow)
_CSS_HARMLESS = frozenset("""
font-weight font-style font-variant font-variant-caps font-variant-numeric font-variant-ligatures
font-variant-east-asian font-stretch font-kerning font-optical-sizing font-synthesis text-decoration-line
text-decoration-style text-decoration-color text-decoration-skip-ink text-underline-position text-emphasis
text-emphasis-style text-emphasis-color text-align text-align-last text-justify text-transform text-rendering
text-overflow word-break line-break overflow-wrap word-wrap hyphens cursor pointer-events user-select caret-color
accent-color list-style list-style-type list-style-image list-style-position border-style border-color border-top-style
border-right-style border-bottom-style border-left-style border-top-color border-right-color border-bottom-color
border-left-color border-radius border-top-left-radius border-top-right-radius border-bottom-left-radius
border-bottom-right-radius border-collapse box-sizing clear caption-side empty-cells z-index transform-origin
transform-style perspective-origin backface-visibility isolation outline-color outline-offset outline-width
flex-grow flex-shrink column-count column-rule column-span column-fill text-orientation quotes break-before
break-after break-inside page-break-before page-break-after page-break-inside orphans widows will-change
object-fit object-position resize scroll-behavior overscroll-behavior touch-action appearance transition
transition-property transition-duration transition-timing-function transition-delay animation animation-name
animation-duration animation-timing-function animation-delay animation-iteration-count animation-direction
animation-fill-mode animation-play-state background-position background-size background-repeat
background-attachment background-origin background-clip background-blend-mode
""".split())  # noqa: SIM905 (a word list reads better as text)
# box sizes: a size smaller than the content lets it overflow over the next boxes (and a stacking context paints it
# on top), so D0 accepts a size only on an element that clips its own overflow; there it hides the element
_CSS_SIZE = frozenset("""
width height min-width min-height max-width max-height inline-size block-size min-inline-size min-block-size
max-inline-size max-block-size flex-basis column-width columns
""".split())  # noqa: SIM905
_SIZE_FREE = frozenset(("auto", "none", "normal", "min-content", "max-content", "fit-content"))
# spacing that only adds room: accepted up to LAYOUT_MAX_PX (more pushes text off-screen: drop)
_CSS_SPACING = frozenset("""
padding padding-top padding-right padding-bottom padding-left padding-inline padding-block padding-inline-start
padding-inline-end padding-block-start padding-block-end border border-top border-right border-bottom border-left
border-width border-top-width border-right-width border-bottom-width border-left-width border-inline border-block
border-spacing gap row-gap column-gap
""".split())  # noqa: SIM905
# decorations that paint across the glyphs or below the line, and shadows that paint beyond the box: small values only
_CSS_DECORATION = frozenset(("text-decoration", "text-decoration-thickness", "text-underline-offset", "text-shadow"))
# alignment that overflows toward the start (negative x or y, out of reach) when the content is too wide or tall
_CSS_ALIGN = frozenset(("justify-content", "justify-items", "justify-self", "align-items", "align-content",
                        "align-self", "place-items", "place-content", "place-self"))
_ALIGN_OK = frozenset(("normal", "start", "flex-start", "self-start", "left", "stretch", "baseline", "first",
                       "space-between", "auto", "legacy", "safe"))
LAYOUT_MAX_PX = 400.0
DECORATION_MAX_PX = 20.0
BIG_FONT_PX = 200.0  # a larger effective font pushes its own line far past the viewport
_SYMBOL_FONTS = re.compile(r"wingdings|webdings|dingbat|symbol|marlett|blank|redact|flow ?block")
_COLOR_FUNC = re.compile(r"(?:rgba?|hsla?)\([^()]*\)|#[0-9a-f]{3,8}\b")
_LEN_TOKEN = re.compile(rf"(?<![\w#.-])({_CSS_NUM})([a-z%]*)(?![\w.])")
_DISPLAY_OK = frozenset(("inline", "block", "inline-block", "flex", "inline-flex", "grid", "inline-grid", "list-item",
                         "contents", "flow-root", "table", "inline-table", "table-row", "table-cell", "table-row-group",
                         "table-header-group", "table-footer-group", "table-caption", "ruby", "ruby-base",
                         "ruby-text", *_GLOBAL))
_DISPLAY_INLINE = frozenset(("inline", "inline-block", "inline-flex", "inline-grid", "inline-table", "contents",
                             "ruby", "ruby-base", "ruby-text"))
_NOWRAP = frozenset(("nowrap", "pre"))  # the element's own lines never wrap: their tail runs off-screen
_NONE_ONLY = {"clip-path": "none", "mask": "none", "mask-image": "none", "mix-blend-mode": "normal",
              "content": "normal", "background-image": "none", "clip": "auto", "line-clamp": "none",
              "text-security": "none"}  # any other value hides (part of) the element's own paint
# any other value can move a box, paint over other text or reorder what is shown: drop
_MOVES = {"transform": "none", "translate": "none", "rotate": "none", "scale": "none", "perspective": "none",
          "offset": "none", "offset-path": "none", "box-shadow": "none", "outline": "none", "outline-style": "none",
          "border-image": "none", "border-image-source": "none", "order": "0", "unicode-bidi": "normal",
          "direction": "ltr", "box-reflect": "none", "filter": "none", "backdrop-filter": "none",
          "writing-mode": "horizontal-tb"}
_FLOAT_OK = frozenset(("none", "left", "inline-start"))  # a right float wider than its container starts off-screen
_OFFSETS = frozenset(("top", "left", "right", "bottom", "inset", "inset-inline", "inset-block", "inset-inline-start",
                      "inset-inline-end", "inset-block-start", "inset-block-end"))


class _El:
    """One open element of the D0 stack and its computed style (only what decides whether its text is shown)."""

    __slots__ = ("bg", "fg", "fill", "font", "hidden", "inline", "opacity", "relative", "tag")

    def __init__(self, tag: str, hidden: bool = False, inline: bool = False, fg=(0.0, 0.0, 0.0, 1.0),
                 bg=(255.0, 255.0, 255.0), font: float = 16.0, opacity: float = 1.0, fill=None):
        self.tag, self.hidden, self.inline = tag, hidden, inline
        self.fg, self.bg, self.font, self.opacity, self.relative = fg, bg, font, opacity, False
        self.fill = fill  # -webkit-text-fill-color (inherited on its own; paints the glyphs whatever `color` says)

    def child(self, tag: str) -> _El:
        return _El(tag, self.hidden, tag in _INLINE, self.fg, self.bg, self.font, self.opacity, self.fill)


_ROOT = _El("#root")  # the page: black text on white, 16px


def _css_unescape(s: str) -> str:
    def rep(m: re.Match) -> str:
        if m.group(1):
            cp = int(m.group(1), 16)
            return chr(cp) if 0 < cp <= 0x10FFFF and not 0xD800 <= cp <= 0xDFFF else "\ufffd"
        return "" if m.group(2) else m.group(3)

    return _CSS_ESCAPE.sub(rep, s)


def _css_split(s: str, sep: str) -> list[str] | None:
    """Split at top-level `sep` (outside strings, parentheses and escapes); None if a string or bracket is left
    open (the browser's recovery is not modelled: the caller drops the field)."""
    parts, cur, depth, quote, i = [], [], 0, None, 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            cur.append(s[i:i + 2])
            i += 2
            continue
        if quote:
            quote = None if c == quote else quote
        elif c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth < 0:
                return None
        elif c == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    if quote or depth:
        return None
    parts.append("".join(cur))
    return parts


def css_declarations(style: str) -> dict[str, str] | None:
    """The declarations of an inline style that win the cascade: property -> value (lower case, `!important`
    removed). An `!important` declaration beats a normal one; otherwise the later one wins. CSS comments and escapes
    ("\\6e one") are resolved as a browser does; a declaration with an invalid property name is ignored, as in CSS.
    None if the text cannot be split safely (an open string or bracket)."""
    parts = _css_split(_CSS_COMMENT.sub(" ", style or ""), ";")
    if parts is None:
        return None
    won: dict[str, tuple[bool, str]] = {}
    for part in parts:
        kv = _css_split(part, ":")
        if kv is None:
            return None
        if len(kv) < 2:
            continue
        name = _css_unescape(kv[0]).strip().lower()
        value = _css_unescape(":".join(kv[1:]))
        if not _CSS_NAME.fullmatch(name):
            continue
        m = _CSS_IMPORTANT.search(value)
        important = m is not None
        value = re.sub(r"\s+", " ", value[: m.start()] if m else value).strip().lower()
        if name not in won or important or not won[name][0]:
            won[name] = (important, value)
    return {k: v for k, (_, v) in won.items()}


def _css_px(v: str, font: float = 16.0, pct_of: float | None = VIEWPORT_PX[0]) -> float | None:
    """A CSS length in px (unitless numbers too, as in quirks mode); em-type units on `font`, percentages on
    `pct_of`, viewport units on a small screen. None for keywords, math functions and garbage."""
    m = _CSS_LEN.match(v.strip())
    if not m:
        return None
    x, unit = float(m.group(1)), m.group(2)
    if not math.isfinite(x):
        return None
    if unit in _UNIT_PX:
        return x * _UNIT_PX[unit]
    if unit in _FONT_UNITS:
        return x * _FONT_UNITS[unit] * font
    if unit in _VIEW_UNITS:
        return x * VIEWPORT_PX[_VIEW_UNITS[unit]] / 100.0
    if unit == "%" and pct_of is not None:
        return x * pct_of / 100.0
    return None


def _lengths_within(v: str, limit: float, font: float, unitless_font: bool = False) -> bool | None:
    """Whether every length in a (shorthand) value is within +-`limit` px; None if the value holds a function other
    than a colour (calc(), var(), url(), ...), which D0 does not evaluate."""
    rest = _COLOR_FUNC.sub(" ", v)
    if "(" in rest:
        return None
    for m in _LEN_TOKEN.finditer(rest):
        x, unit = float(m.group(1)), m.group(2)
        if unitless_font and unit == "":
            px = x * font
        else:
            px = _css_px(m.group(0), font)
            if px is None:
                if unit in ("s", "ms", "deg", "fr", "x", "dppx", "turn", "rad", "grad") or unit == "":
                    continue  # times, angles, grid fractions and plain numbers (flex factors) are not lengths
                return None
        if not math.isfinite(px) or abs(px) > limit:
            return False
    return True


def _css_number(v: str) -> float | None:
    m = re.fullmatch(rf"({_CSS_NUM})(%?)", v.strip())
    if not m:
        return None
    x = float(m.group(1))
    return x / 100.0 if m.group(2) else x


def _hue(v: str) -> float | None:
    m = re.fullmatch(rf"({_CSS_NUM})(deg|rad|grad|turn)?", v)
    if not m:
        return None
    x = float(m.group(1))
    return {"rad": x * 180 / math.pi, "grad": x * 0.9, "turn": x * 360}.get(m.group(2) or "deg", x)


def css_color(v: str) -> tuple[float, float, float, float] | str | None:
    """(r, g, b, alpha) of a CSS colour (0-255, alpha 0-1); "inherit" for inherit/currentcolor; None if unparseable."""
    v = v.strip().lower()
    if v in ("inherit", "currentcolor", "unset", "revert", "revert-layer"):
        return "inherit"
    if v == "initial":
        return (0.0, 0.0, 0.0, 1.0)
    if v == "transparent":
        return (0.0, 0.0, 0.0, 0.0)
    if v in _NAMED_COLORS:
        v = "#" + _NAMED_COLORS[v]
    if v.startswith("#"):
        h = v[1:]
        if not re.fullmatch(r"[0-9a-f]+", h) or len(h) not in (3, 4, 6, 8):
            return None
        if len(h) in (3, 4):
            h = "".join(c * 2 for c in h)
        rgba = [int(h[i:i + 2], 16) for i in range(0, len(h), 2)]
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), rgba[3] / 255.0 if len(rgba) == 4 else 1.0)
    m = re.fullmatch(r"(rgba?|hsla?)\((.*)\)", v)
    if not m:
        return None
    body = m.group(2).replace("/", " / ")
    args = [a for a in re.split(r"[\s,]+", body.strip()) if a]
    alpha = 1.0
    if "/" in args:
        k = args.index("/")
        if k != len(args) - 2:
            return None
        a = _css_number(args[-1])
        if a is None:
            return None
        alpha, args = a, args[:k]
    elif len(args) == 4:
        a = _css_number(args[3])
        if a is None:
            return None
        alpha, args = a, args[:3]
    if len(args) != 3:
        return None
    if m.group(1).startswith("rgb"):
        ch = []
        for a in args:
            x = _css_number(a)
            if x is None:
                return None
            ch.append(x * 255.0 if a.endswith("%") else x)
        r, g, b = (min(255.0, max(0.0, x)) for x in ch)
    else:
        hue, s, lum = _hue(args[0]), _css_number(args[1]), _css_number(args[2])
        if hue is None or s is None or lum is None:
            return None
        r, g, b = (255.0 * x for x in colorsys.hls_to_rgb((hue % 360) / 360.0, min(1, max(0, lum)),
                                                           min(1, max(0, s))))
    return (r, g, b, min(1.0, max(0.0, alpha)))


def _luminance(rgb) -> float:
    def lin(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(x) for x in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _over(fg, alpha: float, bg) -> tuple[float, float, float]:
    return tuple(b + (f - b) * alpha for f, b in zip(fg[:3], bg[:3]))


def contrast(fg, bg) -> float:
    """WCAG contrast ratio of two opaque colours (1 = identical, 21 = black on white)."""
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _font_px(v: str, parent: float) -> float | None:
    v = v.strip()
    if v in _FONT_KEYWORDS:
        return _FONT_KEYWORDS[v]
    if v in ("inherit", "unset", "revert", "revert-layer", "math"):
        return parent
    if v == "initial":
        return 16.0
    if v == "smaller":
        return parent / 1.2
    if v == "larger":
        return parent * 1.2
    return _css_px(v, parent, pct_of=parent)


def _style_verdict(decls: dict[str, str] | None, el: _El) -> int:
    """How an inline style affects the element: `_OK`; `_HIDE` its subtree (effects confined to the element's own
    paint: display, visibility, opacity, colour, tiny or symbol fonts, clipping, clip/mask, no-wrap lines); or `_DROP`
    the field (anything that can move a box, make it overlap or displace other text, or that D0 cannot evaluate).
    Updates `el`'s computed colour, fill, background, font size, opacity and display."""
    if decls is None:
        return _DROP
    verdict = _OK
    pos = decls.get("position", "static")
    if pos == "relative":
        el.relative = True
    elif pos not in ("static", *_GLOBAL):
        return _DROP  # absolute, fixed, sticky: the box can sit over other text
    clips = any(decls.get(k) in ("hidden", "clip", "scroll", "auto", "overlay")
                for k in ("overflow", "overflow-x", "overflow-y", "overflow-block", "overflow-inline"))

    def hide() -> None:
        nonlocal verdict
        verdict = max(verdict, _HIDE)

    for prop, v in decls.items():
        base = re.sub(r"^-(?:webkit|moz|ms|o)-", "", prop)
        if prop.startswith("--") or prop == "position" or base in _CSS_HARMLESS:
            continue
        if v in _GLOBAL:
            # a CSS-wide keyword repeats the parent's value (already judged there) or resets to the initial one;
            # the exceptions compound (opacity, zoom) or take effect only here (offsets of a relative box)
            if (v == "inherit" and base in ("opacity", "zoom")) or (el.relative and base in _OFFSETS
                                                                     and v == "inherit"):
                return _DROP
            continue
        if base in _CSS_SIZE or base == "flex":
            if base == "flex":  # flex: <grow> <shrink>? <basis>?; only the basis is a size
                basis = [x for x in v.split() if not re.fullmatch(r"[+]?(?:\d+\.?\d*|\.\d+)", x)]
                free = all(x in _SIZE_FREE or x in ("0px", "0%") for x in basis)
            else:
                free = all(x in _SIZE_FREE for x in v.split())
            if free:
                continue
            ok = _lengths_within(v, LAYOUT_MAX_PX, el.font)
            if ok is None or not ok or not clips:
                return _DROP  # an unclipped size lets content overflow over the next boxes
            hide()  # a clipping box of bounded size: the overflow check below hides it
        elif base in _CSS_SPACING:
            ok = _lengths_within(v, LAYOUT_MAX_PX, el.font)
            if not ok:
                return _DROP
        elif base in _CSS_DECORATION:
            ok = _lengths_within(v, DECORATION_MAX_PX, el.font)
            if not ok:
                return _DROP
        elif base in _CSS_ALIGN:
            if any(w not in _ALIGN_OK for w in v.split()):
                return _DROP
        elif base == "font-family":
            if _SYMBOL_FONTS.search(v):
                hide()  # letters drawn as symbols
        elif base in ("white-space", "text-wrap", "text-wrap-mode", "white-space-collapse"):
            if v in _NOWRAP or v.startswith("nowrap") or " nowrap" in v:
                hide()
        elif base == "display":
            if v == "none":
                hide()
            elif v in _DISPLAY_OK:
                el.inline = v in _DISPLAY_INLINE
            else:
                return _DROP
        elif base == "visibility":
            if v in ("hidden", "collapse"):
                hide()
            elif v != "visible":
                return _DROP
        elif base == "content-visibility":
            if v == "hidden":
                hide()
            elif v not in ("visible", "auto"):
                return _DROP
        elif base == "opacity":
            x = _css_number(v)
            if x is None:
                return _DROP
            el.opacity *= min(1.0, max(0.0, x))
        elif base == "font-size":
            px = _font_px(v, el.font)
            if px is None or px > BIG_FONT_PX:
                return _DROP
            el.font = px
        elif base == "font":
            hide()  # the shorthand's size is not parsed: fail closed on the element
        elif base == "zoom":
            x = 1.0 if v in ("normal", "reset") else _css_number(v)
            if x is None or x > 4.0:
                return _DROP
            if x <= 0:
                hide()
            else:
                el.font *= min(1.0, x)
        elif base in ("color", "text-fill-color"):
            c = css_color(v)
            if c is None:
                return _DROP
            if base == "text-fill-color":
                el.fill = None if v == "currentcolor" else el.fill if c == "inherit" else c
            elif c != "inherit":
                el.fg = c
        elif base in ("background-color", "background"):
            c = "inherit" if v in ("none", "transparent") else css_color(v)
            if c is None:
                hide()  # an image or gradient behind the text: its contrast is unknown
            elif c != "inherit":
                el.bg = _over(c, c[3], el.bg)
        elif base in ("overflow", "overflow-x", "overflow-y", "overflow-block", "overflow-inline"):
            if v in ("hidden", "clip", "scroll", "auto", "overlay"):
                hide()  # a clipping box
            elif v != "visible":
                return _DROP
        elif base == "line-height":
            if v == "normal":
                continue
            x = _css_number(v) if re.fullmatch(rf"{_CSS_NUM}", v) else None
            px = x * el.font if x is not None else _css_px(v, el.font, pct_of=el.font)
            if px is None or px < 0.8 * el.font or px > LAYOUT_MAX_PX:
                return _DROP  # lines shorter than the font overlap the next boxes; very tall ones push text away
        elif base in ("text-indent", "letter-spacing", "word-spacing", "vertical-align"):
            if v == "normal" or (base == "vertical-align" and v in (
                    "baseline", "sub", "super", "text-top", "text-bottom", "middle", "top", "bottom")):
                continue
            px = _css_px(v, el.font)
            if px is None:
                return _DROP
            if (base == "text-indent" and px < 0) or (base in ("letter-spacing", "word-spacing") and px < -1.0):
                hide()  # the first line pulled off to the left, or glyphs piled on each other
            elif px > (50.0 if base in ("letter-spacing", "word-spacing") else LAYOUT_MAX_PX) or (
                    base == "vertical-align" and px < -LAYOUT_MAX_PX):
                return _DROP  # moves the line (and what shares it) far away
        elif base.startswith("margin"):
            for tok in v.split():
                px = 0.0 if tok == "auto" else _css_px(tok, el.font)
                if px is None or px < 0 or px > LAYOUT_MAX_PX:
                    return _DROP  # a negative margin pulls the box over its neighbours; a huge one pushes them
        elif base in _OFFSETS:
            if el.relative and v != "auto" and _css_px(v, el.font) != 0.0:
                return _DROP
        elif base in _NONE_ONLY or base.startswith("mask"):
            if v != _NONE_ONLY.get(base, "none"):
                hide()
        elif base == "float":
            if v not in _FLOAT_OK:
                return _DROP
        elif base in _MOVES:
            if v != _MOVES[base] and not (base == "outline" and v in ("0", "0px")) and not (
                    base == "list-style-position" and v == "outside"):
                return _DROP
        elif base in ("flex-direction", "flex-flow", "flex-wrap"):
            if "reverse" in v:
                return _DROP  # visual order differs from the text order
        else:
            return _DROP  # a property D0 does not model (grid placement, unknown or vendor-only ones)
    return verdict


def _hides_after_style(el: _El) -> bool:
    """The computed-style checks of an element: font too small, cumulative opacity too low, or text too close in
    colour to its effective background."""
    if el.font < TINY_FONT_PX or el.opacity <= OPACITY_MIN:
        return True
    fg = el.fill if el.fill is not None else el.fg
    shown = _over(fg, fg[3] * el.opacity, el.bg)
    return contrast(shown, el.bg) < MIN_CONTRAST


def css_hides(style: str) -> bool:
    """Whether an inline style (on an element of the page root) hides its element's text, or makes D0 drop the
    field (SAN-1): the computed-style model of `_style_verdict` (display:none; visibility:hidden|collapse;
    content-visibility:hidden; font-size under 4px; opacity at most 0.1; text colour within contrast 1.5 of its
    background (transparent, alpha 0, the background colour); clipping overflow; negative or huge text-indent;
    clip, clip-path, mask, filter; positioning, negative margins and transforms, which drop the field)."""
    el = _ROOT.child("span")
    return _style_verdict(css_declarations(style), el) != _OK or _hides_after_style(el)


# scripting off: <noscript> content is parsed as markup (so a <body hidden> or an open formatting element inside it
# counts, as it does for a reader without scripts), while the <noscript> element itself stays hidden (as it is for a
# reader with scripts): the union of both readers' hidden text
_PARSER_KW = {"scripting": False} if "scripting" in inspect.signature(HTMLParser.__init__).parameters else {}


class _VisibleText(HTMLParser):
    """Tolerant markup reader for D0 (SAN-1). It keeps text only where it can show that a browser renders it:
    - an element hides its subtree by the `hidden`, `aria-hidden="true"` or `popover` attribute, as a closed
      <details> or <dialog>, as a never-rendered element (`_NEVER_RENDERED`), or by an inline style that hides it in
      the computed-style model (`_style_verdict`); `drop` is set when rendering depends on what D0 does not model;
    - the open-element stack follows HTML5 only where the browser surely closes an element: an end tag closes its
      element if nothing HTML5 treats as "special" lies between (the tree builder ignores it otherwise), and inside
      a hidden subtree only when the element is the current node (so reconstructed formatting elements, scope
      rules and the adoption agency can never make D0 show text the browser keeps hidden); an unclosed hidden
      element hides the text up to the end; `<x/>` on a non-void element opens it, as in HTML5;
    - comments, processing instructions, declarations and hidden or never-rendered elements render nothing (not
      even a space), as a browser shows them; block elements separate words, inline ones do not."""

    def __init__(self):
        super().__init__(convert_charrefs=True, **_PARSER_KW)
        self.out: list[str] = []
        self.stack: list[_El] = []
        self.hidden_chars = 0
        self.drop = False

    @property
    def top(self) -> _El:
        return self.stack[-1] if self.stack else _ROOT

    def _count(self, data: str) -> None:
        self.hidden_chars += sum(not c.isspace() for c in data)

    def _open(self, tag: str, attrs) -> _El:
        el = self.top.child(tag)
        a = {}
        verdict = _HIDE if tag in _NEVER_RENDERED or tag in _NOWRAP_ELEMENTS else _OK
        if tag == "font":
            for k, v in attrs:
                if (k or "").lower() == "size" and v:
                    m = re.fullmatch(r"\s*([+-]?)(\d+)\s*", v)
                    if m:
                        n = int(m.group(2))
                        n = 3 + n if m.group(1) == "+" else 3 - n if m.group(1) == "-" else n
                        el.font = _FONT_ATTR_SIZES[min(7, max(1, n)) - 1]
        elif tag in _SMALLER:
            el.font /= 1.2
        for k, v in attrs:
            k, v = (k or "").lower(), v or ""
            a.setdefault(k, v)
            if k in ("hidden", "popover") or (k == "aria-hidden" and v.strip().lower() == "true"):
                verdict = max(verdict, _HIDE)
            elif k == "dir" and v.strip().lower() not in ("", "ltr"):
                verdict = _DROP  # right-to-left (or auto) ordering shows the text in another order than D0 reads
            elif k == "align" and v.strip().lower() == "right" and tag in _FLOATING_ALIGN:
                verdict = _DROP  # a right float
            elif k == "nowrap":
                verdict = max(verdict, _HIDE)
            elif k in _SIZE_ATTRS and v.strip():
                px = _css_px(v.strip().lower(), el.font)
                if px is not None and px > LAYOUT_MAX_PX:
                    verdict = _DROP  # a huge table or cell pushes text off-screen
            elif k == "style":
                verdict = max(verdict, _style_verdict(css_declarations(v), el))
            elif k in ("color", "bgcolor", "text") and tag in ("font", "table", "tr", "td", "th", "body", "marquee"):
                c = css_color(v)
                if c is None or c == "inherit":
                    verdict = max(verdict, _HIDE)
                elif k == "bgcolor":
                    el.bg = _over(c, c[3], el.bg)
                else:
                    el.fg = c
        if tag in ("details", "dialog") and "open" not in a:
            verdict = max(verdict, _HIDE)
        if (tag == "link" and "stylesheet" in a.get("rel", "").lower()) or tag == "frameset":
            verdict = _DROP  # a stylesheet; a frameset document renders no body text at all
        if verdict < _HIDE and _hides_after_style(el):
            verdict = _HIDE
        if verdict == _DROP:
            self.drop = True
        el.hidden = el.hidden or verdict >= _HIDE
        return el

    def _sep(self) -> None:
        self.out.append(" ")

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in _NOT_PUSHED:
            if any((k or "").lower() not in ("lang", "class", "id", "xmlns", "dir") or
                   ((k or "").lower() == "dir" and (v or "").strip().lower() != "ltr") for k, v in attrs):
                self.drop = True  # HTML5 moves these attributes onto the one <html>/<body>: they style everything
            return
        el = self._open(tag, attrs)
        if tag in ("br", "hr"):
            if not el.hidden:
                self._sep()
            return
        if tag in _VOID:
            return  # renders no text and no break (an image or a <wbr> inside a word keeps the word whole)
        if not el.hidden and not el.inline:
            self._sep()
        self.stack.append(el)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)

    def set_cdata_mode(self, elem, **kw):
        # inside <svg>/<math> (foreign content) HTML5 parses <script>, <style>, <title>, <xmp>, ... content as
        # markup, and inside <select> it ignores those start tags: no raw text there
        if not any(e.tag in ("svg", "math", "select") for e in self.stack):
            super().set_cdata_mode(elem, **kw)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "br":  # </br> is <br> in HTML5
            if not self.top.hidden:
                self._sep()
            return
        if tag in _NOT_PUSHED or tag in _NEVER_POPPED:
            return
        k = next((i for i in range(len(self.stack) - 1, -1, -1) if self.stack[i].tag == tag), None)
        if k is None:
            return  # an end tag without an open element is ignored
        if k != len(self.stack) - 1 and (self.top.hidden or any(e.tag in _SPECIAL for e in self.stack[k + 1:])):
            return  # the browser may keep (or reconstruct) the element: keep it open
        el = self.stack[k]
        del self.stack[k:]
        if not el.hidden and not el.inline:
            self._sep()

    def handle_data(self, data):
        top = self.top
        if top.tag == "style" and data.strip():
            self.drop = True  # a stylesheet can hide any element: D0 does not model selectors
        if top.hidden:
            self._count(data)
        else:
            self.out.append(data)

    def handle_comment(self, data):
        self._count(data)

    def unknown_decl(self, data):  # <![CDATA[...]]>: not rendered in HTML
        self._count(data)

    def handle_decl(self, decl):
        self._count(decl)

    def handle_pi(self, data):  # <?...>: a bogus comment in HTML
        self._count(data)


def strip_markup(text: str) -> tuple[str, int]:
    """Visible text of possibly-HTML `text` and the number of hidden (non-space) characters removed. Character
    references are resolved to a fixed point (at most 2 passes) first, so entity-encoded markup is markup. If the
    rendering depends on what D0 does not model (`_VisibleText.drop`), the whole field is removed and counted."""
    t = text
    for _ in range(2):
        u = html.unescape(t)
        if u == t:
            break
        t = u
    if "<" not in t:
        return t, 0
    p = _VisibleText()
    p.feed(_ABRUPT_COMMENT.sub("<!---->", t))  # HTML5 reads "<!-->" and "<!--->" as empty comments
    p.close()
    # a "<" the parser could not read as markup ("margin <5%", "a < b") is text, as a browser shows it
    out = "".join(p.out)
    if p.drop:
        return "", p.hidden_chars + sum(not c.isspace() for c in out)
    return out, p.hidden_chars


def _fold_confusables(t: str) -> tuple[str, int]:
    """Fold look-alike letters (Cyrillic, Greek, Armenian, Cherokee, Latin small capitals, dotless i) to Latin inside
    words made only of ASCII letters and look-alikes (mixed-script obfuscation, red-team family A4); genuine words
    of those scripts are left alone. Returns the text and the number of folded characters."""
    n = 0

    def fold(m: re.Match) -> str:
        nonlocal n
        w = m.group(0)
        if any(ch in _CONFUSABLE_CHARS for ch in w) and all(ch.isascii() or ch in _CONFUSABLE_CHARS for ch in w):
            n += sum(ch in _CONFUSABLE_CHARS for ch in w)
            return w.translate(_CONFUSABLE)
        return w

    return _LETTER_RUN.sub(fold, t), n


def _script(ch: str) -> str | None:
    """Unicode script of a letter, approximated by the first word of its name (LATIN, CYRILLIC, GREEK, CJK, ...);
    None for modifier letters (Lm), which belong to no word script."""
    if unicodedata.category(ch) == "Lm":
        return None
    name = unicodedata.name(ch, "")
    return name.split(" ", 1)[0] if name else None


def mixed_script_tokens(text: str) -> int:
    """Number of letter tokens that mix Latin with another script (SAN-2 residual after the confusable fold)."""
    n = 0
    for m in _LETTER_RUN.finditer(text):
        w = m.group(0)
        if w.isascii():
            continue
        scripts = {s for s in map(_script, w) if s}
        if "LATIN" in scripts and len(scripts) > 1:
            n += 1
    return n


def _invisible(ch: str) -> bool:
    return ord(ch) in _IGNORABLE or unicodedata.category(ch) in _INVISIBLE


def _d0_pass(text: str, max_chars: int) -> tuple[str, D0Flags]:
    t, hidden = strip_markup(text)
    t = unicodedata.normalize("NFKC", t)
    n0 = len(t)
    t = "".join(ch for ch in t if not _invisible(ch))
    invisible = n0 - len(t)
    t = _CONTROL.sub(" ", t)
    t, folded = _fold_confusables(t)
    t = _URL.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    truncated = len(t) > max_chars
    if truncated:
        cut = t[:max_chars]
        t = cut.rsplit(" ", 1)[0] if " " in cut[max_chars // 2:] else cut
    return t, D0Flags(hidden, invisible, folded, 0, truncated)


D0_MAX_PASSES = 8


def d0(text: str | None, max_chars: int) -> Clean:
    """D0 (design_v1 §3.3; design_v2_delta SAN-1..SAN-4) on one text field, with its event flags.

    Order: character references resolved (at most 2 passes), then the fail-closed markup reader (`_VisibleText`)
    keeps only text it can show a browser renders and strips every tag; then NFKC (full-width and compatibility
    forms); the invisible-character drop (every Cf/Co/Cs/Cn and Default_Ignorable code point: zero-width and bidi
    controls, isolates, soft hyphen, the tag block U+E0000-E0FFF and the variation selectors used for "ASCII
    smuggling", Hangul fillers, private use); control characters; the confusable fold (look-alikes inside Latin
    words); URLs; whitespace; the cap. Decoding and NFKC can create new markup ("&amp;lt;", full-width "\uff1c"), so
    the pass repeats until the text is stable; text still changing after `D0_MAX_PASSES` passes (markup nested that
    deep is never news) is removed entirely. Hence `d0(d0(x)) == d0(x)` for every x. The flags record what was
    removed, plus the tokens that still mix Latin with another script."""
    t = "" if text is None or (isinstance(text, float) and math.isnan(text)) else text
    if not isinstance(t, str):
        t = str(t)
    flags = D0Flags()
    for _ in range(D0_MAX_PASSES):
        nxt, f = _d0_pass(t, max_chars)
        flags = flags + f
        if nxt == t:
            break
        t = nxt
    else:
        flags = flags + D0Flags(hidden_chars_removed=sum(not c.isspace() for c in t))
        t = ""
    return Clean(t, D0Flags(flags.hidden_chars_removed, flags.invisible_removed, flags.confusables_folded,
                            mixed_script_tokens(t), flags.truncated))


def sanitize(text: str | None, max_chars: int) -> str:
    """D0 text only (`d0` without the flags)."""
    return str(d0(text, max_chars))


def d0_blind(text: str | None, max_chars: int, blinder, *, symbol: str | None = None, decision_year: int | None = None,
             cutoff=None, tagged=()) -> tuple[str, D0Flags]:
    """The one path from Web text to state text for every reader (SAN-2): D0 first, then blinding (entities of every
    universe firm, levels, times). Blinding a zero-width-split or homoglyph company name only works on D0 output."""
    clean = text if isinstance(text, Clean) else d0(text, max_chars)
    text = blind_clean(clean, blinder, symbol=symbol, decision_year=decision_year, cutoff=cutoff, tagged=tagged)
    return text, clean.flags


def blind_clean(clean: Clean, blinder, *, symbol: str | None = None, decision_year: int | None = None, cutoff=None,
                tagged=()) -> str:
    """Blind a D0 output. Refuses raw text: D0 must run before blinding (SAN-2)."""
    if not isinstance(clean, Clean):
        raise RenderError("blinding needs D0 output (render.d0); raw Web text must be sanitized first (SAN-2)")
    if blinder is None:
        raise RenderError("no blinder in the week context: Web text cannot enter a state unblinded")
    return blinder(str(clean), symbol=symbol, decision_year=decision_year, cutoff=cutoff, tagged=tagged)


def datamark(text: str) -> str:
    """Spotlighting by datamarking (conventions.datamark; RB-5): every whitespace run of a D0-sanitized, blinded news
    string becomes U+02C6. Applied to headline and summary strings only, never to keys or code-rendered fields."""
    return re.sub(r"\s+", DATAMARK, text.strip())


def d0_event_table(items: pd.DataFrame) -> pd.DataFrame:
    """D0 event flags per item of a news table (`id`, `created_at`, `headline`, `summary`), for the SAN-4/FEED-1
    counts by year. Pure code over the archive; no model call. One row per distinct item id."""
    it = items.drop_duplicates("id")
    rows = []
    for iid, created, h, s in zip(it["id"], it["created_at"], it["headline"], it["summary"]):
        f = d0(h, HEADLINE_MAX).flags + d0(s, SUMMARY_MAX).flags
        rows.append({"id": iid, "created_at": created, **f.as_dict()})
    cols = ["id", "created_at", *D0Flags.__dataclass_fields__]
    out = pd.DataFrame(rows, columns=cols)
    out["year"] = pd.to_datetime(out["created_at"], utc=True).dt.year
    return out


# ------------------------------------------------------------------------------------ labels
def prob_bin(p: float | None, bank: Bank) -> str:
    """Probability as a label (bank §A probability bins); an abstention renders as 'not available'."""
    if p is None or not math.isfinite(p):
        return NOT_AVAILABLE
    for lo, label in bank.conventions["probability_bins"]:
        if p >= lo:
            return label
    return bank.conventions["probability_bins"][-1][1]


def size_bucket(rank: int, bank: Bank) -> str:
    """Rank by raw market value within U_t (1 = largest): 1-20 -> 0, 21-60 -> 1, else 2 (bank size_rule)."""
    if rank < 1:
        raise RenderError(f"size rank must be >= 1, got {rank}")
    b = bank.conventions["size_buckets"]
    return b[0] if rank <= 20 else b[1] if rank <= 60 else b[2]


def focal_company(bank: Bank, code_name: str, industry: str, size_rank: int) -> dict[str, str]:
    return {"code_name": code_name, "industry": bank.ff12[industry], "size": size_bucket(size_rank, bank)}


def option_name(label: str) -> str:
    """An option label as words ("not_stated" -> "not stated"), for the bank's '<top option name>' fields."""
    return label.replace("_", " ")


def level_name(level_text: str) -> str:
    """The name of a Score level: its text before the colon ("Clearly harmful: ..." -> "clearly harmful")."""
    return level_text.split(":", 1)[0].strip().lower()


def effect_label(a: Answer, levels: tuple[str, ...], bank: Bank) -> str:
    """'<top level name> (<probability bin of that level>)' (N3 rows, bank §B)."""
    if a.probs is None:
        return NOT_AVAILABLE
    top = a.top
    return f"{level_name(levels[int(top)])} ({prob_bin(a.probs[top], bank)})"


# ------------------------------------------------------------------------------------ times
def relative_time(ts, cutoff_utc) -> str:
    """Time of `ts` relative to the review (bank §A). A time at or after the cutoff is a look-ahead bug."""
    ts, cutoff = pd.Timestamp(ts), pd.Timestamp(cutoff_utc)
    if ts.tzinfo is None or cutoff.tzinfo is None:
        raise RenderError("relative_time needs timezone-aware timestamps")
    if ts >= cutoff:
        raise RenderError(f"{ts} is not before the review cutoff {cutoff}")
    days = (cutoff - ts) / pd.Timedelta(days=1)
    if days < 1:
        return "within the past day"
    if days < 14:
        n = int(days)
        return f"{n} day{'s' if n != 1 else ''} before the review"
    return f"{int(days // 7)} weeks before the review"


# ------------------------------------------------------------------------------------ numbers
def fmt_pct(x: float | None, digits: int = 1) -> str:
    return NOT_AVAILABLE if x is None or not np.isfinite(x) else f"{x:+.{digits}%}"


def hist_bucket(p: float | None) -> str:
    """A value's percentile within its own trailing history, in words."""
    if p is None or not np.isfinite(p):
        return "unknown"
    if p >= 0.9:
        return "top 10%"
    if p >= 0.7:
        return "upper 30%"
    if p > 0.3:
        return "middle"
    if p > 0.1:
        return "lower 30%"
    return "bottom 10%"


def trailing_percentile(history: pd.Series, value: float) -> float | None:
    """Share of `history` at or below `value` (the caller passes only data up to the decision)."""
    h = pd.Series(history).dropna()
    if h.empty or value is None or not np.isfinite(value):
        return None
    return float((h <= value).mean())


def reaction(move: float, daily_vol: float, days: float) -> str:
    """N3 stock_reaction: '<signed %> (<small|moderate|large> move <up|down>)', scaled by vol x sqrt(days)."""
    if not (np.isfinite(move) and np.isfinite(daily_vol) and daily_vol > 0 and days > 0):
        return NOT_AVAILABLE
    z = abs(move) / (daily_vol * math.sqrt(days))
    size = "small" if z < 1 else "moderate" if z <= 2 else "large"
    return f"{fmt_pct(move)} ({size} move {'up' if move >= 0 else 'down'})"


# ------------------------------------------------------------------------------------ state checks
def _strings(x: Any):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _strings(v)


def assert_relative_time(state: Any) -> None:
    """No absolute date or calendar year may appear in a decision state (§5.0). The test is
    `blinding.absolute_time`, the same definition `Blinder.blind_time` rewrites, so blinded text always passes
    and a count such as "2000 jobs" is not mistaken for a year."""
    for s in _strings(state):
        hit = absolute_time(s)
        if hit is not None:
            raise RenderError(f"absolute time {hit!r} in state text {s[:80]!r}")


def assert_no_levels(state: Any) -> None:
    """Trusted numeric states carry ratios and buckets only: no dollar amounts or price levels (§5.0)."""
    for s in _strings(state):
        if _DOLLAR.search(s):
            raise RenderError(f"dollar amount in numeric state text {s[:80]!r}")


# words a typed-only state may contain besides bank text: bins, buckets, relative times, reaction and trend words
_TYPED_WORDS = frozenset("""
very likely uncertain unlikely not available stated unknown small moderate large move up down within past day days
weeks week before the review top upper lower middle bottom compared with years year light heavy harsher mixed easier
rising stable falling yes no none of in and or percentile average code name industry size when
""".split())  # noqa: SIM905 (a word list reads better as text)
_TYPED_SOURCES = {"N3": ("N1.event_type", "N1.effect", "N1.vs_expectations", "N1.problem_kind"),
                  "P2": ("P1.theme", "P1.stage", "P1.profit_conditions", "P1.uncertainty")}
_CODE_NAME = re.compile(r"Company-[0-9A-F]{8}")
_WORD = re.compile(r"[^\W\d_]+")  # every letter run, any script (R3 must see non-Latin text too)
_VOCAB: dict[tuple[str, str], frozenset[str]] = {}


# the conventions typed rows are rendered from; the notes, the evidence fence, the datamark and content-free rules are
# prose (words such as "instructions", "follow", "never") and must not widen the R3 vocabulary (the note is skipped)
TYPED_CONVENTIONS = ("probability_bins", "relative_time", "reaction_buckets", "size_buckets", "masks")


def typed_vocabulary(bank: Bank, program: str) -> frozenset[str]:
    """Words a typed-only state of `program` may use: the bank texts its rows are rendered from (option labels
    and texts, level names, schema, the label conventions `TYPED_CONVENTIONS`, industries) plus the fixed label
    words above."""
    key = (bank.sha256, program)
    if key not in _VOCAB:
        conv = {k: bank.conventions[k] for k in TYPED_CONVENTIONS if k in bank.conventions}
        texts: list[str] = [json.dumps(bank.state_schemas.get(program, {})), json.dumps(conv),
                            json.dumps(bank.ff12)]
        for qid in _TYPED_SOURCES.get(program, ()):
            q = bank.q(qid)
            texts += [json.dumps(q.options or {}), json.dumps(q.levels or [])]
        _VOCAB[key] = frozenset(w.casefold() for t in texts for w in _WORD.findall(t)) | _TYPED_WORDS
    return _VOCAB[key]


def assert_typed_only(state: Any, bank: Bank, program: str) -> None:
    """R3 (§3.2): a synthesis state holds only code-rendered typed labels, never raw item text. Every word of
    every key and value (the fixed note aside) must come from `typed_vocabulary`; raw news text or an injected
    instruction almost surely contains a word outside it."""
    vocab = typed_vocabulary(bank, program)
    body = {k: v for k, v in state.items() if k != "note"} if isinstance(state, dict) else state
    for s in _strings(body):
        for w in _WORD.findall(_CODE_NAME.sub(" ", s)):
            if w.casefold() not in vocab:
                raise RenderError(f"R3: {program} is a typed-only state, but {w!r} in {s[:80]!r} is not a typed label")


def envelope(bank: Bank, program: str, *, content_free: bool = False, typed_only: bool | None = None,
             **fields: Any) -> dict[str, Any]:
    """A state with the bank's note first and the schema's fields in schema order; all fields required.

    `content_free`: a CF probe state (QB-2), whose content-free values bypass the R3 vocabulary check
    (`state_schemas.CF.code`). `typed_only=False` lifts R3 for a registered no-defense variant only (R-nodef's N3
    with raw headlines, design_v1 §13.1); the default applies R3 to N3 and P2."""
    want = [k for k in bank.state_fields(program) if k != "note"]
    extra, missing = set(fields) - set(want), set(want) - set(fields)
    if extra or missing:
        raise RenderError(f"{program} state: unexpected {sorted(extra)}, missing {sorted(missing)}")
    state = {"note": bank.note(program), **{k: fields[k] for k in want}}
    assert_relative_time(state)
    if program in NUMERIC_PROGRAMS:
        assert_no_levels(state)
    r3 = (program in TYPED_ONLY_PROGRAMS) if typed_only is None else typed_only
    if r3 and not content_free:
        assert_typed_only(state, bank, program)
    return state
