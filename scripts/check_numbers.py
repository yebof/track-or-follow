#!/usr/bin/env python3
"""No hand-typed numbers in the paper text (CLAUDE.md rule 2; docs/paper/outline_v1.md §5).

Every number in paper/sections/*.tex must be a macro from paper/generated/numbers.tex (written by
experiments/report/make_numbers.py). This script fails on any digit outside the allow-list:

  * references and locators: \\ref, \\eqref, \\label, \\cite[p.~4]{...}, \\input, \\includegraphics, \\url, ...;
  * layout arguments: \\vspace{..}, \\hspace{..}, \\\\[2pt], \\multicolumn{3}{..}, \\cmidrule(lr){2-3}, figure
    placement, tabular column specs;
  * years (1900-2099) and venue years such as WWW '27;
  * names that contain digits: labels (K1-K4, H1-H3, P0-P5, PD0-PD3, A1-A8, U100, S50, BG-3, MR-1, ...),
    hyphenated names (Random-20, LLaMA-3), SEC forms and items (8-K, 10-K, Item 2.02), S&P 500, AR[0,+1],
    Newey-West(4), and the pinned model names in MODEL_NAMES;
  * inside math: single-digit integers (indices, 1/2, AR windows); any multi-digit or decimal number fails;
  * a line that ends with the pragma "% numbers-ok: <reason>" (every use is listed in the report).

It also reports which number macros used in the text are still placeholders; with --final any used placeholder
fails (an undefined macro already fails the LaTeX build). Finally it regenerates paper/generated/numbers.tex in memory
(experiments/report/make_numbers.py) and warns if the file on disk is stale, i.e. if the paper would print numbers
that no longer match results/ or configs/ (an error with --final).

Usage: python scripts/check_numbers.py [--final] [paths ...]   (default paths: paper/sections/*.tex)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "paper" / "sections"
NUMBERS_TEX = ROOT / "paper" / "generated" / "numbers.tex"
PLACEHOLDER_MACRO = "NumPlaceholder"

PRAGMA_RE = re.compile(r"%\s*numbers-ok\b:?\s*(.*)$")
COMMENT_RE = re.compile(r"(?<!\\)%.*")

# command -> number of mandatory arguments whose content is not prose (all optional [..] args are skipped too)
NONPROSE_ARGS = {
    "cite": 1, "citep": 1, "citet": 1, "citealp": 1, "citealt": 1, "citeauthor": 1, "citeyear": 1,
    "citeyearpar": 1, "citenum": 1, "citeN": 1, "nocite": 1, "shortcite": 1, "Cite": 1, "Citep": 1, "Citet": 1,
    "ref": 1, "eqref": 1, "autoref": 1, "cref": 1, "Cref": 1, "pageref": 1, "nameref": 1, "label": 1,
    "input": 1, "include": 1, "includegraphics": 1, "url": 1, "href": 1, "hyperref": 0,
    "vspace": 1, "hspace": 1, "vskip": 0, "hskip": 0, "setlength": 2, "addtolength": 2, "rule": 2,
    "resizebox": 2, "scalebox": 1, "multicolumn": 2, "multirow": 2, "cline": 1, "cmidrule": 1,
    "begin": 1, "end": 1, "textcolor": 1, "color": 1, "colorbox": 1, "definecolor": 3, "fontsize": 2,
    "setcounter": 2, "addtocounter": 2, "enlargethispage": 1, "captionsetup": 1, "linespread": 1,
    "newcommand": 2, "renewcommand": 2, "providecommand": 2, "arraystretch": 0, "Description": 0,
}
TABULAR_ENVS = {"tabular", "tabular*", "tabularx", "tabulary", "longtable", "array", "supertabular"}
MATH_ENVS = r"equation|equation\*|align|align\*|gather|gather\*|multline|multline\*|eqnarray|eqnarray\*|" \
            r"displaymath|math|flalign|flalign\*|alignat|alignat\*"

COUNT_NOUNS = (r"(?:randomi[sz]ed\s+)?(?:cases|items|units|names|weeks|days|queries|questions|requests|attacks|pairs|"
               r"trials|decisions|stock-weeks|tokens|runs|samples|headlines|filings|documents)\b")
MODEL_NAMES = [
    r"gpt-4\.1-mini(?:-2025-04-14)?", r"GPT-4(?:\.1)?(?:o|-mini)?", r"jev-\d+\.\d+(?:\.\d+)?",
    r"this-that-model-\d+\.\d+", r"LLaMA-\d+(?:\.\d+)?", r"Llama-\d+(?:\.\d+)?(?:-\d+B)?", r"Qwen\d+(?:\.\d+)?(?:-\d+B)?",
]
ALLOWED_TEXT = [
    r"(?<![\d.,])(?:19|20)\d{2}(?![\d.,]*\d)(?!\s*(?:\\%|%|bps|pp\b|" + COUNT_NOUNS + r"))",  # years, not counts
    r"\b[A-Z][A-Za-z]*(?:~|\s)?'\d{2}\b",                                    # WWW '27, KDD '24
    r"\b\d{1,2}-[KQ](?:/A)?\b",                                               # 8-K, 10-K, 10-Q, 8-K/A
    r"\bItems?(?:~|\s)+\d\.\d{2}(?:\s*(?:,|;|/|and|--|–)\s*\d\.\d{2})*",     # SEC 8-K items
    r"\bS\\?&P(?:~|\s)?500\b",
    r"\bAR\s*\[[^\]\n]{1,12}\]",                                               # announcement windows AR[0,+1]
    r"\bNewey(?:--|–|-)West\s*\(\d\)|\bNW\s*\(\d\)",
    r"\b[A-Z][A-Za-z]{0,3}-?\d{1,3}[a-z]?(?:\s*(?:--|–|-|\.\.)\s*(?:[A-Z][A-Za-z]{0,3}-?)?\d{1,3}[a-z]?)?\b",  # labels
    r"τ\d(?:\s*(?:--|–)\s*τ?\d)?",
    r"\b[A-Z][A-Za-z]+-\d+(?:\.\d+)?[A-Za-z]*\b",                             # Random-20, LLaMA-3
] + MODEL_NAMES
ALLOWED_TEXT_RE = re.compile("|".join(f"(?:{p})" for p in ALLOWED_TEXT))
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _blank(chars: list[str], start: int, end: int) -> None:
    for i in range(start, min(end, len(chars))):
        if chars[i] != "\n":
            chars[i] = " "


def _skip_group(text: str, i: int, open_ch: str, close_ch: str) -> int:
    """text[i] == open_ch; return index after the matching close_ch (nesting-aware)."""
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(text)


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\n":
        i += 1
    return i


def blank_nonprose(text: str) -> str:
    """Blank (with spaces, keeping newlines) the arguments of reference/layout commands."""
    chars = list(text)
    for m in re.finditer(r"\\([A-Za-z]+)\*?", text):
        name = m.group(1)
        if name not in NONPROSE_ARGS:
            continue
        start, i = m.start(), m.end()
        n_mand = NONPROSE_ARGS[name]
        env = None
        if name == "cmidrule":
            j = _skip_ws(text, i)
            if j < len(text) and text[j] == "(":
                i = _skip_group(text, j, "(", ")")
        taken = 0
        while True:
            j = _skip_ws(text, i)
            if j < len(text) and text[j] == "[":
                i = _skip_group(text, j, "[", "]")
                continue
            if j < len(text) and text[j] == "{" and taken < n_mand:
                k = _skip_group(text, j, "{", "}")
                if name == "begin" and taken == 0:
                    env = text[j + 1 : k - 1].strip()
                i, taken = k, taken + 1
                continue
            break
        if name == "begin" and env in TABULAR_ENVS:
            # tabular*/tabularx take a width first; every env takes a column spec
            extra = 2 if env in {"tabular*", "tabularx", "tabulary"} else 1
            for _ in range(extra):
                j = _skip_ws(text, i)
                if j < len(text) and text[j] == "{":
                    i = _skip_group(text, j, "{", "}")
        if name in ("vskip", "hskip"):
            mm = re.compile(r"\s*-?[\d.]+\s*[a-z]{2}(?:\s*(?:plus|minus)\s*-?[\d.]+\s*[a-z]{2})*").match(text, i)
            if mm:
                i = mm.end()
        _blank(chars, start, i)
    out = "".join(chars)
    out = re.sub(r"\\\\\*?\s*\[[^\]]*\]", lambda m: " " * len(m.group(0)), out)          # \\[2pt]
    out = re.sub(r"\\renewcommand\{\\arraystretch\}\{[^}]*\}", lambda m: " " * len(m.group(0)), out)
    return out


def math_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    pattern = re.compile(
        r"\$\$.*?\$\$|(?<!\\)\$.*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]|"
        r"\\begin\{(" + MATH_ENVS + r")\}.*?\\end\{\1\}|\\ensuremath\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
        re.DOTALL,
    )
    for m in pattern.finditer(text):
        spans.append((m.start(), m.end()))
    return spans


def line_col(text: str, pos: int) -> tuple[int, int]:
    line = text.count("\n", 0, pos) + 1
    col = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return line, col


def check_file(path: Path) -> tuple[list[str], list[str]]:
    """Return (violations, pragma uses)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = raw.split("\n")
    pragmas, kept = [], []
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    for n, line in enumerate(lines, 1):
        pm = PRAGMA_RE.search(line)
        if pm and not line.lstrip().startswith("%"):
            pragmas.append(f"{rel}:{n}: numbers-ok ({pm.group(1).strip() or 'no reason given'})")
            kept.append(" " * len(line))
        else:
            kept.append(COMMENT_RE.sub(lambda m: " " * len(m.group(0)), line))
    # Math spans are located before blank_nonprose, which blanks \begin{equation}/\end{equation} (the "begin"/"end"
    # arguments are non-prose) and would otherwise hide every display environment from math_spans. Both steps
    # preserve length, so the spans index the blanked text as well.
    spans = math_spans("\n".join(kept))
    text = blank_nonprose("\n".join(kept))
    chars = list(text)
    out = []

    for s, e in spans:
        body = text[s:e]
        for m in NUMBER_RE.finditer(body):
            tok = m.group(0)
            digits = re.sub(r"\D", "", tok)
            if len(digits) > 1 or "." in tok:
                ln, col = line_col(text, s + m.start())
                out.append((ln, col, f"{rel}:{ln}:{col}: number '{tok}' in math -- use a macro from numbers.tex"))
        _blank(chars, s, e)
    text = "".join(chars)

    chars = list(text)
    for m in ALLOWED_TEXT_RE.finditer(text):
        _blank(chars, m.start(), m.end())
    rest = "".join(chars)
    for m in NUMBER_RE.finditer(rest):
        ln, col = line_col(rest, m.start())
        context = raw.split("\n")[ln - 1].strip()
        out.append((ln, col, f"{rel}:{ln}:{col}: hand-typed number '{m.group(0)}' -- \"{context[:100]}\""))
    return [msg for _, _, msg in sorted(out)], pragmas


def generated_files() -> list[Path]:
    """paper/generated/*.tex in the order main.tex inputs them (numbers.tex first), so a later definition wins."""
    main = ROOT / "paper" / "main.tex"
    order = re.findall(r"^\\input\{generated/([A-Za-z0-9_]+)\}", main.read_text(encoding="utf-8"), re.MULTILINE) \
        if main.is_file() else []
    files = [ROOT / "paper" / "generated" / f"{n}.tex" for n in order]
    return [f for f in files if f.is_file()] or [NUMBERS_TEX]


def number_macros(path: Path | None = None) -> dict[str, bool]:
    """{macro name: is_placeholder} from the generated macro files, by each macro's LAST definition (\\newcommand or
    \\renewcommand) in main.tex's input order (editor review L8: numbers_extra.tex renews numbers.tex placeholders)."""
    out: dict[str, bool] = {}
    for f in ([path] if path is not None else generated_files()):
        if not f.is_file():
            continue
        for m in re.finditer(r"^\\(?:re)?newcommand\{\\([A-Za-z]+)\}(?:\[\d\])?\{(.*)\}\s*$", f.read_text(encoding="utf-8"),
                             re.MULTILINE):
            out[m.group(1)] = f"\\{PLACEHOLDER_MACRO}" in m.group(2)
    return out


def numbers_stale() -> str | None:
    """A message if paper/generated/numbers.tex differs from what make_numbers.py would write now; None if current or
    if the generator cannot be imported (e.g. PyYAML missing)."""
    try:
        sys.path.insert(0, str(ROOT / "experiments" / "report"))
        import make_numbers  # noqa: PLC0415
        text, _, _ = make_numbers.generate()
    except Exception:  # noqa: BLE001 -- the staleness check is advisory without the generator's dependencies
        return None
    current = NUMBERS_TEX.read_text(encoding="utf-8") if NUMBERS_TEX.is_file() else ""
    if current != text:
        return "paper/generated/numbers.tex is stale: rerun experiments/report/make_numbers.py (make -C paper prep)"
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", type=Path, help="files to check (default: paper/sections/*.tex)")
    ap.add_argument("--final", action="store_true", help="also fail on placeholder or undefined number macros")
    args = ap.parse_args(argv)
    files = [p.resolve() for p in args.paths] or sorted(SECTIONS.glob("*.tex"))

    violations, pragmas = [], []
    used: dict[str, list[str]] = {}
    macros = number_macros()
    for f in files:
        v, p = check_file(f)
        violations += v
        pragmas += p
        text = "\n".join(COMMENT_RE.sub("", ln) for ln in f.read_text(encoding="utf-8").split("\n"))
        for m in re.finditer(r"\\([A-Za-z]+)", text):
            if m.group(1) in macros:
                used.setdefault(m.group(1), []).append(f"{f.name}:{text.count(chr(10), 0, m.start()) + 1}")

    placeholders = sorted(k for k in used if macros.get(k))
    print(f"check_numbers: {len(files)} files; {len(macros)} macros in the generated files; {len(used)} used "
          f"({len(placeholders)} still placeholders)")
    for v in violations:
        print(f"  ERROR {v}")
    for p in pragmas:
        print(f"  NOTE  {p}")
    final_errors = []
    if args.final:
        final_errors = [f"placeholder macro \\{k} used at {', '.join(used[k][:3])}" for k in placeholders]
        for e in final_errors:
            print(f"  ERROR {e}")
    elif placeholders:
        print(f"  INFO  placeholders in use: {', '.join(placeholders)}")
    stale = numbers_stale()
    if stale:
        (final_errors.append(stale) if args.final else None)
        print(f"  {'ERROR' if args.final else 'WARN '} {stale}")
    failed = bool(violations or final_errors)
    print(f"check_numbers: {'FAIL' if failed else 'OK'} ({len(violations) + len(final_errors)} errors)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
