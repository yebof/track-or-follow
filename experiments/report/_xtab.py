"""Full-width tables that take alternating row colours (user, 2026-10-22).

A tabular* stretched with \\extracolsep{\\fill} leaves uncoloured gaps between cells under \\rowcolors, so the generators
emit a tabularx instead: the first top-level left-aligned (`l`) or paragraph (`p{...}`) column becomes a ragged-right
`X` column that absorbs the free width, and every other column keeps its type."""

from __future__ import annotations

X = ">{\\raggedright\\arraybackslash}X"


def _tokens(spec: str) -> list[str]:
    """Split a column spec into top-level tokens; a token keeps its brace groups (p{..}, >{..}, @{..})."""
    out, i = [], 0
    while i < len(spec):
        j = i + 1
        while j < len(spec) and spec[j] == "{":
            depth, k = 0, j
            while True:
                depth += {"{": 1, "}": -1}.get(spec[k], 0)
                k += 1
                if depth == 0:
                    break
            j = k
        out.append(spec[i:j])
        i = j
    return out


def xcols(spec: str) -> str:
    toks = _tokens(spec)
    for n, t in enumerate(toks):
        if t == "l" or t.startswith("p{"):
            prefix = n > 0 and toks[n - 1].startswith(">{")
            if prefix:
                toks[n - 1] = ""
            toks[n] = X
            return "".join(toks)
    raise ValueError(f"no l or p column to stretch in {spec!r}")


def begin(width: str, spec: str, pos: str = "") -> str:
    """\\begin{tabularx}{width}[pos]{spec} with one stretched column. No @{} at the edges: colortbl paints a row's
    edge cells \\tabcolsep beyond the cell, which with @{} would stick out past the rules."""
    spec = spec.removeprefix("@{}").removesuffix("@{}")
    return f"\\begin{{tabularx}}{{{width}}}{pos}{{{xcols(spec)}}}"


END = "\\end{tabularx}"
