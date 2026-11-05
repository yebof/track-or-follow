#!/usr/bin/env python3
"""Per-model knowledge-cutoff table (paper App. C; design_v2_delta LA-1) -> paper/tables/tableA_models.tex.

Every cell comes from a local file, never from memory (CLAUDE.md rules 1-3):
  * model pins: configs/backends.yaml (backends.llm-logprob / typed-local / jev-cf .model);
  * the emu LLM's documented knowledge cutoff: the archived vendor model page docs/pricing/openai_model_gpt-4.1-mini.html
    ("<date> knowledge cutoff"); its snapshot date: the date suffix of the pinned model id;
  * NG: no cutoff in docs/ng/ (README and the HF API record); first public date = the HF record's createdAt;
  * Jev: no cutoff in docs/jev/; release date = docs/jev/wikipedia_jev.txt ("Stable release" infobox date); the row
    reads "Jev (<status> panel)" once the descriptive Jev-panel output exists (_validity.jev_rows), else "not run";
  * windows: configs/system_v1.yaml windows; weeks after release on the evaluation window: results/analysis/h2.json
    sensitivities.W_release.n_weeks for emu, and 0 for a model first public after W_clean ends.
The script fails if any source is missing or a pattern is not found.

Usage: python experiments/report/table_models.py
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "tables" / "tableA_models.tex"


def _read(rel: str) -> str:
    p = ROOT / rel
    if not p.is_file():
        sys.exit(f"missing source: {rel}")
    return p.read_text(encoding="utf-8", errors="replace")


def _text_of_html(rel: str) -> str:
    t = re.sub(r"<[^>]+>", " ", _read(rel))
    return re.sub(r"\s+", " ", html.unescape(t))


def _tt(model: str) -> str:
    """Model id in typewriter with break points (as make_numbers.py prints \\emuModel); a revision hash is shortened."""
    if "@" in model:
        repo, rev = model.split("@", 1)
        model = f"{repo}@{rev[:8]}"
    s = model.replace("_", r"\_")
    # round-2 figure review A-4: break only before a trailing snapshot date and after "/" or "@", never inside a date
    s = re.sub(r"-(\d{4}-\d{2}-\d{2})$", r"-\\allowbreak{}\1", s)
    return r"\texttt{" + s.replace("/", r"/\allowbreak{}").replace("@", r"@\allowbreak{}") + "}"


def main() -> int:
    backends = yaml.safe_load(_read("configs/backends.yaml"))["backends"]
    system = yaml.safe_load(_read("configs/system_v1.yaml"))
    w_clean = [dt.date.fromisoformat(x) for x in system["windows"]["W_clean"]]
    w_release = dt.date.fromisoformat(system["windows"]["W_release"][0])

    # emu / VERB: pinned snapshot and documented cutoff
    emu_model = backends["llm-logprob"]["model"]
    m = re.search(r"(\d{4}-\d{2}-\d{2})$", emu_model)
    if not m:
        sys.exit(f"no snapshot date in {emu_model}")
    emu_snapshot = dt.date.fromisoformat(m.group(1))
    page = _text_of_html("docs/pricing/openai_model_gpt-4.1-mini.html")
    m = re.search(r"([A-Z][a-z]{2} \d{2}, \d{4}) knowledge cutoff", page)
    if not m:
        sys.exit("no knowledge cutoff on the archived gpt-4.1-mini page")
    emu_cutoff = dt.datetime.strptime(m.group(1), "%b %d, %Y").date()

    # NG: HF record
    ng_model = backends["typed-local"]["model"]
    hf = json.loads(_read("docs/ng/this-that-model-1.0_hf_api_2026-09-26.json"))
    ng_public = dt.date.fromisoformat(hf["createdAt"][:10])
    ng_readme = _read("docs/ng/this-that-model-1.0_README_2026-09-26.md")
    ng_cutoff_doc = re.search(r"(?i)knowledge cut-?off|training cut-?off", ng_readme) is not None

    # Jev: archived encyclopedia infobox
    jev_model = backends["jev-cf"]["model"]
    wiki = _read("docs/jev/wikipedia_jev.txt")
    m = re.search(r"Release\s.{0,80}?\(\s*(\d{4}-\d{2}-\d{2})\s*\)", wiki, re.S)
    if not m:
        sys.exit("no release date in docs/jev/wikipedia_jev.txt")
    jev_release = dt.date.fromisoformat(m.group(1))
    jev_cutoff_doc = any(re.search(r"(?i)knowledge cut-?off", (ROOT / "docs/jev" / f).read_text(errors="replace"))
                         for f in ("docs_introduction.txt", "docs_api.txt", "llms-full.txt"))

    h2 = json.loads(_read("results/analysis/h2.json"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _validity as V
    import _xtab as XT

    has_jev = V.jev_rows() is not None or any(k.lower() == "jev" and isinstance(v, dict) and v.get("status") == "ok"
                                              for k, v in (h2.get("per_implementation") or {}).items())
    emu_release_weeks = int(h2["sensitivities"]["W_release"]["n_weeks"])

    def weeks_after(public: dt.date, n_if_inside: int | None) -> str:
        return "none" if public > w_clean[1] else str(n_if_inside)

    iso = lambda d: d.isoformat()
    rows = [
        (r"emu, VERB, holistic reader and generative-analyst baseline", _tt(emu_model),
         iso(emu_cutoff), iso(emu_snapshot),
         weeks_after(emu_snapshot, emu_release_weeks)),
        (r"NG, also the full-frame price-report coder", _tt(ng_model),
         "documented" if ng_cutoff_doc else "undocumented", iso(ng_public),
         weeks_after(ng_public, None)),
        (r"Jev (\jevPanelStatus{} panel)" if has_jev else r"Jev (not run)", _tt(jev_model),
         "documented" if jev_cutoff_doc else "undocumented", iso(jev_release),
         weeks_after(jev_release, None)),
    ]
    lines = [
        "% GENERATED by experiments/report/table_models.py -- do not edit by hand (CLAUDE.md rule 2).",
        r"{\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        XT.begin(r"\linewidth", r">{\raggedright\arraybackslash}p{0.30\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}llr"),
        r"\toprule",
        r"Reader & Model pin & Cutoff & Release & \shortstack[r]{Weeks\\after} \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [
        r"\bottomrule",
        XT.END,
        r"\par\smallskip",
        r"\begin{minipage}{\linewidth}\scriptsize\raggedright",
        ("Cutoff: documented knowledge cutoff. Release: snapshot date (emu), model-hub creation date (NG), release "
         "date (Jev). Weeks after: decisions of the evaluation window after the release."),
        r"\end{minipage}",
        "}",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
