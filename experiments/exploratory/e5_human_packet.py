#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered): the E5 human audit packet (spec audit.human_codes, D-16) and its ingest.

    .venv/bin/python experiments/exploratory/e5_human_packet.py make --out <dir>     # packet for the human coder(s)
    .venv/bin/python experiments/exploratory/e5_human_packet.py ingest <filled.xlsx> # -> e5_audit_human_codes.csv

`make` writes the spec's 200-item audit draw (same eids as results/exploratory/e5_audit_codes.csv) as an .xlsx with the
item text only: no stratum, no regex flag, no model code, no outcome (spec audit.blind), in an order that does not
reveal the stratum (sha256("E5-HUMAN|<eid>")). The packet holds licensed news text, so it goes to a directory outside
the repository and is never committed. `ingest` reads the coder's 0/1 column and writes eid,code (no text).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5_construct_checks as E5  # noqa: E402

ROOT = E5.C.ROOT
HUMAN = ROOT / "results" / "exploratory" / "e5_audit_human_codes.csv"
SHEET = "items"

README = """E5 human audit: does the item report a stock price move?  (exploratory, not pre-registered)

You will code {n} news items. For each row in items.xlsx, put 1 or 0 in the column `code`:

  1 = the text states that a company's shares or stock rose, fell, jumped, dropped, or are trading higher
      or lower (during the session, premarket, or after hours), or gives the size of such a move.
  0 = the text does not state any move in a company's share price.

Rules
- Judge only the headline and summary shown. Do not look up the item, the date, the stock, or what happened
  to the price afterwards.
- A move of an index, a sector, a commodity or a bond yield alone is 0; a company's share price move is 1,
  even if it is mentioned in passing or is not the item's main topic.
- A price target, rating change, or forecast without a stated move is 0.
- If you cannot decide, pick your best guess and write a word in `note`. Do not leave `code` empty.
- Code alone; do not discuss items with another coder until both of you are done.

Return the filled items.xlsx (rename it items_<initials>.xlsx). The rows are in a random order; the file
contains licensed news text, so do not share it outside the author team.

(The question and the 1/0 definitions are the ones in the E5 spec, audit.coder, verbatim apart from layout.)
"""


def cmd_make(out: Path) -> int:
    spec, _, _ = E5.require_logged_spec()
    g, *_ = E5.build_classified(ROOT, E5.C.Guard(ROOT), spec)
    sample, _ = E5.audit_sample(g, spec)
    model = pd.read_csv(ROOT / "results" / "exploratory" / "e5_audit_codes.csv", comment="#")
    if sorted(model["eid"]) != sorted(sample["eid"]):
        raise SystemExit("audit draw differs from e5_audit_codes.csv")
    s = sample.assign(order=[hashlib.sha256(f"E5-HUMAN|{e}".encode()).hexdigest() for e in sample["eid"]])
    s = s.sort_values("order").reset_index(drop=True)
    items = pd.DataFrame({"n": range(1, len(s) + 1), "eid": s["eid"], "headline": s["headline"].fillna(""),
                          "summary": s["summary"].fillna(""), "code": "", "note": ""})
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "items.xlsx", engine="openpyxl") as xw:
        items.to_excel(xw, sheet_name=SHEET, index=False)
        ws = xw.sheets[SHEET]
        from openpyxl.styles import Alignment
        for col, w in zip("ABCDEF", (5, 22, 60, 90, 7, 20)):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
    (out / "README.txt").write_text(README.format(n=len(items)))
    print(f"wrote {out / 'items.xlsx'} ({len(items)} items) and README.txt")
    return 0


def cmd_ingest(path: Path) -> int:
    d = pd.read_excel(path, sheet_name=SHEET, dtype={"code": "object"})
    code = pd.to_numeric(d["code"], errors="coerce")
    bad = d.loc[~code.isin([0, 1]), "n"].tolist()
    if bad:
        raise SystemExit(f"rows without a 0/1 code: {bad}")
    model = pd.read_csv(ROOT / "results" / "exploratory" / "e5_audit_codes.csv", comment="#")
    if sorted(model["eid"]) != sorted(d["eid"]):
        raise SystemExit("eids differ from the audit draw")
    with open(HUMAN, "w") as fh:
        fh.write("# exploratory (not pre-registered); E5 human audit codes (spec audit.human_codes); "
                 "eids and codes only (no licensed text)\n")
        pd.DataFrame({"eid": d["eid"], "code": code.astype(int)}).to_csv(fh, index=False)
    print(f"wrote {HUMAN.relative_to(ROOT)}: {int(code.sum())} of {len(d)} coded 1")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--out", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("path")
    a = ap.parse_args(argv)
    return cmd_make(Path(a.out)) if a.cmd == "make" else cmd_ingest(Path(a.path))


if __name__ == "__main__":
    sys.exit(main())
