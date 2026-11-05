"""8-K item codes: archive the SEC's official item list and check that the EDGAR submissions archive carries item
codes for every 8-K (design_v2_delta BG-3 filing anchor, QB-13; design_v1 section 13.1 C2 labels).

1. Archive (idempotent): download the official Form 8-K (https://www.sec.gov/files/form8-k.pdf) to docs/sec/, extract
   its text with pdftotext, parse every "Item X.YY <title>" heading into docs/sec/form8k_items.json (source URL,
   retrieval date, SHA-256, the form's OMB expiry line).
2. Check: over every 8-K and 8-K/A in data/raw/sec/submissions (main, predecessor and paginated files), by year:
   filings, share with a non-empty `items` field, codes not on the official list, and filings listing an anchor item
   (`data.earnings.ANCHOR_ITEMS`). Written to docs/sec/8k_item_check.json.

Run: .venv/bin/python scripts/check_8k_items.py [--refresh]
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.data.earnings import ANCHOR_ITEMS, C2_ITEMS, ITEMS_JSON  # noqa: E402

UA = {"User-Agent": "jev-quant-research research@example.com"}
FORM_URL = "https://www.sec.gov/files/form8-k.pdf"
DOCS = ROOT / "docs/sec"
SUBS = ROOT / "data/raw/sec/submissions"
STUDY_FROM = "2019-06-01"  # first news/universe date of the study (bars from 2018)
_HEAD = re.compile(r"^Item (\d\.\d\d) (\S.*)$")


def parse_items(text: str) -> list[dict]:
    """Item headings of the form text: 'Item X.YY Title' at the start of a line, title continued on the next lines
    until it ends with a period. The first occurrence of each code is its heading."""
    lines = text.splitlines()
    out: dict[str, str] = {}
    for i, ln in enumerate(lines):
        m = _HEAD.match(ln.strip())
        if not m or m.group(1) in out or not m.group(2)[:1].isupper():
            continue
        title, j = m.group(2).strip(), i + 1
        while not title.endswith(".") and j < len(lines) and lines[j].strip() and j < i + 4:
            title += " " + lines[j].strip()
            j += 1
        out[m.group(1)] = title
    return [{"code": k, "title": v} for k, v in sorted(out.items())]


def archive(refresh: bool) -> dict:
    DOCS.mkdir(parents=True, exist_ok=True)
    pdf, txt = DOCS / "form8-k.pdf", DOCS / "form8-k.txt"
    if refresh or not pdf.exists():
        r = requests.get(FORM_URL, headers=UA, timeout=60)
        r.raise_for_status()
        pdf.write_bytes(r.content)
    if refresh or not txt.exists():
        subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)
    text = txt.read_text()
    expires = re.search(r"Expires:\s*([A-Za-z]+ \d+, \d{4})", text)
    old = json.loads(ITEMS_JSON.read_text()) if ITEMS_JSON.exists() else {}
    meta = {
        "source_url": FORM_URL,
        "retrieved": old.get("retrieved") if not refresh and old.get("pdf_sha256") == _sha(pdf) else str(date.today()),
        "pdf_sha256": _sha(pdf),
        "omb_expires": expires.group(1) if expires else None,
        "note": "Item headings parsed from the official Form 8-K (General Instructions B and the item text) by "
                "scripts/check_8k_items.py; the EDGAR submissions API lists these codes in its `items` field.",
        "items": parse_items(text),
    }
    ITEMS_JSON.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
    return meta


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(official: set[str]) -> dict:
    by_year = defaultdict(Counter)
    unknown, unknown_by_year, seen = Counter(), defaultdict(Counter), set()
    symbols = set()
    for f in sorted(SUBS.glob("*.json")):
        j = json.loads(f.read_text())
        block = j["filings"]["recent"] if "filings" in j else j
        sym = f.name.split("__")[0].removesuffix(".json")
        for form, acc, accn, it in zip(block.get("form", []), block.get("acceptanceDateTime", []),
                                       block.get("accessionNumber", []), block.get("items", [])):
            if form not in ("8-K", "8-K/A") or (sym, accn) in seen:
                continue
            seen.add((sym, accn))
            symbols.add(sym)
            y = str(acc)[:4]
            codes = [c.strip() for c in (it or "").split(",") if c.strip()]
            c = by_year[y]
            c["filings"] += 1
            c[f"filings_{form}"] += 1
            c["with_items"] += bool(codes)
            c["anchor"] += any(x in ANCHOR_ITEMS for x in codes)
            for x in codes:
                if x in ANCHOR_ITEMS:
                    c[f"anchor_{x}"] += 1
                if x not in official:
                    unknown[x] += 1
                    unknown_by_year[y][x] += 1
    years = {y: dict(sorted(c.items())) for y, c in sorted(by_year.items())}
    tot = Counter()
    for c in by_year.values():
        tot.update(c)
    study = Counter()
    for y, c in by_year.items():
        if y >= STUDY_FROM[:4]:
            study.update(c)
    return {
        "submissions_dir": str(SUBS.relative_to(ROOT)), "symbols": len(symbols),
        "study_years_from": STUDY_FROM[:4],
        "study_total": dict(sorted(study.items())),
        "study_share_with_items": study["with_items"] / study["filings"] if study["filings"] else None,
        "study_codes_not_on_official_list": sum(sum(c.values()) for y, c in unknown_by_year.items()
                                                if y >= STUDY_FROM[:4]),
        "years_with_codes_not_on_official_list": sorted(unknown_by_year),
        "total": dict(sorted(tot.items())),
        "share_with_items": tot["with_items"] / tot["filings"] if tot["filings"] else None,
        "codes_not_on_official_list": dict(unknown.most_common()),
        "by_year": years,
        "anchor_items": list(ANCHOR_ITEMS), "c2_items": list(C2_ITEMS),
        "all_anchor_and_c2_items_official": all(x in official for x in (*ANCHOR_ITEMS, *C2_ITEMS)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download the form")
    args = ap.parse_args()
    meta = archive(args.refresh)
    official = {x["code"] for x in meta["items"]}
    res = check(official)
    res["official_item_list"] = str(ITEMS_JSON.relative_to(ROOT))
    res["checked_on"] = str(date.today())
    (DOCS / "8k_item_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"study years >= {res['study_years_from']}: share with codes {res['study_share_with_items']:.4f}, "
          f"codes off the list {res['study_codes_not_on_official_list']}; years with off-list codes "
          f"{res['years_with_codes_not_on_official_list']}")
    print(f"{len(official)} official item codes; {res['total'].get('filings', 0)} 8-K/8-K/A filings of "
          f"{res['symbols']} symbols; share with item codes {res['share_with_items']:.4f}; "
          f"unknown codes {res['codes_not_on_official_list']}; anchor filings {res['total'].get('anchor', 0)}")
    for y, c in res["by_year"].items():
        print(y, c.get("filings", 0), f"{c.get('with_items', 0) / max(c.get('filings', 1), 1):.4f}", c.get("anchor", 0))


if __name__ == "__main__":
    main()
