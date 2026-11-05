"""Extract the Item 1 "Business" section of every 10-K (2017-06 .. 2026-09) filed by universe companies.

Output: data/raw/sec/business/<SYM>/<filed>_<accession>.json with the section text (point-in-time: a
profile may only be used for decisions after its filing date). Resumable; ~10 requests/s max to SEC.
"""

import json
import re
import sys
import time
from pathlib import Path

import warnings

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "jev-quant-research research@example.com"}
OUT = ROOT / "data/raw/sec/business"
# Section headings stand on their own line ("Item 1. Business"); cross-references inside other sections
# ("see Item 1. Business - Employees") do not, so both patterns are anchored to whole lines.
_SEP = r"[ \t]*[\.:\-\u2014\u2013]?[ \t]*(?:\n\s*){0,2}"
START_ITEM = re.compile(r"(?im)^[ \t]*item[ \t]*1" + _SEP + r"business[ \t]*\.?[ \t]*$")
END_ITEM = re.compile(r"(?im)^[ \t]*item[ \t]*(?:1a" + _SEP + r"risk[ \t]+factors|2" + _SEP + r"properties)[ \t]*\.?[ \t]*$")


def filings():
    for f in sorted((ROOT / "data/raw/sec/submissions").glob("*.json")):
        sym = f.stem.split("__")[0]
        j = json.loads(f.read_text())
        cik = j.get("cik")
        blk = j["filings"]["recent"] if "filings" in j else j
        if cik is None:  # paginated extra file: CIK from the main file of the same stem prefix
            main = json.loads((f.parent / (f.stem.split("__")[0] + ".json")).read_text())
            cik = main.get("cik")
        for form, fd, acc, doc in zip(blk["form"], blk["filingDate"], blk["accessionNumber"], blk["primaryDocument"]):
            if form in ("10-K", "10-K405", "10-KT") and "2017-06-01" <= fd <= "2026-09-24" and doc:
                yield sym, int(cik), fd, acc, doc


def extract_business(html: bytes) -> str:
    text = BeautifulSoup(html, "lxml").get_text("\n")
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    best = ""
    for m in START_ITEM.finditer(text):
        e = END_ITEM.search(text, m.end())
        if e is None:
            continue
        seg = text[m.end() : e.start()].strip()
        if len(seg) > len(best):  # the table-of-contents hit yields a tiny segment; the body is the longest
            best = seg
    if len(best) >= 3000:
        return best
    # Fallbacks for format variants: "Items 1 and 2. Business and Properties", headings split across
    # lines, or a heading line that just says "Business" / "Overview". Take the longest span between a
    # short heading-like line mentioning business and the next short heading line mentioning risk factors.
    lines = text.split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    bus = [i for i, ln in enumerate(lines) if len(ln.strip()) <= 70 and re.search(r"\bbusiness\b", ln, re.I)
           and not re.search(r"risk|forward|combination|segment", ln, re.I)]
    risk = [i for i, ln in enumerate(lines) if len(ln.strip()) <= 70 and re.search(r"risk\s+factors", ln, re.I)]
    for b in bus:
        nxt = [r for r in risk if r > b]
        if not nxt:
            continue
        seg = text[offs[b] + len(lines[b]) : offs[nxt[0]]].strip()
        if 3000 <= len(seg) <= 250000 and len(seg) > len(best):
            best = seg
    return best


def main():
    seen = set()
    todo = [x for x in filings() if (x[0], x[3]) not in seen and not seen.add((x[0], x[3]))]
    print(len(todo), "filings", flush=True)
    for i, (sym, cik, fd, acc, doc) in enumerate(todo):
        p = OUT / sym / f"{fd}_{acc}.json"
        if p.exists():
            if "--redo-short" not in sys.argv or json.loads(p.read_text())["n_chars"] >= 3000:
                continue
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
        for a in range(5):
            try:
                r = requests.get(url, headers=UA, timeout=60)
                if r.status_code in (429, 503):
                    time.sleep(5 * (a + 1))
                    continue
                break
            except requests.RequestException:
                time.sleep(5 * (a + 1))
        else:
            print("FAIL", sym, url, flush=True)
            continue
        biz = extract_business(r.content) if r.ok else ""
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"symbol": sym, "filed": fd, "accession": acc, "url": url, "status": r.status_code,
                                 "n_chars": len(biz), "business": biz[:60000]}))
        if i % 50 == 0:
            print(i, sym, fd, r.status_code, len(biz), flush=True)
        time.sleep(0.15)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
