"""Per-class share counts for companies whose share counts exist only with a share-class dimension (companyfacts
drops dimensioned facts): Berkshire Hathaway (BRK.B) and Visa (V). Needed for market value (V-B1).

For every 10-Q/10-K since 2017 this reads the filing's XBRL instance and keeps
- dei:EntityCommonStockSharesOutstanding per class (cover page, instant = cover date), and
- us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding per class (duration facts),
with the StatementClassOfStockAxis member of each. data/fundamentals.py turns them into one economic share count
(CLASS_SHARES there). Output: data/raw/sec/class_shares/<SYM>.json (rows: accn, filed, form, concept, member,
start, end, val).

Run: .venv/bin/python scripts/download_class_shares.py [SYM ...]
"""
import json
import sys
import time
from pathlib import Path

import requests
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "jev-quant-research research@example.com"}
OUT = ROOT / "data/raw/sec/class_shares"
CONCEPTS = {"EntityCommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"}


def get(url):
    for a in range(5):
        r = requests.get(url, headers=UA, timeout=60)
        if r.status_code in (429, 503):
            time.sleep(2 ** a)
            continue
        time.sleep(0.12)
        return r
    raise RuntimeError(url)


def instance_name(items: list[dict]) -> str | None:
    names = [i["name"] for i in items]
    for n in names:
        if n.endswith("_htm.xml"):
            return n
    skip = ("FilingSummary.xml", "_cal.xml", "_def.xml", "_lab.xml", "_pre.xml", "MetaLinks")
    xs = [n for n in names if n.endswith(".xml") and not any(s in n for s in skip)]
    return xs[0] if xs else None


def parse_instance(xml: bytes) -> list[dict]:
    root = etree.fromstring(xml)
    ctx = {}
    for c in root.iter("{*}context"):
        per = c.find("{*}period")
        inst = per.find("{*}instant")
        start, end = per.find("{*}startDate"), per.find("{*}endDate")
        dims = {m.get("dimension"): (m.text or "").strip() for m in c.iter("{*}explicitMember")}
        ctx[c.get("id")] = {"start": None if start is None else start.text.strip(),
                            "end": (inst if inst is not None else end).text.strip(), "dims": dims}
    rows = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        name = etree.QName(el).localname
        if name not in CONCEPTS or el.text is None:
            continue
        c = ctx.get(el.get("contextRef"))
        if c is None:
            continue
        other = {k: v for k, v in c["dims"].items() if not k.endswith("StatementClassOfStockAxis")}
        if other:
            continue  # only the class axis (or no dimension)
        member = next((v.split(":")[-1] for k, v in c["dims"].items() if k.endswith("StatementClassOfStockAxis")), "")
        rows.append({"concept": name, "member": member, "start": c["start"], "end": c["end"], "val": float(el.text)})
    return rows


def main(symbols: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        sub = json.loads((ROOT / f"data/raw/sec/submissions/{sym}.json").read_text())
        cik = int(sub["cik"])
        b = sub["filings"]["recent"]
        out_p = OUT / f"{sym}.json"
        have = {r["accn"] for r in json.loads(out_p.read_text())} if out_p.exists() else set()
        rows = json.loads(out_p.read_text()) if out_p.exists() else []
        for form, filed, acc in zip(b["form"], b["filingDate"], b["accessionNumber"]):
            if form not in ("10-Q", "10-K", "10-Q/A", "10-K/A") or filed < "2017-01-01" or acc in have:
                continue
            base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}"
            idx = get(base + "/index.json")
            if not idx.ok:
                print("no index", sym, acc)
                continue
            name = instance_name(idx.json()["directory"]["item"])
            if name is None:
                print("no instance", sym, acc)
                continue
            got = parse_instance(get(f"{base}/{name}").content)
            rows += [{"accn": acc, "filed": filed, "form": form, **r} for r in got]
            print(sym, filed, form, len(got), flush=True)
        out_p.write_text(json.dumps(rows, indent=0))
        print(sym, "rows", len(rows))


if __name__ == "__main__":
    main(sys.argv[1:] or ["BRK.B", "V"])
