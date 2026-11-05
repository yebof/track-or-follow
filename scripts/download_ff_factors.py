"""Download the Kenneth R. French data library's daily Fama-French 5 factors (2x3) and daily momentum factor into
data/raw/factors/ (design_v2_delta PE-4, ex-post attribution only; decision Q4 approved 2026-09-26).

Each run stores the zip files under a dated directory (a later re-download may change history, PE-4 caveat 2), the
library's landing page as it was when downloaded, and a manifest with the source URLs, retrieval date, SHA-256 and the
first and last dates in each file. `docs/factors/README.md` documents the vintage used. No factor return ever enters a
decision (caveat 1).

Run: .venv/bin/python scripts/download_ff_factors.py
"""

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.data.factors import FILES, LIBRARY_PAGE, OUT, parse_french_csv  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (research data download)"}


def main() -> None:
    today = str(date.today())
    vdir = OUT / today
    vdir.mkdir(parents=True, exist_ok=True)
    manifest = {"retrieved": today, "library_page": LIBRARY_PAGE, "files": {}}
    page = requests.get(LIBRARY_PAGE, headers=UA, timeout=60)
    page.raise_for_status()
    (vdir / "data_library.html").write_bytes(page.content)
    manifest["library_page_sha256"] = hashlib.sha256(page.content).hexdigest()
    for key, url in FILES.items():
        r = requests.get(url, headers=UA, timeout=120)
        r.raise_for_status()
        p = vdir / url.rsplit("/", 1)[1]
        p.write_bytes(r.content)
        df = parse_french_csv(p)
        manifest["files"][key] = {"url": url, "file": str(p.relative_to(ROOT)), "bytes": len(r.content),
                                  "sha256": hashlib.sha256(r.content).hexdigest(), "columns": list(df.columns),
                                  "first_date": str(df.index.min().date()), "last_date": str(df.index.max().date()),
                                  "n_days": int(len(df))}
        print(key, manifest["files"][key])
    (vdir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (OUT / "CURRENT").write_text(today + "\n")
    print("wrote", vdir)


if __name__ == "__main__":
    main()
