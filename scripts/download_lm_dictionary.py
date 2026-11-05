"""Download the Loughran-McDonald Master Dictionary (baseline B10, design_v1 section 13.3) and archive its source.

Source page: https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (Software Repository for Accounting and Finance,
University of Notre Dame). The page links the CSV on Google Drive; the page itself is archived next to the CSV so the
link and the licence note ("free for use in academic research") can be checked later.

Output: data/raw/lm_dictionary/{Loughran-McDonald_MasterDictionary.csv, sraf_page_<date>.html, README.md}
"""
import re
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/raw/lm_dictionary"
PAGE = "https://sraf.nd.edu/loughranmcdonald-master-dictionary/"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    html = requests.get(PAGE, headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
    (OUT / f"sraf_page_{date.today().isoformat()}.html").write_text(html)
    # the first Drive link after "CSV Format" is the CSV file
    m = re.search(r"CSV Format.*?href=\"(https://drive\.google\.com/file/d/([\w-]+)/[^\"]*)\"[^>]*>([^<]+)<", html, re.S)
    if not m:
        print("CSV link not found on", PAGE)
        return 1
    view_url, file_id, name = m.group(1), m.group(2), m.group(3).strip()
    r = requests.get(f"https://drive.google.com/uc?export=download&id={file_id}", timeout=120,
                     headers={"User-Agent": "Mozilla/5.0"})
    if not r.ok or not r.content[:4].decode(errors="ignore").isalpha():
        print("download failed", r.status_code)
        return 1
    (OUT / "Loughran-McDonald_MasterDictionary.csv").write_bytes(r.content)
    (OUT / "README.md").write_text(
        f"# Loughran-McDonald Master Dictionary\n\n"
        f"- File: `Loughran-McDonald_MasterDictionary.csv` (published as `{name}`), downloaded {date.today()} by "
        f"`scripts/download_lm_dictionary.py`.\n"
        f"- Source page: {PAGE} (archived as `sraf_page_{date.today().isoformat()}.html`).\n"
        f"- File link on that page: {view_url} (direct download id `{file_id}`).\n"
        f"- Licence (source page): free for use in academic research; commercial licences via the authors.\n"
        f"- Use: baseline B10 (net negative-word share of headlines and summaries; `baselines/text.py`). A word counts "
        f"for a decision in year Y only if its category flag (year added) is before Y, or it was removed after Y.\n")
    print("saved", len(r.content), "bytes:", name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
