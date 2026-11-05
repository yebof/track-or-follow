"""Kenneth R. French data library factors (daily FF5 2x3 and momentum), for ex-post style attribution only
(design_v2_delta PE-4). No factor return enters any decision; `scripts/download_ff_factors.py` archives a dated
vintage under data/raw/factors/<retrieval date>/ and `docs/factors/README.md` documents it.

The library's CSV files carry a few header lines, then rows `YYYYMMDD, v1, v2, ...` in PERCENT, then (in some files) a
copyright line. `parse_french_csv` returns decimals (percent / 100) indexed by date; the library's missing-value
codes -99.99 and -999 become NaN.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/raw/factors"
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french"
LIBRARY_PAGE = f"{BASE}/data_library.html"
FILES = {
    "ff5_2x3_daily": f"{BASE}/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "mom_daily": f"{BASE}/ftp/F-F_Momentum_Factor_daily_CSV.zip",
}


def _text(path: Path) -> str:
    path = Path(path)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            return z.read(names[0]).decode("latin-1")
    return path.read_text(encoding="latin-1")


def parse_french_csv(path: Path) -> pd.DataFrame:
    """Daily factor table (decimal returns) from a French-library CSV or its zip."""
    lines = _text(path).splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.strip() and ln.split(",")[0].strip() == ""
                and len(ln.split(",")) > 1)
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    rows = []
    for ln in lines[head + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(cols) + 1 or not (parts[0].isdigit() and len(parts[0]) == 8):
            if rows:
                break  # end of the daily block (copyright line or an annual block)
            continue
        rows.append([parts[0], *map(float, parts[1:])])
    df = pd.DataFrame(rows, columns=["date", *cols])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    return df.mask(df.isin([-99.99, -999.0])) / 100.0


def current_vintage() -> Path:
    return OUT / (OUT / "CURRENT").read_text().strip()


def load_factors(vintage: Path | None = None) -> pd.DataFrame:
    """FF5 (Mkt-RF, SMB, HML, RMW, CMA, RF) joined with momentum (Mom), decimal daily returns."""
    v = vintage or current_vintage()
    ff5 = parse_french_csv(v / FILES["ff5_2x3_daily"].rsplit("/", 1)[1])
    mom = parse_french_csv(v / FILES["mom_daily"].rsplit("/", 1)[1])
    mom.columns = ["Mom" if c.lower().startswith("mom") else c for c in mom.columns]
    return ff5.join(mom, how="inner")

