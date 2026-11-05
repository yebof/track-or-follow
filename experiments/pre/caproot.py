"""A data root whose price bars end before W_clean, for Layer B on the W_pre slice.

`pipeline.layer_b.load_inputs` loads bars up to the window end + 60 days and computes the idiosyncratic-vol panel
and the labels on every loaded session. For the W_pre slice (last decision 2024-05-31) that would reach 2024-07-30,
inside W_clean (first cutoff 2024-07-05 16:00 ET), before the pre-registration unlock. Nothing downstream would read
those values (the last label matures at the 2024-06-10 open, the last 4-week label at the 2024-07-01 open, the last
book is held to 2024-06-07), but they would be computed. `capped_root` builds a mirror of the repository's `data/`
tree (symlinks) whose three bar files are cut at `PRICE_CAP` (2024-07-03, the last session before the first W_clean
decision day), so no return after a W_clean cutoff is ever computed. Everything else (universe, SEC archive, macro,
factors) is the repository's own file, linked.

    root = capped_root(scratch_dir)            # returns the data root; pass it as load_inputs(..., data_root=root)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRICE_CAP = pd.Timestamp("2024-07-03")
BAR_FILES = ("data/raw/bars/sp500_daily_2018_2026.parquet", "data/raw/bars/etf_daily_2018_2026.parquet",
             "data/raw/bars/all_daily_raw_2018_2026.parquet")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def capped_root(dst: Path, cap: pd.Timestamp = PRICE_CAP, src: Path = ROOT) -> Path:
    """Build (or reuse, if its record matches the sources) the capped data root under `dst`; returns `dst`."""
    dst = Path(dst)
    rec_path = dst / "CAPPED.json"
    want = {"cap": str(cap.date()), "sources": {f: _sha(src / f) for f in BAR_FILES}}
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        if rec.get("cap") == want["cap"] and rec.get("sources") == want["sources"]:
            return dst
    (dst / "data" / "raw" / "bars").mkdir(parents=True, exist_ok=True)
    for top in (src / "data").iterdir():  # data/processed, data/cache, ... linked as they are
        if top.name != "raw":
            link = dst / "data" / top.name
            if not link.exists():
                link.symlink_to(top)
    for sub in (src / "data" / "raw").iterdir():
        if sub.name != "bars":
            link = dst / "data" / "raw" / sub.name
            if not link.exists():
                link.symlink_to(sub)
    out = {}
    for f in BAR_FILES:
        df = pd.read_parquet(src / f)
        d = pd.to_datetime(df["date"])
        cut = df[d <= cap]
        cut.to_parquet(dst / f, index=False)
        out[f] = {"rows_in": int(len(df)), "rows_out": int(len(cut)),
                  "max_date_out": str(pd.to_datetime(cut["date"]).max().date())}
    rec_path.write_text(json.dumps({**want, "files": out}, indent=1))
    return dst
