"""FEED-1 (design_v2_delta §3.14): the feed's provenance mix by year 2019..2026 and on W_clean.

Input: `data/processed/news_table.parquet` (built from the raw Benzinga batches; it carries `author` and
`url_section`). Scopes:
- `all`: every distinct item in the table (items tagged to a study symbol or a market ETF);
- `company_le3`: items the news analyst reads (tagged to a study company, <= 3 tags).
Groups: calendar year of `created_at` (UTC) and `W_clean` (created in [c(2024-06-28), c(2026-09-11)), the items the
W_clean decisions can read). Byline classes and the outside-reachable rule: `data/provenance.py`.

The SAN-4 D0 event flags (hidden characters, invisible characters, confusables, mixed script, truncation) are logged
by the Layer-A runner into evidence.parquet (WS-A); they join this table once a judgment run exists.

Run: .venv/bin/python scripts/feed_provenance.py [--out data/processed/feed_provenance.json]
"""

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevtrader.data.news import decision_cutoff  # noqa: E402
from jevtrader.data.provenance import CLASSES, classify_items, desk_bylines, provenance_mix  # noqa: E402

W_CLEAN_READ = (decision_cutoff(pd.Timestamp("2024-06-28")), decision_cutoff(pd.Timestamp("2026-09-11")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(ROOT / "data/processed/news_table.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data/processed/feed_provenance.json"))
    args = ap.parse_args()
    t = pd.read_parquet(args.table, columns=["id", "symbol", "created_at", "n_tags", "author", "url_section"])
    desk = desk_bylines()
    company = t[(t.symbol != "MARKET") & (t.n_tags <= 3)]
    res = {"table": str(Path(args.table).relative_to(ROOT)),
           "table_sha256": hashlib.sha256(Path(args.table).read_bytes()).hexdigest(),
           "computed_on": str(date.today()), "desk_bylines": list(desk), "classes": list(CLASSES),
           "w_clean_created": [str(x) for x in W_CLEAN_READ], "scopes": {}}
    for scope, frame in (("all", t), ("company_le3", company)):
        it = classify_items(frame, desk)
        it = it[(it.created_at >= "2019-01-01") & (it.created_at < "2027-01-01")]
        by_year = provenance_mix(it, it.created_at.dt.year)
        wc = it[(it.created_at >= W_CLEAN_READ[0]) & (it.created_at < W_CLEAN_READ[1])]
        res["scopes"][scope] = {"by_year": by_year,
                                "W_clean": provenance_mix(wc, pd.Series("W_clean", index=wc.index))["W_clean"]}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    for scope, r in res["scopes"].items():
        print(f"[{scope}]")
        for g, v in {**r["by_year"], "W_clean": r["W_clean"]}.items():
            cls = " ".join(f"{c}={v['byline_class_share'][c]:.3f}" for c in CLASSES)
            print(f"  {g}: n={v['n_items']} {cls} reachable={v['outside_reachable_share']:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
