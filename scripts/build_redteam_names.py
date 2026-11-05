"""Frozen display-name table for planted red-team items (audit finding "planted names", 2026-09-27).

A planted item must name the company the way genuine items of the feed name it, so that after D0 and blinding it
reads "Company-XXXXXXXX ..." exactly like a genuine item, with no residue ("CORP /NEW", ", INC.", "de Nemours") and
no unmasked name. The name of a symbol is its most frequent feed short name (`data.leak_probe.feed_short_names`: the
name in the archive's machine-written broker-action and options-activity headlines of its single-ticker items, within
its membership span), with explicit overrides where the templates are too rare (checked by hand in the symbol's own
headlines). Every name is checked against the study blinder: at a cutoff in every month the symbol is a universe
member, `blind(name, tagged=(symbol,))` must be exactly the symbol's code name; any miss or missing name fails.

Writes src/jevtrader/redteam/display_names_v1.json (hashed into `redteam.attacks.templates_sha256`).
Run: .venv/bin/python scripts/build_redteam_names.py [--check]   (--check: verify the frozen table, write nothing)
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

from jevtrader.data.blinding import load_blinder  # noqa: E402
from jevtrader.data.leak_probe import feed_short_names  # noqa: E402
from jevtrader.data.news import decision_cutoff  # noqa: E402
from jevtrader.data.universe import SymbolResolver, load_membership, load_renames  # noqa: E402

OUT = ROOT / "src/jevtrader/redteam/display_names_v1.json"
VERSION = "rt-names-v1"
# Symbols whose templates are too rare; the name is the one their own headlines use (counts in the 2026-09-27 scan).
OVERRIDES = {"APC": "Anadarko Petroleum",  # "Anadarko Petroleum Q2 Adj. EPS ...", 2019 (Anadarko, CIK 773910)
             "BRK.B": "Berkshire Hathaway"}  # 11 of 91 BRK.B headlines


def build() -> dict:
    from jevtrader.programs.render import HEADLINE_MAX, d0

    uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
    syms = sorted(set(uni.symbol))
    news = pd.read_parquet(ROOT / "data/processed/news_table.parquet",
                           columns=["symbol", "created_at", "n_tags", "headline"])
    short = feed_short_names(news[news.symbol.isin(syms)], uni, min_count=3,
                             clean=lambda h: str(d0(h, HEADLINE_MAX)))
    names, source = {}, {}
    for s in syms:
        if s in OVERRIDES:
            names[s], source[s] = OVERRIDES[s], "override"
        elif s in short:
            names[s], source[s] = short[s][0][0], f"feed:{short[s][0][1]}"
    return {"names": names, "source": source, "symbols": syms, "uni": uni}


def check(names: dict[str, str], syms, uni) -> list[str]:
    m = load_membership(ROOT / "data/raw/universe/sp500_hist_components.csv", "2018-01-01")
    res = SymbolResolver(load_renames(ROOT / "data/raw/universe/alpaca_name_changes.json", m))
    b = load_blinder(ROOT / "data/raw/sec/submissions", syms, resolver=res,
                     history=(date(2018, 1, 1), date(2026, 12, 31)))
    bad = [f"{s}: no display name" for s in syms if s not in names]
    months = pd.to_datetime(uni["date"]).groupby(uni["symbol"]).apply(lambda d: sorted(set(d)))
    for s, n in sorted(names.items()):
        for d in months.get(s, []):
            out = b(n, cutoff=decision_cutoff(pd.Timestamp(d) + pd.Timedelta(days=14)), tagged=(s,))
            if out != b.code_name(s):
                bad.append(f"{s}: {n!r} -> {out!r} at {pd.Timestamp(d).date()}")
                break
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        tab = json.loads(OUT.read_text())
        uni = pd.read_parquet(ROOT / "data/processed/universe_monthly_top100.parquet")
        bad = check(tab["names"], sorted(set(uni.symbol)), uni)
    else:
        got = build()
        bad = check(got["names"], got["symbols"], got["uni"])
        if not bad:
            blob = {"version": VERSION, "built": str(date.today()),
                    "rule": "most frequent feed short name (data.leak_probe) within the membership span; overrides",
                    "names": dict(sorted(got["names"].items())), "source": dict(sorted(got["source"].items()))}
            OUT.write_text(json.dumps(blob, indent=1, ensure_ascii=False, sort_keys=False) + "\n")
            print(f"wrote {OUT} ({len(blob['names'])} names), sha256 "
                  f"{hashlib.sha256(OUT.read_bytes()).hexdigest()}")
    for x in bad:
        print("FAIL", x)
    if bad:
        sys.exit(1)
    return bad


if __name__ == "__main__":
    main()
