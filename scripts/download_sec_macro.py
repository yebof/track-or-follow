"""SEC XBRL companyfacts + filing indexes for universe companies; FRED/ALFRED macro series; EPU and GPR indices;
the pre-announced FOMC meeting schedule.

Usage: .venv/bin/python scripts/download_sec_macro.py [sec] [fred] [indices] [fomc] [--refresh] [--root=DIR]
       [--universe=PARQUET]
- `sec`: companyfacts and submissions (with their paginated `files`) per universe company. A file that exists is
  skipped, unless `--refresh` is given (the live refresh): then every file is fetched again and replaced atomically
  (write a temporary file, then rename it over the entry). `--root=DIR` writes under DIR/data/raw/sec instead of the
  archive (the live root, `live/refresh.py`; refused if that would write through a link into the archive).
  `--universe=` names the universe file (default: the archive's). Prints a `SEC_SUMMARY {json}` line (files fetched per
  kind, absent files, failures, newest acceptance time) and exits 1 if any fetch failed.
- `fred`: daily series (current vintage) and revised series (ALFRED, all vintages) from 2005 (design_v1 section 2.4,
  V-m1). A file is re-downloaded when it starts later than FRED_START; `--refresh` forces it (live refresh).
- `indices`: incremental EPU/GPR refresh. Each run stores the raw download as a dated snapshot and appends only new
  dates to `*_archive.csv` (date, value, first_seen). Archived values are never overwritten, so a re-run cannot
  rewrite the history a past decision saw; revised values are only counted.
- `fomc`: archives the Fed's tentative-schedule press releases and calendar pages under data/raw/macro/fomc/ and
  builds data/raw/macro/fomc_calendar.csv (see data/raw/macro/fomc/README.md for the source URLs).
`--root=DIR` applies to `fred`, `indices` and `fomc` too: they then write under DIR/data/raw/macro (the live root,
whose macro directory `live.refresh.link_macro` makes a real directory of per-file links into the archive). Every
write there is a temporary file renamed over the entry, so a link is replaced and the archive file it pointed to stays
byte-identical; a macro directory that is itself a link into the archive is refused. The current-vintage daily FRED
series are re-downloaded whole by `--refresh`, so revised past observations land in the live root only, never in
the archive that the backtests and judgment tables were built from.
"""
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
UA = {"User-Agent": "jev-quant-research research@example.com"}
# CIKs for universe companies no longer in SEC's current ticker file (EDGAR entity search, 2026-09-25).
DELISTED_CIK = {"AGN": 1578845, "ATVI": 718877, "CELG": 816284, "DISCA": 1437107, "EA": 712515,
                "PXD": 1038357, "RTN": 1047122, "TIF": 98246, "TWTR": 1418091, "XLNX": 743988,
                # current ticker file maps APC to an unrelated later issuer (ARKO Petroleum); Anadarko is 773910
                "APC": 773910, "WBA": 1618921}
# Companies that moved to a new holding-company CIK: history lives under the predecessor (EDGAR, 2026-09-25).
PREDECESSOR_CIK = {"XOM": [34088], "BLK": [1364742], "DIS": [1001039], "MRVL": [1058057]}


def get(url, **kw):
    for a in range(6):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** a)
                continue
            return r
        except requests.RequestException:
            time.sleep(2 ** a)
    raise RuntimeError(url.split("?")[0])  # never echo query strings (they may carry an API key)


def _atomic_write(p: Path, data: bytes) -> None:
    """Write via a temporary file and a rename over the entry: a reader never sees a partial file, and a symlink entry
    (the live root's link into the archive) is replaced, never written through."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def _opt(name: str, default=None):
    pre = f"--{name}="
    return next((a[len(pre):] for a in sys.argv[1:] if a.startswith(pre)), default)


def macro_dir(root: Path | None = None) -> Path:
    """data/raw/macro under `root` (default: `--root=` or the archive). Outside the archive, the directory and its
    `fred`, `fomc` and `snapshots` subdirectories must be real directories, never links into the archive."""
    root = Path(root or _opt("root") or ROOT)
    out = root / "data/raw/macro"
    if root.resolve() != ROOT.resolve():
        frozen = (ROOT / "data/raw").resolve()
        for d in (out, out / "fred", out / "fomc", out / "snapshots"):
            if d.is_symlink() or ((d.exists() or d == out) and d.resolve().is_relative_to(frozen)):
                raise RuntimeError(f"{d} links into the archive; run live.refresh.link_macro first")
    return out


def _atomic_text(p: Path, text: str) -> None:
    _atomic_write(p, text.encode())


def sec(root: Path | None = None, refresh: bool | None = None, universe: Path | None = None, fetch=None) -> dict:
    root = Path(root or _opt("root") or ROOT)
    refresh = ("--refresh" in sys.argv) if refresh is None else refresh
    universe = Path(universe or _opt("universe") or ROOT / "data/processed/universe_monthly_top100.parquet")
    fetch = fetch or globals()["get"]
    out_sec = root / "data/raw/sec"
    if root.resolve() != ROOT.resolve():  # never write through the live root's links into the archive
        frozen = (ROOT / "data/raw").resolve()
        for d in (out_sec, out_sec / "submissions", out_sec / "facts"):
            if d.is_symlink() or d.resolve().is_relative_to(frozen):
                raise RuntimeError(f"{d} links into the archive; run live.refresh.link_sec first")
    tick = json.loads((ROOT / "data/raw/sec/company_tickers.json").read_text())
    cik = {v["ticker"].upper().replace("-", "."): int(v["cik_str"]) for v in tick.values()}
    cik.update(DELISTED_CIK)
    syms = sorted(set(pd.read_parquet(universe).symbol))
    mapping = {}
    stats = {"refresh": bool(refresh), "submissions": 0, "facts": 0, "pages": 0, "absent": [], "failed": [],
             "newest_acceptance": ""}
    _atomic_write(out_sec / "predecessor_ciks.json", json.dumps(PREDECESSOR_CIK, indent=1).encode())
    for s in syms:
        c = cik.get(s)
        if c is None:
            print("NO CIK", s)
            continue
        mapping[s] = c
        jobs = [("facts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json", s),
                ("submissions", f"https://data.sec.gov/submissions/CIK{c:010d}.json", s)]
        for k, pc in enumerate(PREDECESSOR_CIK.get(s, [])):
            jobs += [("facts", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{pc:010d}.json", f"{s}__pred{k}"),
                     ("submissions", f"https://data.sec.gov/submissions/CIK{pc:010d}.json", f"{s}__pred{k}")]
        for kind, url, stem in jobs:
            p = out_sec / kind / f"{stem}.json"
            if p.exists() and not refresh:
                continue
            try:
                r = fetch(url, headers=UA)
            except RuntimeError as e:
                stats["failed"].append(f"{stem} {kind}: {e}")
                continue
            if r.ok:
                _atomic_write(p, r.content)
                stats[kind] += 1
                # older filings beyond the first page of `submissions` live in extra files
                if kind == "submissions":
                    j = r.json()
                    acc = (j.get("filings", {}).get("recent", {}).get("acceptanceDateTime") or [""])
                    stats["newest_acceptance"] = max(stats["newest_acceptance"], max(acc))
                    for f in j.get("filings", {}).get("files", []):
                        q = out_sec / "submissions" / f"{stem}__{f['name']}"
                        if q.exists() and not refresh:
                            continue
                        try:
                            rq = fetch("https://data.sec.gov/submissions/" + f["name"], headers=UA)
                        except RuntimeError as e:
                            stats["failed"].append(f"{stem} {f['name']}: {e}")
                            continue
                        if rq.ok:
                            _atomic_write(q, rq.content)
                            stats["pages"] += 1
                        else:
                            stats["failed"].append(f"{stem} {f['name']}: HTTP {rq.status_code}")
                        time.sleep(0.12)
            elif r.status_code == 404 and not p.exists():
                stats["absent"].append(f"{stem} {kind}")  # EDGAR has no such file (e.g. no XBRL facts): not a failure
            else:
                print("FAIL", s, kind, r.status_code)
                stats["failed"].append(f"{stem} {kind}: HTTP {r.status_code}")
            time.sleep(0.12)
    _atomic_write(out_sec / "cik_map.json", json.dumps(mapping, indent=1).encode())
    print("sec done", len(mapping))
    print("SEC_SUMMARY " + json.dumps(stats), flush=True)
    return stats


# Revised series -> ALFRED vintages (all realtime); market series have no revisions. NFCI is dropped (V-m1): it is
# revised weekly and FRED serves only its current vintage. The ICE BofA spreads (BAMLH0A0HYM2, BAMLC0A0CM) are
# replaced by Moody's BAA10Y/AAA10Y: FRED serves only the last 3 years of ICE data (from 2023-09-25), which cannot
# fill a 5-year percentile window.
FRED_VINTAGE = ["CPIAUCSL", "CPILFESL", "UNRATE", "PAYEMS", "INDPRO", "RSAFS", "GDPC1", "PCEPILFE", "ICSA"]
FRED_MARKET = ["DGS10", "DGS2", "DGS3MO", "T10Y2Y", "DFF", "VIXCLS", "BAA10Y", "AAA10Y",
               "DTWEXBGS", "DCOILWTICO", "T10YIE"]
FRED_START = "2005-01-01"


def fred(root: Path | None = None, refresh: bool | None = None, fetch=None):
    key = os.environ["FRED_API_KEY"]
    refresh = ("--refresh" in sys.argv) if refresh is None else refresh
    get = fetch or globals()["get"]
    out = macro_dir(root) / "fred"
    out.mkdir(parents=True, exist_ok=True)
    for s in FRED_VINTAGE + FRED_MARKET:
        p = out / f"{s}.json"
        if p.exists() and not refresh:
            if json.loads(p.read_text()).get("observation_start", "") <= FRED_START:
                continue
        params = {"series_id": s, "api_key": key, "file_type": "json", "observation_start": FRED_START,
                  "limit": 100000}
        if s in FRED_VINTAGE:
            params.update(realtime_start=FRED_START, realtime_end="9999-12-31")
        obs, head = [], None
        while True:  # the API caps a response at 100,000 rows; vintage files can be longer
            r = get("https://api.stlouisfed.org/fred/series/observations", params={**params, "offset": len(obs)})
            if not r.ok:
                raise RuntimeError(f"FRED {s}: HTTP {r.status_code}")
            j = r.json()
            head = head or {k: v for k, v in j.items() if k != "observations"}
            obs += j["observations"]
            if len(obs) >= j["count"] or not j["observations"]:
                break
            time.sleep(0.6)
        head.update(offset=0, downloaded=date.today().isoformat())
        _atomic_text(p, json.dumps({**head, "observations": obs}))  # replaces a link, never writes through it
        print(s, len(obs), "obs from", obs[0]["date"] if obs else None, flush=True)
        time.sleep(0.6)


# --------------------------------------------------------------------------------- EPU / GPR (incremental)
INDICES = {
    "epu": ("https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv", "csv"),
    "gpr": ("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls", "xls"),
}


def _parse_index(name: str, path: Path) -> pd.Series:
    if name == "epu":
        e = pd.read_csv(path)
        return pd.Series(e.daily_policy_index.values, index=pd.to_datetime(dict(year=e.year, month=e.month, day=e.day)))
    g = pd.read_excel(path)
    return pd.Series(g.GPRD.values, index=pd.to_datetime(g.date))


def merge_index_archive(archive: Path, fresh: pd.Series, today: str) -> tuple[pd.DataFrame, int]:
    """Append dates not yet archived (first_seen = today); keep archived values. Returns (archive, n_revised)."""
    fresh = fresh.dropna().sort_index()
    fresh = fresh[~fresh.index.duplicated(keep="last")]
    if archive.exists():
        old = pd.read_csv(archive, parse_dates=["date"])
    else:
        old = pd.DataFrame(columns=["date", "value", "first_seen"])
    known = pd.Series(old.value.values, index=pd.DatetimeIndex(old.date))
    both = fresh.index.intersection(known.index)
    n_revised = int((abs(fresh.loc[both] - known.loc[both]) > 1e-9).sum())
    new = fresh[~fresh.index.isin(known.index)]
    add = pd.DataFrame({"date": new.index, "value": new.values, "first_seen": today})
    merged = pd.concat([old, add], ignore_index=True).sort_values("date").reset_index(drop=True)
    merged["date"] = pd.to_datetime(merged.date).dt.strftime("%Y-%m-%d")
    return merged, n_revised


def indices(root: Path | None = None, fetch=None):
    today = date.today().isoformat()
    get = fetch or globals()["get"]
    macro = macro_dir(root)
    snap_dir = macro / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, ext) in INDICES.items():
        r = get(url, headers={"User-Agent": "Mozilla/5.0"})
        if not r.ok:
            print(name, "HTTP", r.status_code)
            continue
        snap = snap_dir / f"{name}_daily_{today}.{ext}"
        _atomic_write(snap, r.content)
        archive = macro / f"{name}_daily_archive.csv"
        if not archive.exists():  # seed from the first download archived by the project (2026-09-25)
            seed = macro / ("epu_daily.csv" if name == "epu" else "gpr_daily_recent.xls")
            if seed.exists():
                s0, _ = merge_index_archive(archive, _parse_index(name, seed), "2026-09-25")
                _atomic_text(archive, s0.to_csv(index=False))
        merged, n_rev = merge_index_archive(archive, _parse_index(name, snap), today)
        _atomic_text(archive, merged.to_csv(index=False))  # a link to the archive file is replaced, not written
        print(f"{name}: {len(merged)} archived days, last {merged.date.iloc[-1]}, {n_rev} revised values ignored")


# ------------------------------------------------------------------------------------- FOMC schedule
FED = "https://www.federalreserve.gov"
# Tentative-schedule press releases (found in the Board's press-release feed /json/ne-press.json, 2026-09-25):
# release date -> (url path, years it schedules)
FOMC_RELEASES = {
    "2018-05-25": ("/newsevents/pressreleases/monetary20180525a.htm", [2019]),
    "2019-05-17": ("/newsevents/pressreleases/monetary20190517a.htm", [2020]),
    "2020-07-02": ("/newsevents/pressreleases/monetary20200702a.htm", [2021]),
    "2021-06-04": ("/newsevents/pressreleases/monetary20210604a.htm", [2022]),
    "2022-06-24": ("/newsevents/pressreleases/monetary20220624a.htm", [2023]),
    "2023-06-23": ("/newsevents/pressreleases/monetary20230623a.htm", [2024]),
    "2024-08-09": ("/newsevents/pressreleases/monetary20240809a.htm", [2025, 2026]),
    "2025-09-05": ("/newsevents/pressreleases/monetary20250905a.htm", [2027]),
}
FOMC_CALENDARS = ["/monetarypolicy/fomccalendars.htm", "/monetarypolicy/fomchistorical2019.htm",
                  "/monetarypolicy/fomchistorical2020.htm"]
_MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                        "september", "october", "november", "december"], 1)}
_MON = r"(January|February|March|April|May|June|July|August|September|October|November|December)"


def parse_schedule_release(text: str, years: list[int]) -> list[tuple[date, date]]:
    """Meetings (first day, last day) listed in a tentative-schedule press release. Handles the two formats used
    2018-2025: "January 29-30 (Tuesday-Wednesday)" / "April 30-May 1 ..." and "Tuesday, January 28, and Wednesday,
    January 29". A line may carry an explicit year (the following January meeting); otherwise the year is the
    section's ("For 2026:") or the release's single scheduled year."""
    out, cur = [], years[0]
    for line in (x.strip() for x in text.splitlines()):
        m = re.match(r"For (\d{4}):", line)
        if m:
            cur = int(m.group(1))
            continue
        a = re.match(_MON + r" (\d{1,2})-(?:" + _MON + r" )?(\d{1,2})(?:, (\d{4}))? \(", line)
        b = re.match(r"\w+day, " + _MON + r" (\d{1,2}), and \w+day, " + _MON + r" (\d{1,2})(?:, (\d{4}))?$", line)
        if a:
            m1, d1, m2, d2, y = a.group(1), int(a.group(2)), a.group(3) or a.group(1), int(a.group(4)), a.group(5)
        elif b:
            m1, d1, m2, d2, y = b.group(1), int(b.group(2)), b.group(3), int(b.group(4)), b.group(5)
        else:
            continue
        yr = int(y) if y else cur
        out.append((date(yr, _MONTHS[m1.lower()], d1), date(yr, _MONTHS[m2.lower()], d2)))
    return out


def _meeting(year: int, month_txt: str, day_txt: str) -> tuple[date, date, bool] | None:
    """('Apr/May', '30-1') or ('October', '4 (unscheduled)') -> (first day, last day, unscheduled)."""
    abbrev = {k[:3]: v for k, v in _MONTHS.items()}
    months = [abbrev[x[:3].lower()] for x in re.findall(r"[A-Za-z]{3,}", month_txt) if x[:3].lower() in abbrev]
    days = [int(x) for x in re.findall(r"\d{1,2}", day_txt.split("(")[0])]
    if not months or not days:
        return None
    return date(year, months[0], days[0]), date(year, months[-1], days[-1]), "unscheduled" in (month_txt + day_txt).lower()


def parse_calendar_page(html: str) -> list[tuple[date, date, bool]]:
    """Meetings actually held, from the Fed's calendar pages: (first day, last day, unscheduled). The current page
    has one panel per year with month/date cells; the historical pages have one heading per meeting
    ("April/May 30-1 Meeting - 2019", "October 4 (unscheduled) - 2019")."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out = []
    for panel in soup.select("div.panel"):
        head = panel.select_one(".panel-heading")
        y = re.search(r"(\d{4}) FOMC Meetings", head.get_text() if head else "")
        if not y:
            continue
        for row in panel.select("div.fomc-meeting"):
            mo, dd = row.select_one(".fomc-meeting__month"), row.select_one(".fomc-meeting__date")
            if mo is not None and dd is not None:
                m = _meeting(int(y.group(1)), mo.get_text(" ", strip=True), dd.get_text(" ", strip=True))
                if m:
                    out.append(m)
    for h in soup.find_all(["h4", "h5"]):
        t = h.get_text(" ", strip=True)
        g = re.match(r"([A-Za-z/]+) ([\d\-]+(?: \(unscheduled\))?)(?: Meeting)? - (\d{4})$", t)
        if g:
            m = _meeting(int(g.group(3)), g.group(1), g.group(2))
            if m:
                out.append(m)
    return out


def fomc(root: Path | None = None, fetch=None):
    from bs4 import BeautifulSoup

    get = fetch or globals()["get"]
    macro = macro_dir(root)
    d = macro / "fomc"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for released, (path, years) in FOMC_RELEASES.items():
        f = d / Path(path).name
        if not f.exists():
            _atomic_write(f, get(FED + path, headers={"User-Agent": "Mozilla/5.0"}).content)
            time.sleep(0.5)
        art = BeautifulSoup(f.read_text(), "lxml")
        text = (art.select_one("#article") or art).get_text("\n")
        for start, end in parse_schedule_release(text, years):
            if start.year in years:
                rows.append({"meeting_start": start, "decision_date": end, "announced": released,
                             "source_url": FED + path})
    held = []
    for path in FOMC_CALENDARS:
        f = d / Path(path).name
        if not f.exists():
            _atomic_write(f, get(FED + path, headers={"User-Agent": "Mozilla/5.0"}).content)
            time.sleep(0.5)
        held += parse_calendar_page(f.read_text())
    cal = pd.DataFrame(rows).drop_duplicates(["meeting_start", "decision_date"]).sort_values("decision_date")
    held_sched = {e for _, e, u in held if not u}
    cal["held_as_scheduled"] = [e in held_sched if e <= date.today() else None for e in cal.decision_date]
    unsched = sorted({(s, e) for s, e, u in held if u})
    extra = pd.DataFrame([{"meeting_start": s, "decision_date": e, "announced": None, "source_url": None,
                           "held_as_scheduled": None} for s, e in unsched])
    cal["scheduled"] = True
    extra["scheduled"] = False
    cal = pd.concat([cal, extra], ignore_index=True).sort_values("decision_date")
    _atomic_text(macro / "fomc_calendar.csv", cal.to_csv(index=False))
    print(f"fomc: {int(cal.scheduled.sum())} pre-announced meetings {cal.decision_date.min()}..{cal.decision_date.max()};"
          f" {len(extra)} unscheduled (kept for reference, never used); not held as scheduled:",
          cal[(cal.scheduled) & (cal.held_as_scheduled == False)].decision_date.tolist())  # noqa: E712


if __name__ == "__main__":
    what = [a for a in sys.argv[1:] if not a.startswith("--")] or ["sec", "fred", "indices", "fomc"]
    failed = False
    for w in what:
        res = globals()[w]()
        failed |= bool(isinstance(res, dict) and res.get("failed"))
    sys.exit(1 if failed else 0)
