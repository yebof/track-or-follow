"""Ex-post style attribution of book returns (design_v2_delta PE-3, PE-4). Evaluation only: no function here feeds a
decision, and no decision module imports this one (tests/test_attribution_factors_pit.py).

PE-3, own point-in-time factors on U_t. At every weekly decision t, each factor sorts U_t on a trusted code feature
known at c_t (the Layer-B catalog values of `code_features.parquet`, and `size_rank` of `firm_info.parquet`, 1 =
largest raw market value), goes long the top tercile and short the bottom tercile, equal weights, filled at open(e_t)
and held to open(e_{t+1}) like the books (backtest engine, no costs). Daily long-short returns:
    MOM    mom_12_1 high minus low          REV    rev_1m high minus low (past losers, signed so)
    LOWVOL vol_60d low minus high           SIZE   size_rank high minus low (small minus big)
    VALUE  earnings_yield high minus low    PROF   gross_profitability high minus low
plus the market: MKT_EW (EW-U100) and MKT_SPY (SPY), both in excess of the book's cash rate (DFF). A book's daily
excess return is regressed on them with Newey-West (lag 5) standard errors; weekly sums of daily returns are the
robustness check (lag 4). The selection alpha is the intercept, annualized (x252 daily, x52 weekly), with a 95% CI.

PE-4, Fama-French five factors + momentum, daily, from the Kenneth R. French data library files archived under
data/raw/factors/<retrieval date>/ with a CURRENT pointer and a MANIFEST.json of SHA-256s (scripts/download_ff_factors.py,
WS-C). The loader reads the CURRENT vintage and refuses a file whose hash differs from the manifest. Parsed from the
library's CSV layout: free-text header lines, then a line
",Mkt-RF,SMB,HML,RMW,CMA,RF" (",Mom" in the momentum file), then rows "YYYYMMDD, v1, v2, ..." in percent, ended by a
non-date line (copyright). -99.99 / -999 mark missing values. Caveats (delta §3.8): ex post only; the vintage is the
archived file (its SHA-256 is returned); coverage is reported against the evaluated window; the factors are built on
the whole market (read the loadings as tilts; PE-3 is the matched control); close-to-close daily factors against
open-to-open weekly holdings (daily HAC plus weekly aggregation); the book's excess return is over DFF while the
library's RF is the one-month T-bill, which is reported.
"""

from __future__ import annotations

import hashlib
import math
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from jevtrader.backtest.engine import PriceBook, run_backtest
from jevtrader.decide.select import tiebreak_key

OWN_FACTORS: dict[str, tuple[str, int]] = {
    "MOM": ("mom_12_1", 1), "REV": ("rev_1m", 1), "LOWVOL": ("vol_60d", -1), "SIZE": ("size_rank", 1),
    "VALUE": ("earnings_yield", 1), "PROF": ("gross_profitability", 1),
}
FACTORS_DIR = Path("data/raw/factors")
FRENCH_PATTERNS = {"ff5": "5_Factors_2x3_daily", "mom": "Momentum_Factor_daily"}
FRENCH_MISSING = (-99.99, -999.0)


# ------------------------------------------------------------------------------------ PE-3 own factors
def factor_inputs(code_raw: pd.DataFrame, firm_info: pd.DataFrame | None = None) -> pd.DataFrame:
    """Wide (t, symbol) x feature table of the factor sort variables from the long `t, symbol, feature, value` code
    features (raw values) and, for SIZE, firm_info's `size_rank`."""
    feats = {f for f, _ in OWN_FACTORS.values() if f != "size_rank"}
    c = code_raw[code_raw["feature"].isin(feats)]
    wide = c.pivot_table(index=["t", "symbol"], columns="feature", values="value", aggfunc="first")
    if firm_info is not None and "size_rank" in firm_info:
        wide = wide.join(firm_info.set_index(["t", "symbol"])["size_rank"], how="outer")
    wide = wide.reset_index()
    wide["t"] = pd.to_datetime(wide["t"])
    return wide


def tercile_legs(values: pd.Series, sign: int) -> tuple[list[str], list[str]]:
    """(long, short): the top and bottom thirds of sign x value over the names with a finite value; exact ties are
    ordered by the fixed symbol hash. Fewer than three names: no legs."""
    v = pd.to_numeric(values, errors="coerce").astype(float) * sign
    v = v[np.isfinite(v.to_numpy())]
    k = len(v) // 3
    if k < 1:
        return [], []
    order = sorted(v.index, key=lambda s: (-v[s], tiebreak_key(s)))
    return order[:k], order[-k:]


def factor_books(wide: pd.DataFrame, universe: Mapping, dates: Sequence) -> dict[str, tuple[dict, dict]]:
    """factor -> (long targets, short targets), {t: {symbol: weight}}, each leg equal-weighted over U_t members."""
    out: dict[str, tuple[dict, dict]] = {f: ({}, {}) for f in OWN_FACTORS}
    by_t = {pd.Timestamp(t): g.set_index("symbol") for t, g in wide.groupby("t")}
    for t in sorted(pd.Timestamp(d) for d in dates):
        g = by_t.get(t)
        uni = list(dict.fromkeys(universe[t]))
        for f, (col, sign) in OWN_FACTORS.items():
            if g is None or col not in g:
                continue
            lo, sh = tercile_legs(g[col].reindex(uni), sign)
            if lo and sh:
                out[f][0][t] = {s: 1.0 / len(lo) for s in lo}
                out[f][1][t] = {s: 1.0 / len(sh) for s in sh}
    return out


def own_factor_returns(prices: PriceBook, wide: pd.DataFrame, universe: Mapping, dates: Sequence,
                       end: pd.Timestamp | None = None, cash_rate_daily: pd.Series | None = None,
                       delisting_returns: Mapping[str, float] | None = None,
                       ma_exits: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Daily factor returns (sessions x [MOM, REV, LOWVOL, SIZE, VALUE, PROF, MKT_EW, MKT_SPY]); the market columns
    are in excess of `cash_rate_daily`. Legs are held like the books (engine, fills at the next open, no costs)."""
    kw = {"cost_bps": 0.0, "delisting_returns": dict(delisting_returns or {}), "end": end, "ma_exits": ma_exits}
    cols = {}
    for f, (lo, sh) in factor_books(wide, universe, dates).items():
        if lo and sh:
            rl = run_backtest(prices, lo, **kw).daily_returns
            rs = run_backtest(prices, sh, **kw).daily_returns
            cols[f] = rl.sub(rs, fill_value=np.nan)
    booked = sorted(pd.Timestamp(d) for d in dates)
    ew = {t: {s: 1.0 / len(universe[t]) for s in universe[t]} for t in booked}
    r_ew = run_backtest(prices, ew, **kw).daily_returns
    r_spy = run_backtest(prices, {t: {"SPY": 1.0} for t in booked}, **kw).daily_returns
    rf = cash_rate_daily.reindex(r_ew.index).fillna(0.0) if cash_rate_daily is not None else 0.0
    cols["MKT_EW"] = r_ew - rf
    cols["MKT_SPY"] = r_spy - rf
    return pd.DataFrame(cols).sort_index()


def hac_ols(y: pd.Series, X: pd.DataFrame, lags: int = 5) -> dict:
    """OLS with an intercept and Newey-West (Bartlett, `lags`) standard errors. Returns alpha, betas, SEs, t, R2."""
    d = pd.concat([y.rename("__y"), X], axis=1, join="inner").dropna()
    n = len(d)
    k = X.shape[1] + 1
    if n <= k + 2:
        return {"n": n}
    Y = d["__y"].to_numpy(dtype=float)
    Z = np.column_stack([np.ones(n), d[list(X.columns)].to_numpy(dtype=float)])
    ZZi = np.linalg.pinv(Z.T @ Z)
    b = ZZi @ Z.T @ Y
    e = Y - Z @ b
    g = Z * e[:, None]
    S = g.T @ g
    for L in range(1, min(lags, n - 1) + 1):
        w = 1.0 - L / (lags + 1.0)
        G = g[L:].T @ g[:-L]
        S += w * (G + G.T)
    V = ZZi @ S @ ZZi
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    tss = float(((Y - Y.mean()) ** 2).sum())
    names = ["alpha", *X.columns]
    return {"n": n, "coef": dict(zip(names, map(float, b))), "se": dict(zip(names, map(float, se))),
            "t": dict(zip(names, map(float, b / np.where(se > 0, se, np.nan)))),
            "r2": 1.0 - float(e @ e) / tss if tss > 0 else float("nan"), "lags": lags}


def _weekly(x: pd.DataFrame | pd.Series):
    iso = x.index.isocalendar()
    return x.groupby([iso.year.values, iso.week.values]).sum()


def style_attribution(book_returns: pd.Series, rf: pd.Series | float, factors: pd.DataFrame, lags: int = 5,
                      weekly_lags: int = 4, market: str = "MKT_EW") -> dict:
    """PE-3 regression of the book's daily excess return on the own factors (one market column, `market`) with
    NW(lags); the weekly sums are the robustness check. Selection alpha annualized with a 95% normal CI."""
    ex = (book_returns - (rf.reindex(book_returns.index).fillna(0.0) if isinstance(rf, pd.Series) else rf)).rename("book")
    keep = [c for c in factors.columns if not c.startswith("MKT_") or c == market]
    out = {}
    for label, y, X, L, per in (("daily", ex, factors[keep], lags, 252),
                                ("weekly", _weekly(ex), _weekly(factors[keep]), weekly_lags, 52)):
        r = hac_ols(y, X, L)
        if "coef" in r:
            a, s = r["coef"]["alpha"], r["se"]["alpha"]
            r["alpha_annualized"] = a * per
            r["alpha_ci95_annualized"] = ((a - 1.96 * s) * per, (a + 1.96 * s) * per)
            r["alpha_p_two_sided"] = float(2 * stats.norm.sf(abs(a / s))) if s > 0 else float("nan")
        out[label] = r
    out["factors"] = keep
    return out


# ------------------------------------------------------------------------------------ PE-4 French library
def _isnum(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def parse_french_csv(text: str) -> pd.DataFrame:
    """The first daily table of a Kenneth R. French library CSV (module doc), as decimals indexed by date."""
    lines = text.replace("﻿", "").splitlines()
    header: list[str] | None = None
    rows: list[list] = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            if len(cells) >= 2 and cells[0] == "" and all(c and not _isnum(c) for c in cells[1:] if c):
                nxt = next((ln for ln in lines[i + 1:] if ln.strip()), "")
                if re.match(r"^\s*\d{8}\s*,", nxt):
                    header = [c for c in cells[1:] if c]
            continue
        if re.fullmatch(r"\d{8}", cells[0] or ""):
            rows.append([cells[0], *[float(c) if _isnum(c) else np.nan for c in cells[1:1 + len(header)]]])
        elif rows:
            break
    if header is None or not rows:
        raise ValueError("no daily table in the French library file")
    df = pd.DataFrame(rows, columns=["date", *header])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    df = df.mask(df.isin(FRENCH_MISSING))
    return df / 100.0


def _read_text(path: Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next(m for m in z.namelist() if m.lower().endswith((".csv", ".txt")))
            return z.read(member).decode("latin-1")
    return path.read_bytes().decode("latin-1")


def find_french_file(directory: Path, kind: str) -> Path | None:
    pat = FRENCH_PATTERNS[kind].lower()
    hits = sorted(p for p in Path(directory).glob("*") if pat in p.name.lower()
                  and p.suffix.lower() in (".csv", ".txt", ".zip"))
    csvs = [p for p in hits if p.suffix.lower() != ".zip"]
    return (csvs or hits or [None])[0]


def vintage_dir(directory: Path = FACTORS_DIR) -> Path:
    """The archived vintage to read: `directory/<CURRENT>` when `directory` holds a CURRENT pointer (the layout of
    scripts/download_ff_factors.py: data/raw/factors/<retrieval date>/ plus CURRENT), else `directory` itself."""
    directory = Path(directory)
    cur = directory / "CURRENT"
    return directory / cur.read_text().strip() if cur.exists() else directory


def load_french(directory: Path = FACTORS_DIR, ff5: Path | None = None, mom: Path | None = None) -> pd.DataFrame:
    """Daily Mkt-RF, SMB, HML, RMW, CMA, RF and Mom (decimals), inner-joined on dates. `attrs['provenance']` holds
    the vintage directory and each file's name, SHA-256 and first/last date (the vintage record of PE-4); when the
    vintage has a MANIFEST.json, every file's SHA-256 must equal the recorded one."""
    vdir = vintage_dir(directory)
    ff5 = ff5 or find_french_file(vdir, "ff5")
    mom = mom or find_french_file(vdir, "mom")
    if ff5 is None or mom is None:
        raise FileNotFoundError(f"French library files not found in {vdir} (need {sorted(FRENCH_PATTERNS.values())})")
    a, b = parse_french_csv(_read_text(ff5)), parse_french_csv(_read_text(mom))
    b = b.rename(columns={c: "Mom" for c in b.columns if c.lower().startswith("mom")})
    df = a.join(b[["Mom"]], how="inner")
    prov = {str(Path(p).name): {"sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest(),
                                "first": str(f.index.min().date()), "last": str(f.index.max().date())}
            for p, f in ((ff5, a), (mom, b))}
    man = Path(vdir) / "MANIFEST.json"
    if man.exists():
        import json

        recorded = {Path(v["file"]).name: v.get("sha256") for v in json.loads(man.read_text()).get("files", {}).values()}
        for name, rec in prov.items():
            if name in recorded and recorded[name] != rec["sha256"]:
                raise ValueError(f"{name}: SHA-256 differs from the vintage MANIFEST (file changed after download)")
    df.attrs["provenance"] = {"vintage_dir": str(vdir), **prov}
    return df


def french_attribution(book_returns: pd.Series, rf_book: pd.Series | float, french: pd.DataFrame, lags: int = 5,
                       window_end: pd.Timestamp | None = None) -> dict:
    """PE-4: the book's daily excess return (over its own cash rate, DFF) on Mkt-RF, SMB, HML, RMW, CMA and Mom, NW
    (lags), on the dates both cover; coverage against the evaluated window is reported (delta §3.8 caveat 3)."""
    ex = book_returns - (rf_book.reindex(book_returns.index).fillna(0.0) if isinstance(rf_book, pd.Series) else rf_book)
    cols = [c for c in ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom") if c in french]
    r = hac_ols(ex, french[cols], lags)
    if "coef" in r:
        a, s = r["coef"]["alpha"], r["se"]["alpha"]
        r["alpha_annualized"] = a * 252
        r["alpha_ci95_annualized"] = ((a - 1.96 * s) * 252, (a + 1.96 * s) * 252)
    last = french.index.max()
    r["coverage"] = {"factor_last_date": str(last.date()) if len(french) else None,
                     "book_first": str(book_returns.index.min().date()) if len(book_returns) else None,
                     "book_last": str(book_returns.index.max().date()) if len(book_returns) else None,
                     "window_end": str(pd.Timestamp(window_end).date()) if window_end is not None else None,
                     "covers_window": bool(window_end is None or last >= pd.Timestamp(window_end))}
    r["risk_free"] = {"book": "DFF/252 (FRED, point-in-time)", "factors": "one-month T-bill (library RF column)",
                      "mean_diff_daily": float((french["RF"].reindex(book_returns.index).dropna()
                                                - (rf_book.reindex(book_returns.index).dropna()
                                                   if isinstance(rf_book, pd.Series) else rf_book)).mean())
                      if "RF" in french and isinstance(rf_book, pd.Series) else None}
    r["provenance"] = french.attrs.get("provenance")
    return r


def alpha_ci(res: dict) -> tuple[float, float] | None:
    ci = res.get("alpha_ci95_annualized")
    return (float(ci[0]), float(ci[1])) if ci is not None and all(map(math.isfinite, ci)) else None
