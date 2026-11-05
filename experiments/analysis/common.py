"""Shared plumbing of the pre-registered analysis entry points (docs/design/preregistration_v1.md §14.3, P-8).

Every entry point under experiments/analysis/ follows the same four rules:

1. **Files only.** It reads result files (judgment tables, Layer-B run directories, RT-S score tables, the trial
   registry) and data files (bars, universe, SEC archive, macro), never a model or the network.
2. **The W_clean lock.** Before it reads anything that carries an outcome of a decision inside the locked window
   (a return, a label, an abnormal return, an 8-K label, an IC, a book, a learned coefficient, an attack success), it
   calls `Guard.check` on those decision dates, which is `pipeline.lock.assert_unlocked_for`: the call raises
   `LockedWindowError` until the pre-registration is committed and the key is issued. While the lock is closed, price
   bars are also read through a capped data root whose bars end on `PRICE_CAP` (experiments/pre/caproot.py), so no
   return after a W_clean cutoff is ever computed, even by accident.
3. **One output per analysis.** `write_result` writes `<out>/<name>.json` with a `meta` block: the script, its
   SHA-256 and its closure SHA-256 (the script plus every repository file it imports, as code_hashes.py), the
   window, the lock state, every input file with its SHA-256, the arguments and the time. Runs on the
   registered window (W_clean) go to results/analysis/<name>.json; any other window goes to
   results/analysis/<window>/<name>.json, so a development run can never be mistaken for a registered result
   (experiments/report/make_numbers.py fills a macro only from a registered-window file).
4. **No hand-set numbers.** Seeds, bootstrap sizes, lags and thresholds are module constants of the entry points and
   are hashed with the code (experiments/analysis/code_hashes.py -> results/analysis/CODE_HASHES.json).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "experiments" / "pre", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from jevtrader.pipeline.lock import LOCKED_WINDOW, LockedWindowError, assert_unlocked_for, unlock_status  # noqa: E402

REGISTERED_WINDOW = "W_clean"
ANALYSIS_DIR = Path("results/analysis")
SYSTEM_YAML = Path("configs/system_v1.yaml")
LEARNING_YAML = Path("configs/learning_v1.yaml")
BARS_FILES = ("data/raw/bars/sp500_daily_2018_2026.parquet", "data/raw/bars/etf_daily_2018_2026.parquet")
RAW_BARS_FILES = ("data/raw/bars/all_daily_raw_2018_2026.parquet",)
UNIVERSE_FILE = Path("data/processed/universe_monthly_top100.parquet")
SUBMISSIONS = Path("data/raw/sec/submissions")
PRIOR_LOOKS = Path("results/prereg/prior_look_stock_weeks.csv")
D1_RESULT = ANALYSIS_DIR / "d1.json"
CAPROOT_DIR = Path("data/cache/analysis_caproot")  # gitignored; rebuilt when the bar files change
PRICE_PAD_BEFORE = pd.Timedelta(days=550)  # as pipeline.layer_b.load_inputs (252-session betas, 12-1 momentum)
PRICE_PAD_AFTER = pd.Timedelta(days=60)
ET = "America/New_York"

__all__ = ["LOCKED_WINDOW", "ROOT", "Guard", "LockedWindowError", "out_path", "write_result"]


# ------------------------------------------------------------------------------------------------ small helpers
def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def plain(obj):
    """JSON-safe copy: numpy scalars to Python, NaN/inf to None, timestamps to ISO dates, tuples to lists."""
    if isinstance(obj, Mapping):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if (obj.tzinfo is not None or obj.time() != dt.time(0)) else obj.date().isoformat()
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return [plain(v) for v in obj.tolist()]
    return obj


def rel(path: Path | str, root: Path = ROOT) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------------------------------------ windows
def windows(root: Path = ROOT) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    cfg = Path(root) / SYSTEM_YAML
    if not cfg.exists():
        cfg = ROOT / SYSTEM_YAML
    w = yaml.safe_load(cfg.read_text())["windows"]
    return {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in w.items()}


def resolve_window(window: str, root: Path = ROOT) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A named window of configs/system_v1.yaml, or 'YYYY-MM-DD:YYYY-MM-DD'."""
    if ":" in window:
        a, b = window.split(":")
        return pd.Timestamp(a), pd.Timestamp(b)
    w = windows(root)
    if window not in w:
        raise KeyError(f"unknown window {window!r}; known: {sorted(w)}")
    return w[window]


def window_tag(window: str) -> str:
    return window.replace(":", "_")


def release_start(root: Path = ROOT) -> pd.Timestamp:
    return windows(root)["W_release"][0]


# ------------------------------------------------------------------------------------------------ the lock
@dataclass
class Guard:
    """The W_clean test lock for analysis code (module doc, rule 2)."""

    root: Path = ROOT

    def status(self) -> dict:
        ok, why = unlock_status(Path(self.root))
        return {"open": bool(ok), "why": why, "locked_window": [str(LOCKED_WINDOW[0].date()),
                                                                 str(LOCKED_WINDOW[1].date())]}

    @property
    def open(self) -> bool:
        return bool(unlock_status(Path(self.root))[0])

    def check(self, dates: Iterable, what: str) -> None:
        """Raise LockedWindowError if any date lies in the locked window and the lock is closed."""
        dates = [pd.Timestamp(d) for d in dates if d is not None and not pd.isna(d)]
        dates = [d.tz_convert(None) if d.tzinfo is not None else d for d in dates]
        try:
            assert_unlocked_for(dates, Path(self.root))
        except LockedWindowError as e:
            raise LockedWindowError(f"{what}: refused before reading any W_clean outcome ({e})") from None

    def check_window(self, window: str, what: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Refuse a window whose decisions or last holding week touch W_clean (as layer_b.evaluate)."""
        start, end = resolve_window(window, self.root)
        lo, hi = LOCKED_WINDOW
        if start <= hi and end + pd.Timedelta(days=7) >= lo:
            self.check([max(start, lo)], what)
        return start, end


# ------------------------------------------------------------------------------------------------ output
def registered_windows(root: Path = ROOT) -> set[str]:
    """The windows whose outputs are the registered results: W_clean, and, when the D1 rule moved the start, the
    moved window 'start:end' on which the primary cell and the H3 contrast must then be replayed (prereg §7.2)."""
    out = {REGISTERED_WINDOW}
    d1 = d1_start(root)
    w0, w1 = windows(root)[REGISTERED_WINDOW]
    if d1["applied"] and pd.Timestamp(d1["start"]) != w0:
        out.add(f"{pd.Timestamp(d1['start']).date()}:{w1.date()}")
    return out


def is_registered(window: str, root: Path = ROOT) -> bool:
    return window in registered_windows(root)


def out_path(name: str, window: str, out_dir: Path | str | None = None, root: Path = ROOT) -> Path:
    if out_dir is not None:
        return Path(out_dir) / f"{name}.json"
    base = Path(root) / ANALYSIS_DIR
    return base / f"{name}.json" if is_registered(window, root) else base / window_tag(window) / f"{name}.json"


def write_result(path: Path, payload: Mapping, *, script: str | Path, window: str, guard: Guard,
                 inputs: Mapping[str, Path | str | None] = (), argv: Sequence[str] | None = None,
                 extra_meta: Mapping | None = None) -> Path:
    """Write the analysis JSON with its provenance block (module doc, rule 3)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ins = {}
    for k, p in dict(inputs).items():
        if p is None:
            ins[k] = None
            continue
        p = Path(p)
        if p.is_file():
            ins[k] = {"path": rel(p, guard.root), "sha256": sha256_file(p)}
        elif p.is_dir():
            ins[k] = {"path": rel(p, guard.root), "dir": True}
        else:
            ins[k] = {"path": rel(p, guard.root), "missing": True}
    meta = {"script": rel(script), "script_sha256": sha256_file(script),
            "closure_sha256": closure_sha256(script), "window": window,
            "registered_window": is_registered(window, guard.root), "lock": guard.status(), "inputs": ins,
            "argv": list(argv) if argv is not None else list(sys.argv[1:]),
            "created_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), **dict(extra_meta or {})}
    path.write_text(json.dumps(plain({"meta": meta, **dict(payload)}), indent=1, sort_keys=False, allow_nan=False))
    return path


def closure_sha256(script: Path | str) -> str | None:
    """The entry point's closure hash as experiments/analysis/code_hashes.py computes it (the script and every
    repository file it imports, transitively): ties an output to the exact code that made it, not only to the entry
    script (audit 2026-09-28). None if the script is outside the repository."""
    import code_hashes as CH

    p = Path(script).resolve()
    try:
        own = CH.rel(p)
    except ValueError:
        return None
    files = {CH.rel(f): CH.sha256_file(f) for f in CH.closure(p)}
    return CH.group_hash({own: CH.sha256_file(p), **files})


def read_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------------------------------------ data root
def data_root(guard: Guard, requested: Path | str | None = None) -> Path:
    """Where bars are read from: the repository (or `requested`) when the lock is open; otherwise a capped mirror
    whose bar files end on caproot.PRICE_CAP, the last session before the first W_clean decision day."""
    import caproot

    base = Path(requested) if requested is not None else Path(guard.root)
    if guard.open:
        return base
    if not all((base / f).exists() for f in caproot.BAR_FILES):
        return base  # synthetic roots carry their own bars; load_prices still caps them
    return caproot.capped_root(Path(guard.root) / CAPROOT_DIR, src=base)


def price_cap(guard: Guard) -> pd.Timestamp | None:
    import caproot

    return None if guard.open else caproot.PRICE_CAP


def _read_bars(files: Sequence[Path], lo: pd.Timestamp | None, hi: pd.Timestamp | None,
               columns: Sequence[str] | None = None) -> pd.DataFrame:
    frames = []
    for f in files:
        if not Path(f).exists():
            continue
        df = pd.read_parquet(f, columns=list(columns) if columns else None)
        d = pd.to_datetime(df["date"])
        keep = np.ones(len(df), dtype=bool)
        if lo is not None:
            keep &= (d >= lo).to_numpy()
        if hi is not None:
            keep &= (d <= hi).to_numpy()
        df = df[keep].copy()
        df["date"] = d[keep]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no bar file among {[str(f) for f in files]}")
    b = pd.concat(frames, ignore_index=True)
    b["symbol"] = b["symbol"].astype(str)
    return b


@dataclass
class Prices:
    """A PriceBook over a padded window plus the weekly decision calendar and U_t, read from files."""
    book: object  # backtest.engine.PriceBook
    dates: list[pd.Timestamp]
    universe: dict[pd.Timestamp, tuple[str, ...]]
    cap: pd.Timestamp | None
    root: Path


def load_prices(guard: Guard, start, end, *, droot: Path | None = None, pad_before=PRICE_PAD_BEFORE,
                pad_after=PRICE_PAD_AFTER) -> Prices:
    """Adjusted bars over [start - pad_before, end + pad_after] cut at each symbol's life end, as
    pipeline.layer_b.load_inputs; the weekly decision dates in [start, end] and U_t. With the lock closed the bars end
    on the price cap whatever the window (module doc, rule 2)."""
    from jevtrader.backtest.engine import PriceBook, weekly_decision_dates
    from jevtrader.data.prices import cut_at, life_ends, sessions_from
    from jevtrader.pipeline.layer_b import universe_by_date

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    droot = Path(droot) if droot is not None else data_root(guard)
    cap = price_cap(guard)
    hi = end + pad_after
    if cap is not None:
        hi = min(hi, cap)
    lo = start - pad_before
    bars = _read_bars([droot / f for f in BARS_FILES], None, cap)
    raw_files = [droot / f for f in RAW_BARS_FILES]
    life_src = (_read_bars(raw_files, None, cap, columns=["symbol", "date"])
                if all(f.exists() for f in raw_files) else bars[["symbol", "date"]])
    bars = cut_at(bars, life_ends(life_src, sessions_from(life_src)))
    bars = bars[(bars["date"] >= lo) & (bars["date"] <= hi)]
    book = PriceBook(bars)
    dates = weekly_decision_dates(book.sessions, str(start.date()), str(end.date()))
    uni_path = droot / UNIVERSE_FILE
    universe = universe_by_date(pd.read_parquet(uni_path), dates) if uni_path.exists() else {}
    return Prices(book=book, dates=dates, universe=universe, cap=cap, root=droot)


def load_labels(prices: Prices, dates: Sequence | None = None, universe: Mapping | None = None, *,
                with_sigma: bool = False):
    """T1 labels (learn.labels.LabelStore) of `dates` over `universe` (default: the window's U_t), with the M&A
    exits of the SEC archive as pipeline.layer_b.load_inputs; `with_sigma` adds the ex-ante idiosyncratic vol (T3)."""
    from jevtrader.learn.labels import LabelStore, idio_vol_weekly
    from jevtrader.pipeline import inputs as I

    universe = dict(universe if universe is not None else prices.universe)
    dates = [pd.Timestamp(d) for d in (dates if dates is not None else sorted(universe))]
    universe = {pd.Timestamp(k): tuple(v) for k, v in universe.items() if pd.Timestamp(k) in set(dates)}
    ma = I.ma_exit_set(prices.root, sorted({s for v in universe.values() for s in v}))
    sigma = None
    if with_sigma:
        lab_cfg = yaml.safe_load((ROOT / LEARNING_YAML).read_text())["labels"]
        iv = lab_cfg["idio_vol"]
        sigma = idio_vol_weekly(prices.book.close, market=iv["market"], com=iv["ewma_center_of_mass_days"],
                                beta_window=iv["beta_window_sessions"], min_periods=iv["beta_min_periods"],
                                ewm_min_periods=iv["ewma_min_periods"])
    return LabelStore.build(prices.book, sorted(universe), universe, sigma, ma_exits=ma)


def cash_rate(prices: Prices) -> pd.Series | None:
    from jevtrader.pipeline.layer_b import load_cash_rate

    return load_cash_rate(prices.root, prices.book.sessions)


# ------------------------------------------------------------------------------------------------ Layer-B run dirs
def eval_dirs(root: Path, window: str, *, name: str | None = None, gen_id: str | None = None) -> list[Path]:
    """Layer-B run directories results/eval/<cfg16>/<window>__<inputs12>/ with a config.json (optionally filtered by
    the config name and the generation)."""
    out = []
    for d in sorted((Path(root) / "results/eval").glob(f"*/{window_tag(window)}__*")):
        c = d / "config.json"
        if not c.exists():
            continue
        cfg = json.loads(c.read_text())
        if name is not None and cfg.get("name") != name:
            continue
        if gen_id is not None and cfg.get("gen_id") != gen_id:
            continue
        out.append(d)
    return out


def find_eval_dir(root: Path, window: str, name: str, gen_id: str) -> Path:
    found = eval_dirs(root, window, name=name, gen_id=gen_id)
    if len(found) != 1:
        raise FileNotFoundError(f"expected one Layer-B run '{name}' of {gen_id} on {window}, found "
                                f"{[rel(p, root) for p in found]}; pass the directory explicitly")
    return found[0]


@dataclass
class EvalRun:
    """One Layer-B run directory, read after the lock check on its decision dates."""
    path: Path
    config: dict
    metrics: dict
    predictions: pd.DataFrame
    window: str

    @property
    def name(self) -> str:
        return str(self.config.get("name"))

    @property
    def dates(self) -> list[pd.Timestamp]:
        return sorted(pd.to_datetime(self.predictions["t"]).unique())

    @property
    def learned(self) -> pd.DataFrame:
        p = self.predictions
        return p[~p["cold"].astype(bool)]

    def file(self, name: str) -> Path:
        return self.path / name


def read_eval_run(path: Path | str, guard: Guard, what: str) -> EvalRun:
    """The run's window comes from its manifest (never from the caller); the lock check runs on that window before
    any outcome-bearing file (predictions, metrics, weights, returns) is opened."""
    path = Path(path)
    man = json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else {}
    window = man.get("window") or path.name.split("__")[0]
    if ":" not in window and "_" in window and window not in windows(guard.root):
        window = window.replace("_", ":", 1) if window.count("-") == 4 else window
    guard.check_window(window, f"{what} ({rel(path, guard.root)})")
    cfg = json.loads((path / "config.json").read_text())
    metrics = json.loads((path / "metrics.json").read_text())
    pred = pd.read_parquet(path / "predictions.parquet")
    pred["t"] = pd.to_datetime(pred["t"])
    guard.check(pred["t"].unique(), f"{what} ({rel(path, guard.root)})")
    return EvalRun(path=path, config=cfg, metrics=metrics, predictions=pred, window=window)


# ------------------------------------------------------------------------------------------------ registered inputs
def d1_start(root: Path = ROOT, path: Path | None = None) -> dict:
    """The confirmatory window start after the D1 rule (prereg §7.2): results/analysis/d1.json when it exists and the
    rule was applied, else the first decision of W_clean."""
    p = Path(path) if path is not None else Path(root) / D1_RESULT
    w0 = windows(root)["W_clean"][0]
    if not p.exists():
        return {"start": w0, "applied": False, "source": None, "reason": "results/analysis/d1.json not written"}
    d = json.loads(p.read_text())
    rule = d.get("rule", {})
    return {"start": pd.Timestamp(rule.get("start", w0)), "applied": bool(rule.get("applied", False)),
            "source": rel(p, root), "reason": rule.get("reason")}


def require_d1_start(first_date, root: Path, what: str, window: str) -> dict:
    """With the D1 rule applied on the registered window, a confirmatory frame must begin at the rule's start (prereg
    §7.2: the primary cell, the H1 unit frame, the H2 item frame and the H3 weeks all move with it). Returns the rule
    record; exits if `first_date` (the frame's first decision) differs from the start."""
    if window != REGISTERED_WINDOW:
        return {"applied": False, "reason": "not the registered window"}
    d1 = d1_start(root)
    if d1["applied"] and pd.Timestamp(first_date) != pd.Timestamp(d1["start"]):
        raise SystemExit(f"{what}: the D1 rule starts the confirmatory window on {pd.Timestamp(d1['start']).date()}, "
                         f"but the input starts on {pd.Timestamp(first_date).date()}; rerun it on the moved window")
    return d1


def prior_look_keys(root: Path = ROOT) -> set[tuple[pd.Timestamp, str]]:
    p = Path(root) / PRIOR_LOOKS
    if not p.exists():
        p = ROOT / PRIOR_LOOKS
    df = pd.read_csv(p)
    return {(pd.Timestamp(t), str(s)) for t, s in zip(df["t"], df["symbol"])}


def subset(universe: Sequence[str], name: str, root: Path = ROOT) -> tuple[str, ...]:
    """S30 / S50 of U_t by the design's hash rule (pipeline.inputs.subset_names with the configured salt)."""
    from jevtrader.pipeline.inputs import load_subsets, subset_names

    cfg = Path(root) / SYSTEM_YAML
    salt, n = load_subsets(cfg if cfg.exists() else ROOT / SYSTEM_YAML)[name]
    return subset_names(universe, n, salt)


def first_n_weeks_dropped(weeks: Sequence, n: int = 13) -> list[pd.Timestamp]:
    """Weeks after the first `n` of the given weeks (a count; the registered drop-13 sensitivity uses `drop_start`,
    a date)."""
    w = sorted(pd.Timestamp(x) for x in weeks)
    return w[n:]


DROP_FIRST = 13  # prereg §7.1: the drop-13 sensitivity


def decision_calendar(root: Path, start, end) -> list[pd.Timestamp]:
    """Weekly decision dates (the last session of each ISO week) in [start, end] from the session calendar (the date
    column of SPY in the ETF bar file; no price is read)."""
    from jevtrader.backtest.engine import weekly_decision_dates

    f = Path(root) / BARS_FILES[1]
    if not f.exists():
        f = ROOT / BARS_FILES[1]
    b = pd.read_parquet(f, columns=["symbol", "date"])
    ses = pd.DatetimeIndex(sorted(pd.to_datetime(b.loc[b["symbol"] == "SPY", "date"]).unique()))
    return weekly_decision_dates(ses, str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))


def drop_start(window: str, root: Path = ROOT, n: int = DROP_FIRST) -> pd.Timestamp:
    """The first decision kept by the drop-13 sensitivity (prereg §7.1), a DATE fixed by the configs: the (n+1)-th
    weekly decision of W_clean (configs/system_v1.yaml windows.W_clean on the session calendar) = 2024-10-04 for the
    registered window, whether or not the D1 rule moved the start (§7.2: the drop-13 sensitivity is unchanged by a
    move). It is a sub-series of the primary run (the learner still starts cold at the window start; decisions before
    this date leave the statistic), not a cold-started replay. A development window uses its own (n+1)-th decision."""
    name = REGISTERED_WINDOW if is_registered(window, root) else window
    w0, w1 = resolve_window(name, root)
    cal = decision_calendar(root, w0, w1)
    return cal[n] if len(cal) > n else pd.Timestamp(w1) + pd.Timedelta(days=1)


def load_config(name: Path, root: Path = ROOT) -> dict:
    p = Path(root) / name
    if not p.exists():
        p = ROOT / name
    return yaml.safe_load(p.read_text())
