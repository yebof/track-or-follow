"""Trial registry `results/trials.jsonl` (design §6.6, §13.6; evaluation_first E5).

Every return series Layer B computes is appended with its config hash and its *per-period* (daily, not annualized)
Sharpe ratio, because `eval.metrics.deflated_sharpe` needs the per-period SRs of every trial tried. The registry is
append-only; `trial_sharpes` counts each distinct return series of a window once (same window, dates and returns
hash), so neither a re-run nor a config change that leaves the returns unchanged inflates the trial count.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _plain(obj):
    """JSON-safe, canonical form: dataclasses -> dicts, tuples -> lists, non-finite floats -> strings."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else ("inf" if f > 0 else "-inf" if f < 0 else "nan")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def canonical_json(obj) -> str:
    return json.dumps(_plain(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def per_period_sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = (r - rf).dropna()
    sd = ex.std(ddof=1)
    return float(ex.mean() / sd) if len(ex) > 1 and sd > 0 else float("nan")


def register_trial(
    registry: Path, *, cfg_sha: str, name: str, window: str, returns: pd.Series, rf: pd.Series | float = 0.0,
    periods_per_year: int = 252, extra: dict | None = None,
) -> dict:
    """Append one trial record and return it."""
    ex = (returns - rf).dropna()
    sr = per_period_sharpe(returns, rf)
    rec = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cfg_sha": cfg_sha, "name": name, "window": window,
        "start": ex.index.min().isoformat() if len(ex) else None, "end": ex.index.max().isoformat() if len(ex) else None,
        "n_periods": len(ex), "sr_per_period": sr, "sr_annualized": sr * math.sqrt(periods_per_year),
        "mean": float(ex.mean()) if len(ex) else float("nan"), "sd": float(ex.std(ddof=1)) if len(ex) > 1 else float("nan"),
        "skew": float(stats.skew(ex)) if len(ex) > 2 else float("nan"),
        "kurtosis": float(stats.kurtosis(ex, fisher=False)) if len(ex) > 3 else float("nan"),
        "returns_sha": hashlib.sha256(np.ascontiguousarray(ex.to_numpy(dtype=float)).tobytes()).hexdigest()[:16],
        **(extra or {}),
    }
    registry = Path(registry)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a") as fh:
        fh.write(canonical_json(rec) + "\n")
    return rec


def load_trials(registry: Path) -> pd.DataFrame:
    p = Path(registry)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in p.read_text().splitlines() if line.strip()])


def trial_sharpes(registry: Path, window: str | None = None, names: Iterable[str] | None = None) -> list[float]:
    """Per-period SRs of the distinct registered trials (optionally one window / a set of trial names)."""
    tr = load_trials(registry)
    if tr.empty:
        return []
    if window is not None:
        tr = tr[tr["window"] == window]
    if names is not None:
        tr = tr[tr["name"].isin(list(names))]
    tr = tr.drop_duplicates(["window", "start", "end", "returns_sha"], keep="last")
    return [float(x) for x in pd.to_numeric(tr["sr_per_period"], errors="coerce") if np.isfinite(x)]
