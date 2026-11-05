"""W_clean test lock (design §2.5, §13.6; evaluation_first E4).

No outcome-dependent Layer-B computation may touch a decision inside the locked window until the pre-registration is
committed. The key is `results/locks/clean_unlocked`, a JSON file holding the path and SHA-256 of the pre-registration
(`configs/prereg_v1.yaml`). The lock opens only while that file exists *and* the pre-registration still has the
recorded hash, so editing the pre-registration after the unlock closes the lock again until the change is logged and
the key re-issued (docs/design/prereg_deviations.md). The check is on decision dates, not window names, so a custom
window that overlaps W_clean is locked too (W_release lies inside W_clean).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from jevtrader.pipeline.registry import file_sha256

LOCKED_WINDOW = (pd.Timestamp("2024-07-05"), pd.Timestamp("2026-09-11"))
LOCK_FILE = Path("results/locks/clean_unlocked")
PREREG = Path("configs/prereg_v1.yaml")


class LockedWindowError(RuntimeError):
    pass


def unlock_status(root: Path, lock_file: Path = LOCK_FILE) -> tuple[bool, str]:
    key = Path(root) / lock_file
    if not key.exists():
        return False, f"{lock_file} missing"
    try:
        rec = json.loads(key.read_text())
        prereg = Path(root) / rec["prereg"]
        want = rec["sha256"]
    except (ValueError, KeyError) as e:
        return False, f"malformed lock key: {e}"
    if not prereg.exists():
        return False, f"pre-registration {rec['prereg']} missing"
    have = file_sha256(prereg)
    if have != want:
        return False, f"pre-registration changed since unlock ({have[:12]} != {want[:12]})"
    return True, "unlocked"


def touches_locked(dates: Iterable, window: tuple[pd.Timestamp, pd.Timestamp] = LOCKED_WINDOW) -> bool:
    lo, hi = window
    return any(lo <= pd.Timestamp(d) <= hi for d in dates)


def range_touches_locked(start, end, window: tuple[pd.Timestamp, pd.Timestamp] = LOCKED_WINDOW) -> bool:
    return pd.Timestamp(start) <= window[1] and pd.Timestamp(end) >= window[0]


def assert_unlocked_for(dates: Iterable, root: Path, window: tuple[pd.Timestamp, pd.Timestamp] = LOCKED_WINDOW,
                        lock_file: Path = LOCK_FILE) -> None:
    """Raise LockedWindowError if any decision date lies in the locked window and the lock is closed."""
    dates = list(dates)
    if not touches_locked(dates, window):
        return
    ok, why = unlock_status(root, lock_file)
    if not ok:
        raise LockedWindowError(
            f"decisions in the locked window {window[0].date()}..{window[1].date()} require the pre-registration "
            f"unlock ({why})"
        )


def write_unlock(root: Path, prereg: Path = PREREG, lock_file: Path = LOCK_FILE) -> Path:
    """Issue the key after the pre-registration commit (run once by the orchestrator on 10-02)."""
    root = Path(root)
    p = root / prereg
    if not p.exists():
        raise FileNotFoundError(p)
    key = root / lock_file
    key.parent.mkdir(parents=True, exist_ok=True)
    key.write_text(json.dumps({
        "prereg": str(prereg), "sha256": file_sha256(p),
        "unlocked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1))
    return key
