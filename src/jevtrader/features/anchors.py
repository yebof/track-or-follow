"""Trusted anchor events for anchor-gated exclusion (design_v2_delta BG-3; question bank code_constants.exclusion_anchor).

A weight-0 exclusion of a name (sentinel or System-2) needs a trusted anchor A_{i,t} = A^price OR A^filing. Without
one, the exclusion becomes x0.5 (decide.risk). This module computes A from TRUSTED inputs only: adjusted closes (the
stock and SPY) and SEC 8-K acceptance times with item codes. Its functions take no news, no Layer-A answer and no
risk-table column, so no untrusted input can reach A (Proposition P0: A is part of the trusted objects Theta_t).

Price anchor. Design rule: with d0 the session of the sentinel item k* (created before 16:00 ET -> that session,
else the next), r_ex = cumulative return of i minus SPY from the close of d0 - 1 to the close of d_t, n = sessions
from d0 to d_t, sd = sd of i's daily returns in excess of SPY over the 60 sessions ending at d0 - 1;
A^price = 1 iff r_ex <= -2 sd sqrt(n). The session d0 comes from the creation time of an untrusted item, which an
attacker chooses. So that A stays a function of trusted inputs, d0 ranges over EVERY session an item of the week's
news window [c_{t-1}, c_t) can map to (the sessions after d_{t-1} up to d_t), and A^price = 1 iff the rule holds for
at least one of them. This is the attacker's best case of the design rule (any item time is reachable by planting
an item then), so it can only fire more often than the per-item rule, never less, and it cannot move under attack.

Filing anchor. An 8-K or 8-K/A of the issuer with acceptanceDateTime in [c_t - 7 calendar days, c_t) that lists any
item in {1.03, 2.04, 2.06, 3.01, 4.01, 4.02} (bankruptcy; triggering events that accelerate an obligation; material
impairments; delisting notice; change of accountant; non-reliance on financial statements). EDGAR acceptance times
are UTC (data.earnings.load_8k_items), so a filing counts from its acceptance instant, strictly before the cutoff.

Point in time: A_{i,t} uses closes up to d_t (known at c_t = 16:00 ET on d_t, the close) and filings accepted before
c_t (tests/test_anchor_rule.py perturbs everything after).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from jevtrader.data.news import decision_cutoff

SEVERE_8K_ITEMS = ("1.03", "2.04", "2.06", "3.01", "4.01", "4.02")
# The pre-registered text of the price anchor (question bank code_constants.exclusion_anchor.price_anchor, QB-13) is
# "<formula>; <d0 clause>". tests/test_anchor_rule.py compares both parts with the bank exactly. The formula is the
# design rule; the d0 clause is the rule implemented here (module doc): the bank's rc3 text still has the per-item d0
# ("d0 = session of the sentinel item (C2 rule)"), whose d0 an attacker chooses, so that test is a strict xfail until
# the bank is updated (docs/impl/day2_wsB2.md, cross-workstream requests).
PRICE_ANCHOR_FORMULA = ("cumulative return of the stock minus SPY from the close of d0-1 to the close of d_t <= "
                        "-2 * sd_60(daily excess return, sessions ending d0-1) * sqrt(sessions from d0 to d_t)")
PRICE_ANCHOR_D0 = ("d0 = every session an item of the week's news window [c_{t-1}, c_t) can map to (the sessions "
                   "after d_{t-1} up to d_t); A^price = 1 if the inequality holds for at least one such d0")
ANCHOR_COLUMNS =["t", "symbol", "anchor_price", "anchor_filing", "anchor", "z_min", "d0_min", "accessions"]


@dataclass(frozen=True)
class AnchorConfig:
    k: float = 2.0  # A^price: r_ex <= -k sd sqrt(n)
    sd_window: int = 60  # sessions ending at d0 - 1
    sd_min_periods: int = 40  # fewer excess returns in the window -> no price anchor for that d0
    market: str = "SPY"
    filing_window_days: int = 7  # [c_t - 7 calendar days, c_t)
    items: tuple[str, ...] = SEVERE_8K_ITEMS
    forms: tuple[str, ...] = ("8-K", "8-K/A")
    use_filing: bool = True  # QB-13 status: removed before the freeze if 8-K item codes are unavailable


def week_sessions(sessions: pd.DatetimeIndex, calendar: Sequence[pd.Timestamp], t) -> pd.DatetimeIndex:
    """Sessions an item of decision t's news window [c_{t-1}, c_t) can map to: after d_{t-1}, up to d_t. `calendar`
    is the weekly decision calendar of `sessions` (backtest.engine.weekly_decision_dates)."""
    t = pd.Timestamp(t)
    prev = [d for d in calendar if d < t]
    lo = prev[-1] if prev else sessions[0] - pd.Timedelta(days=1)
    return sessions[(sessions > lo) & (sessions <= t)]


def price_anchors(close: pd.DataFrame, decisions: Sequence, universe: Mapping, cfg: AnchorConfig = AnchorConfig(),
                  calendar: Sequence[pd.Timestamp] | None = None) -> pd.DataFrame:
    """A^price per (t, symbol in U_t) from adjusted closes (sessions x symbols, incl. the market column) only.
    Columns: t, symbol, anchor_price, z_min (min over d0 of r_ex / (sd sqrt(n)); NaN if no d0 is computable),
    d0_min (the d0 attaining z_min)."""
    from jevtrader.backtest.engine import weekly_decision_dates

    if cfg.market not in close:
        raise KeyError(f"market column {cfg.market!r} missing from the closes")
    close = close.sort_index()
    sessions = close.index
    cal = list(calendar) if calendar is not None else weekly_decision_dates(sessions, sessions[0], sessions[-1])
    c = close.to_numpy(dtype=float)
    col = {s: j for j, s in enumerate(close.columns)}
    jm = col[cfg.market]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = c[1:] / c[:-1] - 1.0
    ret = np.vstack([np.full((1, c.shape[1]), np.nan), ret])
    ex = pd.DataFrame(ret - ret[:, [jm]], index=sessions, columns=close.columns)
    sd = ex.rolling(cfg.sd_window, min_periods=cfg.sd_min_periods).std(ddof=1).to_numpy()  # sd[b]: returns b-59..b
    pos = {d: k for k, d in enumerate(sessions)}
    rows = []
    for t in sorted(pd.Timestamp(x) for x in decisions):
        if t not in pos:
            raise ValueError(f"{t.date()} is not a session of the price panel")
        kt = pos[t]
        syms = [s for s in dict.fromkeys(universe[t])]
        js = np.array([col.get(s, -1) for s in syms])
        have = js >= 0
        jj = np.where(have, js, 0)
        best_z = np.full(len(syms), np.inf)
        best_d0 = np.array([None] * len(syms), dtype=object)
        for d0 in week_sessions(sessions, cal, t):
            b = pos[d0] - 1  # close of d0 - 1
            if b < 0:
                continue
            n = kt - pos[d0] + 1
            with np.errstate(divide="ignore", invalid="ignore"):
                r_ex = (c[kt, jj] / c[b, jj] - 1.0) - (c[kt, jm] / c[b, jm] - 1.0)
                z = r_ex / (sd[b, jj] * math.sqrt(n))
            z = np.where(have & np.isfinite(z), z, np.inf)
            better = z < best_z
            best_z = np.where(better, z, best_z)
            best_d0[better] = d0
        z_min = np.where(np.isfinite(best_z), best_z, np.nan)
        hit = np.isfinite(best_z) & (best_z <= -cfg.k)
        rows += [(t, s, bool(h), float(z), d) for s, h, z, d in zip(syms, hit, z_min, best_d0)]
    out = pd.DataFrame(rows, columns=["t", "symbol", "anchor_price", "z_min", "d0_min"])
    return out


def load_filings(sub_dir: Path, symbols: Sequence[str], cfg: AnchorConfig = AnchorConfig()) -> dict[str, pd.DataFrame]:
    """symbol -> 8-K / 8-K/A filings listing any anchor item (accepted_utc, form, accession, items), from the archived
    EDGAR submissions (data.earnings.load_8k_items, incl. predecessor-entity and paginated files)."""
    from jevtrader.data.earnings import load_8k_items

    out = {}
    for s in sorted(set(symbols)):
        f = load_8k_items(Path(sub_dir), s, tuple(cfg.items))
        out[s] = f[f["form"].isin(cfg.forms)].reset_index(drop=True)
    return out


def filing_anchors(filings: Mapping[str, pd.DataFrame], decisions: Sequence, universe: Mapping,
                   cfg: AnchorConfig = AnchorConfig()) -> pd.DataFrame:
    """A^filing per (t, symbol in U_t): an anchor-item 8-K accepted in [c_t - window, c_t). Columns: t, symbol,
    anchor_filing, accessions (comma-joined)."""
    items = set(cfg.items)
    rows = []
    for t in sorted(pd.Timestamp(x) for x in decisions):
        c = decision_cutoff(t)
        lo = c - pd.Timedelta(days=cfg.filing_window_days)
        for s in dict.fromkeys(universe[t]):
            f = filings.get(s)
            acc = []
            if f is not None and len(f):
                at = pd.to_datetime(f["accepted_utc"], utc=True)
                ok = ((at >= lo) & (at < c) & f["form"].isin(cfg.forms)).to_numpy(copy=True)
                ok = ok & np.array([any(x in items for x in (it or ())) for it in f["items"]], dtype=bool)
                acc = list(f.loc[ok, "accession"])
            rows.append((t, s, bool(acc), ",".join(acc)))
    return pd.DataFrame(rows, columns=["t", "symbol", "anchor_filing", "accessions"])


def anchor_panel(close: pd.DataFrame, decisions: Sequence, universe: Mapping,
                 filings: Mapping[str, pd.DataFrame] | None, cfg: AnchorConfig = AnchorConfig(),
                 calendar: Sequence[pd.Timestamp] | None = None) -> pd.DataFrame:
    """A = A^price OR A^filing per (t, symbol in U_t). `filings=None` means the filing source is unavailable: the
    panel then has anchor_filing = False everywhere and `attrs['filing_source'] = False`, which Layer B's input check
    reports when an exclusion is at stake. Trusted inputs only (module doc)."""
    p = price_anchors(close, decisions, universe, cfg, calendar)
    if cfg.use_filing and filings is not None:
        f = filing_anchors(filings, decisions, universe, cfg)
        src = True
    else:
        f = p[["t", "symbol"]].assign(anchor_filing=False, accessions="")
        src = not cfg.use_filing  # the filing anchor is switched off by configuration, not missing
    out = p.merge(f, on=["t", "symbol"], how="left", validate="one_to_one")
    out["anchor_filing"] = out["anchor_filing"].fillna(False).astype(bool)
    out["accessions"] = out["accessions"].fillna("")
    out["anchor"] = out["anchor_price"] | out["anchor_filing"]
    out = out[ANCHOR_COLUMNS]
    out.attrs["filing_source"] = src
    return out


def anchors_for(panel: pd.DataFrame | None, t, names: Sequence[str]) -> pd.Series:
    """A_{i,t} as a 0/1 Series over `names`; a name without a row (or no panel) has no anchor."""
    if panel is None or panel.empty:
        return pd.Series(0.0, index=list(names))
    g = panel[panel["t"] == pd.Timestamp(t)].set_index("symbol")["anchor"]
    return g.reindex(list(names)).fillna(False).astype(float)
