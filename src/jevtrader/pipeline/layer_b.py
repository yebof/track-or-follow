"""Layer B: replay the decision layer over Layer A's judgment tables (design §4.1-4.2, §12.3). Pure Python, no network
(tests/test_pipeline_no_network_layer_b.py).

Weekly cycle for decision t (cutoff c_t = 16:00 ET on d_t):
  1. standardize this week's raw features (causal standardizer; values are logged, never recomputed);
  2. settle and learn: fit the trusted and untrusted learners on ledger rows whose labels matured before c_t;
  3. score: L^T, L^W, cap, L, P, contributions; log L^T as the as-of offset;
  4. decide: D-sel-EW (band guard) and D-sized (incumbents, vetoes, inverse-vol, caps, turnover); cards.
Until the first label matures (decision 3 under P-clean) every book holds EW-U100 (§2.5).

Inputs are the judgment tables only (plus prices and the universe): `answers.parquet` (trusted S1 answers, mapped by
learn.features.features_from_answers), `features.parquet` (Layer A's code-aggregated features such as the news,
sentinel and policy features: `t, symbol, feature, value, covered`) and optionally `code_features.parquet`.

Missing inputs never degrade silently (`input_report`): a decision week without judgment rows, a configured feature
with no data, a coverage hole, an unmapped judgment-table name, or a missing industry map, risk table, cash rate, sigma
or price panel, or a risk table that does not line up with the decisions, stops the run unless the config waives that
check in `allow_missing` (recorded with its trials). A trusted catalog feature read from a row that Layer A marked as
tainted (not T) is refused outright (`load_judgment_features`).

Protocol: Layer B runs P-clean only (§2.5): every learner cold-starts at the first decision of the evaluated window and
learns only from that window's matured labels. P-full (expanding window from 2020) is used only by the non-LLM
baselines (baselines/ml.py), so it is not a Layer-B option.

Day 2 (design_v2_delta): D-sized uses reference-anchored sizing (BG-2) and anchor-gated exclusion (BG-3) with the
trusted anchor panel of features/anchors.py; `decide_week` is the single decision function that the replay, the
bound checks (eval/bound.py) and the per-reader checks share; per-agent as-of heads are logged for the composition
variants (learn/heads.py, fusion/pools.py); `load_inputs` assembles the industry map, the risk table, the M&A exits
and the anchors from files (pipeline/inputs.py).
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from jevtrader.backtest.engine import DelistingError, PriceBook, run_backtest, weekly_decision_dates
from jevtrader.data.news import decision_cutoff
from jevtrader.data.prices import RAW_BARS, cut_at, life_ends, sessions_from
from jevtrader.decide.cards import build_cards
from jevtrader.decide.risk import (
    RiskConfig,
    Weights,
    equal_weights,
    risk_multipliers,
    size,
    vetoed_names,
)
from jevtrader.decide.select import TIEBREAK_SALT, Selection, select
from jevtrader.eval import metrics as M
from jevtrader.features.anchors import AnchorConfig, anchors_for
from jevtrader.learn.allocator import Allocator, AllocatorConfig, Ledger, attribution_bound, attribution_residual
from jevtrader.learn.features import (
    CATALOG,
    S1_TRUSTED_MAP,
    Standardizer,
    catalog_name,
    features_from_answers,
    specs_for,
)
from jevtrader.learn.heads import AgentHeads
from jevtrader.learn.labels import ET, LabelStore, open_time_utc
from jevtrader.pipeline import inputs as I
from jevtrader.pipeline.lock import LOCK_FILE, LOCKED_WINDOW, PREREG, assert_unlocked_for, range_touches_locked
from jevtrader.pipeline.registry import _plain, canonical_json, config_hash, file_sha256, register_trial

ROOT = Path(__file__).resolve().parents[3]
LEARNING_YAML = ROOT / "configs/learning_v1.yaml"
SYSTEM_YAML = ROOT / "configs/system_v1.yaml"
RAW_COLUMNS = ["t", "symbol", "feature", "value", "covered"]


@dataclass(frozen=True)
class DecisionConfig:
    name: str = "primary"
    gen_id: str = ""
    allocator: AllocatorConfig = AllocatorConfig()
    risk: RiskConfig = RiskConfig()
    K: int = 20
    J: float = 10.0
    cost_bps: float = 15.0
    std_trailing_weeks: int = 52
    std_min_history: int = 8
    std_floor: float = 0.05
    std_clip: float = 4.0
    noul_clip: float = 4.0
    allow_missing: tuple[str, ...] = ()  # input checks (INPUT_CHECKS) waived for this run; recorded with its trials
    config_files_sha: tuple[str, ...] = ()  # SHA-256 of learning_v1.yaml and system_v1.yaml when loaded from them
    anchors: AnchorConfig = AnchorConfig()  # BG-3 trusted anchor constants (question bank exclusion_anchor)
    heads: bool = True  # fit and log the per-agent as-of heads (composition variants, design §6.7)
    freeze_after_weeks: int | None = None  # V-static: slopes frozen once fitted on this many matured weeks

    @property
    def sha(self) -> str:
        return config_hash(self)

    @classmethod
    def from_yaml(cls, learning: Path = LEARNING_YAML, system: Path = SYSTEM_YAML, **overrides) -> DecisionConfig:
        lc, sc = yaml.safe_load(Path(learning).read_text()), yaml.safe_load(Path(system).read_text())
        check_frozen_constants(lc, sc)
        a, cap, st = lc["allocator"], lc["text_cap"], lc["standardization"]
        alloc = AllocatorConfig(
            trusted=tuple(lc["features"]["trusted"]), untrusted=tuple(lc["features"]["untrusted"]),
            prior_sd_beta=float(a["prior_sd_beta"]), prior_sd_gamma=float(a["prior_sd_gamma"]),
            half_life_weeks=float(a["half_life_weeks"]), cap_mult=float(cap["multiplier"]), cap_floor=float(cap["floor"]),
        )
        r, s = sc["risk"], sc["selection"]
        risk = RiskConfig(
            name_cap=r["name_cap"], industry_cap=r["industry_cap"], turnover_cap=r["turnover_cap"],
            earnings_mult=r["earnings_mult"], earnings_sessions=r["earnings_sessions"], sentinel_high=r["sentinel_high"],
            sentinel_single_mult=r["sentinel_single_mult"], corroboration=r["corroboration"],
            exclusion_kinds=tuple(r["exclusion_kinds"]), other_kind_mult=r["other_kind_mult"],
            incumbent_buffer=s["incumbent_buffer"], sizing=r.get("sizing", "reference"),
            exclusion_rule=r.get("exclusion_rule", "anchored"), key_column=r.get("key_column", "n_prob"),
            unanchored_mult=r.get("unanchored_mult", 0.5), s2_reduce_mult=r.get("s2_reduce_mult", 0.5),
        )
        an = sc.get("anchors", {})
        anchors = AnchorConfig(
            k=float(an.get("price_k", 2.0)), sd_window=int(an.get("sd_window_sessions", 60)),
            sd_min_periods=int(an.get("sd_min_periods", 40)), market=an.get("market", "SPY"),
            filing_window_days=int(an.get("filing_window_days", 7)),
            items=tuple(str(x) for x in an.get("filing_items", AnchorConfig().items)),
            forms=tuple(an.get("filing_forms", AnchorConfig().forms)), use_filing=bool(an.get("use_filing", True)),
        )
        base = cls(
            allocator=alloc, risk=risk, K=int(s["K"]), J=float(s["J"]), cost_bps=float(sc["costs"]["bps_per_side"]),
            std_trailing_weeks=int(st["trailing_weeks"]), std_min_history=int(st["min_history_weeks"]),
            std_floor=float(st["scale_floor"]), std_clip=float(st["clip"]),
            noul_clip=float(lc["feature_map"]["noul_logit_clip"]),
            config_files_sha=(file_sha256(learning), file_sha256(system)), anchors=anchors,
        )
        return dataclasses.replace(base, **overrides)


def check_frozen_constants(learning_cfg: dict, system_cfg: dict) -> None:
    """Settings of the frozen YAML files that the code keeps as module constants must equal them, so that editing the
    YAML (which changes the config hash) cannot leave behaviour unchanged, and vice versa. Raises ValueError."""
    sc, lc = system_cfg, learning_cfg

    def win(xs) -> tuple:
        return tuple(pd.Timestamp(x) for x in xs)

    cut = decision_cutoff(pd.Timestamp("2024-01-05")).tz_convert(ET).strftime("%H:%M")
    avail = open_time_utc(pd.Timestamp("2024-01-05")).tz_convert(ET).strftime("%H:%M")
    pairs = {
        "selection.tiebreak_salt": (sc["selection"]["tiebreak_salt"], TIEBREAK_SALT),
        "lock.locked_window": (win(sc["lock"]["locked_window"]), LOCKED_WINDOW),
        "windows.W_clean": (win(sc["windows"]["W_clean"]), LOCKED_WINDOW),
        "lock.lock_file": (Path(sc["lock"]["lock_file"]), LOCK_FILE),
        "lock.prereg": (Path(sc["lock"]["prereg"]), PREREG),
        "clock.cutoff_local": (str(sc["clock"]["cutoff_local"]), cut),
        "clock.label_available_local": (str(sc["clock"]["label_available_local"]), avail),
        "labels.primary": (lc["labels"]["primary"], "T1"),
    }
    bad = {k: v for k, (v, want) in pairs.items() if v != want}
    if bad:
        raise ValueError(f"frozen YAML settings differ from the code constants they must equal: {bad}")


def windows(system: Path = SYSTEM_YAML) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    w = yaml.safe_load(Path(system).read_text())["windows"]
    return {k: (pd.Timestamp(a), pd.Timestamp(b)) for k, (a, b) in w.items()}


def resolve_window(window: str, system: Path = SYSTEM_YAML) -> tuple[pd.Timestamp, pd.Timestamp]:
    """A named window of configs/system_v1.yaml, or 'YYYY-MM-DD:YYYY-MM-DD'."""
    if ":" in window:
        a, b = window.split(":")
        return pd.Timestamp(a), pd.Timestamp(b)
    w = windows(system)
    if window not in w:
        raise KeyError(f"unknown window {window!r}; known: {sorted(w)}")
    return w[window]


# ------------------------------------------------------------------------------------ inputs
@dataclass
class LayerBInputs:
    raw: pd.DataFrame  # t, symbol, feature, value, covered: raw (unstandardized) features of every week
    universe: Mapping[pd.Timestamp, Sequence[str]]  # U_t per decision date
    labels: LabelStore
    sigma: pd.DataFrame | None = None  # sessions x symbols, ex-ante weekly idiosyncratic vol (PIT)
    industry: Mapping[str, str] | None = None  # symbol -> FF12 code
    risk: pd.DataFrame | None = None  # t, symbol, sessions_to_earnings, h, n_prob, problem_kind, s2_mult
    prices: PriceBook | None = None
    cash_rate_daily: pd.Series | None = None  # DFF / 252, point-in-time
    delisting_returns: Mapping[str, float] | None = None  # explicit returns of non-M&A delistings (V-m14)
    ma_exits: frozenset[str] = frozenset()  # known M&A exits (cash at the last close)
    evidence: Mapping[pd.Timestamp, Mapping[str, dict]] | None = None  # card evidence blocks from Layer A
    provenance: Mapping[pd.Timestamp, Mapping[str, dict]] | None = None
    anchors: pd.DataFrame | None = None  # trusted anchor panel (features.anchors): t, symbol, anchor, ...
    # the names Layer A was asked about per decision date (a subset run: S50/S30 or a symbol list; pipeline.inputs.
    # load_focus); None = every U_t name. Only the coverage gate reads it (input_report); decisions never do.
    focus: Mapping[pd.Timestamp, tuple[str, ...]] | None = None


def load_judgment_features(gen_dir: Path, industry: Mapping[str, str], noul_clip: float = 4.0) -> pd.DataFrame:
    """Raw long features `t, symbol, feature, value, covered` from a judgment-table directory.

    Layer A's φ-mapped `features.parquet` (names mapped onto the catalog; a null value is an abstention) takes
    precedence; the trusted S1 features it lacks are φ-mapped here from `answers.parquet`; `code_features.parquet`
    (code factors) is added if present. Names outside the catalog are dropped and listed in `raw.attrs['unmapped']`.

    Trust is checked against the recorded taint, not only the name: when a table carries Layer A's per-row `taint`
    (programs/runner.py), a row that maps onto a trusted (channel T) catalog feature must have taint T, else
    ValueError. Without this, a W-tainted '<program>.x_trend' would become the trusted x_trend (P0 rests on it).

    `raw.attrs['layer_a_features']` lists the features read from Layer A's tables (features.parquet, answers.parquet):
    their values exist only for the run's focus names, so a subset run's coverage gate measures them over the focus;
    code_features.parquet covers every U_t name. `raw.attrs['feature_programs']` maps each Layer-A feature to the
    Layer-A program(s) that produced it (features.parquet's `program` column; the answers' program for the trusted S1
    features), so the gate can name the program whose weeks are missing (`input_report`, check `layer_a_weeks`)."""
    gen_dir = Path(gen_dir)
    frames, unmapped, layer_a = [], set(), set()
    programs: dict[str, set[str]] = {}
    for name in ("features.parquet", "code_features.parquet"):
        if (gen_dir / name).exists():
            f = pd.read_parquet(gen_dir / name)
            mapped = f["feature"].astype(str).map(catalog_name)
            unmapped |= set(f.loc[mapped.isna(), "feature"].astype(str))
            if "taint" in f:
                trusted = mapped.map(lambda c: c is not None and CATALOG[c].channel == "T").astype(bool)
                bad = f[trusted & f["taint"].astype(str).ne("T")]  # a missing taint is not T either
                if len(bad):
                    ex = sorted({f"{a} (taint {b})" for a, b in zip(bad["feature"].astype(str), bad["taint"])})[:5]
                    raise ValueError(f"{name}: {len(bad)} rows map onto a trusted catalog feature but carry a "
                                     f"non-T taint: {ex}")
            f = f.assign(feature=mapped)[mapped.notna()]
            f["value"] = pd.to_numeric(f["value"], errors="coerce")
            if "covered" not in f:
                f["covered"] = np.isfinite(f["value"].to_numpy(dtype=float))
            if name == "features.parquet":
                layer_a |= set(f["feature"])
                if "program" in f:
                    for feat, prog in f[["feature", "program"]].drop_duplicates().itertuples(index=False):
                        programs.setdefault(feat, set()).add(str(prog))
            frames.append(f[RAW_COLUMNS])
    have = set(pd.concat(frames)["feature"]) if frames else set()
    if (gen_dir / "answers.parquet").exists():
        # the forecast-type views (tech_view, fund_view) are mapped too: only V-outlook configures them, and without
        # them that registered variant could never run (integration check 09-27)
        missing = [f for f in S1_TRUSTED_MAP if f not in have]
        if missing:
            fa = features_from_answers(pd.read_parquet(gen_dir / "answers.parquet"), industry, missing,
                                       clip=noul_clip)
            layer_a |= set(fa["feature"])
            for feat in set(fa["feature"]):
                programs.setdefault(feat, set()).add(S1_TRUSTED_MAP[feat][0])
            frames.append(fa)
    if not frames:
        raise FileNotFoundError(f"no judgment tables in {gen_dir}")
    raw = pd.concat(frames, ignore_index=True)
    raw["t"] = pd.to_datetime(raw["t"])
    dup = raw.duplicated(["t", "symbol", "feature"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate (t, symbol, feature) rows across judgment tables")
    raw.attrs["unmapped"] = sorted(unmapped)
    raw.attrs["layer_a_features"] = sorted(layer_a)
    raw.attrs["feature_programs"] = {f: sorted(p) for f, p in sorted(programs.items()) if f in layer_a}
    return raw


BARS_FILES = ("data/raw/bars/sp500_daily_2018_2026.parquet", "data/raw/bars/etf_daily_2018_2026.parquet")
UNIVERSE_FILE = "data/processed/universe_monthly_top100.parquet"


def universe_by_date(uni: pd.DataFrame, dates: Sequence[pd.Timestamp], top: int = 100) -> dict[pd.Timestamp, tuple]:
    """U_t = the monthly PIT top-`top` membership in force at d_t (latest reconstitution date <= d_t)."""
    u = uni.copy()
    u["date"] = pd.to_datetime(u["date"])
    u = u[u["rank"] < top]
    recon = np.array(sorted(u["date"].unique()))
    by = {d: tuple(g.sort_values("rank")["symbol"]) for d, g in u.groupby("date")}
    out = {}
    for t in dates:
        k = recon.searchsorted(np.datetime64(t), side="right") - 1
        if k >= 0:
            out[pd.Timestamp(t)] = by[pd.Timestamp(recon[k])]
    return out


def symbol_lives(root: Path, adjusted: pd.DataFrame) -> pd.Series:
    """symbol -> end of its life (first gap > 10 sessions, V-m2), so that a reused ticker (APC, FB, NFX) is never
    forward-filled into the old company's life (the PriceBook precondition). Computed on the raw bars when they are
    on disk, the same cut as data.prices.build_panels and the baselines; otherwise on the adjusted bars."""
    raw_files = [Path(root) / f for f in RAW_BARS]
    src = (pd.concat([pd.read_parquet(f, columns=["symbol", "date"]) for f in raw_files], ignore_index=True)
           if all(f.exists() for f in raw_files) else adjusted[["symbol", "date"]])
    src = src.assign(date=pd.to_datetime(src["date"]))
    return life_ends(src, sessions_from(src))


def load_cash_rate(root: Path, sessions: pd.DatetimeIndex) -> pd.Series | None:
    """DFF/252 per session from the archived FRED series (features.macro.MacroStore, PIT); None if not on disk."""
    from jevtrader.features.macro import MacroStore

    try:
        return MacroStore(Path(root) / "data/raw/macro").cash_rate_daily(sessions)
    except (FileNotFoundError, KeyError):
        return None


_UNSET = object()


def load_inputs(cfg: DecisionConfig, start, end, root: Path = ROOT, industry: Mapping[str, str] | None = None,
                learning: Path = LEARNING_YAML, delisting_returns: Mapping[str, float] | None = None,
                ma_exits: frozenset[str] | None = None, cash_rate_daily: pd.Series | None = None,
                risk: pd.DataFrame | None = None, anchors: pd.DataFrame | None = None,
                evidence: Mapping[pd.Timestamp, Mapping[str, dict]] | None = None, *, gen_dir: Path | None = None,
                data_root: Path | None = None, focus=_UNSET, system: Path = SYSTEM_YAML) -> LayerBInputs:
    """Inputs from disk: judgment tables `results/judgments/<gen_id>/` (or `gen_dir`, e.g. the stacked directory of a
    live decision, pipeline/live.py), adjusted bars (stocks + ETFs), the PIT universe and the DFF cash rate; sigma and
    labels are computed here. Data files (bars, universe, SEC archive, macro) are read under `data_root` (default
    `root`; the live decisions read the refreshed live root, which mirrors the archive layout). Inputs not passed in
    are assembled from files (pipeline/inputs.py): the industry map and the trusted earnings column from
    `firm_info.parquet`, the sentinel columns from `risk.parquet`, the M&A exits from the SEC archive, the trusted
    anchor panel from the prices and the archived 8-K filings, the run's focus names from the Layer-A manifest (a subset
    run's coverage gate), and the news analyst's card evidence blocks from the blackboard archive (cards only).
    `input_report` flags whatever is still missing."""
    from jevtrader.learn.labels import idio_vol_weekly

    root = Path(root)
    droot = Path(data_root) if data_root is not None else root
    gen_dir = Path(gen_dir) if gen_dir is not None else root / "results/judgments" / cfg.gen_id
    lab_cfg = yaml.safe_load(Path(learning).read_text())["labels"]
    iv = lab_cfg["idio_vol"]
    bars = pd.concat([pd.read_parquet(droot / f) for f in BARS_FILES if (droot / f).exists()], ignore_index=True)
    bars["date"] = pd.to_datetime(bars["date"])
    bars = cut_at(bars, symbol_lives(droot, bars))
    lo, hi = pd.Timestamp(start) - pd.Timedelta(days=550), pd.Timestamp(end) + pd.Timedelta(days=60)
    prices = PriceBook(bars[(bars["date"] >= lo) & (bars["date"] <= hi)])
    dates = weekly_decision_dates(prices.sessions, str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date()))
    universe = universe_by_date(pd.read_parquet(droot / UNIVERSE_FILE), dates)
    firm = I.load_firm_info(gen_dir)
    if industry is None:
        industry = I.industry_map(firm)
    if risk is None:
        risk = I.load_risk_table(gen_dir, firm)
    if focus is _UNSET:
        focus = I.load_focus(gen_dir, universe, I.load_subsets(system))
    if ma_exits is None:
        ma_exits = I.ma_exit_set(droot, sorted({s for t in universe for s in universe[t]}))
    sigma = idio_vol_weekly(prices.close, market=iv["market"], com=iv["ewma_center_of_mass_days"],
                            beta_window=iv["beta_window_sessions"], min_periods=iv["beta_min_periods"],
                            ewm_min_periods=iv["ewma_min_periods"])
    labels = LabelStore.build(prices, list(universe), universe, sigma, tail_k=lab_cfg["tail_k"],
                              delisting_returns=delisting_returns, ma_exits=ma_exits)
    raw = load_judgment_features(gen_dir, industry or {}, cfg.noul_clip)
    if cash_rate_daily is None:
        cash_rate_daily = load_cash_rate(droot, prices.sessions)
    if anchors is None:
        anchors = I.anchors_from_files(droot, prices.close, list(universe), universe, cfg.anchors)
    if evidence is None:
        evidence = I.load_card_evidence(gen_dir)
    return LayerBInputs(raw=raw, universe=universe, labels=labels, sigma=sigma, industry=industry, risk=risk,
                        prices=prices, cash_rate_daily=cash_rate_daily, delisting_returns=delisting_returns,
                        ma_exits=frozenset(ma_exits), anchors=anchors, evidence=evidence, focus=focus)


# ------------------------------------------------------------------------------------ input checks
INPUT_CHECKS = ("weeks", "features", "coverage", "unmapped", "industry", "risk", "cash_rate", "sigma", "prices",
                "anchors", "focus", "layer_a_weeks")
RISK_COLUMNS = ("sessions_to_earnings", "h", "n_prob", "problem_kind")
# subset runs: a week's value coverage over the focus names is a hole only if it is also this improbable for a
# Binomial(n_t focus names, median rate) draw (one-sided), so that a quiet news week among 30 names is not an
# outage; full runs keep the plain relative rule (never weakened)
SUBSET_HOLE_ALPHA = 1e-4
# `layer_a_weeks` absolute row rule (DEV #2 audit of B-1): a Layer-A feature whose program ran in the window must have
# a row (a value or an abstention) for at least this share of its population in every evaluated week. Real Layer-A
# programs write a row for every name they were asked about (DEV #1: row coverage 1.0 for all 22 Layer-A features in
# each of its 10 weeks); the relative rules are blind once the median itself is low (a partial crash every week)
LAYER_A_MIN_ROWS = 0.9


class InputError(ValueError):
    pass


def layer_a_features(raw: pd.DataFrame, feats: Sequence[str]) -> set[str]:
    """Features whose values Layer A produced (they exist only for its focus names): as recorded by
    `load_judgment_features`, else (hand-built inputs) every S1 or untrusted feature; trusted code features come from
    the code tables over all of U_t."""
    rec = raw.attrs.get("layer_a_features")
    if rec is not None:
        return {f for f in feats if f in set(rec)}
    return {f for f in feats if not (CATALOG[f].kind == "code" and CATALOG[f].channel == "T")}


def feature_program(raw: pd.DataFrame, feature: str) -> str:
    """The Layer-A program(s) that produced `feature` (as `load_judgment_features` recorded them), else its agent."""
    progs = (raw.attrs.get("feature_programs") or {}).get(feature)
    return "/".join(progs) if progs else CATALOG[feature].agent


def missing_layer_a_weeks(inputs: LayerBInputs, dates: Sequence[pd.Timestamp], feats: Sequence[str],
                          rowcov: pd.DataFrame, skip: Sequence[pd.Timestamp] = (),
                          skip_features: Sequence[str] = (), min_rows: float = LAYER_A_MIN_ROWS) -> dict[str, dict]:
    """Layer-A weeks missing or partial inside the evaluated window, by producing program (dev1_report §11.4, B-1,
    and the DEV #2 audit of B-1).

    A Layer-A feature of the configuration with **no row at all** for its population (U_t, or the declared focus of a
    subset run) in an evaluated week is missing that week, whatever its median: real Layer-A programs write a row for
    every name they were asked about in every week they ran (a null value when there is no news). The relative
    coverage rule cannot see this once more than half of a window is empty (its median is then 0): DEV #1's 44-week
    span had Layer-A rows in 10 weeks and passed. A week in which such a feature has rows for some names but fewer
    than `min_rows` of its population is **partial**: one stray row must not make a crashed week look present, and a
    systematic partial crash (rows for 40% of U_t every week) lowers the median itself, so the relative rule is blind
    to it. Weeks in `skip` (every feature empty: the `weeks` check) and features absent from every week (the
    `features` check) are reported there, not here. Returns {program: {features, missing_weeks, n_missing,
    partial_weeks, n_partial, min_row_share, n_weeks, present_weeks}} for every program with a missing or partial
    week (`present_weeks` counts the weeks with a row, partial ones included)."""
    la = layer_a_features(inputs.raw, feats)
    skip, skip_features = set(skip), set(skip_features)
    out: dict[str, dict] = {}
    for f in feats:
        if f not in la or f in skip_features or float(rowcov[f].sum()) == 0.0:
            continue
        miss = [t for t in dates if t not in skip and float(rowcov.loc[t, f]) == 0.0]
        part = [t for t in dates if t not in skip and 0.0 < float(rowcov.loc[t, f]) < min_rows]
        if not miss and not part:
            continue
        p = out.setdefault(feature_program(inputs.raw, f), {"features": [], "missing": set(), "partial": set(),
                                                            "low": 1.0})
        p["features"].append(f)
        p["missing"] |= set(miss)
        p["partial"] |= set(part)
        if part:
            p["low"] = min(p["low"], min(float(rowcov.loc[t, f]) for t in part))
    res = {}
    for prog, p in sorted(out.items()):
        miss = sorted(p["missing"])
        part = sorted(p["partial"] - p["missing"])
        res[prog] = {"features": p["features"], "missing_weeks": [t.date().isoformat() for t in miss],
                     "n_missing": len(miss), "partial_weeks": [t.date().isoformat() for t in part],
                     "n_partial": len(part), "min_row_share": p["low"] if part else None,
                     "n_weeks": len(dates), "present_weeks": len(dates) - len(miss)}
    return res


def _layer_a_week_problems(missing: Mapping[str, dict]) -> list[str]:
    """One self-contained line per program and kind (the InputError message shows the first five lines)."""
    def shown(w):
        return ", ".join(w[:6]) + (f", ... {w[-1]}" if len(w) > 6 else "")

    lines = []
    for prog, m in missing.items():
        feats = ", ".join(m["features"][:4]) + (f", +{len(m['features']) - 4} more" if len(m["features"]) > 4 else "")
        if m["n_missing"]:
            lines.append(f"program {prog} ({feats}): no Layer-A rows in {m['n_missing']} of {m['n_weeks']} evaluated "
                         f"weeks [{shown(m['missing_weeks'])}]")
        if m.get("n_partial"):
            lines.append(f"program {prog} ({feats}): Layer-A rows for less than {LAYER_A_MIN_ROWS:.0%} of the "
                         f"population in {m['n_partial']} of {m['n_weeks']} evaluated weeks (lowest "
                         f"{m['min_row_share']:.1%}) [{shown(m['partial_weeks'])}]")
    return lines


def _coverage_tables(inputs: LayerBInputs, dates: list[pd.Timestamp], feats: list[str]):
    """(value coverage, row coverage, population size, population kind per feature, focus problems): per decision
    date and feature, the share of the feature's population with a covered finite value, and with any row at all
    (covered or an abstention). The population is U_t, except for Layer-A features of a subset run: their focus
    names (`LayerBInputs.focus`), since Layer A asked nothing about the other names."""
    uni = pd.DataFrame([(t, s) for t in dates for s in dict.fromkeys(inputs.universe[t])], columns=["t", "symbol"])
    n_u = uni.groupby("t").size().reindex(dates).astype(float)
    r = inputs.raw[inputs.raw["feature"].isin(feats)].merge(uni, on=["t", "symbol"])
    ok = np.isfinite(pd.to_numeric(r["value"], errors="coerce").to_numpy(dtype=float))
    if "covered" in r:
        ok &= r["covered"].astype("boolean").fillna(False).astype(bool).to_numpy()
    r = r.assign(ok=ok)
    la = layer_a_features(inputs.raw, feats) if inputs.focus is not None else set()
    fproblems: list[str] = []
    pop = {f: ("focus" if f in la else "universe") for f in feats}
    in_pop = np.ones(len(r), dtype=bool)
    n_f = n_u
    if la:
        missing = [t for t in dates if t not in inputs.focus]
        fproblems += [f"no focus declaration@{t.date().isoformat()}" for t in missing]
        fset = {t: set(inputs.focus.get(t, inputs.universe[t])) & set(inputs.universe[t]) for t in dates}
        n_f = pd.Series({t: float(len(fset[t])) for t in dates}).reindex(dates)
        fproblems += [f"empty focus@{t.date().isoformat()}" for t in dates if n_f[t] == 0]
        member = np.fromiter((s in fset[t] for t, s in zip(r["t"], r["symbol"])), dtype=bool, count=len(r))
        is_la = r["feature"].isin(la).to_numpy()
        out = r[is_la & ~member]
        if len(out):
            ex = out.iloc[0]
            fproblems.append(f"{len(out)} Layer-A rows for names outside the declared focus (first "
                             f"{ex['t'].date()}:{ex['symbol']}:{ex['feature']})")
        in_pop = member | ~is_la
    rp = r[in_pop]
    val = rp.groupby(["t", "feature"])["ok"].sum().unstack("feature").reindex(index=dates, columns=feats).fillna(0.0)
    rows = (rp.groupby(["t", "feature"])["symbol"].nunique().unstack("feature")
            .reindex(index=dates, columns=feats).fillna(0.0))
    n = pd.DataFrame({f: (n_f if pop[f] == "focus" else n_u) for f in feats}, index=dates)
    safe = n.where(n > 0)
    return (val / safe).fillna(0.0), (rows / safe).fillna(0.0), n, val, pop, fproblems


def input_report(cfg: DecisionConfig, inputs: LayerBInputs, dates: Sequence[pd.Timestamp],
                 min_rel_coverage: float = 0.5) -> dict:
    """Per-feature weekly coverage and every input problem, keyed by INPUT_CHECKS. Without these checks a crashed
    Layer-A shard (a week without rows) makes every x~ = 0, so R^T becomes pure symbol-hash order; a feature absent
    from every table keeps beta = 0; and a missing industry map, risk table or cash rate silently switches off the FF12
    cap, the risk multipliers or the excess-return convention.

    Coverage of a feature in a week is the share of its population with a covered, finite value. The population is
    U_t, except for the Layer-A features of a subset run (PRE on S30, DEV on S50, a symbol list): Layer A asked only
    about the run's focus names (`LayerBInputs.focus`), so they are measured over the focus names (design_v2_delta
    B-3). A coverage hole is a week whose coverage is below `min_rel_coverage` x the feature's median weekly coverage;
    over a focus population it must also be improbable for a Binomial(n_t, median) draw (`SUBSET_HOLE_ALPHA`), since a
    quiet week among 30 names is not an outage. Row coverage (the share of the population with any row, covered or an
    abstention) below `min_rel_coverage` x its median is a hole too, in every run: that is a crashed or partial Layer-A
    shard whatever the news flow. A full run's value rule is unchanged. A Layer-A row for a name outside the declared
    focus, a week without a focus declaration and an empty focus are `focus` problems.

    `layer_a_weeks` (dev1_report §11.4, B-1): a Layer-A feature of this configuration with no row at all for its
    population in an evaluated week means that its program did not run (or crashed) that week, whatever the median;
    rows for fewer than LAYER_A_MIN_ROWS of its population (DEV #2 audit) mean a partial run or crash. The relative
    rules miss both once the median itself is low; this check does not. Each problem line names the program, its
    affected features and the missing or partial weeks; `report["layer_a_weeks"]` has the full lists.
    Configuration variants are gated on their own features (`run_decisions` runs this report per configuration).

    The risk table must also land on the evaluated stock-weeks (`pipeline.inputs.risk_coverage`: rows on non-decision
    dates or outside U_t, an earnings panel below its floor in some week, a week with news features but no sentinel
    row) and carry the exclusion rule's key column. This is a data gate: it can stop a run, it never changes a
    decision."""
    from scipy.stats import binom

    dates = sorted(pd.Timestamp(d) for d in dates)
    feats = list(cfg.allocator.features)
    uni = pd.DataFrame([(t, s) for t in dates for s in dict.fromkeys(inputs.universe[t])], columns=["t", "symbol"])
    cov, rowcov, npop, k, pop, fproblems = _coverage_tables(inputs, dates, feats)
    problems: dict[str, list[str]] = {}
    empty = [t for t in dates if cov.loc[t].sum() == 0]
    if empty:
        problems["weeks"] = [t.date().isoformat() for t in empty]
    absent = [f for f in feats if cov[f].sum() == 0]
    if absent:
        problems["features"] = absent
    med, rmed = cov.median(), rowcov.median()
    holes = []
    for f in feats:
        if f in absent:
            continue
        low = cov[f] < min_rel_coverage * med[f]
        if pop[f] == "focus":  # a subset run: the low week must also be improbable for n_t names at the median rate
            tail = binom.cdf(k[f].to_numpy(), npop[f].to_numpy().astype(int), float(med[f]))
            low &= pd.Series(tail < SUBSET_HOLE_ALPHA, index=cov.index)
        low |= rowcov[f] < min_rel_coverage * rmed[f]
        holes += [f"{f}@{t.date().isoformat()}" for t in cov.index[low.to_numpy()] if t not in empty]
    if holes:
        problems["coverage"] = holes
    missing = missing_layer_a_weeks(inputs, dates, feats, rowcov, skip=empty, skip_features=absent)
    if missing:
        problems["layer_a_weeks"] = _layer_a_week_problems(missing)
    if fproblems:
        problems["focus"] = fproblems
    unmapped = list(inputs.raw.attrs.get("unmapped", []))
    if unmapped:
        problems["unmapped"] = unmapped
    ind = dict(inputs.industry or {})
    names = sorted(set(uni["symbol"]))
    if not ind:
        problems["industry"] = ["no industry map: FF12 cap off, x_mac from answers dropped"]
    elif any(s not in ind for s in names):
        problems["industry"] = [s for s in names if s not in ind]
    risk_cov: dict = {}
    if inputs.risk is None or inputs.risk.empty:
        problems["risk"] = ["no risk-officer table: every multiplier is 1"]
    else:
        # the exclusion rule's key column (V-excl-collapsed: n_prob_collapsed; V-excl-items: n_prob_items) is required
        # too, so a BG-3c row is never computed from a missing column
        need = list(dict.fromkeys([*RISK_COLUMNS, cfg.risk.key_col]))
        rp = [f"column {c} missing" for c in need if c not in inputs.risk]
        text = list(cfg.allocator.untrusted)
        text_weeks = [t for t in dates if text and float(cov.loc[t, text].sum()) > 0]
        risk_cov, align = I.risk_coverage(inputs.risk, inputs.universe, dates, text_weeks)
        rp += align
        if rp:
            problems["risk"] = rp
    if inputs.prices is None:
        problems["prices"] = ["no price panel: no drift, no backtest"]
    if inputs.cash_rate_daily is None:
        problems["cash_rate"] = ["no DFF series: cash earns 0 and Sharpe uses raw returns"]
    elif inputs.prices is not None:
        sess = inputs.prices.sessions[(inputs.prices.sessions >= dates[0]) & (inputs.prices.sessions <= dates[-1])]
        gap = sess.difference(inputs.cash_rate_daily.dropna().index)
        if len(gap):
            problems["cash_rate"] = [f"{len(gap)} sessions without a cash rate, first {gap[0].date()}"]
    if inputs.sigma is None:
        problems["sigma"] = ["no sigma: inverse-vol weights fall back to equal weights"]
    if cfg.risk.exclusion_rule == "anchored":
        at_stake = I.exclusion_eligible(inputs.risk, dates, cfg.risk.corroboration, cfg.risk.exclusion_kinds,
                                        key=cfg.risk.key_col)
        if at_stake and inputs.anchors is None:
            problems["anchors"] = [f"no anchor panel: {at_stake} exclusion-eligible risk rows would all become x0.5"]
        elif at_stake and not inputs.anchors.attrs.get("filing_source", True):
            problems["anchors"] = [f"no 8-K filing source: the filing anchor is off for {at_stake} "
                                   "exclusion-eligible risk rows"]
        elif inputs.anchors is not None:
            miss = I.anchor_coverage(inputs.anchors, inputs.universe, dates)
            if miss:
                problems["anchors"] = miss
    unknown = set(cfg.allow_missing) - set(INPUT_CHECKS)
    if unknown:
        raise ValueError(f"unknown allow_missing entries {sorted(unknown)}; known: {INPUT_CHECKS}")
    return {
        "coverage": {f: {"mean": float(cov[f].mean()), "min": float(cov[f].min()), "median": float(med[f]),
                         "population": pop[f], "rows_min": float(rowcov[f].min()),
                         "n_min": int(npop[f].min()) if len(npop) else 0}
                     for f in feats},
        "focus": None if inputs.focus is None else {"n_min": int(min(len(inputs.focus.get(t, ())) for t in dates)),
                                                     "n_max": int(max(len(inputs.focus.get(t, ())) for t in dates))},
        "risk_coverage": risk_cov,
        "layer_a_weeks": missing,
        "problems": problems, "allowed": sorted(set(problems) & set(cfg.allow_missing)),
    }


# ------------------------------------------------------------------------------------ decision loop
@dataclass
class DecisionLog:
    dates: list[pd.Timestamp]
    predictions: pd.DataFrame
    targets: dict[str, dict[pd.Timestamp, dict[str, float]]]
    snapshots: list[dict] = field(default_factory=list)
    scales: list[dict] = field(default_factory=list)
    cards: list[dict] = field(default_factory=list)
    ledger: Ledger | None = None
    inputs: dict = field(default_factory=dict)  # input_report
    sizing: list[dict] = field(default_factory=list)  # per learned week: phi, one-way turnover (realized, reference), cash
    sizing_names: list[dict] = field(default_factory=list)  # per learned week and name (BG-2 log, sizing.jsonl)
    heads: pd.DataFrame | None = None  # t, symbol, one as-of head logit per agent (design §6.7)
    head_snapshots: list[dict] = field(default_factory=list)
    states: list[WeekState] = field(default_factory=list)  # per learned week: what decide_week read (bound checks)
    multipliers: pd.DataFrame | None = None  # t, symbol, one column per risk cause (learned weeks, U_t)
    frozen_from: pd.Timestamp | None = None  # V-static: first decision scored with the frozen slopes
    # the card's agent line sum_a c_a = L over every scored name (learn.allocator.attribution_bound): count, largest
    # residual, largest a priori bound, names above the bound (0 expected) and above 1e-12
    attribution: dict = field(default_factory=dict)


ATTRIBUTION_TOL = 1e-12  # the stated precision of "the agents' contributions sum to L" (dev2_ops §4)


@dataclass
class WeekState:
    """Everything `decide_week` reads at decision t (learned weeks). The bound checks (eval/bound.py) replay decisions
    from it with worst-case untrusted inputs; everything here except `L_W_bar` and `risk` is a trusted object."""
    t: pd.Timestamp
    universe: tuple[str, ...]
    L_T: pd.Series
    L_W_bar: pd.Series
    kappa: float
    w_cur: pd.Series
    sigma: pd.Series
    risk: pd.DataFrame  # untrusted sentinel / S2 columns plus the trusted earnings column, index = U_t
    anchors: pd.Series  # trusted A_{i,t} over U_t
    industry: pd.Series


@dataclass
class WeekDecision:
    sel_ew: Selection
    w_ew: pd.Series
    sel_sz: Selection
    sized: Weights
    mult: pd.DataFrame


def decide_week(cfg: DecisionConfig, L_T: pd.Series, L: pd.Series, w_cur: pd.Series, sigma: pd.Series | None,
                risk_rows: pd.DataFrame, anchors: pd.Series, industry: pd.Series, universe: Sequence[str]) -> WeekDecision:
    """One learned decision: the band-guard selections and both books (the only decision code path; the replay, the
    bound checks and the per-reader checks all call it). `L_T` and `L` are the trusted and combined logits over U_t
    (ranks with the fixed symbol-hash tie-break); `risk_rows` the risk table rows over U_t; `anchors` the trusted A."""
    uni = list(universe)
    fr = pd.DataFrame({"L_T": L_T.reindex(uni).astype(float), "L": L.reindex(uni).astype(float)})
    sel_ew = select(fr, cfg.K, cfg.J)
    w_ew = equal_weights(sel_ew.names)
    mult = risk_multipliers(risk_rows.reindex(uni), cfg.risk, anchors=anchors.reindex(uni))
    held = list(w_cur.index[w_cur > 0])
    sel_sz = select(fr, cfg.K, cfg.J, incumbents=held, vetoed=vetoed_names(mult),
                    incumbent_buffer=cfg.risk.incumbent_buffer)
    sized = size(sel_sz, sigma if sigma is not None else pd.Series(dtype=float), sel_sz.trusted_topK, mult, cfg.risk,
                 w_cur, industry, uni)
    return WeekDecision(sel_ew, w_ew, sel_sz, sized, mult)


def drift_weights(
    w: pd.Series, prices: PriceBook | None, fill_session, asof_session, *, w_before: pd.Series | None = None,
    cost_bps: float = 0.0, cash_rate_daily: pd.Series | None = None,
    delisting_returns: Mapping[str, float] | None = None, ma_exits: frozenset[str] | set[str] = frozenset(),
) -> pd.Series:
    """The engine's book at the close of `asof_session` for target `w` filled at open(fill_session), all known at the
    cutoff of that session. As in backtest.engine: an unfillable target stays in cash; the fill costs `cost_bps` per
    side on sum |w - w_before| (w_before = the book before the fill), paid from cash; cash accrues `cash_rate_daily`;
    a name whose life ended in between exits at its last close x (1 + delisting return), which must be explicit unless
    it is a known M&A exit (V-m14), and its proceeds accrue cash from the exit session."""
    if prices is None or w.empty or fill_session is None:
        return w.copy()
    p0 = prices.open.reindex(columns=w.index).loc[fill_session].to_numpy(dtype=float)
    p1 = prices.close.reindex(columns=w.index).loc[asof_session].to_numpy(dtype=float)
    filled = np.isfinite(p0) & (np.abs(w.to_numpy(dtype=float)) > 1e-12)
    wf = pd.Series(np.where(filled, w.to_numpy(dtype=float), 0.0), index=w.index)
    before = w_before if w_before is not None else pd.Series(dtype=float)
    cost = cost_bps / 1e4 * float(wf[filled].sub(before, fill_value=0.0).abs().sum())
    rc = cash_rate_daily.reindex(prices.sessions).fillna(0.0) if cash_rate_daily is not None else None

    def growth(d) -> float:  # cash growth over the sessions d..asof (the engine accrues from the day cash arises)
        return 1.0 if rc is None else float(np.prod(1.0 + rc.loc[d:asof_session].to_numpy()))

    cash = (1.0 - float(wf.sum()) - cost) * growth(fill_session)
    alive = filled & np.isfinite(p1)
    for k in np.flatnonzero(filled & ~np.isfinite(p1)):
        s = w.index[k]
        if delisting_returns is not None and s in delisting_returns:
            dr = float(delisting_returns[s])
        elif s in ma_exits:
            dr = 0.0
        else:
            raise DelistingError(f"held symbol {s} stopped trading before {pd.Timestamp(asof_session).date()} and is "
                                 "not a known M&A exit; supply an explicit delisting return (V-m14)")
        last = prices.last[s]
        cash += wf.iloc[k] * prices.close.at[last, s] / p0[k] * (1.0 + dr) * growth(prices.next_session(last))
    val = pd.Series(wf.to_numpy()[alive] * p1[alive] / p0[alive], index=w.index[alive])
    total = val.sum() + cash
    return val / total if total > 0 else val


def run_decisions(cfg: DecisionConfig, inputs: LayerBInputs, dates: Sequence[pd.Timestamp],
                  holdings: Mapping[pd.Timestamp, pd.Series] | None = None) -> DecisionLog:
    """The decision loop (module doc). `holdings` replaces the replay's drifted D-sized book w_cur at the given
    decision dates by given weights: a live decision starts from the live account's actual holdings at d's close
    (pipeline/live.py), and its cards then state each action relative to them. Every other week, and every run
    without `holdings`, is unchanged; the band-guard bound holds for any w_cur (design_v2_delta BG-1)."""
    dates = sorted(pd.Timestamp(d) for d in dates)
    if not dates:
        raise ValueError("no decision dates")
    report = input_report(cfg, inputs, dates)
    bad = {k: v for k, v in report["problems"].items() if k not in cfg.allow_missing}
    if bad:
        raise InputError("Layer-B inputs are incomplete (waive a check with DecisionConfig.allow_missing): "
                         + "; ".join(f"{k}: {v[:5]}{' ...' if len(v) > 5 else ''}" for k, v in bad.items()))
    labels = inputs.labels.restrict(dates[0], dates[-1])  # P-clean: learn only from this window's labels
    feats = cfg.allocator.features
    std = Standardizer(specs_for(feats), cfg.std_trailing_weeks, cfg.std_min_history, cfg.std_floor, cfg.std_clip)
    # calibration snapshots are namespaced by the judgment generation (backend, pinned model, bank, prompt, blinding)
    acfg = dataclasses.replace(cfg.allocator, namespace=cfg.allocator.namespace or cfg.gen_id)
    alloc = Allocator(acfg, labels)
    heads = AgentHeads(acfg, labels) if cfg.heads else None
    ledger = Ledger(feats)
    raw = inputs.raw[inputs.raw["feature"].isin(feats)]
    raw_by_t = {pd.Timestamp(t): g for t, g in raw.groupby("t", sort=False)}
    empty = raw.iloc[0:0]
    risk_by_t = {pd.Timestamp(t): g.set_index("symbol") for t, g in inputs.risk.groupby("t")} if inputs.risk is not None else {}
    anc_by_t = ({pd.Timestamp(t): g for t, g in inputs.anchors.groupby("t")} if inputs.anchors is not None else {})
    industry = pd.Series(dict(inputs.industry or {}), dtype=object)
    targets: dict[str, dict] = {"sel_ew": {}, "sized": {}}
    preds, snaps, scales, cards, sizing, sizing_names, head_rows, head_snaps, states, mults = ([] for _ in range(10))
    prev_ew, prev_sz = pd.Series(dtype=float), pd.Series(dtype=float)
    prev_cur = pd.Series(dtype=float)  # the drifted book the previous D-sized target was traded from
    prev_t = None
    frozen = None  # V-static
    frozen_from = None
    att = {"n": 0, "n_nonzero": 0, "n_above_bound": 0, "n_above_tol": 0, "max_residual": 0.0, "max_bound": 0.0,
           "tol": ATTRIBUTION_TOL}
    for t in dates:
        c = decision_cutoff(t)
        uni = list(dict.fromkeys(inputs.universe[t]))
        x = std.transform(t, raw_by_t.get(t, empty), uni)
        snap = alloc.fit(ledger, c)
        if cfg.freeze_after_weeks is not None:
            if frozen is None and snap.n_weeks >= cfg.freeze_after_weeks:
                frozen, frozen_from = snap, t
            snap = frozen if frozen is not None else snap
        sc = alloc.score(x, snap)
        res_a, bnd_a = attribution_residual(sc).to_numpy(), attribution_bound(sc).to_numpy()
        att["n"] += len(res_a)
        att["n_nonzero"] += int((res_a > 0).sum())
        att["n_above_bound"] += int((res_a > bnd_a).sum())
        att["n_above_tol"] += int((res_a > ATTRIBUTION_TOL).sum())
        if len(res_a):
            att["max_residual"] = max(att["max_residual"], float(res_a.max()))
            att["max_bound"] = max(att["max_bound"], float(bnd_a.max()))
        if heads is not None:
            hs = heads.fit(ledger, c)
            hl = heads.score(x, hs)
            head_rows.append(hl.assign(t=t, symbol=hl.index, head_snapshot_id=hs.snapshot_id).reset_index(drop=True))
            head_snaps.append({"t": t.date().isoformat(), **hs.to_json()})
        ledger.append(t, x, sc.frame["L_T"], snap.snapshot_id)
        fill_prev = inputs.prices.next_session(prev_t) if (inputs.prices is not None and prev_t is not None) else None
        w_cur = drift_weights(prev_sz, inputs.prices, fill_prev, t, w_before=prev_cur, cost_bps=cfg.cost_bps,
                              cash_rate_daily=inputs.cash_rate_daily, delisting_returns=inputs.delisting_returns,
                              ma_exits=inputs.ma_exits)
        card_prev_sz = prev_sz
        if holdings is not None and t in holdings:  # a live week: the account's holdings, not the replay's book
            w_cur = pd.Series(holdings[t], dtype=float)
            w_cur = w_cur[w_cur > 0].sort_index()
            card_prev_sz = w_cur
        sel_ew = sel_sz = sized = None
        A = anchors_for(anc_by_t.get(t), t, uni)
        if snap.cold:
            w_ew = equal_weights(uni)
            w_sz = equal_weights(uni)
        else:
            rk = risk_by_t.get(t)
            rows = rk.reindex(uni) if rk is not None else pd.DataFrame(index=uni)
            sig = None
            if inputs.sigma is not None:
                past = inputs.sigma.loc[:t]
                sig = past.iloc[-1] if len(past) else None
            wd = decide_week(cfg, sc.frame["L_T"], sc.frame["L"], w_cur, sig, rows, A, industry, uni)
            sel_ew, w_ew, sel_sz, sized = wd.sel_ew, wd.w_ew, wd.sel_sz, wd.sized
            w_sz = sized.final
            sizing.append({"t": t.date().isoformat(), "phi": sized.phi, "turnover": sized.turnover,
                           "turnover_ref": sized.turnover_ref, "cash": sized.cash, "sizing": sized.sizing,
                           "holdings_sum": math.fsum(w_cur.to_numpy(dtype=float)), "holding_scale": sized.holding_scale,
                           "F": sized.F, "n_entrants": len(sized.entrants), "n_displaced": len(sized.displaced),
                           "n_excluded": len(sized.excluded)})
            sizing_names.extend({"t": t.date().isoformat(), **r} for r in sized.log)
            mults.append(wd.mult.assign(t=t, symbol=wd.mult.index).reset_index(drop=True))
            states.append(WeekState(t=t, universe=tuple(uni), L_T=sc.frame["L_T"].copy(),
                                    L_W_bar=sc.frame["L_W_bar"].copy(), kappa=float(sc.kappa), w_cur=w_cur.copy(),
                                    sigma=(sig if sig is not None else pd.Series(dtype=float)).copy(),
                                    risk=rows.copy(), anchors=A.copy(), industry=industry))
        targets["sel_ew"][t] = {s: float(v) for s, v in w_ew.items()}
        targets["sized"][t] = {s: float(v) for s, v in w_sz.items()}
        p = sc.frame.copy()
        p = p.join(sc.agents.add_prefix("c_"))
        p.insert(0, "symbol", p.index)
        p.insert(0, "t", t)
        p["cold"], p["snapshot_id"], p["kappa"], p["s_T"] = snap.cold, snap.snapshot_id, sc.kappa, sc.s_T
        p["frozen"] = frozen is not None
        p["anchor"] = p["symbol"].map(A).fillna(0.0).astype(bool)
        p["w_sel_ew"] = p["symbol"].map(w_ew).fillna(0.0)
        p["w_sized"] = p["symbol"].map(w_sz).fillna(0.0)
        preds.append(p.reset_index(drop=True))
        snaps.append({"t": t.date().isoformat(), **snap.to_json(), "kappa": sc.kappa, "s_T": sc.s_T})
        scales.append({"t": t.date().isoformat(), **{f: dataclasses.asdict(v) for f, v in std.log[t].items()}})
        ev = (inputs.evidence or {}).get(t)
        pv = (inputs.provenance or {}).get(t)
        cards.extend(build_cards(t, sc, sel_ew, w_ew, sel_sz, sized, prev_ew, card_prev_sz, K=cfg.K, J=cfg.J,
                                 evidence=ev, provenance=pv, anchors=A))
        prev_ew, prev_sz, prev_cur, prev_t = w_ew, w_sz, w_cur, t
    hd = pd.concat(head_rows, ignore_index=True) if head_rows else None
    if hd is not None:
        hd = hd[["t", "symbol", *[c_ for c_ in hd.columns if c_ not in ("t", "symbol", "head_snapshot_id")],
                 "head_snapshot_id"]]
    return DecisionLog(dates=dates, predictions=pd.concat(preds, ignore_index=True), targets=targets,
                       snapshots=snaps, scales=scales, cards=cards, ledger=ledger, inputs=report, sizing=sizing,
                       sizing_names=sizing_names, heads=hd, head_snapshots=head_snaps, states=states,
                       multipliers=pd.concat(mults, ignore_index=True) if mults else None, frozen_from=frozen_from,
                       attribution=att)


# ------------------------------------------------------------------------------------ evaluation
@dataclass
class EvalResult:
    cfg_sha: str
    window: str
    log: DecisionLog
    returns: dict[str, pd.Series]
    metrics: dict
    unfilled: list[tuple[str, str, str, str]]  # (book, decision date, fill date, symbol)
    out_dir: Path | None = None
    inputs_sha: str = ""
    hold_end: pd.Timestamp | None = None


def prediction_metrics(log: DecisionLog, labels: LabelStore, start: pd.Timestamp | None = None) -> dict:
    """Weekly rank-IC of L vs r - m and pooled scores of P from the first learned decision (after the fact; from
    `start` for V-static), with the scoring hygiene of VC-6 and the VC-1 reference forecasters (constant 0.5 and the
    trailing base rate of matured labels) as skill scores."""
    pr = log.predictions[~log.predictions["cold"]]
    if start is not None:
        pr = pr[pr["t"] >= pd.Timestamp(start)]
    if pr.empty:
        return {"n_weeks_scored": 0}
    lab = labels.table[["t", "symbol", "r", "m", "y"]]
    j = pr.merge(lab, on=["t", "symbol"], how="inner").dropna(subset=["r", "y"])
    if j.empty:
        return {"n_weeks_scored": 0}
    j["rel"] = j["r"] - j["m"]
    sc = j.pivot(index="t", columns="symbol", values="L")
    fw = j.pivot(index="t", columns="symbol", values="rel")
    ic = M.information_coefficient(sc, fw)
    p, y = j["P"].to_numpy(), j["y"].to_numpy()
    base = M.trailing_base_rate(labels.restrict(log.dates[0], log.dates[-1]), j["t"])
    table = M.score_table(p, y)
    return {
        "n_weeks_scored": int(ic.notna().sum()), "ic_mean": float(ic.mean()), "ic_sd": float(ic.std(ddof=1)),
        "ic_t_nw4": M.t_stat_newey_west(ic, 4), "log_loss": M.log_loss(p, y), "brier": M.brier(p, y),
        "auc": M.auc(p, y), "scores": table,
        "skill": {"vs_constant": M.skill_scores(p, y, np.full(len(y), 0.5)),
                  "vs_trailing_base_rate": M.skill_scores(p, y, base)},
        "ic_by_week": {k.date().isoformat(): float(v) for k, v in ic.items()},
    }


def portfolio_metrics(res, bench, cost_bps: float) -> dict:
    """Portfolio estimates of one book (design §13.5, PE-2): summary on excess returns, CER, CVaR, average and
    conditional drawdown, turnover, and the break-even per-side cost against the EW-U100 book `bench`."""
    r, rf = res.daily_returns, res.rf
    return {
        **M.summarize(r, rf=rf, benchmark=bench.daily_returns if bench is not res else None),
        "cer_gamma3": M.cer(r, 3.0, rf), "cvar95": M.cvar(r - rf),
        "avg_drawdown": M.average_drawdown(r), "cdd80": M.conditional_drawdown(r, 0.8),
        "avg_turnover": float(res.turnover.mean()) if len(res.turnover) else 0.0,
        "avg_turnover_one_way": float(0.5 * res.turnover.mean()) if len(res.turnover) else 0.0,
        "break_even_bps_vs_ew": M.break_even_cost(r, res.turnover, bench.daily_returns, bench.turnover, cost_bps),
    }


def holding_end(prices: PriceBook, last_decision) -> pd.Timestamp:
    """End of the evaluated return series: the close of d_{T+1}, the weekly decision date after the last decision.
    The last book is held from open(e_T) until it would be replaced at open(e_{T+1}) and is marked to market at the
    close before that (no liquidation cost; the final overnight is left out). Without a later week in the data, the
    last session. Without this cut the engine would hold the last book to the end of the loaded bars (weeks past the
    window, possibly inside the locked W_clean)."""
    cal = weekly_decision_dates(prices.sessions, prices.sessions[0], prices.sessions[-1])
    later = [d for d in cal if d > pd.Timestamp(last_decision)]
    return pd.Timestamp(later[0]) if later else pd.Timestamp(prices.sessions[-1])


def _sha_frame(obj) -> str:
    if obj is None:
        return "none"
    f = obj.to_frame() if isinstance(obj, pd.Series) else obj
    h = hashlib.sha256(pd.util.hash_pandas_object(f, index=True).to_numpy().tobytes())
    h.update(canonical_json([list(map(str, f.columns)), list(map(str, f.dtypes))]).encode())
    return h.hexdigest()


def inputs_hash(inputs: LayerBInputs) -> str:
    """SHA-256 over every input a Layer-B run reads (judgment features, universe, labels, sigma, industry, risk
    table, prices, cash rate, delisting returns, M&A exits, card evidence and provenance). The config hash does not
    cover data, so outputs and trial records carry this hash too."""
    p = inputs.prices
    parts = {
        "raw": _sha_frame(inputs.raw), "labels": _sha_frame(inputs.labels.table), "sigma": _sha_frame(inputs.sigma),
        "risk": _sha_frame(inputs.risk), "cash_rate": _sha_frame(inputs.cash_rate_daily),
        "open": _sha_frame(p.open if p is not None else None), "close": _sha_frame(p.close if p is not None else None),
        "universe": config_hash({str(pd.Timestamp(t).date()): list(v) for t, v in inputs.universe.items()}),
        "industry": config_hash(dict(inputs.industry or {})),
        "delisting": config_hash([dict(inputs.delisting_returns or {}), sorted(inputs.ma_exits)]),
        "evidence": config_hash({str(k): v for k, v in (inputs.evidence or {}).items()}),
        "provenance": config_hash({str(k): v for k, v in (inputs.provenance or {}).items()}),
        "anchors": _sha_frame(inputs.anchors[[c for c in inputs.anchors.columns if c != "d0_min"]]
                              if inputs.anchors is not None else None),
    }
    if inputs.focus is not None:  # a subset run's focus (the coverage gate reads it); full runs keep their old hash
        parts["focus"] = config_hash({str(pd.Timestamp(t).date()): list(v) for t, v in inputs.focus.items()})
    return config_hash(parts)


# every file evaluate() and experiments/run_layer_b.py may write into a run directory; the directory is keyed by config
# and inputs only, so a rerun under changed code would otherwise leave an earlier run's files next to the new ones
EVAL_OUTPUTS = ("predictions.parquet", "weights.parquet", "daily_returns.parquet", "cards.jsonl", "snapshots.jsonl",
                "scales.jsonl", "sizing.jsonl", "sizing_weeks.jsonl", "heads.parquet", "head_snapshots.jsonl",
                "multipliers.parquet", "metrics.json", "config.json")
SIDE_OUTPUTS = ("inputs_manifest.json", "variants.json", "exclusion_rules.json", "composition.json",
                "bound_check.json", "attribution.json", "sharpe_tests.json")
MANIFEST = "manifest.json"
PACKAGE_DIR = Path(__file__).resolve().parents[1]  # src/jevtrader


@functools.lru_cache(maxsize=4)
def code_hash(package: Path = PACKAGE_DIR) -> str:
    """SHA-256 over the package source (sorted relative paths and contents of every .py file): which code wrote a
    run's outputs."""
    h = hashlib.sha256()
    for f in sorted(Path(package).rglob("*.py")):
        h.update(f.relative_to(package).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def clear_outputs(out_dir: Path) -> None:
    """Remove the Layer-B output files (EVAL_OUTPUTS, SIDE_OUTPUTS, the manifest) an earlier run left in `out_dir`.
    Only these names are touched."""
    for name in (*EVAL_OUTPUTS, *SIDE_OUTPUTS, MANIFEST):
        (Path(out_dir) / name).unlink(missing_ok=True)


def write_manifest(out_dir: Path, files: Sequence[str], **extra) -> dict:
    """Add `files` (names inside `out_dir`) to the run manifest `manifest.json` with their SHA-256, together with the
    code hash and `extra`. The manifest lists exactly the files this run wrote; report scripts read only those."""
    out_dir = Path(out_dir)
    path = out_dir / MANIFEST
    man = json.loads(path.read_text()) if path.exists() else {"files": {}}
    man["files"].update({f: file_sha256(out_dir / f) for f in files})
    man.update({"code_sha": code_hash(), **_plain(extra)})
    path.write_text(json.dumps(man, indent=1, sort_keys=True))
    return man


def evaluate(
    cfg: DecisionConfig, window: str, inputs: LayerBInputs | None = None, *, root: Path = ROOT,
    register: bool = True, write: bool = True, system: Path = SYSTEM_YAML, learning: Path = LEARNING_YAML,
    load_kw: dict | None = None,
) -> EvalResult:
    """Replay Layer B over `window` and score it. Refuses, before the pre-registration unlock, any window whose
    decisions or last holding week touch the locked W_clean; appends every book's return series to the trial
    registry; writes outputs under results/eval/<cfg_sha16>/<window>__<inputs_sha12>/, first removing any Layer-B
    output an earlier run left there, and lists them with their SHA-256 and the code hash in `manifest.json`."""
    root = Path(root)
    sys_cfg = yaml.safe_load(Path(system).read_text())
    check_frozen_constants(yaml.safe_load(Path(learning).read_text()), sys_cfg)
    start, end = resolve_window(window, system)
    # checked on the window range plus its last holding week before anything is loaded, and again on the actual
    # decision dates and the end of the last holding period
    if range_touches_locked(start, end + pd.Timedelta(days=7)):
        assert_unlocked_for([max(start, LOCKED_WINDOW[0])], root)
    if inputs is None:
        inputs = load_inputs(cfg, start, end, root, learning=learning, **(load_kw or {}))
    dates = [pd.Timestamp(t) for t in sorted(inputs.universe) if start <= pd.Timestamp(t) <= end]
    if not dates:
        raise ValueError(f"no decision dates in window {window}")
    hold_end = holding_end(inputs.prices, dates[-1]) if inputs.prices is not None else dates[-1]
    assert_unlocked_for([*dates, hold_end], root)
    log = run_decisions(cfg, inputs, dates)
    cfg_sha, data_sha = cfg.sha, inputs_hash(inputs)
    score_from = log.frozen_from if cfg.freeze_after_weeks is not None else None  # V-static: after its fitting window
    metrics: dict = {"window": window, "cfg_sha": cfg_sha, "inputs_sha": data_sha, "code_sha": code_hash(),
                     "n_decisions": len(dates),
                     "hold_end": hold_end.date().isoformat(), "inputs": log.inputs,
                     "attribution_ledger": log.attribution,
                     "prediction": prediction_metrics(log, inputs.labels, score_from),
                     "scored_from": score_from.date().isoformat() if score_from is not None else None}
    returns: dict[str, pd.Series] = {}
    rf: dict[str, pd.Series] = {}
    unfilled: list = []
    if inputs.prices is not None and not (cfg.freeze_after_weeks is not None and score_from is None):
        booked = [t for t in dates if score_from is None or t >= score_from]
        bench = {t: {s: 1.0 / len(inputs.universe[t]) for s in inputs.universe[t]} for t in booked}
        books = {**{b: {t: tg[t] for t in booked} for b, tg in log.targets.items()}, "ew_u100": bench}
        results = {}
        for book, tg in books.items():
            # the engine keeps unfillable targets in cash and logs them (V-M2); decisions never see e_t
            res = run_backtest(inputs.prices, tg, cost_bps=cfg.cost_bps, cash_rate_daily=inputs.cash_rate_daily,
                               delisting_returns=dict(inputs.delisting_returns or {}), end=hold_end,
                               ma_exits=inputs.ma_exits)
            results[book] = res
            returns[book], rf[book] = res.daily_returns, res.rf
            un = res.meta.get("unfilled", pd.DataFrame())
            unfilled += [(book, str(pd.Timestamp(r.decision).date()), str(pd.Timestamp(r.fill).date()), r.symbol)
                         for r in un.itertuples()]
        for book, res in results.items():
            metrics[f"portfolio_{book}"] = portfolio_metrics(res, results["ew_u100"], cfg.cost_bps)
            metrics[f"portfolio_{book}"]["n_unfilled"] = len(res.meta.get("unfilled", pd.DataFrame()))
        if register:
            reg = root / sys_cfg.get("registry", "results/trials.jsonl")
            for book in ("sel_ew", "sized"):
                if book in returns:
                    register_trial(reg, cfg_sha=cfg_sha, name=f"{cfg.name}/{book}", window=window,
                                   returns=returns[book], rf=rf[book],
                                   extra={"gen_id": cfg.gen_id, "protocol": "P-clean", "inputs_sha": data_sha,
                                          "code_sha": code_hash(), "hold_end": hold_end.date().isoformat(),
                                          "allow_missing": list(cfg.allow_missing),
                                          "input_problems_waived": {k: log.inputs["problems"][k]
                                                                    for k in log.inputs["allowed"]}})
    out_dir = None
    if write:
        out_dir = root / "results/eval" / cfg_sha[:16] / f"{window.replace(':', '_')}__{data_sha[:12]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        clear_outputs(out_dir)  # nothing an earlier run wrote survives next to this run's files
        written = ["predictions.parquet", "weights.parquet", "cards.jsonl", "snapshots.jsonl", "scales.jsonl",
                   "sizing.jsonl", "sizing_weeks.jsonl", "metrics.json", "config.json"]
        log.predictions.to_parquet(out_dir / "predictions.parquet")
        wl = [(b, t, s, w) for b, tg in log.targets.items() for t, ws in tg.items() for s, w in ws.items()]
        pd.DataFrame(wl, columns=["book", "t", "symbol", "w"]).to_parquet(out_dir / "weights.parquet")
        if returns:
            pd.DataFrame(returns).to_parquet(out_dir / "daily_returns.parquet")
            written.append("daily_returns.parquet")
        (out_dir / "cards.jsonl").write_text("".join(canonical_json(c) + "\n" for c in log.cards))
        (out_dir / "snapshots.jsonl").write_text("".join(canonical_json(s) + "\n" for s in log.snapshots))
        (out_dir / "scales.jsonl").write_text("".join(canonical_json(s) + "\n" for s in log.scales))
        (out_dir / "sizing.jsonl").write_text("".join(canonical_json(s) + "\n" for s in log.sizing_names))
        (out_dir / "sizing_weeks.jsonl").write_text("".join(canonical_json(s) + "\n" for s in log.sizing))
        if log.heads is not None:
            log.heads.to_parquet(out_dir / "heads.parquet")
            (out_dir / "head_snapshots.jsonl").write_text("".join(canonical_json(s) + "\n" for s in log.head_snapshots))
            written += ["heads.parquet", "head_snapshots.jsonl"]
        if log.multipliers is not None:
            log.multipliers.to_parquet(out_dir / "multipliers.parquet")
            written.append("multipliers.parquet")
        (out_dir / "metrics.json").write_text(json.dumps(_plain(metrics), indent=1, sort_keys=True))
        (out_dir / "config.json").write_text(json.dumps(_plain(cfg), indent=1, sort_keys=True))
        write_manifest(out_dir, written, cfg_sha=cfg_sha, inputs_sha=data_sha, window=window)
    return EvalResult(cfg_sha=cfg_sha, window=window, log=log, returns=returns, metrics=metrics, unfilled=unfilled,
                      out_dir=out_dir, inputs_sha=data_sha, hold_end=hold_end)
