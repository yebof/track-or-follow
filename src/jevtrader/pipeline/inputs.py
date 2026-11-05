"""Layer-B inputs assembled from files (Day-1 integration open items 3 and 4), so that no real run relies on inputs
passed by hand.

Files, all under the judgment-table directory `results/judgments/<gen_id>/` unless stated:
- `risk.parquet` (Layer A, WS-A): the untrusted sentinel / System-2 columns per (t, symbol):
    h                 max P(serious_problem) over about-focal items
    n_prob            distinct provenance keys (author bylines, SAN-5) among kept, non-promotional items with
                      P(serious_problem) >= .8
    n_prob_collapsed  the same with the desk bylines of code_constants.corroboration collapsed into one key (optional)
    n_prob_items      distinct such items (design_v1 item rule; optional, BG-3c replay only)
    problem_kind      top `problem_kind` label of the item attaining h
    s2_mult, s2_keys  System-2 action (1, .5, 0) and distinct provenance keys cited by its verified claims (P1)
  Any `sessions_to_earnings` or anchor column in it is ignored: those are trusted inputs with their own source.
- `firm_info.parquet` (WS-C, scripts/build_code_features.py): `t, symbol, industry, size_rank, sector_etf,
  sessions_to_earnings`. It gives the FF12 industry map and the trusted earnings column of the risk table.
- `code_features.parquet` (WS-C) is read with the judgment tables (pipeline.layer_b.load_judgment_features).
- `data/raw/sec/submissions/` (WS-C): 8-K item codes for the M&A exit classification (data.universe.ma_exits) and
  the filing anchor (features.anchors).
- `memory/work.jsonl` (Layer A's blackboard archive): the news analyst's card evidence blocks (key `N23.card.<sym>`,
  written by programs.news; design_v1 §10.1 "Evidence"), passed through to the decision cards (never read by a
  decision rule).
- `MANIFEST.json` (Layer A, programs/runner.py): its `context` (programs.context.ContextBuilder.describe: `subset`,
  `symbols`, `top`) declares the run's focus names, i.e. the names whose stock-level requests Layer A made. A subset
  run (S50/S30 or an explicit symbol list) has S1 features for its focus names only, so Layer B measures their
  coverage over the focus, not over U_t (`load_focus`; layer_b.input_report). A stacked directory (pipeline/live.py)
  carries one rule per source generation in `STACK.json`.
Every loader returns None when its file is absent; `layer_b.input_report` then stops the run unless the check is
waived, so a missing file is never a silent default.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from jevtrader.pipeline.registry import file_sha256

RISK_FILE, FIRM_FILE = "risk.parquet", "firm_info.parquet"
MANIFEST_FILE, STACK_FILE = "MANIFEST.json", "STACK.json"
WORK_FILE = Path("memory/work.jsonl")
CARD_PREFIX = "N23.card."  # programs.news.CARD_KEY (Layer B never imports Layer A; tests/test_integration_day2.py ties them)
SUBMISSIONS = Path("data/raw/sec/submissions")
UNTRUSTED_RISK_COLUMNS = ("h", "n_prob", "n_prob_collapsed", "n_prob_items", "problem_kind", "s2_mult", "s2_keys")
TRUSTED_RISK_COLUMNS = ("sessions_to_earnings",)
EARNINGS_FLOOR = 0.8  # share of U_t with a finite sessions_to_earnings that a per-(t, U_t) panel reaches every week
UNIVERSE_TOP = 100  # U_t = the monthly PIT top-100 (layer_b.universe_by_date); a focus rule must be over the same U_t


# ------------------------------------------------------------------------------------ focus (subset runs)
def subset_names(universe: Sequence[str], n: int, salt: str) -> tuple[str, ...]:
    """The `n` names of U_t with the smallest sha256(salt + symbol), in U_t order: design_v1 §2.2's S50/S30 rule, the
    same rule as programs.context.subset_names (Layer B does not import Layer A; tests/test_pipeline_focus_coverage.py
    ties the two)."""
    uni = list(universe)
    keep = set(sorted(uni, key=lambda s: hashlib.sha256(f"{salt}{s}".encode()).hexdigest())[:n])
    return tuple(s for s in uni if s in keep)


def load_subsets(system_yaml: Path) -> dict[str, tuple[str, int]]:
    """{subset name: (salt, n)} from configs/system_v1.yaml `subsets`."""
    import yaml

    sub = yaml.safe_load(Path(system_yaml).read_text()).get("subsets", {})
    return {k: (str(v["salt"]), int(v["n"])) for k, v in sub.items()}


def focus_rule(context: Mapping | None) -> dict | None:
    """The focus rule a Layer-A context description declares (`subset`, `symbols`, `top`), or None for a run over
    every U_t name (no subset, no symbol list, or no description at all)."""
    if not context:
        return None
    subset, symbols = context.get("subset"), context.get("symbols")
    if subset is None and symbols is None:
        return None
    return {"subset": subset, "symbols": None if symbols is None else sorted(symbols),
            "top": int(context.get("top", UNIVERSE_TOP))}


def generation_context(gen_dir: Path) -> dict | None:
    """MANIFEST.json's `context` of a Layer-A generation (None without a manifest or a description)."""
    p = Path(gen_dir) / MANIFEST_FILE
    return json.loads(p.read_text()).get("context") if p.exists() else None


def focus_names(rule: Mapping | None, universe: Sequence[str], subsets: Mapping[str, tuple[str, int]]) -> tuple[str, ...]:
    """The focus names of one week under `rule` (programs.context.ContextBuilder.focus): the subset of U_t, then the
    symbol list; all of U_t without a rule."""
    names = tuple(universe)
    if rule is None:
        return names
    if int(rule.get("top", UNIVERSE_TOP)) != UNIVERSE_TOP:
        raise ValueError(f"focus rule over a top-{rule['top']} universe; Layer B's U_t is the top-{UNIVERSE_TOP}")
    if rule.get("subset") is not None:
        if rule["subset"] not in subsets:
            raise ValueError(f"unknown subset {rule['subset']!r}; known: {sorted(subsets)}")
        salt, n = subsets[rule["subset"]]
        names = subset_names(names, n, salt)
    if rule.get("symbols") is not None:
        want = set(rule["symbols"])
        names = tuple(s for s in names if s in want)
    return names


def focus_rules(gen_dir: Path) -> list[tuple[frozenset[pd.Timestamp] | None, dict | None]]:
    """(weeks, rule) pairs of a judgment directory: one pair (None = every week) for a Layer-A generation, one per
    source for a stacked directory (`STACK.json`, pipeline/live.py)."""
    gen_dir = Path(gen_dir)
    st = gen_dir / STACK_FILE
    if st.exists():
        return [(frozenset(pd.Timestamp(w) for w in s["weeks"]), s.get("focus_rule"))
                for s in json.loads(st.read_text())["sources"]]
    return [(None, focus_rule(generation_context(gen_dir)))]


def load_focus(gen_dir: Path, universe: Mapping[pd.Timestamp, Sequence[str]],
               subsets: Mapping[str, tuple[str, int]]) -> dict[pd.Timestamp, tuple[str, ...]] | None:
    """{decision date: focus names} of the generation(s) behind `gen_dir`, or None when every week is a full run (the
    coverage gate then works over U_t exactly as before)."""
    rules = focus_rules(gen_dir)
    if all(r is None for _, r in rules):
        return None
    out = {}
    for t, names in universe.items():
        t = pd.Timestamp(t)
        rule = next((r for w, r in rules if w is None or t in w), None)
        out[t] = focus_names(rule, names, subsets)
    return out


def _read(path: Path) -> pd.DataFrame | None:
    if not Path(path).exists():
        return None
    f = pd.read_parquet(path)
    f["t"] = pd.to_datetime(f["t"])
    return f


def load_firm_info(gen_dir: Path) -> pd.DataFrame | None:
    return _read(Path(gen_dir) / FIRM_FILE)


def industry_map(firm_info: pd.DataFrame | None) -> dict[str, str] | None:
    """symbol -> FF12 code from firm_info (current SIC, declared non-PIT; constant per symbol). Raises if a symbol
    has two different codes."""
    if firm_info is None or "industry" not in firm_info:
        return None
    f = firm_info.dropna(subset=["industry"])
    many = f.groupby("symbol")["industry"].nunique()
    if (many > 1).any():
        raise ValueError(f"symbols with more than one industry code: {sorted(many.index[many > 1])[:5]}")
    return {s: str(g) for s, g in f.groupby("symbol")["industry"].first().items()}


def load_risk_table(gen_dir: Path, firm_info: pd.DataFrame | None = None) -> pd.DataFrame | None:
    """The risk-officer table `t, symbol, sessions_to_earnings, h, n_prob, problem_kind, ...`: the untrusted columns
    of `risk.parquet` joined (outer, on t and symbol) with the trusted earnings column of `firm_info.parquet`. None if
    neither file exists. A symbol-week without a Layer-A row has no sentinel input (multiplier 1)."""
    gen_dir = Path(gen_dir)
    fi = firm_info if firm_info is not None else load_firm_info(gen_dir)
    rk = _read(gen_dir / RISK_FILE)
    if rk is None and fi is None:
        return None
    parts = []
    if rk is not None:
        if rk.duplicated(["t", "symbol"]).any():
            raise ValueError(f"{RISK_FILE}: duplicate (t, symbol) rows")
        parts.append(rk[["t", "symbol", *[c for c in UNTRUSTED_RISK_COLUMNS if c in rk]]])
    if fi is not None and "sessions_to_earnings" in fi:
        parts.append(fi[["t", "symbol", "sessions_to_earnings"]])
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["t", "symbol"], how="outer", validate="one_to_one")
    out = out.sort_values(["t", "symbol"], kind="mergesort").reset_index(drop=True)
    # provenance for the coverage check (risk_coverage): firm_info's earnings column is a per-(t, U_t) panel
    out.attrs["earnings_source"] = FIRM_FILE if (fi is not None and "sessions_to_earnings" in fi) else None
    out.attrs["sentinel_source"] = RISK_FILE if rk is not None else None
    return out


def load_card_evidence(gen_dir: Path) -> dict[pd.Timestamp, dict[str, dict]] | None:
    """Card evidence blocks {decision date: {symbol: block}} from `<gen_dir>/memory/work.jsonl` (the entries
    `N23.card.<symbol>`; the same reading as programs.news.card_evidence), for `LayerBInputs.evidence`. None when the
    generation has no blackboard archive or no card entry (then cards carry no evidence block, as before)."""
    f = Path(gen_dir) / WORK_FILE
    if not f.exists():
        return None
    out: dict[pd.Timestamp, dict[str, dict]] = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)["record"]
        if rec["key"].startswith(CARD_PREFIX):
            t = pd.Timestamp(pd.Timestamp(rec["t"]).date())  # the store keeps d_t as UTC midnight
            out.setdefault(t, {})[rec["key"][len(CARD_PREFIX):]] = rec["value"]
    return out or None


def ma_exit_set(root: Path, symbols: Sequence[str]) -> frozenset[str]:
    """Known M&A exits among `symbols` (data.universe.ma_exits on the raw bars and the archived 8-Ks); empty if the
    SEC archive or the raw bars are not on disk."""
    from jevtrader.data.prices import RAW_BARS
    from jevtrader.data.universe import ma_exits

    root = Path(root)
    raw = [root / f for f in RAW_BARS]
    if not (root / SUBMISSIONS).exists() or not all(f.exists() for f in raw):
        return frozenset()
    bars = pd.concat([pd.read_parquet(f, columns=["symbol", "date"]) for f in raw], ignore_index=True)
    return frozenset(s for s, k in ma_exits(sorted(set(symbols)), bars, root / SUBMISSIONS).items() if k == "mna")


def anchors_from_files(root: Path, close: pd.DataFrame, dates: Sequence, universe: Mapping, cfg) -> pd.DataFrame:
    """Trusted anchor panel (features.anchors) from the price panel and, if archived, the 8-K filings."""
    from jevtrader.features.anchors import anchor_panel, load_filings

    sub = Path(root) / SUBMISSIONS
    syms = sorted({s for t in dates for s in universe[t]})
    filings = load_filings(sub, syms, cfg) if (cfg.use_filing and sub.exists()) else None
    return anchor_panel(close, dates, universe, filings, cfg)


def input_manifest(root: Path, gen_id: str, gen_dir: Path | None = None) -> dict:
    """Which input files a real run read, with their SHA-256 (recorded by experiments/run_layer_b.py and
    experiments/run_live_decision.py; `gen_dir` overrides results/judgments/<gen_id>, e.g. a stacked directory)."""
    root = Path(root)
    gen = Path(gen_dir) if gen_dir is not None else root / "results/judgments" / gen_id
    out = {}

    def rel(p: Path) -> str:
        return str(p.relative_to(root)) if p.is_relative_to(root) else str(p)

    for name in ("answers.parquet", "features.parquet", "code_features.parquet", RISK_FILE, FIRM_FILE,
                 "evidence.parquet", MANIFEST_FILE, STACK_FILE, str(WORK_FILE)):
        p = gen / name
        out[name] = {"path": rel(p), "sha256": file_sha256(p)} if p.exists() else None
    sub = root / SUBMISSIONS
    out["sec_submissions"] = {"path": str(SUBMISSIONS), "n_files": len(list(sub.glob("*.json")))} if sub.exists() else None
    return out


def exclusion_eligible(risk: pd.DataFrame | None, dates: Sequence, corroboration: int = 2,
                       kinds: Sequence[str] = (), key: str = "n_prob") -> int:
    """Number of risk rows (in `dates`) where an exclusion is at stake, i.e. where the anchor decides between 0 and
    x0.5: >= `corroboration` keys (column `key`) with an exclusion kind, or an S2 exclusion (s2_mult = 0)."""
    if risk is None or risk.empty:
        return 0
    r = risk[risk["t"].isin({pd.Timestamp(d) for d in dates})]
    n = pd.to_numeric(r.get(key, pd.Series(0.0, index=r.index)), errors="coerce").fillna(0.0)
    kind = r.get("problem_kind", pd.Series(None, index=r.index))
    s2 = pd.to_numeric(r.get("s2_mult", pd.Series(1.0, index=r.index)), errors="coerce").fillna(1.0)
    return int(((n >= corroboration) & kind.isin(list(kinds))).sum() + (s2 <= 0).sum())


def anchor_coverage(anchors: pd.DataFrame | None, universe: Mapping, dates: Sequence) -> list[str]:
    """(t, symbol) of U_t without an anchor row, as 'YYYY-MM-DD:SYM' strings."""
    if anchors is None:
        return []
    have = set(zip(pd.to_datetime(anchors["t"]), anchors["symbol"]))
    miss = [f"{pd.Timestamp(t).date()}:{s}" for t in dates for s in universe[t] if (pd.Timestamp(t), s) not in have]
    return miss



def risk_coverage(risk: pd.DataFrame, universe: Mapping, dates: Sequence, text_weeks: Sequence = (),
                  earnings_floor: float = EARNINGS_FLOOR) -> tuple[dict, list[str]]:
    """Whether the risk table lands on the evaluated stock-weeks: (per-week coverage summary, problems). A table that
    does not line up with the decisions switches the sentinel, S2 and earnings rules off without any other sign
    (run_decisions gives multiplier 1 to a (t, symbol) without a row), so this is a data gate like the feature
    coverage check. Problems:
    - alignment: rows dated inside the window on a day that is not a decision date (an off-by-one date, another
      holiday calendar), and rows on a decision date for a symbol outside U_t;
    - earnings: the trusted `sessions_to_earnings` column is a per-(t, U_t) panel (firm_info.parquet has one row per
      member, finite for ~all of them). If it came from firm_info (`attrs['earnings_source']`) or reaches
      `earnings_floor` of U_t in some evaluated week, every evaluated week must reach the floor; finite values that
      all lie outside the evaluated weeks are a problem too. A sparse hand-built column is reported, not gated;
    - sentinel: a week of `text_weeks` (weeks with covered untrusted features) without a single sentinel row in U_t
      (Layer A writes one for every stock-week with news, so a crashed or misdated shard shows up here)."""
    dates = sorted(pd.Timestamp(d) for d in dates)
    dset = set(dates)
    uni = {t: set(universe[t]) for t in dates}
    n_t = pd.Series({t: len(uni[t]) for t in dates}, dtype=float)
    r = risk.assign(t=pd.to_datetime(risk["t"]))
    inwin = r[(r["t"] >= dates[0]) & (r["t"] <= dates[-1])]
    problems: list[str] = []
    off = inwin[~inwin["t"].isin(dset)]
    if len(off):
        problems.append(f"{len(off)} of {len(inwin)} risk rows inside the window are not on a decision date (first "
                        f"{off['t'].min().date()}): misaligned dates")
    on = inwin[inwin["t"].isin(dset)]
    member = np.array([s in uni[t] for t, s in zip(on["t"], on["symbol"])], dtype=bool)
    if (~member).any():
        o = on[~member].iloc[0]
        problems.append(f"{int((~member).sum())} risk rows on a decision date name a symbol outside U_t (first "
                        f"{o['t'].date()}:{o['symbol']})")
    on = on[member]
    cov: dict = {}

    def summary(share: pd.Series) -> dict:
        return {"mean": float(share.mean()), "min": float(share.min()), "median": float(share.median())}

    if "sessions_to_earnings" in r:
        e_all = np.isfinite(pd.to_numeric(r["sessions_to_earnings"], errors="coerce").to_numpy(dtype=float))
        e_on = np.isfinite(pd.to_numeric(on["sessions_to_earnings"], errors="coerce").to_numpy(dtype=float))
        share = (on[e_on].groupby("t")["symbol"].nunique().reindex(dates).fillna(0.0) / n_t).fillna(0.0)
        cov["sessions_to_earnings"] = summary(share)
        if e_all.any() and not e_on.any():
            problems.append("the earnings column has values, but none on an evaluated stock-week: the earnings rule "
                            "is off for the whole window")
        elif risk.attrs.get("earnings_source") == FIRM_FILE or (share >= earnings_floor).any():
            low = share[share < earnings_floor]
            problems += [f"earnings@{t.date()}: {x:.2f} of U_t" for t, x in low.items()]
    present = [c for c in UNTRUSTED_RISK_COLUMNS if c in r]
    if present:
        has = on[present].notna().any(axis=1).to_numpy()
        rows = on[has].groupby("t")["symbol"].nunique().reindex(dates).fillna(0.0)
        cov["sentinel_rows"] = summary((rows / n_t).fillna(0.0))
        problems += [f"no sentinel row@{pd.Timestamp(t).date()} (a week with news features)"
                     for t in sorted(pd.Timestamp(x) for x in text_weeks) if rows.get(t, 0.0) == 0]
    return cov, problems
