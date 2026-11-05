"""Layer B for one live decision (design_v1 §4.2 steps 4-5, §9.5, §10.1, §13.11): the frozen decision layer replayed
from the first decision of its history window to the live decision d on every matured label, then d's book, cards
and bound check. Pure Python like the rest of Layer B: no network, no model, no broker (tests/
test_pipeline_no_network_layer_b.py covers this module); experiments/run_live_decision.py runs Layer A and the
dry-run executor around it.

Judgment tables of several generations. The history weeks (MAIN on W_clean, the pseudo-forward weeks, earlier live
weeks) and the live week d come from different Layer-A generations. `plan_sources` assigns every week to exactly one
generation (the history generations cover [start, d), the live generation covers d only; an overlap is an error) and
`stack_generations` copies each generation's rows of its weeks into one directory: features, answers, code features,
firm info, risk table and the card evidence entries of the blackboard archive, plus `STACK.json` (the sources, their
weeks, their manifest and table hashes, and each source's focus rule for the coverage gate). Sources must come from
the same backend, pinned model, prompt version, question bank and program versions: the frozen slopes are only
meaningful on one judgment process. Stacking reads files and copies rows; it evaluates nothing.

The book. `decide_live` runs `run_decisions` (the replay's decision loop, the one `decide_week`) over every decision
date of [start, d] with P-clean learning (labels mature at t + 2, every learner cold-starts at `start`). The live
targets are the D-sized book at d. Its holdings w_cur are the live account's actual holdings at d's close (weights at
the raw closes, from our reconciled account record; all cash before the first fill), passed as `holdings`: the
incumbent buffer and the turnover speed then act on what the account holds, and a cash account buys the selected
names only (the replay's own drifted book would carry its sold-down residuals into a cash account). Learning never
reads w_cur (P0, P5), so the slopes, logits and ranks are those of the pure replay. Without `holdings` the book is
the replay's own (the pure-replay semantics). The holdings are logged with the decision, so the whole decision is
reproducible from files (the independent fresh-process replay behind the pre-trade rule LIVE-HASH recomputes it).
The W_clean lock applies to the history window before anything is loaded (pipeline/lock.py): a live decision on a
W_clean history needs the pre-registration unlock.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from jevtrader.eval import bound as B
from jevtrader.learn.allocator import trusted_not_T
from jevtrader.pipeline import inputs as I
from jevtrader.pipeline.layer_b import (
    DecisionConfig,
    DecisionLog,
    InputError,
    code_hash,
    inputs_hash,
    load_inputs,
    run_decisions,
)
from jevtrader.pipeline.lock import LOCKED_WINDOW, assert_unlocked_for, range_touches_locked
from jevtrader.pipeline.registry import canonical_json, config_hash, file_sha256

JUDGMENTS = Path("results/judgments")
STACKED_TABLES = ("features.parquet", "answers.parquet", "code_features.parquet", I.FIRM_FILE, I.RISK_FILE)
IDENTITY_KEYS = ("backend", "model_pinned", "prompt_version")  # MANIFEST backend fields every source must share


class SourceError(ValueError):
    """The judgment generations of a live decision do not form one consistent history."""


@dataclass(frozen=True)
class Source:
    gen_id: str
    weeks: tuple[str, ...]  # ISO decision dates this generation provides

    def to_json(self) -> dict:
        return {"gen_id": self.gen_id, "weeks": list(self.weeks)}


def gen_path(root: Path, gen_id: str) -> Path:
    return Path(root) / JUDGMENTS / gen_id


def read_manifest(gen_dir: Path) -> dict:
    p = Path(gen_dir) / I.MANIFEST_FILE
    if not p.exists():
        raise SourceError(f"{gen_dir} has no {I.MANIFEST_FILE} (not a finished Layer-A generation)")
    return json.loads(p.read_text())


def generation_weeks(gen_dir: Path) -> list[pd.Timestamp]:
    """The decision weeks a Layer-A generation was run for (MANIFEST.json `spec.weeks`)."""
    return sorted(pd.Timestamp(w) for w in read_manifest(gen_dir)["spec"]["weeks"])


def plan_sources(root: Path, decision, start, history_gens: Sequence[str], live_gen: str | None) -> list[Source]:
    """Every week of [start, d) from the history generations (in the order given) and d from the live generation
    (None: the history part only, as checked before Layer A runs). A listed history generation without a week in
    [start, d), a live generation without d, or a week two sources provide is refused; gaps are left to the Layer-B
    input check (`weeks`), which stops the run."""
    d, start = pd.Timestamp(decision), pd.Timestamp(start)
    if start > d:
        raise SourceError(f"history start {start.date()} is after the decision {d.date()}")
    out, seen = [], {}
    for g in history_gens:
        weeks = [w for w in generation_weeks(gen_path(root, g)) if start <= w < d]
        if not weeks:
            raise SourceError(f"history generation {g} has no decision week in [{start.date()}, {d.date()})")
        for w in weeks:
            if w in seen:
                raise SourceError(f"week {w.date()} is provided by both {seen[w]} and {g}")
            seen[w] = g
        out.append(Source(g, tuple(w.date().isoformat() for w in weeks)))
    if live_gen is None:
        return out
    if d not in generation_weeks(gen_path(root, live_gen)):
        raise SourceError(f"live generation {live_gen} was not run for the decision {d.date()}")
    out.append(Source(live_gen, (d.date().isoformat(),)))
    return out


def stack_id(sources: Sequence[Source], root: Path) -> str:
    """Identity of a stack: the sources, their weeks and their manifests' SHA-256."""
    return "stack-" + config_hash([{**s.to_json(), "manifest_sha256": file_sha256(gen_path(root, s.gen_id) /
                                                                               I.MANIFEST_FILE)}
                                   for s in sources])[:16]


def _identity(man: dict) -> dict:
    b = man.get("backend") or {}
    return {"bank_sha256": (man.get("bank") or {}).get("sha256"), **{k: b.get(k) for k in IDENTITY_KEYS}}


def _card_lines(gen_dir: Path, weeks: set[pd.Timestamp]) -> list[str]:
    f = Path(gen_dir) / I.WORK_FILE
    if not f.exists():
        return []
    out = []
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)["record"]
        if rec["key"].startswith(I.CARD_PREFIX) and pd.Timestamp(pd.Timestamp(rec["t"]).date()) in weeks:
            out.append(line)
    return out


def stack_generations(root: Path, sources: Sequence[Source], out_dir: Path) -> dict:
    """Write the stacked judgment directory `out_dir` (see the module doc) and return its STACK.json record. Any file
    an earlier stack left in `out_dir` is replaced."""
    root, out_dir = Path(root), Path(out_dir)
    mans = {s.gen_id: read_manifest(gen_path(root, s.gen_id)) for s in sources}
    ids = {g: _identity(m) for g, m in mans.items()}
    first = next(iter(ids.values()))
    bad = {g: v for g, v in ids.items() if v != first}
    if bad:
        raise SourceError(f"sources come from different backends or banks: {first} vs {bad}")
    versions: dict[str, tuple[str, str]] = {}
    for g, m in mans.items():
        for p, v in (m.get("programs") or {}).items():
            if p in versions and versions[p][1] != v:
                raise SourceError(f"program {p} has version {versions[p][1]} in {versions[p][0]} but {v} in {g}")
            versions.setdefault(p, (g, v))
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_sources = []
    parts: dict[str, list[pd.DataFrame]] = {n: [] for n in STACKED_TABLES}
    cards: list[str] = []
    for s in sources:
        gdir = gen_path(root, s.gen_id)
        weeks = {pd.Timestamp(w) for w in s.weeks}
        files = {}
        for name in STACKED_TABLES:
            p = gdir / name
            if not p.exists():
                continue
            files[name] = file_sha256(p)
            f = pd.read_parquet(p)
            t = pd.to_datetime(f["t"])
            parts[name].append(f.assign(t=t)[t.isin(weeks).to_numpy()])
        lines = _card_lines(gdir, weeks)
        if (gdir / I.WORK_FILE).exists():
            files[str(I.WORK_FILE)] = file_sha256(gdir / I.WORK_FILE)
        cards += lines
        rec_sources.append({**s.to_json(), "manifest_sha256": file_sha256(gdir / I.MANIFEST_FILE), "files": files,
                            "focus_rule": I.focus_rule(mans[s.gen_id].get("context")), "n_cards": len(lines)})
    for name in STACKED_TABLES:
        dst = out_dir / name
        if parts[name]:
            pd.concat(parts[name], ignore_index=True).to_parquet(dst, index=False)
        else:
            dst.unlink(missing_ok=True)
    wf = out_dir / I.WORK_FILE
    if cards:
        wf.parent.mkdir(parents=True, exist_ok=True)
        wf.write_text("".join(line + "\n" for line in cards))
    else:
        wf.unlink(missing_ok=True)
    rec = {"stack_id": stack_id(sources, root), "sources": rec_sources, "identity": first,
           "program_versions": {p: v for p, (_, v) in sorted(versions.items())},
           "files": {n: file_sha256(out_dir / n) for n in (*STACKED_TABLES, str(I.WORK_FILE))
                     if (out_dir / n).exists()}}
    (out_dir / I.STACK_FILE).write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec


# ------------------------------------------------------------------------------------ the live decision
@dataclass
class LiveDecision:
    decision: pd.Timestamp
    start: pd.Timestamp
    cfg: DecisionConfig
    log: DecisionLog
    inputs_sha: str
    universe: tuple[str, ...]
    targets: dict[str, float]  # D-sized final weights at d (the live book), names with a positive weight
    targets_sel_ew: dict[str, float]  # D-sel-EW at d (the signal book; recorded, not traded)
    cards: list[dict]
    predictions: pd.DataFrame  # d's rows of the logit ledger
    cold: bool
    bound: dict
    sizing: list[dict] = field(default_factory=list)  # d's BG-2 per-name sizing log
    multipliers: pd.DataFrame | None = None
    holdings: dict[str, float] | None = None  # w_cur at d: the account's weights (None: the replay's own book)

    def summary(self) -> dict:
        w = pd.Series(self.targets, dtype=float)
        return {"decision": self.decision.date().isoformat(), "start": self.start.date().isoformat(),
                "n_decisions": len(self.log.dates), "cold": self.cold, "cfg_sha": self.cfg.sha,
                "config_files_sha": list(self.cfg.config_files_sha), "inputs_sha": self.inputs_sha,
                "code_sha": code_hash(), "n_targets": int((w > 0).sum()), "invested": float(math.fsum(w)),
                "max_weight": float(w.max()) if len(w) else 0.0, "n_universe": len(self.universe),
                "bound_passed": bool(self.bound.get("passed")), "input_problems_waived":
                    {k: self.log.inputs["problems"][k] for k in self.log.inputs.get("allowed", [])},
                "holdings": "replay book" if self.holdings is None else
                    {"n": len(self.holdings), "invested": float(math.fsum(self.holdings.values()))}}


def live_bound_check(log: DecisionLog, cfg: DecisionConfig, decision) -> dict:
    """BG-5 worst-case reader on the live week alone (eval.bound; the same replays as `bound_check`): no name outside
    the band enters, and no unanchored protected name loses more than half its reference weight, whatever the
    untrusted inputs of week d. Vacuous (and recorded as such) in a cold-start week."""
    d = pd.Timestamp(decision)
    st = [s for s in log.states if pd.Timestamp(s.t) == d]
    ne, ve, npr, vp = 0, [], 0, []
    for s in st:
        ne, ve = B.worst_case_entry(cfg, s)
        npr, vp = B.worst_case_protection(cfg, s)
    not_T = trusted_not_T(cfg.allocator.trusted)
    return {"t": d.date().isoformat(), "learned": bool(st), "n_entry_checks": ne, "entry_violations": ve,
            "n_protection_checks": npr, "protection_violations": vp, "trusted_not_T": not_T,
            "passed": not ve and not vp and not not_T, "K": cfg.K, "J": cfg.J}


def decide_live(cfg: DecisionConfig, decision, start, root: Path, gen_dir: Path, data_root: Path | None = None,
                holdings: Mapping[str, float] | None = None, **load_kw) -> LiveDecision:
    """The live decision d from the judgment tables in `gen_dir` (a stacked directory) and the data under
    `data_root`: P-clean replay over [start, d], the D-sized book at d starting from `holdings` (the account's weights
    at d's close; {} = all cash; None = the replay's own book), d's cards and the live-week bound check. The W_clean
    lock is checked on [start, d] before any input is loaded and again on the actual decision dates."""
    d, start = pd.Timestamp(decision), pd.Timestamp(start)
    if start > d:
        raise ValueError(f"history start {start.date()} is after the decision {d.date()}")
    if range_touches_locked(start, d):  # before loading anything (labels over the window are outcomes)
        assert_unlocked_for([max(start, LOCKED_WINDOW[0])], root)
    inputs = load_inputs(cfg, start, d, root, gen_dir=gen_dir, data_root=data_root, **load_kw)
    dates = sorted(pd.Timestamp(t) for t in inputs.universe if start <= pd.Timestamp(t) <= d)
    if not dates or dates[-1] != d:
        raise InputError(f"{d.date()} is not a decision date of the loaded data (last: "
                         f"{dates[-1].date() if dates else None})")
    assert_unlocked_for(dates, root)
    hold = None if holdings is None else {d: pd.Series(dict(holdings), dtype=float)}
    log = run_decisions(cfg, inputs, dates, holdings=hold)
    pr = log.predictions[log.predictions["t"] == d].reset_index(drop=True)
    tg = {s: float(w) for s, w in log.targets["sized"][d].items() if w > 0}
    ew = {s: float(w) for s, w in log.targets["sel_ew"][d].items() if w > 0}
    iso = d.date().isoformat()
    mult = log.multipliers
    return LiveDecision(
        decision=d, start=start, cfg=cfg, log=log, inputs_sha=inputs_hash(inputs),
        universe=tuple(dict.fromkeys(inputs.universe[d])), targets=dict(sorted(tg.items())),
        targets_sel_ew=dict(sorted(ew.items())), cards=[c for c in log.cards if c["t"] == iso], predictions=pr,
        cold=bool(pr["cold"].iloc[0]) if len(pr) else True, bound=live_bound_check(log, cfg, d),
        sizing=[r for r in log.sizing_names if r["t"] == iso],
        multipliers=None if mult is None else mult[mult["t"] == d].reset_index(drop=True),
        holdings=None if holdings is None else {s: float(w) for s, w in sorted(dict(holdings).items())})


LIVE_OUTPUTS = ("cards.jsonl", "predictions.parquet", "weights.json", "sizing.jsonl", "multipliers.parquet",
                "bound_check.json", "layer_b.json")


def write_live_outputs(out_dir: Path, dec: LiveDecision, extra: dict | None = None) -> dict[str, str]:
    """d's Layer-B outputs under `out_dir` (results/live/<d>/): cards, the logit ledger, both books, the sizing log,
    the risk multipliers, the live-week bound check and a summary. Returns {file: SHA-256}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in LIVE_OUTPUTS:
        (out_dir / name).unlink(missing_ok=True)
    (out_dir / "cards.jsonl").write_text("".join(canonical_json(c) + "\n" for c in dec.cards))
    dec.predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "weights.json").write_text(json.dumps({"sized": dec.targets, "sel_ew": dec.targets_sel_ew},
                                                     indent=1, sort_keys=True))
    (out_dir / "sizing.jsonl").write_text("".join(canonical_json(r) + "\n" for r in dec.sizing))
    if dec.multipliers is not None:
        dec.multipliers.to_parquet(out_dir / "multipliers.parquet", index=False)
    (out_dir / "bound_check.json").write_text(json.dumps(dec.bound, indent=1, sort_keys=True, default=str))
    (out_dir / "layer_b.json").write_text(json.dumps({**dec.summary(), "inputs": dec.log.inputs, **(extra or {})},
                                                     indent=1, sort_keys=True, default=str))
    return {n: file_sha256(out_dir / n) for n in LIVE_OUTPUTS if (out_dir / n).exists()}


def cards_sha(cards: Sequence[dict]) -> str:
    return config_hash([canonical_json(c) for c in cards])
