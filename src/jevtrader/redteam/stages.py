"""RT-S after the unlock: the registered sample, the stages and arms, the clean reference runs, and resumable attacked
Layer-A generation on the attacked units only (pre-registration draft §3.1, §10, §13; design_v2_delta RB-5, RB-15).

Order (pre-registration draft §13 item 3; each paid step runs only if its own preflight fits):
1. `in_band_frame`: E_t, the in-band names R^T <= K + J of every learned decision of the typed clean Layer-B run (P-1).
2. Clean readers on E_t (`clean_reader_spec`): H, H-spot, H-guard, each its own generation and run tag (H, HSPOT,
   HGUARD; sub-caps 1.50 / 2.00 / 2.00). H-guard reads N01's rows of the same week, so its generation also runs N01
   (MAIN's requests: response-cache hits).
3. The C1 sample (`redteam.run.sample_units` with `REGISTERED_SAMPLE`, guarded by the W_clean lock): 200 promotion
   units x A1-A7 (and the stage-5 demotion strata), from the typed and holistic clean logs.
4. NODEF clean on the units' week-t stock-weeks (`nodef_clean_spec`, tag NODEF, sub-cap 3.00): the reference and the
   memory seed of the nodef arm.
5. The attacked stages (`STAGES`, all under the RT-S sub-cap 12.00, tags `RT-S/<arm>/<program>`), in order:
   `1_typed_H` (N01, N23, H, LM), `2_guard_spot` (N01 again, from the cache, for H-guard; H-guard, H-spot),
   `3_ta_lite` (TA-lite under attack: NOT built here, reported "not run" by the driver), `4_nodef` (N01, N23 with the
   nodef variant, seeded from the NODEF clean generation), `5_demotion` (A8-1, A8-2 on the demotion units; runs only
   if the headroom after stages 1-4 covers its preflight). HS is cut (sub-cap 0): no arm runs it.

`run_stage` runs `redteam.run.arun_batch` batch by batch and skips a batch whose directory already holds a MANIFEST
for the same batch, arm and templates hash (resume). Nothing here decides a sample size or a family: they are the
registered ones.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from jevtrader.redteam import attacks as A
from jevtrader.redteam import run as RT

ROOT = RT.ROOT
REGISTERED_SAMPLE = {"n_promotion": 200, "n_bound": 0, "n_demotion": 100, "n_anchor": 25, "seed": RT.SEED}
DEMOTION_FAMILIES = ("A8-1", "A8-2")  # pre-registration draft §10: demotion reduced to A8-1, A8-2 (stage 5)
CLEAN_READERS = {"H": "H", "H-spot": "HSPOT", "H-guard": "HGUARD"}  # reader -> run tag (its own sub-cap)
NODEF_TAG = "NODEF"


@dataclass(frozen=True)
class Stage:
    name: str
    arm: str  # output directory and tag segment: results/redteam/rts/<arm>/<batch>, ledger RT-S/<arm>/<program>
    programs: tuple[str, ...]
    families: Mapping[str, tuple[str, ...]]  # unit kind -> families
    variant: frozenset[str] = frozenset()
    clean: str = "main"  # the clean generation that seeds N23's memory: "main" | "nodef"
    readers: tuple[str, ...] = ()  # the readers scored from this arm's tables
    built: bool = True
    note: str = ""


STAGES: tuple[Stage, ...] = (
    Stage("1_typed_H", "typed", ("N01", "N23", "H", "LM"), {RT.PROMOTION: A.C1_PROMOTION},
          readers=("typed", "H", "LM", "R-med", "R-vote")),
    Stage("2_guard_spot", "guards", ("N01", "H-guard", "H-spot"), {RT.PROMOTION: A.C1_PROMOTION},
          readers=("H-guard", "H-spot"), note="N01 is re-run for H-guard's dependency; its requests are stage 1's"),
    Stage("3_ta_lite", "ta-lite", (), {RT.PROMOTION: ("A1", "A2", "A3", "A6")}, built=False,
          note="TA-lite under attack (100 TAL stock-weeks x A1, A2, A3, A6): not built in this driver"),
    Stage("4_nodef", "nodef", ("N01", "N23"), {RT.PROMOTION: A.C1_PROMOTION}, frozenset({"nodef"}), clean="nodef",
          readers=("nodef",)),
    Stage("5_demotion", "demotion", ("N01", "N23", "H", "LM"), {RT.DEMOTION: DEMOTION_FAMILIES},
          readers=("typed", "H", "LM"), note="runs only if the headroom after stages 1-4 covers its preflight"),
)
STAGE = {s.name: s for s in STAGES}


def stage_units(stage: Stage, units: pd.DataFrame) -> pd.DataFrame:
    return units[units["kind"].isin(list(stage.families))].reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ E_t and logs
def in_band_frame(log, K: int, J: float, root: Path | None = ROOT) -> pd.DataFrame:
    """E_t (P-1): the names with R^T <= K + J at every learned (not cold) decision of the typed clean log, with R^T.
    R^T is learned, so this refuses before the unlock on W_clean weeks (`run.check_unit_lock`)."""
    RT.check_unit_lock({"typed": log}, root)
    p = log.predictions
    p = p[~p["cold"].astype(bool) & (p["R_T"] <= K + J)][["t", "symbol", "R_T"]].copy()
    p["t"] = pd.to_datetime(p["t"])
    return p.sort_values(["t", "R_T", "symbol"]).reset_index(drop=True)


def focus_by_week(frame: pd.DataFrame) -> dict[pd.Timestamp, tuple[str, ...]]:
    return {pd.Timestamp(t): tuple(sorted(g["symbol"])) for t, g in frame.groupby("t")}


def stack_raw(raw: pd.DataFrame, gen_dirs: Sequence[Path], industry: Mapping[str, str]) -> pd.DataFrame:
    """The clean raw features of the typed generation plus the Layer-A features of other generations (a reader's
    clean run on E_t): rows of the extra generations for features the base does not have. A reader's feature exists
    only where its generation ran (E_t); elsewhere it is uncovered (P-1)."""
    from jevtrader.pipeline.layer_b import load_judgment_features

    parts = [raw]
    have = set(raw["feature"])
    for g in gen_dirs:
        f = load_judgment_features(Path(g), industry)
        f = f[~f["feature"].isin(have)]
        have |= set(f["feature"])
        parts.append(f)
    out = pd.concat(parts, ignore_index=True)
    out.attrs = dict(raw.attrs)
    return out


def reader_log(base_cfg, reader: str, inputs, window: str, root: Path = ROOT, gen_dirs: Sequence[Path] = (),
               allow_missing: Sequence[str] = ()):
    """A reader's clean Layer-B run (variant V-reader-<reader>, or the base for "typed") on `inputs` with the reader's
    generation stacked in; not registered, not written. The typed reader's own window check applies (`evaluate`
    refuses a locked window before the unlock)."""
    from jevtrader.pipeline.layer_b import evaluate
    from jevtrader.pipeline.variants import variant_config

    cfg = base_cfg if reader == "typed" else variant_config(base_cfg, f"V-reader-{reader}")
    if allow_missing:
        cfg = dataclasses.replace(cfg, allow_missing=tuple(sorted(set(cfg.allow_missing) | set(allow_missing))))
    if gen_dirs:
        inputs = dataclasses.replace(inputs, raw=stack_raw(inputs.raw, gen_dirs, inputs.industry or {}))
    res = evaluate(cfg, window, inputs, root=root, register=False, write=False)
    return cfg, res, inputs


# ------------------------------------------------------------------------------------------------ clean runs
def focused_factory(builder: Callable[[pd.Timestamp], Any], focus: Mapping[pd.Timestamp, Sequence[str]]):
    """A context factory whose week-t focus is `focus[t]` (names outside U_t dropped)."""

    def build(t):
        ctx = builder(pd.Timestamp(t))
        names = tuple(s for s in focus.get(pd.Timestamp(t), ()) if s in set(ctx.universe))
        return dataclasses.replace(ctx, focus=names)

    build.describe = lambda: {"focused": getattr(builder, "describe", lambda: {})(),  # noqa: E731
                              "focus_sha": _focus_sha(focus)}
    return build


def _focus_sha(focus: Mapping[pd.Timestamp, Sequence[str]]) -> str:
    import hashlib

    blob = json.dumps({str(pd.Timestamp(t).date()): sorted(v) for t, v in sorted(focus.items())}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def clean_reader_spec(reader: str, focus: Mapping[pd.Timestamp, Sequence[str]], builder, s1, out_root: Path,
                      gen_id: str | None = None, data_freeze: str = "", program_objects=None):
    """The GeneratorSpec of one clean reader run on E_t (module doc, step 2). H-guard also runs N01 (cache hits)."""
    from jevtrader.programs.runner import GeneratorSpec

    if reader not in CLEAN_READERS:
        raise ValueError(f"clean reader runs: {sorted(CLEAN_READERS)}")
    programs = ["N01", reader] if reader == "H-guard" else [reader]
    tag = CLEAN_READERS[reader]
    return GeneratorSpec(gen_id=gen_id or f"{tag.lower()}_emu_wclean_et", weeks=sorted(focus), programs=programs,
                         run_tag=tag, data_freeze=data_freeze, out_root=Path(out_root), s1=s1,
                         context_factory=focused_factory(builder, focus), program_objects=program_objects,
                         notes=f"redteam/stages clean {reader} on E_t (P-1); focus sha {_focus_sha(focus)[:16]}")


def nodef_clean_spec(units: pd.DataFrame, builder, s1, out_root: Path, gen_id: str = "nodef_emu_wclean_units",
                     data_freeze: str = "", program_objects=None):
    """NODEF clean on the units' week-t stock-weeks (module doc, step 4): N01 + N23 with the nodef variant."""
    from jevtrader.programs.runner import GeneratorSpec

    focus = focus_by_week(units[["t", "symbol"]].drop_duplicates())
    return GeneratorSpec(gen_id=gen_id, weeks=sorted(focus), programs=["N01", "N23"], run_tag=NODEF_TAG,
                         variant=frozenset({"nodef"}), data_freeze=data_freeze, out_root=Path(out_root), s1=s1,
                         context_factory=focused_factory(builder, focus), program_objects=program_objects,
                         notes=f"redteam/stages NODEF clean on the C1 units' stock-weeks; focus sha "
                               f"{_focus_sha(focus)[:16]}")


# ------------------------------------------------------------------------------------------------ attacked stages
@dataclass
class StagePlan:
    stage: Stage
    batches: list
    skipped: pd.DataFrame
    units: int = 0
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"stage": self.stage.name, "arm": self.stage.arm, "programs": list(self.stage.programs),
                "families": {k: list(v) for k, v in self.stage.families.items()}, "units": self.units,
                "batches": len(self.batches), "unit_families": sum(len(b.units) for b in self.batches),
                "attacked_stock_weeks": sum(len(b.affected) * len(b.weeks) for b in self.batches),
                "skipped": self.skipped.groupby("reason").size().to_dict() if len(self.skipped) else {},
                "templates_sha256": A.templates_sha256()}


def plan_stage(stage: Stage, news: pd.DataFrame, units: pd.DataFrame, calendar, *, eligible_later=None,
               names=None) -> StagePlan:
    u = stage_units(stage, units)
    if not stage.built or not len(u):
        return StagePlan(stage, [], pd.DataFrame(columns=["unit_id", "family", "reason"]), len(u))
    batches, skipped = RT.plan_batches(news, u, calendar, families=dict(stage.families),
                                       eligible_later=eligible_later, names=names)
    return StagePlan(stage, batches, skipped, len(u))


def _batch_done(d: Path, batch, arm: str) -> bool:
    m = d / "MANIFEST.json"
    if not m.exists():
        return False
    try:
        man = json.loads(m.read_text())
    except ValueError:
        return False
    return (man.get("arm") == arm and man.get("templates_sha256") == A.templates_sha256()
            and man.get("batch", {}).get("batch_id") == batch.batch_id
            and man.get("batch", {}).get("units") == json.loads(json.dumps(batch.to_json()["units"], default=str)))


async def arun_stage(plan: StagePlan, news: pd.DataFrame, context_factory, s1, out_root: Path,
                     clean_gen: Path | None, root: Path = ROOT, run_tag: str = "RT-S", program_objects=None,
                     resume: bool = True, on_batch: Callable[[Path], None] | None = None) -> list[Path]:
    """Attacked Layer A for every batch of a stage (sequential; `run.arun_batch`), skipping finished batches."""
    st = plan.stage
    cfg = RT.AttackRunConfig(out_root=Path(out_root), arm=st.arm, programs=st.programs, variant=st.variant,
                             clean_gen=clean_gen, run_tag=run_tag, root=Path(root), program_objects=program_objects)
    dirs = []
    for b in plan.batches:
        d = Path(out_root) / st.arm / b.batch_id
        if not (resume and _batch_done(d, b, st.arm)):
            d = await RT.arun_batch(b, news, context_factory, s1, cfg)
        dirs.append(d)
        if on_batch is not None:
            on_batch(d)
    return dirs


def write_plan(plan: StagePlan, out_root: Path) -> Path:
    d = Path(out_root) / plan.stage.arm
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps({**plan.to_json(), "batch_list": [b.to_json() for b in plan.batches]},
                                            indent=1, default=str))
    plan.skipped.to_parquet(d / "skipped.parquet", index=False)
    return d


# ------------------------------------------------------------------------------------------------ nodef scoring
def score_nodef_batch(batch_dir: Path | str, typed_run, clean_feats: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    """The nodef arm's replay (pre-registration draft §3.1, stated deviation): nodef has no clean U100 run, so its
    features are scored with the TYPED reader's learned slopes, scales and cap (`typed_run`: its config, log and raw);
    competitors keep their typed clean features. The clean reference of a unit is the typed week with the unit's
    affected names carrying their NODEF clean features (`clean_feats`: the NODEF clean generation's features,
    `run.attacked_features` of that directory); the attacked state carries the attacked nodef features instead.
    Success = the name enters D-sel-EW under attack; `eligible` = it is not in the nodef-clean book. One row per unit
    (offset 0 only). Checks the W_clean lock on the scored weeks first."""
    from jevtrader.eval.bound import attacked_week_logits, replay_units
    from jevtrader.pipeline.lock import assert_unlocked_for

    batch_dir = Path(batch_dir)
    man = json.loads((batch_dir / "MANIFEST.json").read_text())
    b = man["batch"]
    eval_t = pd.Timestamp(b["eval_t"])
    assert_unlocked_for([eval_t], root)
    att = RT.attacked_features(batch_dir)
    cfg, log, raw = typed_run.cfg, typed_run.log, typed_run.raw
    if not any(st.t == eval_t for st in log.states):
        return pd.DataFrame([{"unit_id": u["unit_id"], "family": b["family"], "kind": u["kind"], "reader": "nodef",
                              "t_attack": pd.Timestamp(b["t"]), "eval_t": eval_t, "symbol": u["symbol"],
                              "scored": False, "eligible": None, "success": None} for u in b["units"]])
    unt = cfg.allocator.untrusted
    rows, lw_clean, lw_att = [], {}, {}
    for pos, u in enumerate(b["units"]):
        ov_c = RT.unit_overrides(clean_feats, unt, eval_t, u["affected"])
        ov_a = RT.unit_overrides(att, unt, eval_t, u["affected"])
        lw_clean[pos] = attacked_week_logits(cfg, raw, log, eval_t, ov_c)["L_W_bar"]
        lw_att[pos] = attacked_week_logits(cfg, raw, log, eval_t, {**ov_c, **ov_a})["L_W_bar"]
        rows.append({"reader": "nodef", "t": eval_t, "symbol": u["symbol"]})
    frame = pd.DataFrame(rows)
    ref = replay_units(log, cfg, frame, week_logits=lw_clean).to_dict("records")
    hit = replay_units(log, cfg, frame, week_logits=lw_att).to_dict("records")
    out = []
    for u, rc, ra in zip(b["units"], ref, hit):
        clean_in = bool(rc["in_ew_att"])
        out.append({"unit_id": u["unit_id"], "family": b["family"], "kind": u["kind"], "batch_id": b["batch_id"],
                    "t_attack": pd.Timestamp(b["t"]), "eval_t": eval_t, "offset": 0, "symbol": u["symbol"],
                    "reader": "nodef", "scored": True,
                    "in_ew_clean": clean_in, "in_ew_att": bool(ra["in_ew_att"]),
                    "in_sized_clean": bool(rc["in_sized_att"]), "in_sized_att": bool(ra["in_sized_att"]),
                    "entered_ew": bool(ra["in_ew_att"]) and not clean_in,
                    "entered_sized": bool(ra["in_sized_att"]) and not bool(rc["in_sized_att"]),
                    "eligible": not clean_in, "success": bool(ra["in_ew_att"]) and not clean_in,
                    "n_overrides_clean": len(RT.unit_overrides(clean_feats, unt, eval_t, u["affected"])),
                    "n_overrides_att": len(RT.unit_overrides(att, unt, eval_t, u["affected"]))})
    return pd.DataFrame(out)
