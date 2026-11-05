"""FAITH (S4): faithfulness interventions with null arms (design_v1 §10.3, §13.8; pre-registration draft §9, reduced).

Registered arms, in this order (arms that do not fit the FAITH sub-cap are cut from the END of the list and reported
"not run"; `projection.cut_from_end`):

| Arm | Sample | Intervention (each unit) | Null arm | Paid requests per unit |
|---|---|---|---|---|
| T3 | 500 N3 states of MAIN (W_clean) | `mirror`: every this-week effect level k -> 4 - k; `complement`: the key event's `important` bin -> its complement | the unedited N3 state re-asked (no cache) | 3 N3 |
| T2 | 250 learned W_clean stock-weeks with >= 3 items, stratified by the card's action | delete the card's top-1 / top-3 items, a random 1 / 3 (seeded), and keep only the top 3 (sufficiency); aggregates recomputed in code, N2/N3 re-asked | N3 re-asked on the unedited stock-week | <= 5 N3 (+ N2 of changed clusters) + 1 |
| T4-TAL | 250 of TAL's S30 x last-60 stock-weeks | the news analyst's report replaced by a neutral report, or its stance line inverted; bull, bear, trader, risk re-run | the unedited downstream chain re-run (no cache) | <= 12 generative calls |
| T4-H | 250 E_t stock-weeks with >= 2 items | H re-asked without the item of largest |E[effect] - 2| x P(important), or without a random item | H re-asked on the unedited state (no cache) | 3 H |

Definitions frozen here (they are registered by this module's hash, `faith_sha256`):
- T3 `mirror` rewrites the level name of every `this_week_events[i].effect` ("clearly harmful" <-> "clearly
  helpful", "mildly harmful" <-> "mildly helpful"; "neutral or mixed" and "not available" unchanged; the probability
  bin kept). Implied direction of the change in E[net_effect]: -sign(sum of (level - 2) over this week's events); no
  implied direction when the sum is 0.
- T3 `complement`: the key event is the this-week row with the largest |level - 2| x P(important bin midpoint)
  (ties: the first row); its `important` bin is replaced by the complementary bin (very likely <-> very unlikely,
  likely <-> unlikely). Implied direction of the change in |E[net_effect] - 2|: down if the bin was likely or very
  likely, up if unlikely or very unlikely; a key event with an `uncertain` or unavailable bin has no complement (the
  unit gets no complement request).
- T2's card ranking: an item's weight is |rho m d (1 - u/2)(1 + ln(w)/4)| x novelty of its kept cluster
  (`programs.news.influence` x N2 novelty, from the N23 card), items outside kept clusters weigh 0; ties by eid. Random
  items: the smallest sha256(seed | t | symbol | eid). Metrics (after the unlock): |dL| of the stock-week's untrusted
  logit (typed slopes and cap), R_k = median |dL_top-k| / median |dL_random-k|, flip rate of D-sel-EW membership
  above the null arm's, sufficiency |dL_keep-top-3|.
- T4-TAL's neutral report is `NEUTRAL_REPORT` (below); `invert` swaps "Stance: bullish" and "Stance: bearish" in the
  news report (a neutral stance has no inversion: no request). Metric: final-decision flip rate and dp_outperform
  above the null arm's.
- T4-H: the deleted item is removed from H's evidence (the state is rebuilt by `readers.HolisticReader` without it).
  Metrics: |dE[outlook]| top vs random (R_k) and the flip of the top outlook level above the null arm's.
- Null arm (validity M8): the unedited state is re-asked through a second instance of the same backend WITHOUT the
  response cache (`projection.ReaskProxy`): the identical prompt, answered again.

Sampling reads Layer-A tables only for T3 and T4-TAL. T2 and T4-H need learned quantities (the card's action, E_t),
so their samplers check the W_clean lock first. Replays of unchanged requests are response-cache hits; a T3 state is
used only if its rebuilt N3 request key equals MAIN's (exact replay), walking the hash order until 500.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import math
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from jevtrader.redteam import run as RT

ROOT = RT.ROOT
FAITH_VERSION = "faith-v1"
ARMS = ("T3", "T2", "T4-TAL", "T4-H")  # the registered order (cut from the end)
SIZES = {"T3": 500, "T2": 250, "T4-TAL": 250, "T4-H": 250}
SEEDS = {a: f"FAITH-{a}-www27" for a in ARMS}
RUN_TAG = "FAITH"
MIRROR = {"clearly harmful": "clearly helpful", "clearly helpful": "clearly harmful",
          "mildly harmful": "mildly helpful", "mildly helpful": "mildly harmful"}
LEVELS = ("clearly harmful", "mildly harmful", "neutral or mixed", "mildly helpful", "clearly helpful")
BIN_COMPLEMENT = {"very likely": "very unlikely", "very unlikely": "very likely", "likely": "unlikely",
                  "unlikely": "likely"}
BIN_MID = {"very likely": 0.9, "likely": 0.7, "uncertain": 0.5, "unlikely": 0.3, "very unlikely": 0.1}
NEUTRAL_REPORT = ("News review: the items this week are routine coverage with no material new information about the "
                  "company's business, results or outlook. Nothing in them changes the investment picture.\n"
                  "Stance: neutral")
_EFFECT = re.compile(r"^(?P<level>[a-z ]+?) \((?P<bin>[a-z ]+)\)$")


def faith_sha256() -> str:
    """This module's source plus the frozen constants (registered with the pre-registration, P-8)."""
    blob = {"source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "version": FAITH_VERSION,
            "arms": list(ARMS), "sizes": SIZES, "seeds": SEEDS, "mirror": MIRROR, "bins": BIN_COMPLEMENT,
            "neutral_report": NEUTRAL_REPORT}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()


def _h(*key) -> str:
    return hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()


# ================================================================================================= T3 interventions
def parse_effect(label: str) -> tuple[str | None, str | None]:
    """'clearly helpful (likely)' -> ('clearly helpful', 'likely'); 'not available' -> (None, None)."""
    m = _EFFECT.match(str(label).strip())
    return (m.group("level"), m.group("bin")) if m and m.group("level") in LEVELS else (None, None)


def level_of(label: str) -> int | None:
    lv, _ = parse_effect(label)
    return None if lv is None else LEVELS.index(lv)


def mirror_state(state: Mapping) -> tuple[dict, int | None]:
    """(mirrored state, implied sign of dE[net_effect]) (module doc)."""
    st = json.loads(json.dumps(state))
    total = 0
    for row in st.get("this_week_events", []):
        lv, b = parse_effect(row.get("effect", ""))
        if lv is None:
            continue
        total += LEVELS.index(lv) - 2
        row["effect"] = f"{MIRROR.get(lv, lv)} ({b})"
    return st, (None if total == 0 else -int(math.copysign(1, total)))


def complement_state(state: Mapping) -> tuple[dict | None, int | None, int | None]:
    """(state with the key event's `important` bin complemented, key row index, implied sign of d|E - 2|), or
    (None, key, None) when the key event's bin has no complement (module doc)."""
    rows = state.get("this_week_events", [])
    best, key = -1.0, None
    for i, row in enumerate(rows):
        lv = level_of(row.get("effect", ""))
        w = (abs(lv - 2) if lv is not None else 0) * BIN_MID.get(str(row.get("important", "")), 0.0)
        if w > best:
            best, key = w, i
    if key is None:
        return None, None, None
    b = str(rows[key].get("important", ""))
    if b not in BIN_COMPLEMENT:
        return None, key, None
    st = json.loads(json.dumps(state))
    st["this_week_events"][key]["important"] = BIN_COMPLEMENT[b]
    return st, key, (-1 if b in ("very likely", "likely") else 1)


# ================================================================================================= replay (Layer A)
class Capture:
    """Backend proxy recording every request whose questions include `marker` (e.g. N3's net_effect)."""

    def __init__(self, s1, marker: str = "net_effect"):
        self._s1, self.marker, self.seen = s1, marker, []

    def __getattr__(self, k):
        return getattr(self._s1, k)

    async def ask(self, state, questions, tag: str = "", **kw):
        r = await self._s1.ask(state, questions, tag=tag, **kw)
        if self.marker in questions:
            self.seen.append({"state": state, "questions": dict(questions), "tag": tag, "key": r.request_key,
                              "response": r})
        return r


async def replay_week(t, symbols: Sequence[str], news: pd.DataFrame, context_factory, s1, *,
                      programs: Sequence[str] = ("N01", "N23"), clean_gen: Path | None = None,
                      delete_ids: Sequence = (), variant: frozenset[str] = frozenset(), tag: str = RUN_TAG,
                      program_objects=None, memory_dir: Path | None = None) -> dict:
    """Layer A of one week for `symbols` on a COPY of the news table without `delete_ids` (FAITH T2) or unchanged
    (T3 capture): the context of the clean run, focus = the symbols, N23's memory seeded from the clean generation
    (`run.seed_memory`: the records the clean N23 read). Returns the program outputs' frames and blackboards. No lock:
    this is a Layer-A replay (the W_CLEAN rule allows Layer A); outcome-dependent callers check the lock themselves."""
    import jevtrader.programs.runner as RN
    from jevtrader.data.news import NewsStore
    from jevtrader.memory.store import MemoryStore

    t = pd.Timestamp(t)
    table = news if not len(delete_ids) else news[~news["id"].isin(list(delete_ids))]
    tmp = None
    if memory_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="faith_mem_")
        memory_dir = Path(tmp.name)
    try:
        store = MemoryStore(Path(memory_dir), enforce_gate="gate_off" not in variant)
        if clean_gen is not None:
            RT.seed_memory(store, clean_gen, symbols, t)
        progs = RN.topo_order([RN.resolve_program(n, program_objects) for n in programs])
        names = {p.name for p in progs}
        tagged = RN._Tagged(s1, tag)
        ctx0 = context_factory(t)
        focus = tuple(sorted(set(symbols) & set(ctx0.universe)))
        ctx0 = dataclasses.replace(ctx0, news=NewsStore(table), focus=focus, variant=frozenset(variant))
        frames: dict[str, list[pd.DataFrame]] = {k: [] for k in RT.TABLES}
        boards: dict[str, Any] = {}
        for p in progs:
            deps = [d for d in getattr(p, "depends_on", ()) if d in names]
            readable = {*getattr(p, "depends_on", ()), *getattr(p, "memory_reads", ()),
                        *((p.name,) if getattr(p, "week_ordered", False) else ())}
            view = store.view(t, ctx0.cutoff_utc, deps=deps, history=None, readable=readable,
                              clearance=RN._clearance(p))
            ctx = dataclasses.replace(ctx0, memory=view)
            outs = await p.run(ctx, tagged)
            RN._validate(outs, p, ctx)
            RN._apply_memory(store, outs, p, ctx)
            for o in outs:
                boards.update(o.blackboard or {})
            for kk, df in RN._frames(outs, getattr(p, "taint", "W"), tagged.run_tag, "").items():
                if len(df):
                    frames[kk].append(df)
        out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=RN.COLUMNS[k]))
               for k, v in frames.items()}
        out["blackboard"] = boards
        return out
    finally:
        if tmp is not None:
            tmp.cleanup()


# ================================================================================================= T3
def sample_t3(main_requests: pd.DataFrame, n: int | None = None, seed: str = SEEDS["T3"],
              window: tuple[str, str] = ("2024-07-05", "2026-09-11")) -> pd.DataFrame:
    """MAIN's N3 stock-weeks in W_clean, in the registered order: allocated over weeks in proportion to their N3 count
    (`run.stratified_sample` rule, for the whole frame: every row keeps its within-week rank), then walked by
    (allocation round, rank). Columns t, symbol, request_key, order. Layer A only."""
    r = main_requests[main_requests["program"] == "N3"][["t", "symbol", "request_key"]].copy()
    r["t"] = pd.to_datetime(r["t"])
    r = r[(r["t"] >= pd.Timestamp(window[0])) & (r["t"] <= pd.Timestamp(window[1]))]
    r = r.drop_duplicates(["t", "symbol"]).reset_index(drop=True)
    r["_rank"] = [_h(seed, pd.Timestamp(t).date(), s) for t, s in zip(r["t"], r["symbol"])]
    r = r.sort_values(["t", "_rank"])
    r["_k"] = r.groupby("t").cumcount()
    size = r.groupby("t")["symbol"].transform("size")
    r["_pos"] = (r["_k"] + 0.5) / size  # position within the week: proportional allocation when walked in order
    r = r.sort_values(["_pos", "_rank"]).reset_index(drop=True)
    r["order"] = range(len(r))
    r = r.drop(columns=["_rank", "_k", "_pos"])
    return r if n is None else r.head(n)


async def capture_t3(units: pd.DataFrame, news: pd.DataFrame, context_factory, replay_s1, clean_gen: Path,
                     n: int, on_week: Callable[[str, int], None] | None = None) -> list[dict]:
    """Rebuild N3 states by replaying MAIN week by week (cache hits), keeping a state only if its request key equals
    MAIN's (exact replay), in the sample's order until `n` (module doc). Returns records with the state."""
    want = units.sort_values("order")
    keep: dict[int, dict] = {}
    for t, g in want.groupby("t", sort=True):
        cap = Capture(replay_s1)
        await replay_week(t, list(g["symbol"]), news, context_factory, cap, clean_gen=clean_gen, tag="FAITH/T3cap")
        got = {c["key"]: c for c in cap.seen}
        for u in g.itertuples(index=False):
            c = got.get(u.request_key)
            if c is not None:
                keep[int(u.order)] = {"order": int(u.order), "t": pd.Timestamp(t), "symbol": u.symbol,
                                      "request_key": u.request_key, "state": c["state"], "questions": c["questions"],
                                      "clean": c["response"]}
        if on_week is not None:
            on_week(str(pd.Timestamp(t).date()), len(keep))
    return [keep[k] for k in sorted(keep)][:n]


async def ask_t3(recs: Sequence[dict], s1, null_s1) -> list[dict]:
    """The T3 requests: mirror, complement (if defined) and the null re-ask per state."""

    async def one(r):
        st, qs = r["state"], r["questions"]
        mst, msign = mirror_state(st)
        cst, key, csign = complement_state(st)
        jobs = {"mirror": s1.ask(mst, qs, tag=f"{RUN_TAG}/T3/N3")}
        if cst is not None:
            jobs["complement"] = s1.ask(cst, qs, tag=f"{RUN_TAG}/T3/N3")
        jobs["null"] = null_s1.ask(st, qs, tag=f"{RUN_TAG}/T3-null/N3")
        res = dict(zip(jobs, await asyncio.gather(*jobs.values())))
        e0 = _E(r["clean"], "net_effect")
        row = {"order": r["order"], "t": r["t"], "symbol": r["symbol"], "request_key": r["request_key"],
               "E_clean": e0, "mirror_sign": msign, "complement_sign": csign, "key_row": key,
               "n_events": len(st.get("this_week_events", []))}
        for k, resp in res.items():
            row[f"E_{k}"] = _E(resp, "net_effect")
            row[f"key_{k}"] = resp.request_key
            row[f"cached_{k}"] = bool(resp.cached)
            row[f"usd_{k}"] = float(resp.cost_usd or 0.0) if not resp.cached else 0.0
        return row

    return list(await asyncio.gather(*(one(r) for r in recs)))


def _E(resp, q: str) -> float | None:
    a = resp.answers.get(q) if resp is not None else None
    return None if a is None or a.probs is None else float(a.expected_score)


def t3_summary(rows: pd.DataFrame) -> dict:
    """Descriptive T3 results (Layer-A answers only): mean |dE| per intervention and for the null arm, the share of
    units moving in the implied direction (intervention vs null, on units with an implied direction)."""
    out = {"n": int(len(rows))}
    d_null = rows["E_null"] - rows["E_clean"]
    out["null_abs_dE_mean"] = _f(d_null.abs().mean())
    for k, sign in (("mirror", "mirror_sign"), ("complement", "complement_sign")):
        if f"E_{k}" not in rows:
            continue
        d = rows[f"E_{k}"] - rows["E_clean"]
        ok = rows[sign].notna() & d.notna()
        if k == "complement":
            mag = (rows[f"E_{k}"] - 2).abs() - (rows["E_clean"] - 2).abs()
            mag_null = (rows["E_null"] - 2).abs() - (rows["E_clean"] - 2).abs()
            moved = (mag * rows[sign] > 0)[ok]
            moved_null = (mag_null * rows[sign] > 0)[ok]
        else:
            moved = (d * rows[sign] > 0)[ok]
            moved_null = (d_null * rows[sign] > 0)[ok]
        out[k] = {"n": int(ok.sum()), "abs_dE_mean": _f(d.abs().mean()), "share_implied": _f(moved.mean()),
                  "share_implied_null": _f(moved_null.mean())}
    return out


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


# ================================================================================================= T2
def card_ranking(card: Mapping) -> list[tuple[str, float]]:
    """Items of an N23 card by their kept cluster's weight (module doc); items outside kept clusters weigh 0."""
    w: dict[str, float] = {it["eid"]: 0.0 for it in card.get("items", [])}
    for c in card.get("clusters", []):
        d = (float(c.get("effect_E", 2.0)) - 2.0) / 2.0
        wt = abs(float(c.get("rho", 1.0)) * float(c.get("p_important") or 0.0) * d
                 * (1.0 - 0.5 * float(c.get("p_unconfirmed") or 0.0))
                 * (1.0 + 0.25 * math.log(max(int(c.get("width", 1)), 1)))) * float(c.get("novelty", 1.0))
        for e in c.get("items", []):
            w[e] = max(w.get(e, 0.0), wt)
    return sorted(w.items(), key=lambda x: (-x[1], x[0]))


def t2_deletions(card: Mapping, t, symbol: str, seed: str = SEEDS["T2"]) -> dict[str, list[str]]:
    """The eids each T2 run deletes: top1, top3, rand1, rand3 and keep3 (every item except the top 3)."""
    ranked = [e for e, _ in card_ranking(card)]
    rnd = sorted(ranked, key=lambda e: _h(seed, pd.Timestamp(t).date(), symbol, e))
    return {"top1": ranked[:1], "top3": ranked[:3], "rand1": rnd[:1], "rand3": rnd[:3], "keep3": ranked[3:]}


def sample_t2(log, n01_counts: Mapping[tuple, int], n: int = SIZES["T2"], seed: str = SEEDS["T2"],
              root: Path | None = ROOT, min_items: int = 3) -> pd.DataFrame:
    """Learned stock-weeks with >= 3 items, stratified by the card's action (entered / held / exited / in band, from
    the typed clean log's books), `run.stratified_sample` by action. Refuses before the unlock (the books are learned)."""
    RT.check_unit_lock({"typed": log}, root)
    rows = []
    for st in log.states:
        t = pd.Timestamp(st.t)
        new = log.targets["sel_ew"].get(st.t, {})
        prev = {s for s, w in st.w_cur.items() if w > 0}
        for s in st.universe:
            if n01_counts.get((t, s), 0) < min_items:
                continue
            inb, was = s in new, s in prev
            action = "entered" if inb and not was else "held" if inb else "exited" if was else "in_band"
            rows.append({"t": t, "symbol": s, "action": action})
    fr = pd.DataFrame(rows, columns=["t", "symbol", "action"])
    return RT.stratified_sample(fr, n, seed, by="action")


def eid_to_ids(news: pd.DataFrame, symbol: str, eids: Sequence[str]) -> list:
    """The feed item ids behind evidence ids (`news.item_eid`) among the symbol's rows."""
    from jevtrader.programs.news import item_eid

    want = set(eids)
    ids = news.loc[news["symbol"] == symbol, "id"].unique()
    return [i for i in ids if item_eid(i) in want]


def _unit_logit(typed_run, t, symbol: str, feats: pd.DataFrame) -> tuple[float, bool]:
    """(the symbol's untrusted logit L-bar^W, its D-sel-EW membership) at week t with the symbol's untrusted features
    replaced by `feats` (typed slopes, scales and cap; eval.bound)."""
    from jevtrader.eval.bound import attacked_week_logits, replay_units

    cfg, log, raw = typed_run.cfg, typed_run.log, typed_run.raw
    feats = feats.assign(feature=feats["feature"].astype(str).map(_catalog))
    feats = feats[feats["feature"].notna()]
    ov = RT.unit_overrides(feats.assign(t=pd.to_datetime(feats["t"])), cfg.allocator.untrusted, t, [symbol])
    lw = attacked_week_logits(cfg, raw, log, t, ov)["L_W_bar"]
    r = replay_units(log, cfg, pd.DataFrame([{"reader": "typed", "t": pd.Timestamp(t), "symbol": symbol}]),
                     week_logits={0: lw}).iloc[0]
    return float(lw.get(symbol, 0.0)), bool(r["in_ew_att"])


def _catalog(name: str):
    from jevtrader.learn.features import catalog_name

    return catalog_name(name)


async def t2_unit(t, symbol: str, card: Mapping, news: pd.DataFrame, context_factory, s1, null_s1, clean_gen: Path,
                  typed_run, program_objects=None, root: Path | None = ROOT) -> dict:
    """One T2 stock-week (module doc): the five deletion runs and the null run, each a Layer-A replay of N01 + N23 for
    the symbol (unchanged requests are cache hits; the null run re-asks N3 without the cache), then dL and the book
    membership against the clean week under the typed slopes. Checks the lock (the logits are learned)."""
    from jevtrader.pipeline.lock import assert_unlocked_for
    from jevtrader.redteam.projection import ReaskProxy

    if root is not None:
        assert_unlocked_for([pd.Timestamp(t)], Path(root))
    t = pd.Timestamp(t)
    st = next(x for x in typed_run.log.states if pd.Timestamp(x.t) == t)
    l0 = float(st.L_W_bar.get(symbol, 0.0))
    in0 = symbol in typed_run.log.targets["sel_ew"].get(st.t, {})
    dels = t2_deletions(card, t, symbol)
    row: dict[str, Any] = {"t": t, "symbol": symbol, "L_clean": l0, "in_ew_clean": in0,
                           "n_items": len(card.get("items", []))}
    runs = {k: dict(s1=s1, delete=eid_to_ids(news, symbol, v)) for k, v in dels.items() if v}
    runs["null"] = dict(s1=ReaskProxy(s1, null_s1, programs=("N3",)), delete=[])
    for k, r in runs.items():
        out = await replay_week(t, [symbol], news, context_factory, r["s1"], clean_gen=clean_gen,
                                delete_ids=r["delete"], tag=f"{RUN_TAG}/T2" if k != "null" else f"{RUN_TAG}/T2",
                                program_objects=program_objects)
        f = out["features"]
        f = f[(f["symbol"] == symbol)]
        lk, ink = _unit_logit(typed_run, t, symbol, f)
        req = out["requests"]
        row.update({f"dL_{k}": lk - l0, f"flip_{k}": ink != in0, f"n_deleted_{k}": len(r["delete"]),
                    f"usd_{k}": float(req.loc[~req["cached"].astype(bool), "usd"].sum()) if len(req) else 0.0})
    return row


def t2_summary(rows: pd.DataFrame) -> dict:
    """R_k = median |dL_top-k| / median |dL_random-k| (k = 1, 3), flip rates against the null arm's, sufficiency."""
    out = {"n": int(len(rows))}
    for k in ("top1", "top3", "rand1", "rand3", "keep3", "null"):
        if f"dL_{k}" in rows:
            out[k] = {"median_abs_dL": _f(rows[f"dL_{k}"].abs().median()), "flip_rate": _f(rows[f"flip_{k}"].mean())}
    for k in (1, 3):
        a, b = out.get(f"top{k}", {}), out.get(f"rand{k}", {})
        if a.get("median_abs_dL") is not None and b.get("median_abs_dL"):
            out[f"R_{k}"] = a["median_abs_dL"] / b["median_abs_dL"]
    return out


# ================================================================================================= T4-TAL
def invert_stance(report: str) -> str | None:
    """'Stance: bullish' <-> 'Stance: bearish' in the report's stance line(s); None for a neutral or missing stance."""
    from jevtrader.baselines.parse import parse_analyst

    p = parse_analyst(report)
    if not p.valid or p.value == "neutral":
        return None
    swap = {"bullish": "bearish", "bearish": "bullish"}
    return re.sub(r"(?im)^(\s*Stance:\s*)(bullish|bearish)\s*$",
                  lambda m: m.group(1) + swap[m.group(2).lower()], report)


def sample_t4_tal(tal_answers: pd.DataFrame, n: int = SIZES["T4-TAL"], seed: str = SEEDS["T4-TAL"]) -> pd.DataFrame:
    fr = tal_answers[["t", "symbol"]].drop_duplicates().copy()
    fr["t"] = pd.to_datetime(fr["t"])
    return RT.stratified_sample(fr, n, seed)


async def ta_downstream(p: Mapping, reports: Mapping[str, str], llm) -> dict:
    """TA-lite's bull, bear, trader and risk calls on the given analyst reports (baselines.llm.TaLite.run's second
    half, verbatim templates), the final decision by the one parser."""
    from jevtrader.baselines.llm import TaLite, _fill
    from jevtrader.baselines.parse import last_json_object, ta_final

    def call(role, user, kind):
        return llm.complete(role, p["system"], user, int(p["max_tokens"][kind]), float(p["decoding"]["temperature"]))

    rep = TaLite.reports_text(dict(reports))
    bull, bear = await asyncio.gather(call("bull", _fill(p["bull_template"], reports=rep), "bull"),
                                      call("bear", _fill(p["bear_template"], reports=rep), "bear"))
    trader = await call("trader", _fill(p["trader_template"], reports=rep, bull=bull.text.strip(),
                                        bear=bear.text.strip()), "trader")
    obj = last_json_object(trader.text)
    proposal = json.dumps(obj, ensure_ascii=False) if obj is not None else trader.text.strip()
    risk = await call("risk", _fill(p["risk_template"], trader_json=proposal, reports=rep), "risk")
    fin = ta_final(trader.text, risk.text)
    calls = [bull, bear, trader, risk]
    return {"decision": fin.decision, "p": fin.p_outperform, "invalid": bool(fin.invalid),
            "usd": float(sum(c.usd for c in calls if not c.cached)), "cached": [bool(c.cached) for c in calls]}


def tal_reports(outputs: pd.DataFrame, requests: pd.DataFrame, t, symbol: str) -> dict[str, str]:
    from jevtrader.baselines.llm import ROLES

    r = requests[(pd.to_datetime(requests["t"]) == pd.Timestamp(t)) & (requests["symbol"] == symbol)]
    txt = dict(zip(outputs["request_key"], outputs["text"]))
    out = {}
    for role in ROLES:
        k = r.loc[r["role"] == role, "request_key"]
        out[role] = txt.get(k.iloc[0], "") if len(k) else ""
    return out


# ================================================================================================= T4-H
def h_deletions(rows: Sequence[Mapping], t, symbol: str, seed: str = SEEDS["T4-H"]) -> dict[str, str | None]:
    """T4-H's deleted items from the typed N01 rows of the stock-week: `top` = largest |E[effect] - 2| x
    P(important) (items without an effect answer weigh 0; ties by eid), `rand` = the smallest sha256(seed | t | symbol
    | eid) among the other items."""
    def weight(r):
        eff = r.get("effect") or {}
        e = sum(int(k) * float(v) for k, v in eff.items()) if eff else 2.0
        imp = (r.get("p") or {}).get("important") or 0.0
        return abs(e - 2.0) * float(imp)

    eids = [r["eid"] for r in rows]
    if len(eids) < 2:
        return {"top": None, "rand": None}
    top = sorted(rows, key=lambda r: (-weight(r), r["eid"]))[0]["eid"]
    rest = sorted((e for e in eids if e != top), key=lambda e: _h(seed, pd.Timestamp(t).date(), symbol, e))
    return {"top": top, "rand": rest[0]}


# ================================================================================================= projection
@dataclass
class ArmProjection:
    arm: str
    high: float
    point: float
    basis: str
    units: int
    detail: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)
