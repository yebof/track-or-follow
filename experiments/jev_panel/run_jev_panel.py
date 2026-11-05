"""Jev panel on Cloudflare Workers AI (design_v1 §11.3 "Jev panel rules"; decisions 2026-09-29 D-13).

Which generations are re-asked on Jev, what that costs, how each row is labelled, and the exact commands. The paid path
refuses to run without Cloudflare credentials and an explicit, positive JEV-CF sub-cap.

    .venv/bin/python experiments/jev_panel/run_jev_panel.py preflight                  # free: no model call, no network
    .venv/bin/python experiments/jev_panel/run_jev_panel.py prepare --component forward  # free: rebuild + verify states
    .venv/bin/python experiments/jev_panel/run_jev_panel.py prepare --component h2frame  # free (CPU, about 115 weeks)
    .venv/bin/python experiments/jev_panel/run_jev_panel.py dry --component aligned2000  # free: paid path, mock transport
    .venv/bin/python experiments/jev_panel/run_jev_panel.py run --component canary       # PAID; refuses unless authorized
    .venv/bin/python experiments/jev_panel/run_jev_panel.py validate --component aligned2000

**Components** (preflight order; each is its own generation `results/judgments/jevcf_jev-1.13.0_<name>/`; requests
are exactly the source generation's (state, questions), rebuilt and checked against its emulator request keys):
- `canary`: the N1.attributes requests of the first N_CANARY units of the aligned sample, re-asked without the response
  cache (§11.3 identity: a silent model change shows as changed answers); `run` re-asks it first when the latest canary
  is older than CANARY_MAX_AGE_DAYS.
- `aligned2000`: the NG/VERB aligned 2,000-item N1 sample (`verb_llm-verbalized_wclean_items2000`; NG's 500 nested),
  the frozen bank's five N1 requests per unit, from VERB's verified states.
- `probes`: every D1 (recall) and D2 (re-identification) job of `probe_emu_d1d2` and `probe_emu_d1_reask`.
- `forward`: N1 items created after the release date: MAIN's pseudo-forward week 2026-09-18 now, then each live
  decision's week once its generation (`results/judgments/live_<d>_*`) exists (scheduled weeks from the NYSE calendar
  of `jevtrader.live.jobs` up to `--forward-until`).
- `h2frame`: the registered H2 frame (MAIN's C2-eligible W_clean item units; `results/analysis/h2_frame.json`), five N1
  requests each; the aligned 2,000 are nested, so their requests are response-cache hits.

**Labels** (§11.3). The docs give no training cutoff; the release date of jev-1.13.0 (2026-09-15, docs/jev) bounds its
training data. An item created on a later UTC day is `post-release`; every other row is `contamination unverified`.
No Jev-vs-emulator performance test is run on W_clean; admissible Jev evidence per component is listed in the output.

**Tokens and money.** Base = the backend's own estimate (o200k proxy of the `input` JSON, bytes / 3.5, plus the
per-request overhead measured on the archived examples); high = bytes / 2.5 plus the same overhead (an assumption: a
smaller-vocabulary tokenizer on JSON). Price $0.042 per 1M input tokens, output free (docs/jev/cloudflare_jev.md).
Where states are not rebuilt yet (h2frame, forward), tokens are MAIN's measured emulator tokens of the same requests
times the per-group Jev/emu ratio measured on the aligned sample. Requests already in the response cache under `jev-cf`
are free; in preflight order, a request counted by an earlier component is a cache hit later (`incremental`).

**Budget.** Jev is billed to the user's Cloudflare account outside the $250 LLM budget (CLAUDE.md; D-13); its spend is
booked in `backends.jev-cf.ledger_path`, whose hard cap and sub-cap are `ledger.sub_caps_usd.JEV-CF` (0 = not
authorized). `run` refuses (exit 2) without CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN (values never printed) or a
positive JEV-CF sub-cap, and aborts (exit 3) if the component's base projection exceeds the sub-cap headroom; the paid
process is bounded by a guard of that headroom and the backend's own per-call checks.

Output of `preflight`: experiments/jev_panel/out/preflight.json. Nothing here reads a return, a label or any outcome.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT / "experiments", ROOT / "experiments" / "ng"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_gen_common as C

from jevtrader.s1 import cloudflare_jev as CF
from jevtrader.s1.types import canonical_json, state_sha256

log = logging.getLogger("run_jev_panel")
RUN = CF.RUN
BACKEND = "jev-cf"
OUT_DIR = ROOT / "experiments" / "jev_panel" / "out"
PREFLIGHT = OUT_DIR / "preflight.json"
JUDG = ROOT / "results" / "judgments"
MAIN_DIR = JUDG / "main_emu_wclean_u100"
VERB_DIR = JUDG / "verb_llm-verbalized_wclean_items2000"
PROBE_DIRS = (JUDG / "probe_emu_d1d2", JUDG / "probe_emu_d1_reask")
H2_FRAME = ROOT / "results" / "analysis" / "h2_frame.json"
LIVE_ROOT = ROOT / "data" / "live" / "root"
SCRIPT = "experiments/jev_panel/run_jev_panel.py"
PY = ".venv/bin/python"
COMPONENTS = ("canary", "aligned2000", "probes", "forward", "h2frame")
GEN_SUFFIX = {"canary": "canary", "aligned2000": "wclean_items2000", "probes": "d1d2", "forward": "fwd",
              "h2frame": "wclean_h2frame"}
GROUPS_ALL = ("about_focal", "instructs_reader", "promotional", "new_development", "attributes")
W_CLEAN = ("2024-07-05", "2026-09-11")  # configs/system_v1.yaml windows.W_clean (checked in `frame_units`)
EXCLUDED_TYPE = "stock_activity_or_opinion"
N_CANARY = 50
CANARY_MAX_AGE_DAYS = 7
SHARD = 500
LABEL_PRE = "contamination unverified"
LABEL_POST = "post-release"
HIGH_BYTES_PER_TOKEN = 2.5
FORWARD_UNTIL = "2026-10-16"  # the last planned live decision (docs/impl/live_runbook.md); `--forward-until`
RULES = {  # design_v1 §11.3 "Jev panel rules" (validity M5, reviewer B3), as the output states them
    "labels": f"rows on pre-release weeks are '{LABEL_PRE}': the docs give no training cutoff; an item created after "
              f"the release date ({CF.RELEASE_DATE}) is '{LABEL_POST}'",
    "no_performance_test": "no Jev-vs-emulator performance test is run on W_clean",
    "admissible": ["W_fwd weeks after the logged release_date (descriptive)",
                   "SEC-verifiable concept agreement (C2 secondary: 8-K item codes)",
                   "C1 robustness, including transfer of the emulator's adaptive attacks", "cost and latency",
                   "faithfulness, with the Jev re-ask floor measured first", "D1/D2 probes as lower bounds"],
    "forward_trading": "not planned: live decisions stay on the emulator (Plan B); a Jev slope would be capped at the "
                       "emulator's W_clean slope first",
}
ADMISSIBLE = {
    "canary": ["identity: a silent model change behind typesafe/jev (answers compared launch to launch)"],
    "aligned2000": ["8-K concept agreement on the aligned units (C2 secondary)",
                    "agreement with emu, VERB and NG on identical requests (descriptive)", "cost and latency"],
    "probes": ["D1 recall and D2 re-identification as lower bounds"],
    "forward": ["post-release item validity, E[effect] vs later reactions (descriptive)", "cost and latency"],
    "h2frame": ["8-K concept agreement on the registered frame (C2 secondary)", "cost and latency"],
}
NOT_ADMISSIBLE = {
    "aligned2000": ["a Jev-vs-emulator test of the E[effect]-vs-AR statistic (W_clean, contamination unverified)"],
    "h2frame": ["H2 on Jev as a confirmatory or comparative test (W_clean, contamination unverified; descriptive only)"],
    "probes": ["a null D1/D2 result as proof of no memorization"],
}
NOT_PLANNED = [
    {"what": "full Layer-A Jev shadow book on the forward weeks (every program, then Layer B)",
     "why": "not in the D-13 panel; needs the live Layer-A runner on backend jev-cf and a new calibration namespace"},
    {"what": "H, HS and the FID/V3 fidelity set on Jev (design_v1 §11.3 `make jev-rows`)",
     "why": "not in the D-13 panel; the Jev re-ask floor would come first for any faithfulness row"},
]


# ------------------------------------------------------------------------------------------ config, env, labels
def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "backends.yaml").read_text())


def load_env() -> dict[str, str]:
    """The environment with .env underneath (values are used for presence checks only, never printed)."""
    env: dict[str, str] = {}
    try:
        from dotenv import dotenv_values

        env.update({k: v for k, v in dotenv_values(ROOT / ".env").items() if v is not None})
    except ImportError:
        pass
    env.update(os.environ)
    return env


def authorization(cfg: Mapping, env: Mapping[str, str]) -> list[str]:
    """Reasons to refuse a paid run; empty = authorized."""
    out = [f"{k} is not set (add it to .env; its value is never printed)"
           for k in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN") if not env.get(k)]
    caps = (cfg.get("ledger") or {}).get("sub_caps_usd") or {}
    if RUN not in caps:
        out.append(f"configs/backends.yaml has no ledger.sub_caps_usd.{RUN}")
    elif not float(caps[RUN]) > 0:
        out.append(f"ledger.sub_caps_usd.{RUN} is {float(caps[RUN]):.2f}: not authorized (the user sets a positive "
                   f"amount)")
    if BACKEND not in (cfg.get("backends") or {}):
        out.append(f"configs/backends.yaml has no backends.{BACKEND} entry")
    return out


def release_label(created_at: Any) -> str:
    """`post-release` iff the item was created on a UTC day after the release date (the release is a date, not a time)."""
    ts = pd.Timestamp(created_at)
    ts = ts.tz_convert("UTC") if ts.tzinfo is not None else ts
    return LABEL_POST if ts.date() > dt.date.fromisoformat(CF.RELEASE_DATE) else LABEL_PRE


def gen_id(component: str) -> str:
    return f"jevcf_{CF.JEV_MODEL}_{GEN_SUFFIX[component]}"


def jev_ledger_path(cfg: Mapping) -> Path:
    return ROOT / cfg["backends"][BACKEND]["ledger_path"]


def jev_spent(path: Path) -> float:
    if not path.exists():
        return 0.0
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        (usd,) = c.execute("SELECT COALESCE(SUM(usd), 0) FROM spend WHERE tag = ? OR tag LIKE ?",
                           (RUN, f"{RUN}/%")).fetchone()
    except sqlite3.Error:
        usd = 0.0
    c.close()
    return float(usd)


def frozen_bank():
    from jevtrader.programs.bank_v1 import FROZEN_SHA256, load_bank

    bank = load_bank()  # JEV-CF is not a FROZEN_RUN_TAGS tag: the frozen hash is enforced here instead
    if bank.sha256 != FROZEN_SHA256:
        raise SystemExit(f"bank {bank.sha256[:12]} is not the frozen v1.0 bank {FROZEN_SHA256[:12]}")
    return bank


def keyer(bank_hash: str) -> CF.CloudflareJevBackend:
    """A jev-cf instance for request keys only (injected dummy client: it can never reach the network)."""
    return CF.CloudflareJevBackend(bank_hash=bank_hash, client=object())


# ------------------------------------------------------------------------------------------ source rows
def item_rows(gen_dir: Path, t_min: str | None = None, t_max: str | None = None) -> dict[str, dict]:
    """A generation's N01 item rows (M-work blackboard `N01.items.<symbol>`) keyed by unit id t|symbol|eid."""
    from jevtrader.programs import news as N

    out = {}
    with open(gen_dir / "memory" / "work.jsonl") as f:
        for line in f:
            rec = json.loads(line)["record"]
            if not rec["key"].startswith(N.ITEMS_KEY):
                continue
            t = rec["t"][:10]
            if (t_min and t < t_min) or (t_max and t > t_max):
                continue
            sym = rec["key"][len(N.ITEMS_KEY):]
            for r in rec["value"]:
                uid = f"{t}|{sym}|{r['eid']}"
                if uid in out:
                    raise RuntimeError(f"duplicate item row {uid}")
                out[uid] = {"t": t, "symbol": sym, **r}
    return out


def top_label(probs: dict | None) -> str | None:
    return None if not probs else max(probs, key=probs.get)  # first maximal key: bank order


def eligible(r: dict) -> bool:
    """C2 eligibility by the source generation's own answers (experiments/analysis/h2_validity.py item frame)."""
    top = top_label(r.get("event_type"))
    return bool(r["status"]["kept"]) and top is not None and top != EXCLUDED_TYPE


def frame_units(rows: dict[str, dict]) -> list[str]:
    win = yaml.safe_load((ROOT / "configs" / "system_v1.yaml").read_text())["windows"]["W_clean"]
    if tuple(win) != W_CLEAN:
        raise SystemExit(f"W_clean in configs/system_v1.yaml is {win}, not {W_CLEAN}")
    units = sorted(u for u, r in rows.items() if W_CLEAN[0] <= r["t"] <= W_CLEAN[1] and eligible(r))
    reg = json.loads(H2_FRAME.read_text())["frame"]["units"]
    if len(units) != reg:
        raise SystemExit(f"rebuilt H2 frame has {len(units)} units, results/analysis/h2_frame.json {reg}")
    return units


def read_states(*dirs: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.jsonl")):
            with open(p) as fh:
                for line in fh:
                    x = json.loads(line)
                    out.setdefault(x["unit"], x)
    return out


def verb_sample() -> pd.DataFrame:
    smp = pd.read_parquet(VERB_DIR / "sample.parquet").sort_values("order")
    want = json.loads((VERB_DIR / "sample.json").read_text())["sample_sha256"]
    got = hashlib.sha256("\n".join(sorted(smp["unit"])).encode()).hexdigest()
    if got != want:
        raise SystemExit("VERB sample hash mismatch")
    return smp


def live_generations() -> dict[str, Path]:
    """Live decision generations present: decision date -> directory (results/judgments/live_<d>_<context id>)."""
    out = {}
    for d in sorted(JUDG.glob("live_*")):
        parts = d.name.split("_")
        if len(parts) >= 2 and (d / "MANIFEST.json").exists():
            out[parts[1]] = d
    return out


def scheduled_weeks(after: str, until: str) -> list[str]:
    from jevtrader.live.jobs import sessions
    from jevtrader.programs.context import decision_dates

    if until <= after:
        return []
    s = sessions(pd.Timestamp(after) - pd.Timedelta(days=14), pd.Timestamp(until) + pd.Timedelta(days=14))
    return [str(t.date()) for t in decision_dates(s, pd.Timestamp(after) + pd.Timedelta(days=1), until)]


# ------------------------------------------------------------------------------------------ requests
@dataclasses.dataclass
class Planned:
    uid: str
    group: str  # N1 group, or the probe stage
    label: str
    request: Any  # programs.base.Request
    meta: dict


def n1_requests(bank, st: dict, groups: tuple[str, ...], emu: Mapping, label: str, component: str) -> list[Planned]:
    """The frozen bank's N1 requests of one unit, exactly the source generation's (checked by its emulator keys)."""
    import items500 as NG

    from jevtrader.programs import news as N
    from jevtrader.programs.base import Request
    from jevtrader.programs.news import NewsEvidence

    e = st["evidence"]
    evid = NewsEvidence(eid=e["eid"], available_at_utc=pd.Timestamp(e["available_at_utc"]), kept=True,
                        quarantined=False, text=None, d0_flags=e["d0_flags"], raw_sha256=e["raw_sha256"])
    t = pd.Timestamp(st["t"])
    out = []
    for g in N.n1_plan(bank).groups():
        if g.name not in groups:
            continue
        rq = Request.build("N1", f"N1.{g.name}", st["symbol"], t, st["n1_state"], NG.n1_questions(bank, g.ask),
                           [evid], tag="N1")
        if NG.emu_key(emu, rq.state, rq.questions) != st["emu_keys"][f"N1.{g.name}"]:
            raise SystemExit(f"{st['unit']} {g.name}: not the source generation's request")
        out.append(Planned(f"{st['unit']}|{g.name}", g.name, label, rq,
                           {"component": component, "unit": st["unit"], "group": g.name, "label": label,
                            "use": list(g.use), "stratum": st.get("stratum")}))
    return out


def probe_requests(bank) -> list[Planned]:
    import run_gen_probe as P

    out = []
    for d in PROBE_DIRS:
        emu = json.loads((d / "MANIFEST.json").read_text())["backend"]
        keys = dict(pd.read_parquet(d / "jobs.parquet", columns=["uid", "request_key"]).itertuples(index=False))
        with gzip.open(d / "plan" / "jobs.jsonl.gz", "rt") as fh:
            recs = [json.loads(x) for x in fh]
        from jevtrader.s1.types import request_key

        for rec in recs:
            rq = P.build_request(bank, rec)
            k = request_key("llm-logprob", emu["model_pinned"], emu["prompt_version"], emu["cfg_hash"],
                            emu["bank_hash"], rq.state, rq.questions)
            if keys.get(rec["uid"]) != k:
                raise SystemExit(f"{d.name} {rec['uid']}: rebuilt probe differs from the generation's request")
            period = rec["state"].get("period") if isinstance(rec["state"], dict) else None
            last = max([rec["t"], *(period.values() if isinstance(period, dict) else [])])
            label = release_label(last)
            out.append(Planned(f"{d.name}|{rec['uid']}", rec["stage"], label, rq,
                               {"component": "probes", "source": d.name, "uid": rec["uid"], "label": label,
                                **rec["meta"]}))
    return out


def aligned_requests(bank, groups=GROUPS_ALL, component="aligned2000", limit: int | None = None) -> list[Planned]:
    smp = verb_sample()
    states = read_states(VERB_DIR / "states")
    emu = json.loads((MAIN_DIR / "MANIFEST.json").read_text())["backend"]
    rows = item_rows(MAIN_DIR, *W_CLEAN)
    units = list(smp["unit"])[: limit or None]
    out = []
    for u in units:
        out += n1_requests(bank, states[u], groups, emu, release_label(rows[u]["created_at"]), component)
    return out


def canary_requests(bank) -> list[Planned]:
    return aligned_requests(bank, ("attributes",), "canary", N_CANARY)


def prepared_dir(component: str) -> Path:
    return JUDG / gen_id(component) / "states"


def h2frame_requests(bank) -> list[Planned]:
    rows = item_rows(MAIN_DIR, *W_CLEAN)
    units = frame_units(rows)
    states = read_states(VERB_DIR / "states", prepared_dir("h2frame"))
    missing = [u for u in units if u not in states]
    if missing:
        raise SystemExit(f"h2frame: {len(missing)} of {len(units)} states missing; run `prepare --component h2frame`")
    emu = json.loads((MAIN_DIR / "MANIFEST.json").read_text())["backend"]
    return [p for u in units for p in n1_requests(bank, states[u], GROUPS_ALL, emu, LABEL_PRE, "h2frame")]


def forward_sources() -> list[dict]:
    """Post-release forward weeks with a generation: MAIN's pseudo-forward week(s) after W_clean, then live decisions."""
    out = [{"week": t, "gen_dir": MAIN_DIR, "data_root": None}
           for t in sorted({r["t"] for r in item_rows(MAIN_DIR, t_min="2026-09-12").values()})]
    for d, gdir in live_generations().items():
        if d > W_CLEAN[1]:
            out.append({"week": d, "gen_dir": gdir, "data_root": LIVE_ROOT})
    return out


def forward_units(src: dict) -> tuple[list[str], dict[str, dict], int]:
    """(post-release eligible units, rows, pre-release eligible units excluded) of one forward week."""
    rows = item_rows(src["gen_dir"], src["week"], src["week"])
    el = [u for u, r in rows.items() if eligible(r)]
    post = sorted(u for u in el if release_label(rows[u]["created_at"]) == LABEL_POST)
    return post, rows, len(el) - len(post)


def forward_requests(bank) -> list[Planned]:
    states = read_states(prepared_dir("forward"))
    out = []
    for src in forward_sources():
        units, _, _ = forward_units(src)
        missing = [u for u in units if u not in states]
        if missing:
            raise SystemExit(f"forward {src['week']}: {len(missing)} states missing; run `prepare --component forward`")
        emu = json.loads((src["gen_dir"] / "MANIFEST.json").read_text())["backend"]
        for u in units:
            out += n1_requests(bank, states[u], GROUPS_ALL, emu, LABEL_POST, "forward")
    return out


BUILDERS = {"canary": canary_requests, "aligned2000": aligned_requests, "probes": probe_requests,
            "forward": forward_requests, "h2frame": h2frame_requests}


# ------------------------------------------------------------------------------------------ prepare (free, CPU)
def rebuild(units: list[str], rows: dict[str, dict], gen_dir: Path, data_root: Path | None, sdir: Path,
            weeks_limit: int | None = None) -> int:
    """Rebuild N1 item states exactly as the source generation built them (its context builder, ItemSource, N0
    decision) and verify each against the generation's emulator keys, `n1_sha256` and archived text hash; any mismatch
    stops (experiments/run_gen_verb.py `prepare`, generalized to a generation and a data root)."""
    import items500 as NG

    from jevtrader.programs import news as N
    from jevtrader.programs.context import ContextBuilder, RealData

    bank = frozen_bank()
    man = json.loads((gen_dir / "MANIFEST.json").read_text())
    emu = man["backend"]
    if emu["bank_hash"] != bank.sha256:
        raise SystemExit(f"{gen_dir.name} ran another bank")
    sdir.mkdir(parents=True, exist_ok=True)
    have = read_states(sdir)
    todo = [u for u in units if u not in have]
    by_week: dict[str, list[str]] = defaultdict(list)
    for u in todo:
        by_week[rows[u]["t"]].append(u)
    if not by_week:
        return 0
    if data_root is None:  # MAIN: its context (fact symbols of its weeks), checked by context id and archive
        dev = json.loads((gen_dir / "dev_run.json").read_text())
        data = RealData()
        data.fact_symbols = data.symbols_for_weeks([pd.Timestamp(w) for w in dev["weeks"]])
        builder = ContextBuilder(data, subset=None)
        if hashlib.sha256(canonical_json(builder.describe()).encode()).hexdigest()[:10] != dev["context_id"] \
                or data.archive_id() != man["spec"]["data_freeze"]:
            raise SystemExit(f"context or archive differs from {gen_dir.name}'s")
    else:  # a live decision: the live data root, as experiments/run_live_decision.py builds it
        data = RealData(root=data_root)
        data.fact_symbols = data.symbols_for_weeks([pd.Timestamp(w) for w in by_week])
        builder = ContextBuilder(data, subset=None, strict=True)
    plan = N.n1_plan(bank)
    ev = pd.read_parquet(gen_dir / "evidence.parquet", columns=["request_key", "eid", "text_sha256"])
    n0keys = {rows[u]["request_keys"]["N0"] for u in todo}
    ev = ev[ev["request_key"].isin(n0keys)]
    ev_sha = {(k, e): s for k, e, s in ev.itertuples(index=False)}
    n, t0 = 0, time.time()
    for i, (week, us) in enumerate(sorted(by_week.items())):
        if weeks_limit is not None and i >= weeks_limit:
            break
        wt = pd.Timestamp(week)
        ctx = builder(wt)
        src = N.ItemSource(ctx, bank)
        lines = []
        for u in us:
            main = rows[u]
            it = {x.eid: x for x in src.items(main["symbol"])}.get(main["eid"])
            if it is None:
                raise SystemExit(f"{u}: item not rebuilt")
            fc = N.focal(bank, ctx, main["symbol"])
            r0 = N.n0_request(bank, it, wt)
            keep = [k in set(main["kept_sentences"]) for k in range(len(it.n0_sentences))]
            n1 = N.n1_item_state(it, keep)
            checks = {"N0_key": NG.emu_key(emu, r0.state, r0.questions) == main["request_keys"]["N0"],
                      "n_sentences": len(it.n0_sentences) == main["n_sentences"],
                      "n1_sha256": N.item_hash(n1) == main["n1_sha256"],
                      "text_sha256": ev_sha.get((main["request_keys"]["N0"], it.eid)) == it.raw_sha256}
            n1_state = None
            for grp in plan.groups():
                rq = N.n1_group_request(bank, it, fc, n1, wt, grp.ask)
                n1_state = rq.state if n1_state is None else n1_state
                checks[f"N1.{grp.name}_key"] = (rq.state == n1_state and NG.emu_key(emu, rq.state, rq.questions)
                                                == main["request_keys"].get(f"N1.{grp.name}"))
            if not all(checks.values()):
                raise SystemExit(f"{u}: rebuilt state differs from {gen_dir.name}'s: {checks}")
            e = it.evidence()
            lines.append({"unit": u, "t": week, "symbol": main["symbol"], "eid": main["eid"], "stratum": None,
                          "n_sentences": len(it.n0_sentences), "n1_state": n1_state,
                          "evidence": {"eid": e.eid, "available_at_utc": pd.Timestamp(e.available_at_utc).isoformat(),
                                       "d0_flags": e.d0_flags, "raw_sha256": e.raw_sha256},
                          "emu_keys": {k: main["request_keys"][k] for k in ("N0", *NG.N1_KEYS)},
                          "n1_state_sha256": state_sha256(n1_state), "checks": checks, "source": gen_dir.name})
        NG.fsync_write(sdir / f"{week}.jsonl", "".join(json.dumps(x) + "\n" for x in lines))
        n += len(lines)
        print(f"prepare {gen_dir.name} {week}: {len(lines)} state(s) verified ({time.time() - t0:.0f}s)", flush=True)
    return n


def cmd_prepare(a) -> int:
    if a.component == "h2frame":
        rows = item_rows(MAIN_DIR, *W_CLEAN)
        have = read_states(VERB_DIR / "states")
        units = [u for u in frame_units(rows) if u not in have]
        n = rebuild(units, rows, MAIN_DIR, None, prepared_dir("h2frame"), a.limit_weeks)
    elif a.component == "forward":
        n = 0
        for src in forward_sources():
            units, rows, _ = forward_units(src)
            n += rebuild(units, rows, src["gen_dir"], src["data_root"], prepared_dir("forward"), a.limit_weeks)
    else:
        print(f"{a.component}: states come from the source generation; nothing to prepare")
        return 0
    print(f"prepare {a.component}: {n} new state(s)", flush=True)
    return 0


# ------------------------------------------------------------------------------------------ projection
def cached_keys(cfg: Mapping, keys: set[str]) -> set[str]:
    """Keys already in the real response cache (read-only)."""
    p = ROOT / cfg["cache"]["path"]
    if not p.exists() or not keys:
        return set()
    c = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    hit: set[str] = set()
    ks = sorted(keys)
    for i in range(0, len(ks), 900):
        chunk = ks[i:i + 900]
        hit |= {r[0] for r in c.execute(f"SELECT key FROM cache WHERE key IN ({','.join('?' * len(chunk))})", chunk)}
    c.close()
    return hit


def tokens(state, questions) -> tuple[int, int]:
    """(base, high) input tokens of one Jev request."""
    b = CF.input_bytes(state, questions)
    return CF.estimate_input_tokens(state, questions), math.ceil(b / HIGH_BYTES_PER_TOKEN) + CF.REQUEST_OVERHEAD_TOKENS


def usd(tok: float) -> float:
    return tok * CF.PRICE_IN / 1e6


def exact_rows(planned: list[Planned], kb: CF.CloudflareJevBackend) -> pd.DataFrame:
    recs = []
    for p in planned:
        base, high = tokens(p.request.state, p.request.questions)
        recs.append({"uid": p.uid, "group": p.group, "label": p.label, "key": kb.key(p.request.state, p.request.questions),
                     "base": base, "high": high})
    return pd.DataFrame(recs)


def emu_tokens() -> dict[str, int]:
    req = pd.read_parquet(MAIN_DIR / "requests.parquet", columns=["request_key", "program", "input_tokens"])
    req = req[(req.program == "N1") & (req.input_tokens > 0)]
    return req.groupby("request_key").input_tokens.max().to_dict()


def ratios(aligned: pd.DataFrame, emu_tok: Mapping[str, int], emu_keys: Mapping[str, str]) -> dict:
    """Per N1 group: Jev tokens (base, high) per emulator token, measured on the aligned sample's requests."""
    out = {}
    a = aligned.assign(emu=[emu_tok.get(emu_keys.get(u)) for u in aligned.uid])
    a = a[a.emu.notna()]
    for g, d in a.groupby("group"):
        out[g] = {"n": len(d), "base": float(d.base.sum() / d.emu.sum()), "high": float(d.high.sum() / d.emu.sum()),
                  "emu_mean": float(d.emu.mean()), "jev_base_mean": float(d.base.mean())}
    return out


def projected_rows(units: list[str], rows: Mapping[str, dict], emu_tok: Mapping[str, int], rat: Mapping[str, dict],
                   label_of, exact: pd.DataFrame | None, kb_map: Mapping[str, str]) -> pd.DataFrame:
    """Rows for units without rebuilt states: tokens = emulator tokens of the same request x the group ratio. The Jev
    key is unknown until the state is rebuilt, so the emulator key stands in (one-to-one: both hash the same (state,
    questions) under a fixed configuration)."""
    have = set(exact.uid) if exact is not None and len(exact) else set()
    recs = []
    mean = {g: r["emu_mean"] for g, r in rat.items()}
    for u in units:
        for g in GROUPS_ALL:
            uid = f"{u}|{g}"
            if uid in have:
                continue
            ek = rows[u]["request_keys"][f"N1.{g}"]
            e = emu_tok.get(ek, mean[g])
            recs.append({"uid": uid, "group": g, "label": label_of(u), "key": kb_map.get(uid, "emu:" + ek),
                         "base": math.ceil(e * rat[g]["base"]), "high": math.ceil(e * rat[g]["high"])})
    df = pd.DataFrame(recs, columns=["uid", "group", "label", "key", "base", "high"])
    return pd.concat([exact, df], ignore_index=True) if exact is not None and len(exact) else df


def summarize(df: pd.DataFrame, seen: set[str], cached: set[str], rpm: float) -> dict:
    d = df.drop_duplicates("key")
    paid = d[~d.key.isin(cached)]
    inc = paid[~paid.key.isin(seen)]
    return {
        "requests": len(df), "distinct_requests": len(d), "already_cached": int(d.key.isin(cached).sum()),
        "paid_calls_standalone": len(paid), "paid_calls_incremental": len(inc),
        "tokens_base": int(paid.base.sum()), "tokens_high": int(paid.high.sum()),
        "usd_base": round(usd(paid.base.sum()), 4), "usd_high": round(usd(paid.high.sum()), 4),
        "usd_base_incremental": round(usd(inc.base.sum()), 4), "usd_high_incremental": round(usd(inc.high.sum()), 4),
        "tokens_per_request_base": round(float(paid.base.mean()), 1) if len(paid) else None,
        "hours_at_rpm_incremental": round(len(inc) / rpm / 60.0, 2),
        "max_est_input_tokens": int(df.base.max()) if len(df) else 0,
        "over_context_guard": int((df.base > CF.CONTEXT_TOKENS).sum()),
        "labels": {k: int(v) for k, v in Counter(df.label).items()},
    }


def cmd(*args: str) -> str:
    return " ".join([PY, SCRIPT, *args])


def preflight(forward_until: str = FORWARD_UNTIL, write: bool = True) -> dict:
    cfg = load_cfg()
    env = load_env()
    bank = frozen_bank()
    kb = keyer(bank.sha256)
    jcfg = cfg["backends"][BACKEND]
    rpm = float(jcfg.get("rpm") or 240)
    t0 = time.time()
    comps: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}

    # exact components (states on disk)
    aligned_pl = aligned_requests(bank)
    frames["aligned2000"] = exact_rows(aligned_pl, kb)
    frames["canary"] = frames["aligned2000"][frames["aligned2000"].group == "attributes"].head(N_CANARY).copy()
    probe_pl = probe_requests(bank)
    frames["probes"] = exact_rows(probe_pl, kb)

    # projected components (states rebuilt by `prepare`; exact where they already are)
    rows = item_rows(MAIN_DIR, *W_CLEAN)
    emu_tok = emu_tokens()
    emu_keys = {p.uid: rows[p.meta["unit"]]["request_keys"][f"N1.{p.group}"] for p in aligned_pl}
    rat = ratios(frames["aligned2000"], emu_tok, emu_keys)
    units = frame_units(rows)
    st_h2 = read_states(VERB_DIR / "states", prepared_dir("h2frame"))
    emu_main = json.loads((MAIN_DIR / "MANIFEST.json").read_text())["backend"]
    ready = [u for u in units if u in st_h2]
    ex_h2 = exact_rows([p for u in ready for p in n1_requests(bank, st_h2[u], GROUPS_ALL, emu_main, LABEL_PRE,
                                                             "h2frame")], kb)
    frames["h2frame"] = projected_rows(units, rows, emu_tok, rat, lambda u: LABEL_PRE, ex_h2, {})

    fwd_parts, fwd_weeks, st_fwd = [], [], read_states(prepared_dir("forward"))
    for src in forward_sources():
        us, frows, n_pre = forward_units(src)
        femu = json.loads((src["gen_dir"] / "MANIFEST.json").read_text())["backend"]
        rdy = [u for u in us if u in st_fwd]
        ex = exact_rows([p for u in rdy for p in n1_requests(bank, st_fwd[u], GROUPS_ALL, femu, LABEL_POST, "forward")],
                        kb)
        ftok = emu_tok if src["gen_dir"] == MAIN_DIR else {}
        part = projected_rows(us, frows, ftok, rat, lambda u: LABEL_POST, ex, {})
        fwd_parts.append(part)
        fwd_weeks.append({"week": src["week"], "source": src["gen_dir"].name, "status": "available",
                          "post_release_units": len(us), "pre_release_units_excluded": n_pre,
                          "states_ready": len(rdy), "requests": len(part),
                          "usd_base": round(usd(part.base.sum()), 4)})
    last = max([w["week"] for w in fwd_weeks] + [W_CLEAN[1]])
    per_week_units = json.loads(H2_FRAME.read_text())["frame"]["units_per_week"]["median"]
    tok_req = float(frames["h2frame"].base.mean())
    tok_req_high = float(frames["h2frame"].high.mean())
    for w in scheduled_weeks(last, forward_until):
        n_req = int(per_week_units * len(GROUPS_ALL))
        fwd_weeks.append({"week": w, "source": f"results/judgments/live_{w}_* (not yet generated)",
                          "status": "scheduled", "estimate": "W_clean median eligible units per week x 5 requests x "
                          "the h2frame mean tokens per request", "requests": n_req,
                          "usd_base": round(usd(n_req * tok_req), 4), "usd_high": round(usd(n_req * tok_req_high), 4)})
    frames["forward"] = pd.concat(fwd_parts, ignore_index=True) if fwd_parts else \
        pd.DataFrame(columns=["uid", "group", "label", "key", "base", "high"])

    cached = cached_keys(cfg, set().union(*(set(f.key) for f in frames.values())))
    seen: set[str] = set()
    states_status = {
        "canary": "ready (VERB's verified states)", "aligned2000": "ready (VERB's verified states)",
        "probes": "ready (the probe generations' plans, rebuilt and key-checked)",
        "forward": f"{sum(w.get('states_ready', 0) for w in fwd_weeks)} of "
                   f"{sum(w.get('post_release_units', 0) for w in fwd_weeks)} available units rebuilt; "
                   f"`prepare --component forward` rebuilds the rest",
        "h2frame": f"{len(ready)} of {len(units)} units ready (the aligned 2,000 are VERB's); "
                   f"`prepare --component h2frame` rebuilds the rest",
    }
    sources = {
        "canary": f"{VERB_DIR.name}: N1.attributes of its first {N_CANARY} units (sample order)",
        "aligned2000": f"{VERB_DIR.name} (sample {json.loads((VERB_DIR / 'sample.json').read_text())['sample_sha256'][:12]}"
                       f"; NG's 500 nested)",
        "probes": " + ".join(d.name for d in PROBE_DIRS),
        "forward": f"{MAIN_DIR.name} weeks after W_clean, then live_<d>_* generations",
        "h2frame": f"{MAIN_DIR.name} N01 item rows, W_clean, C2-eligible by emu (results/analysis/h2_frame.json)",
    }
    projection = {"canary": "exact payload", "aligned2000": "exact payload", "probes": "exact payload",
                  "forward": "exact payload where rebuilt, else MAIN's emulator tokens x the aligned-sample ratio",
                  "h2frame": "exact payload where rebuilt, else MAIN's emulator tokens x the aligned-sample ratio"}
    for c in COMPONENTS:
        if c == "canary":  # asked without the response cache: every request is paid, and none fills it for later
            s = summarize(frames[c], set(), set(), rpm)
        else:
            s = summarize(frames[c], seen, cached, rpm)
            seen |= set(frames[c].key.drop_duplicates())
        comps[c] = {"gen_id": gen_id(c), "source": sources[c], "states": states_status[c],
                    "projection": projection[c], **s, "admissible_use": ADMISSIBLE[c],
                    "not_admissible": NOT_ADMISSIBLE.get(c, []), "command": cmd("run", "--component", c)}
    comps["forward"]["weeks"] = fwd_weeks
    sched = [w for w in fwd_weeks if w["status"] == "scheduled"]
    comps["forward"]["scheduled_usd_base"] = round(sum(w["usd_base"] for w in sched), 4)
    comps["forward"]["scheduled_usd_high"] = round(sum(w["usd_high"] for w in sched), 4)

    tot_base = sum(comps[c]["usd_base_incremental"] for c in COMPONENTS) + comps["forward"]["scheduled_usd_base"]
    tot_high = sum(comps[c]["usd_high_incremental"] for c in COMPONENTS) + comps["forward"]["scheduled_usd_high"]
    calls = sum(comps[c]["paid_calls_incremental"] for c in COMPONENTS) + sum(w["requests"] for w in sched)
    cap = float(((cfg.get("ledger") or {}).get("sub_caps_usd") or {}).get(RUN, 0.0))
    lpath = jev_ledger_path(cfg)
    spent = jev_spent(lpath)
    refusals = authorization(cfg, env)
    out = {
        "what": "Jev panel on Cloudflare Workers AI: free preflight (design_v1 §11.3; decisions 2026-09-29 D-13)",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "model_calls": 0, "network": False,
        "wall_s": None,
        "backend": {"id": BACKEND, "model_pinned": jcfg["model"], "cf_model": jcfg["cf_model"],
                    "price_usd_per_m": {"input": CF.PRICE_IN, "output": CF.PRICE_OUT, "cached_input": CF.PRICE_CACHED},
                    "context_tokens": CF.CONTEXT_TOKENS, "overhead_tokens": CF.REQUEST_OVERHEAD_TOKENS, "rpm": rpm,
                    "release_date": CF.RELEASE_DATE, "release_date_sources": list(CF.RELEASE_DATE_SOURCES),
                    "sources": ["docs/jev/cloudflare_jev.md", "docs/jev/cloudflare_jev_schema-input.json",
                                "docs/jev/cloudflare_platform_limits.md", "docs/jev/cloudflare_platform_errors.md"]},
        "authorization": {"credentials_present": {k: bool(env.get(k)) for k in ("CLOUDFLARE_ACCOUNT_ID",
                                                                                "CLOUDFLARE_API_TOKEN")},
                          "sub_cap_usd": cap, "ledger": str(lpath.relative_to(ROOT)), "spent_usd": round(spent, 6),
                          "can_run": not refusals, "refusals": refusals},
        "token_model": {"base": "bytes of the `input` JSON / 3.5 + the archived per-request overhead "
                                f"({CF.REQUEST_OVERHEAD_TOKENS})",
                        "high": f"bytes / {HIGH_BYTES_PER_TOKEN} + the same overhead (assumption: a smaller-vocabulary "
                                "tokenizer on JSON)",
                        "ratio_jev_per_emu_token_by_group": rat},
        "components": comps, "order": list(COMPONENTS),
        "totals": {"paid_calls": int(calls), "usd_base": round(tot_base, 4), "usd_high": round(tot_high, 4),
                   "hours_at_rpm": round(calls / rpm / 60.0, 2),
                   "note": "incremental in preflight order, scheduled forward weeks included"},
        "suggested_sub_cap_usd": math.ceil(1.5 * tot_high * 100) / 100,
        "suggested_sub_cap_rule": "1.5 x the high projection (the config's 1.5x rule applied to the high case)",
        "rules": RULES, "not_planned": NOT_PLANNED,
        "commands": [
            cmd("preflight") + "   # free",
            ("# user: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN (Workers AI Read + Edit) in .env; "
             f"ledger.sub_caps_usd.{RUN} in configs/backends.yaml >= the suggested amount"),
            cmd("prepare", "--component", "forward") + "   # free: rebuild and verify states",
            cmd("prepare", "--component", "h2frame") + "   # free: CPU, about one pass over W_clean's weeks",
            cmd("dry", "--component", "aligned2000") + "   # free: the paid path on a mock transport",
            *[cmd("run", "--component", c) for c in COMPONENTS],
            *[cmd("validate", "--component", c) for c in COMPONENTS if c != "canary"],
        ],
    }
    out["wall_s"] = round(time.time() - t0, 1)
    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        PREFLIGHT.write_text(json.dumps(out, indent=1, default=str))
    return out


def print_preflight(p: dict) -> None:
    a = p["authorization"]
    print(f"JEV PANEL PREFLIGHT (free; {p['model_calls']} model calls; network {p['network']}): backend {BACKEND} "
          f"({p['backend']['cf_model']} -> {p['backend']['model_pinned']}), ${CF.PRICE_IN}/M input, output free, "
          f"{p['backend']['rpm']:.0f} rpm, context {CF.CONTEXT_TOKENS}")
    print(f"authorization: credentials {a['credentials_present']}, {RUN} sub-cap ${a['sub_cap_usd']:.2f}, "
          f"spent ${a['spent_usd']:.4f} in {a['ledger']}; can run: {a['can_run']}")
    for r in a["refusals"]:
        print(f"  refuse: {r}")
    print(f"{'component':<12} {'requests':>9} {'paid(inc)':>10} {'tok/req':>8} {'$base(inc)':>11} {'$high(inc)':>11} "
          f"{'hours':>6}  labels")
    for c in p["order"]:
        x = p["components"][c]
        print(f"{c:<12} {x['requests']:>9} {x['paid_calls_incremental']:>10} {x['tokens_per_request_base'] or 0:>8} "
              f"{x['usd_base_incremental']:>11.4f} {x['usd_high_incremental']:>11.4f} "
              f"{x['hours_at_rpm_incremental']:>6}  {x['labels']}")
    f = p["components"]["forward"]
    for w in f["weeks"]:
        print(f"  forward {w['week']}: {w['status']}, {w['requests']} requests, ${w['usd_base']:.4f} ({w['source']})")
    t = p["totals"]
    print(f"TOTAL {t['paid_calls']} paid calls, ${t['usd_base']:.2f} base / ${t['usd_high']:.2f} high, "
          f"{t['hours_at_rpm']} h at {p['backend']['rpm']:.0f} rpm; suggested {RUN} sub-cap "
          f"${p['suggested_sub_cap_usd']:.2f} ({p['suggested_sub_cap_rule']})")
    print("rules: " + p["rules"]["labels"] + "; " + p["rules"]["no_performance_test"])
    print("commands:")
    for line in p["commands"]:
        print("  " + line)
    print(f"written: {PREFLIGHT.relative_to(ROOT)}")


# ------------------------------------------------------------------------------------------ dry / run / validate
class FakeCloudflare:
    """Mock Workers AI transport for dry runs: plausible answers, usage = the backend's own base estimate."""

    def __init__(self):
        self.calls = 0

    def transport(self):
        import httpx2

        from jevtrader.s1.fakes import jev_answer

        def route(request):
            body = json.loads(request.content)
            self.calls += 1
            qs = body["input"]["questions"]
            n_in = CF.estimate_tokens(canonical_json(body["input"])) + CF.REQUEST_OVERHEAD_TOKENS
            res = {"model": CF.JEV_MODEL, "answers": {k: jev_answer(q) for k, q in qs.items()},
                   "usage": {"input_tokens": n_in, "output_tokens": 8 * len(qs)}}
            return httpx2.Response(200, json={"result": res, "success": True, "errors": [], "messages": []})

        return httpx2.MockTransport(route)


def data_freeze(component: str) -> str:
    if component == "probes":
        vals = set()
        for d in PROBE_DIRS:
            vals |= set(pd.read_parquet(d / "requests.parquet", columns=["data_freeze"]).data_freeze.unique())
        if len(vals) != 1:
            raise SystemExit(f"probe generations have data freezes {sorted(vals)}")
        return vals.pop()
    return json.loads((MAIN_DIR / "MANIFEST.json").read_text())["spec"]["data_freeze"]


def shards_of(planned: list[Planned]) -> dict[str, list]:
    jobs = [C.Job(p.uid, p.request, p.meta) for p in planned]
    return {f"s{i // SHARD:03d}": jobs[i:i + SHARD] for i in range(0, len(jobs), SHARD)}


def latest_canary() -> Path | None:
    d = JUDG / gen_id("canary")
    runs = sorted(x for x in d.glob("*") if (x / "MANIFEST.json").exists()) if d.exists() else []
    return runs[-1] if runs else None


def compare_canary(new: Path, old: Path | None) -> dict:
    if old is None:
        return {"previous": None}
    a = pd.read_parquet(new / "answers.parquet")
    b = pd.read_parquet(old / "answers.parquet")
    key = ["request_key", "qid"]
    m = a.merge(b, on=key, suffixes=("", "_prev"))
    diffs = []
    for x, y in zip(m.probs_json, m.probs_json_prev):
        if x is None or y is None:
            continue
        px, py = json.loads(x), json.loads(y)
        diffs.append(max(abs(px[k] - py.get(k, 0.0)) for k in px))
    return {"previous": old.name, "matched_answers": len(m), "max_abs_prob_change": max(diffs) if diffs else None,
            "answers_changed_over_0.01": int(sum(d > 0.01 for d in diffs))}


def cmd_run(a, dry: bool) -> int:
    cfg = load_cfg()
    if not dry:  # refuse before anything else: no data read, no backend built, no network
        refusals = authorization(cfg, load_env())
        if refusals:
            for r in refusals:
                print(f"REFUSED ({a.component}): {r}", flush=True)
            return 2
    bank = frozen_bank()
    planned = BUILDERS[a.component](bank)
    kb = keyer(bank.sha256)
    df = exact_rows(planned, kb)
    proj_base = usd(df.drop_duplicates("key").base.sum())
    proj_high = usd(df.drop_duplicates("key").high.sum())
    out = JUDG / gen_id(a.component)
    if a.component == "canary":
        out = out / pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    fp = {"gen_id": gen_id(a.component), "bank": bank.sha256, "backend": f"{BACKEND}/{CF.JEV_MODEL}",
          "component": a.component, "requests_sha256": hashlib.sha256("\n".join(sorted(df.key)).encode()).hexdigest()}
    from jevtrader.s1.backends import make_backend
    from jevtrader.s1.budget import SpendLedger

    if dry:
        scratch = Path(a.scratch or tempfile.mkdtemp(prefix="jevcf_dry_"))
        led = SpendLedger(scratch / "dry_spend.sqlite", hard_cap_usd=float("inf"), sub_caps={RUN: float("inf")})
        fake = FakeCloudflare()
        s1 = make_backend(backend=BACKEND, cache=False, ledger=led, transport=fake.transport(), account_id="dry",
                          api_token="dry", ledger_path=None, rpm=None)
        s1.cache = None if a.component == "canary" else C.ReadOnlyCache(ROOT / cfg["cache"]["path"])
        out, guard_usd = scratch / out.relative_to(JUDG), float("inf")
    else:
        cap = float(cfg["ledger"]["sub_caps_usd"][RUN])
        spent = jev_spent(jev_ledger_path(cfg))
        print(f"PREFLIGHT {RUN}/{a.component}: {len(df)} requests ({df.key.nunique()} distinct), projected "
              f"${proj_base:.4f} (high ${proj_high:.4f}) before cache hits vs sub-cap ${cap:.2f} (spent ${spent:.4f}, "
              f"headroom ${cap - spent:.4f}); ledger {cfg['backends'][BACKEND]['ledger_path']}", flush=True)
        # a resumed run pays only for requests not yet in the response cache (added 2026-10-09: a resume of h2frame
        # with 6 of 276 shards left was refused on the whole component's projection)
        rc = C.ReadOnlyCache(ROOT / cfg["cache"]["path"])
        todo = df.drop_duplicates("key")
        todo = todo[[rc.get(k) is None for k in todo.key]]
        proj_left = usd(todo.base.sum())
        print(f"PREFLIGHT {RUN}/{a.component}: {len(todo)} requests not yet cached, projected ${proj_left:.4f}", flush=True)
        if spent + proj_left > cap:
            print(f"ABORT: the base projection exceeds the {RUN} headroom", flush=True)
            return 3
        if a.component != "canary":
            last = latest_canary()
            age = None if last is None else (pd.Timestamp.now(tz="UTC") - pd.Timestamp(last.name)).days
            if age is None or age > CANARY_MAX_AGE_DAYS:
                print(f"canary: {'none yet' if last is None else f'{age} days old'}; re-asking it first", flush=True)
                rc = cmd_run(argparse.Namespace(**{**vars(a), "component": "canary"}), dry=False)
                if rc != 0:
                    return rc
                spent = jev_spent(jev_ledger_path(cfg))
        s1 = make_backend(backend=BACKEND, cache=a.component != "canary")
        if s1.ledger is None or Path(s1.ledger.path).resolve() != jev_ledger_path(cfg).resolve():
            raise SystemExit("jev-cf must book into its own ledger file")
        guard_usd = cap - spent
    g = C.guarded(s1, RUN, guard_usd)
    t0 = time.time()
    shards = shards_of(planned)
    st = C.run_shards(out, shards, g, RUN, data_freeze(a.component), fp, a.concurrency)
    if s1.ledger is not None:
        s1.ledger.flush()
    info = {**st, "dry": dry, "component": a.component, "process_spent_usd": g.spent, "process_calls": g.calls,
            "refused": g.refused, "http_retries": s1.http_retries, "overflows": s1.overflows,
            "wall_s": round(time.time() - t0, 1), "projected_usd": proj_base, "projected_high_usd": proj_high}
    C.launch_record(out, info)
    print(json.dumps(info, default=str), flush=True)
    if st["status"] != "complete":
        return 3
    man = C.finalize(out, sorted(shards), {"run": RUN, "backend": s1.describe(), "component": a.component,
                                          "labels": dict(Counter(df.label)), "rules": RULES,
                                          "admissible_use": ADMISSIBLE[a.component],
                                          "not_admissible": NOT_ADMISSIBLE.get(a.component, []),
                                          "tags": sorted({f"{RUN}/{p.request.tag}" for p in planned}), "dry": dry})
    if a.component == "canary":
        cmp = compare_canary(out, None if dry else _previous_canary(out))
        (out / "compare.json").write_text(json.dumps(cmp, indent=1))
        print(f"canary compare: {cmp}", flush=True)
    print(f"finalized {man['rows']} validity {man['validity_rate']} -> {out}", flush=True)
    return 0


def _previous_canary(current: Path) -> Path | None:
    runs = sorted(x for x in current.parent.glob("*") if (x / "MANIFEST.json").exists() and x != current)
    return runs[-1] if runs else None


def cmd_validate(a) -> int:
    cfg = load_cfg()
    out = latest_canary() if a.component == "canary" else JUDG / gen_id(a.component)
    if out is None or not (out / "MANIFEST.json").exists():
        raise SystemExit(f"{gen_id(a.component)}: no finalized generation")
    bank = frozen_bank()
    planned = BUILDERS[a.component](bank)
    jobs = pd.read_parquet(out / "jobs.parquet")
    req = pd.read_parquet(out / "requests.parquet")
    ans = pd.read_parquet(out / "answers.parquet")
    man = json.loads((out / "MANIFEST.json").read_text())
    paid = ~req.cached.astype(bool)
    spent = jev_spent(jev_ledger_path(cfg))
    meta = pd.DataFrame([json.loads(x) for x in jobs.meta_json])
    chk = {
        "complete": set(jobs.uid) == {p.uid for p in planned},
        "pinned": bool((req.model_pinned == CF.JEV_MODEL).all() and (req.response_model[paid] == CF.JEV_MODEL).all()
                       and (req.backend == BACKEND).all() and (req.prompt_version == "jev-api-v1").all()
                       and (req.bank_hash == bank.sha256).all()),
        "tags": all(t.startswith(f"{RUN}/") for t in req.tag.unique()),
        "labels_attached": bool(meta.label.isin([LABEL_PRE, LABEL_POST]).all()),
        "validity_ge_98pct": bool(1 - ans.probs_json.isna().mean() >= 0.98),
        "ledger_covers_paid_rows": spent + 1e-9 >= float(req.usd[paid].sum()),
        "not_dry": not man.get("dry", False),
    }
    res = {"gen_id": out.name, "checks": chk, "pass": all(chk.values()),
           "summary": {"requests": len(req), "paid": int(paid.sum()), "paid_usd_rows": float(req.usd[paid].sum()),
                       "degraded_answers": int(ans.probs_json.isna().sum()),
                       "labels": meta.label.value_counts().to_dict(), "jev_ledger_usd": spent},
           "manifest_sha256": man["sha256"], "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    (out / "validation.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"pass": res["pass"], "checks": chk}, default=str))
    return 0 if res["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["preflight", "prepare", "dry", "run", "validate"], nargs="?", default="preflight")
    ap.add_argument("--component", choices=COMPONENTS)
    ap.add_argument("--forward-until", default=FORWARD_UNTIL)
    ap.add_argument("--limit-weeks", type=int, default=None, help="prepare: rebuild at most this many weeks")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--scratch", default=None, help="dry: output directory (default: a new temporary directory)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if a.cmd != "preflight" and a.component is None:
        ap.error(f"{a.cmd} needs --component")
    if a.cmd == "preflight":
        print_preflight(preflight(a.forward_until))
        return 0
    if a.cmd == "prepare":
        return cmd_prepare(a)
    if a.cmd in ("dry", "run"):
        return cmd_run(a, dry=a.cmd == "dry")
    return cmd_validate(a)


if __name__ == "__main__":
    sys.exit(main())
