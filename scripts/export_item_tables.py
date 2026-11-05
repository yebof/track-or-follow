#!/usr/bin/env python3
"""Export the hashed-identifier item tables released with the artifact (no licensed text, no raw vendor ids).

    .venv/bin/python scripts/export_item_tables.py            # -> results/release/*.parquet + MANIFEST.json
    .venv/bin/python scripts/export_item_tables.py --check    # re-evaluate H2 from the exported tables only

Tables (one row per item unit `t|symbol|eid`; `eid` = "news:" + SHA-256(vendor item id)[:16], programs.news.news_eid):
- `h2_frame_emu.parquet`: the registered H2 frame on W_clean (h2_validity.item_rows + c2_frame on MAIN) with emu's typed
  judgments (E_effect, top event type, the event-type and vs-expectations distributions as JSON) and the labels of
  h2_validity: d0, AR[0,+1], placebo AR[-5,-1] and the 8-K concept indicators.
- `readers_<name>.parquet`: every N1 answer of a reader generation (jobs.parquet joined to answers.parquet by request key;
  request keys and state digests are SHA-256 values; no state or prompt text); for NG, its items.parquet.

`--check` rebuilds the H2 statistics from `h2_frame_emu.parquet` alone (h2_validity.analyse) and compares them with
results/analysis/h2.json; it needs no price, filing or news data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "experiments" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import common as C  # noqa: E402
import h2_validity as H  # noqa: E402

OUT = ROOT / "results" / "release"
FRAME = "h2_frame_emu.parquet"
READERS = {  # released name -> generation
    "jev_aligned2000": "jevcf_jev-1.13.0_wclean_items2000",
    "jev_h2frame": "jevcf_jev-1.13.0_wclean_h2frame",
    "jev_forward": "jevcf_jev-1.13.0_fwd",
    "verb_aligned2000": "verb_llm-verbalized_wclean_items2000",
    "ng_items500": "ng_typed-local_wclean_items500",
}
JSON_COLS = ("p_type", "vs")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_frame(root: Path) -> pd.DataFrame:
    gen = root / "results/judgments" / H.DEFAULT_GEN
    start, end = C.resolve_window(C.REGISTERED_WINDOW, root)
    frame = H.c2_frame(H.item_rows(gen, start, end))
    guard = C.Guard(root)
    guard.check(frame["t"].unique(), "release: H2 labels")
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    frame = H.add_abnormal_returns(frame, px.book)
    frame = H.sec_labels(frame, px.root / C.SUBMISSIONS)
    for c in JSON_COLS:
        frame[c] = [json.dumps(v, sort_keys=True) for v in frame[c]]
    return frame


def export_reader(gen_dir: Path) -> pd.DataFrame:
    if (gen_dir / "items.parquet").exists():  # NG (experiments/ng/items500.py): one row per unit, no text
        return pd.read_parquet(gen_dir / "items.parquet")
    jobs = pd.read_parquet(gen_dir / "jobs.parquet") if (gen_dir / "jobs.parquet").exists() else None
    ans = pd.read_parquet(gen_dir / "answers.parquet")
    keep = [c for c in ("request_key", "t", "symbol", "program", "stage", "qid", "kind", "probs_json") if c in ans]
    ans = ans[keep]
    if jobs is None:
        return ans
    jobs = jobs[[c for c in ("uid", "request_key", "state_sha256", "meta_json") if c in jobs]]
    return jobs.merge(ans, on="request_key", how="left")


def load_frame(path: Path) -> pd.DataFrame:
    f = pd.read_parquet(path)
    for c in JSON_COLS:
        f[c] = [json.loads(v) for v in f[c]]
    return f


def check(root: Path, out: Path) -> int:
    f = load_frame(out / FRAME)
    d1 = C.d1_start(root)
    ref = json.loads((root / "results/analysis/h2.json").read_text())
    drop = ref["sensitivities"][f"drop_first_{H.DROP_FIRST}"]["from"]  # needs the bar calendar otherwise
    res = H.analyse(f, start=pd.Timestamp(d1["start"]), release=C.release_start(root), prior=C.prior_look_keys(root),
                    impls={}, drop_from=pd.Timestamp(drop))
    rows = [("confirmatory", res["confirmatory"], ref["confirmatory"]),
            ("placebo", res["placebo_ar_m5_m1"], ref["placebo_ar_m5_m1"]),
            ("pre_open", res["controls"]["pre_open_only"], ref["controls"]["pre_open_only"]),
            ("W_release", res["sensitivities"]["W_release"], ref["sensitivities"]["W_release"])]
    bad = 0
    for name, a, b in rows:
        ok = a["n_units"] == b["n_units"] and np.isclose(a["mean"], b["mean"], rtol=0, atol=1e-12) and \
            np.isclose(a["t"], b["t"], rtol=0, atol=1e-9)
        bad += not ok
        print(f"{name:12s} mean {a['mean']:.6f} (h2.json {b['mean']:.6f}) units {a['n_units']} t {a['t']:.3f} "
              f"{'OK' if ok else 'MISMATCH'}")
    return int(bad > 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="re-evaluate H2 from the exported frame only")
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    if a.check:
        return check(root, out)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    frame = export_frame(root)
    frame.to_parquet(out / FRAME, index=False, compression="zstd")
    files[FRAME] = {"rows": len(frame), "source": H.DEFAULT_GEN}
    for name, gen in READERS.items():
        t = export_reader(root / "results/judgments" / gen)
        t.to_parquet(out / f"readers_{name}.parquet", index=False, compression="zstd")
        files[f"readers_{name}.parquet"] = {"rows": len(t), "source": gen}
    for k in files:
        files[k]["sha256"] = sha256(out / k)
    (out / "MANIFEST.json").write_text(json.dumps({"script": "scripts/export_item_tables.py", "files": files},
                                                  indent=1) + "\n")
    print(json.dumps(files, indent=1))
    return check(root, out)


if __name__ == "__main__":
    sys.exit(main())
