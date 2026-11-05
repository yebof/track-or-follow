#!/usr/bin/env python3
"""Holm over H1-H3 and the claim register (prereg §3.4, §4.2).

    .venv/bin/python experiments/analysis/confirmatory.py            # reads results/analysis/{h1,h2,h3,bound}.json

Family-wise alpha = .05, Holm step-down (eval.metrics.holm) over the three one-sided confirmatory p-values, in the
presentation order H1 -> H2 -> H3. A hypothesis whose analysis file is missing or did not produce a p-value enters
the family with p = 1 (it cannot be claimed, and m stays 3), and the output says so. Each claim is stated only if its
Holm-adjusted p < .05; otherwise the pre-registered fallback wording applies (positioning_v2 §2.7 as amended; the
wording itself lives in experiments/report/make_numbers.py, which reads `claims` from this file). Only files of the
registered window are accepted unless --window names another one (development runs).

The claim register (prereg §4.2) also carries BOUND, outside the Holm family: `claims.BOUND` is True iff bound.json
(experiments/analysis/bound_claim.py: BG-5 finds 0 violations and the P0-P5 property tests pass) says so; a missing
bound.json gives False (the bound clause of the result sentences is then dropped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

NAME = "confirmatory"
ALPHA = 0.05
FAMILY = (("H1", "h1", ("confirmatory", "p_one_sided"), ("confirmatory", "estimate")),
          ("H2", "h2", ("confirmatory", "p"), ("confirmatory", "mean")),
          ("H3", "h3", ("confirmatory", "p"), ("confirmatory", "mean")))


def _get(d: dict, path: tuple[str, ...]):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def decide(results: dict[str, dict | None], alpha: float = ALPHA, bound: dict | None = None) -> dict:
    from jevtrader.eval.metrics import holm

    rows = []
    for label, _, p_path, est_path in FAMILY:
        d = results.get(label)
        p = _get(d, p_path) if d else None
        rows.append({"hypothesis": label, "p_one_sided": p, "estimate": _get(d, est_path) if d else None,
                     "ci": _get(d, (p_path[0], "ci")) if d else None, "available": p is not None})
    ps = [r["p_one_sided"] if r["p_one_sided"] is not None else 1.0 for r in rows]
    adj = holm(ps)
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        r["claim"] = "stated" if (r["available"] and a < alpha) else "fallback"
    b = _get(bound, ("claim", "bound")) if bound else None
    return {"alpha": alpha, "method": "Holm step-down, m = 3, one-sided p-values", "family": rows,
            "claims": {**{r["hypothesis"]: r["claim"] == "stated" for r in rows}, "BOUND": b is True},
            "bound": {"available": bound is not None, "claim": b is True,
                      "bg5_violations": _get(bound, ("bg5", "n_violations")) if bound else None,
                      "property_tests": _get(bound, ("property_tests", "summary")) if bound else None,
                      "rule": "outside the Holm family: BG-5 finds 0 violations and the property tests pass"},
            "complete": all(r["available"] for r in rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", default=C.REGISTERED_WINDOW)
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--in-dir", default=None, help="directory of h1/h2/h3.json (default: by window)")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    guard = C.Guard(root)
    guard.check_window(a.window, "Holm over H1-H3")
    results, inputs = {}, {}
    bound = None
    for label, name, _, _ in (*FAMILY, ("BOUND", "bound", None, None)):
        p = (Path(a.in_dir) / f"{name}.json") if a.in_dir else C.out_path(name, a.window, None, root)
        inputs[label] = p
        if p.exists():
            d = C.read_json(p)
            ok = (d["meta"].get("registered_window") if C.is_registered(a.window, root)
                  else d["meta"]["window"] == a.window)
            if not ok:
                raise SystemExit(f"{p} is for window {d['meta']['window']}, not {a.window}")
            if label == "BOUND":
                bound = d
            else:
                results[label] = d
        elif label != "BOUND":
            results[label] = None
    res = decide(results, bound=bound)
    path = C.out_path(NAME, a.window, a.out_dir, root)
    C.write_result(path, res, script=__file__, window=a.window, guard=guard, inputs=inputs, argv=argv)
    print(" ".join(f"{r['hypothesis']}: p {r['p_one_sided']} holm {r['p_holm']:.4g} {r['claim']}"
                   for r in res["family"]), "->", C.rel(path, root))
    return 0


if __name__ == "__main__":
    sys.exit(main())
