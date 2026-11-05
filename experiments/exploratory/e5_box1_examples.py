#!/usr/bin/env python3
"""EXPLORATORY (not pre-registered), E5 Box 1: two frame items chosen by the fixed rule of e5_spec.yaml (box1),
paraphrased and blinded (outline_v2 §3 "Box 1"). Illustrative items chosen by a fixed rule; not evidence.

    .venv/bin/python experiments/exploratory/e5_box1_examples.py select       # the rule -> results/exploratory/
                                                                              #   e5_box1_examples.json (no text)
    .venv/bin/python experiments/exploratory/e5_box1_examples.py show-text    # print the two raw texts to the
                                                                              #   terminal for the author (never saved)
    .venv/bin/python experiments/exploratory/e5_box1_examples.py render       # -> paper/tables/box1_examples.tex

Selection (spec box1.rule): item A = smallest sha256("BOX1-A|<unit>") among frame units whose top event type is
earnings_results, whose top effect level is 0 or 4 and whose text does not match the price-report classifier; item B =
smallest sha256("BOX1-B|<unit>") among units whose text matches. No redraw.

What the box shows per item: the author's one-sentence paraphrase (experiments/exploratory/e5_box1_paraphrases.yaml:
keyed by the item's unit hash, written with "{firm}" for the focal firm, which the script replaces by the firm's
blinding pseudonym, data.blinding.pseudonym; the paraphrase may contain no digit, so no number is typed by hand);
the publication time relative to the decision cutoff; the reader's top event type and its probability; the five
E[effect] level probabilities and E[effect]; the top vs_expectations answer; AR[0,+1] and AR[-5,-1] in %; and the
classifier flag. Every number comes from the registered judgment tables and prices through this script. The licensed
text is printed to the terminal only (show-text) and never written to a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _xcommon as X  # noqa: E402
import e5_construct_checks as E5  # noqa: E402

import common as C  # noqa: E402  (registered, read-only)
import h2_validity as H2  # noqa: E402  (registered, read-only)

NAME = "e5_box1_examples"
PARA = E5.HERE / "e5_box1_paraphrases.yaml"
TABLE = C.ROOT / "paper" / "tables" / "box1_examples.tex"
LEVEL_NAMES = ("clearly harmful", "mildly harmful", "neutral or mixed", "mildly helpful", "clearly helpful")
TYPE_NAMES = {"earnings_results": "earnings results", "guidance_change": "guidance change",
              "analyst_action": "analyst action", "deal_or_investment": "deal or investment",
              "product_or_contract": "product or contract", "capital_return": "capital return",
              "financing_or_credit": "financing or credit", "legal_or_regulatory": "legal or regulatory",
              "management_or_labor": "management or labor", "operations_disruption": "operations disruption",
              "industry_or_macro_readthrough": "industry or macro read-through", "other": "other development"}


def unit_hash(unit: str) -> str:
    return hashlib.sha256(unit.encode()).hexdigest()[:16]


def effect_dists(root: Path, units: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict]:
    """The N1 answers (effect, event_type, vs_expectations distributions) of the given (t, symbol, eid) units."""
    out = {}
    tag = f'"key": "{H2.ITEMS_KEY}'
    syms = {s for _, s, _ in units}
    with open(root / "results" / "judgments" / E5.GEN / "memory" / "work.jsonl") as fh:
        for line in fh:
            if tag not in line:
                continue
            rec = json.loads(line)["record"]
            sym = rec["key"][len(H2.ITEMS_KEY):]
            if sym not in syms:
                continue
            t = rec["t"][:10]
            for r in rec["value"]:
                k = (t, sym, r["eid"])
                if k in units and k not in out:
                    out[k] = {"effect": r.get("effect") or {}, "event_type": r.get("event_type") or {},
                              "vs_expectations": r.get("vs_expectations") or {}}
    return out


def select(root: Path, argv) -> int:
    spec, sha, log = E5.require_logged_spec()
    guard = C.Guard(root)
    g, start, end, d1 = E5.build_classified(root, guard, spec)
    g["unit_key"] = list(zip(g["t"].dt.strftime("%Y-%m-%d"), g["symbol"], g["eid"]))
    cand_a = g[(g["top_type"] == "earnings_results") & ~g["price"] & g["has_text"]]
    dists = effect_dists(root, set(cand_a["unit_key"]) | set(g.loc[g["price"], "unit_key"]))
    top_eff = cand_a["unit_key"].map(lambda k: max(dists[k]["effect"], key=dists[k]["effect"].get)
                                     if dists.get(k, {}).get("effect") else None)
    cand_a = cand_a[top_eff.isin(["0", "4"]).to_numpy()]
    cand_b = g[g["price"]]
    picks = {}
    for tag, cand in (("A", cand_a), ("B", cand_b)):
        r = cand.assign(rank=[hashlib.sha256(f"BOX1-{tag}|{u}".encode()).hexdigest() for u in cand["unit"]])
        picks[tag] = r.sort_values("rank").iloc[0]
    sel = pd.DataFrame([picks["A"], picks["B"]]).reset_index(drop=True)
    guard.check(sel["t"].unique(), "E5 Box 1 labels (AR)")
    px = C.load_prices(guard, start, end + pd.Timedelta(days=7))
    sel = H2.add_abnormal_returns(sel, px.book)
    from jevtrader.data.blinding import pseudonym
    from jevtrader.data.news import decision_cutoff

    items = {}
    for tag, r in zip(("A", "B"), sel.itertuples(index=False)):
        dd = dists[r.unit_key]
        eff = {k: float(v) for k, v in dd["effect"].items()}
        et, vs = dd["event_type"], dd["vs_expectations"]
        cut = decision_cutoff(r.t)
        hours = (cut - r.created_at).total_seconds() / 3600.0
        items[tag] = {"unit_hash": unit_hash(r.unit), "pseudonym": pseudonym(r.symbol),
                      "rule": "earnings_results, top effect level 0 or 4, no price report" if tag == "A"
                      else "text reports the price move",
                      "hours_before_cutoff": hours, "top_event_type": r.top_type,
                      "p_top_event_type": float(et.get(r.top_type, np.nan)),
                      "effect_probs": [eff.get(str(k), 0.0) for k in range(5)], "E_effect": float(r.E_effect),
                      "top_effect_level": int(max(eff, key=eff.get)),
                      "vs_expectations_top": max(vs, key=vs.get) if vs else None,
                      "p_vs_expectations_top": float(max(vs.values())) if vs else None,
                      "ar_0_1": float(r.ar), "ar_m5_m1": float(r.ar_placebo), "price_report": bool(r.price),
                      "created_before_open_et": r.created_at.tz_convert(C.ET).strftime("%H:%M") < H2.PRE_OPEN}
    payload = {"label": X.LABEL, "note": spec["box1"]["label"], "rule": spec["box1"]["rule"],
               "spec": {"sha256": sha, "logged_utc": log["current"]["logged_utc"]},
               "candidates": {"A": int(len(cand_a)), "B": int(len(cand_b))}, "items": items,
               "private_keys": {tag: {"unit": picks[tag]["unit"]} for tag in picks}}
    p = X.write(NAME, payload, script=__file__, window="W_clean", guard=guard,
                inputs={"work.jsonl": root / "results/judgments" / E5.GEN / "memory" / "work.jsonl", "spec": E5.SPEC},
                argv=argv, why="Box 1 illustration rule specified in e5_spec.yaml after the unlock")
    print(f"Box 1 selected: A {items['A']['unit_hash']} ({len(cand_a)} candidates), B {items['B']['unit_hash']} "
          f"({len(cand_b)} candidates) -> {C.rel(p)}")
    return 0


def show_text(root: Path) -> int:
    d = E5.X.OUT_DIR / f"{NAME}.json"
    box = json.loads(d.read_text())
    nt = pd.read_parquet(E5.NEWS, columns=["id", "headline", "summary"])
    nt["eid"] = ["news:" + hashlib.sha256(str(i).encode()).hexdigest()[:16] for i in nt["id"]]
    for tag, v in box["private_keys"].items():
        t, sym, eid = v["unit"].split("|")
        r = nt[nt["eid"] == eid].iloc[0]
        it = box["items"][tag]
        print(f"=== {tag} unit_hash {it['unit_hash']} focal {sym} -> {it['pseudonym']}")
        print("HEADLINE:", r["headline"])
        print("SUMMARY:", r["summary"])
    return 0


def _pct(x: float) -> str:
    s = f"{abs(100 * x):.1f}"
    return ("\\ensuremath{-}" if x < 0 and s != "0.0" else "+" if x > 0 and s != "0.0" else "") + s


def _p(x: float) -> str:
    return f"{x:.2f}"


def _when(hours: float) -> str:
    if hours < 24:
        h = int(round(hours))
        return f"{h} hour{'s' if h != 1 else ''} before the decision cutoff"
    d = int(round(hours / 24))
    return f"{d} day{'s' if d != 1 else ''} before the decision cutoff"


def render() -> int:
    box = X.OUT_DIR / f"{NAME}.json"
    d = json.loads(box.read_text())
    if not (d.get("meta", {}).get("exploratory") is True and d["meta"].get("label") == X.LABEL):
        raise SystemExit(f"{box}: not an exploratory output")
    para = yaml.safe_load(PARA.read_text())
    rows = []
    for tag in ("A", "B"):
        it = d["items"][tag]
        pp = para.get(tag) or {}
        if pp.get("unit_hash") != it["unit_hash"]:
            raise SystemExit(f"paraphrase {tag} is for unit {pp.get('unit_hash')}, the rule selected {it['unit_hash']}")
        text = str(pp.get("paraphrase", "")).strip()
        if not text or re.search(r"\d", text) or "{firm}" not in text:
            raise SystemExit(f"paraphrase {tag}: must be non-empty, contain {{firm}} and no digit")
        text = text.replace("{firm}", it["pseudonym"]).replace("&", "\\&").replace("%", "\\%")
        head = ("(A) Earnings item, no price report" if tag == "A" else "(B) Item that reports the price move")
        eff = " / ".join(_p(x) for x in it["effect_probs"])
        vs = (f"{it['vs_expectations_top'].replace('_', ' ')} ({_p(it['p_vs_expectations_top'])})"
              if it.get("vs_expectations_top") else "---")
        rows += [f"\\textbf{{{head}.}} \\emph{{{text}}} Published {_when(it['hours_before_cutoff'])}.\\par",
                 f"\\hspace*{{1em}}Reader: {TYPE_NAMES.get(it['top_event_type'], it['top_event_type'])} "
                 f"({_p(it['p_top_event_type'])}); effect levels 0 (clearly harmful) to 4 (clearly helpful): {eff}; "
                 f"\\Eeff{{}} {it['E_effect']:.2f}; vs.\\ expectations: {vs}.\\par",
                 f"\\hspace*{{1em}}Records: \\ARzo{{}} {_pct(it['ar_0_1'])}\\%, placebo \\ARpl{{}} "
                 f"{_pct(it['ar_m5_m1'])}\\%; reports the price move: {'yes' if it['price_report'] else 'no'}.\\par"]
        if tag == "A":
            rows.append("\\vskip\\medskipamount")  # a primitive skip: acmart warns on \\vspace in floats
    # round-2 figure review B1-1/B1-2: a light framed box (not a table), ragged right, the levels named, no note that
    # repeats the caption
    note = ("Paraphrased, with the firm's blinding pseudonym; reader: the confirmatory reader's typed judgments "
            "(probabilities in parentheses). Exploratory, not pre-registered.")
    text = "\n".join([f"% GENERATED by experiments/exploratory/e5_box1_examples.py ({X.LABEL}; e5_spec.yaml box1) -- "
                      "do not edit by hand (CLAUDE.md rule 2).",
                      "{\\footnotesize\\setlength{\\fboxsep}{4pt}\\setlength{\\fboxrule}{0.4pt}%",
                      "\\fcolorbox{black!35}{black!4}{\\begin{minipage}{\\dimexpr\\linewidth-2\\fboxsep-2\\fboxrule"
                      "\\relax}\\raggedright", *rows, "\\vskip\\smallskipamount{\\scriptsize " + note + "\\par}",
                      "\\end{minipage}}%", "}"]) + "\n"
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    TABLE.write_text(text)
    print(f"wrote {C.rel(TABLE)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("select", "show-text", "render"))
    ap.add_argument("--root", default=str(C.ROOT))
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.cmd == "select":
        return select(root, argv if argv is not None else sys.argv[1:])
    if a.cmd == "show-text":
        return show_text(root)
    return render()


if __name__ == "__main__":
    sys.exit(main())
