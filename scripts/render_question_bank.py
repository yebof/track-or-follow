"""Render docs/design/question_bank_v1.md from docs/design/question_bank_v1.json (the single source of truth).

The Markdown is generated so that the text of every question, option, level and prompt is byte-identical to the JSON
that the code loads (`src/jevtrader/programs/bank_v1.py`, to be built). Do not edit the Markdown by hand.

Run: .venv/bin/python scripts/render_question_bank.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/design/question_bank_v1.json"
DST = ROOT / "docs/design/question_bank_v1.md"

PROGRAMS = {
    "N0": "News: sentence screening. One request per item; one Noul pair per sentence (headline + up to 6 sentences).",
    "N1": "News: item fan-out, including the sentinel. Since v1.0-rc6 the questions are asked in separate requests per "
          "(item, focal company), in two stages (`code_constants.N1.requests`, section D): about_focal, "
          "instructs_reader and promotional each alone; then, for an item about the focal company that is not "
          "quarantined, new_development alone and (since v1.0-rc7) the unchanged 13-question request, from which "
          "only the nine attribute questions are read.",
    "N2": "News: novelty against episodic memory (M-epi). One request per kept cluster that has prior records; "
          "one Noul per prior record (at most 5).",
    "N3": "News: weekly synthesis. One request per stock-week with at least one kept cluster. Typed-only state "
          "(taint barrier): no raw text.",
    "T": "Technical analyst. One request per stock-week. Trusted buckets only.",
    "FA": "Fundamentals analyst, 10-K business profile (writes M-sem). One request per 10-K per exposure group "
          "(customers, trade, regulation, macro, technology; see section D).",
    "FB": "Fundamentals analyst, filing read. One request per new 10-Q/10-K or valuation-bucket change. Trusted buckets.",
    "P1": "Policy analyst, item read. One request per pre-filtered policy item (at most 60 per week); 6 questions plus "
          "one `affects_industry_k` per FF12 industry (12).",
    "P2": "Policy analyst, weekly read. One request per week. Typed-only state.",
    "M": "Macro analyst. One request per week; 6 questions plus one `industry_tailwind_k` per FF12 industry (12).",
    "V": "System-2 claim verification (P1). One request per escalation; the three questions are asked once per claim "
         "(at most 6 claims), with the claim text in the instruction field.",
    "B": "Alias selection for blinding (P1, setup only).",
    "D": "Look-ahead probes (diagnostics only; never used in decisions). Recall probes are deliberately unblinded.",
    "H": "Baseline: holistic reader of the same news evidence (one request per stock-week). The same question serves "
         "the readers H-spot (state `H_spot`: datamarked evidence) and H-guard (state `H_guard`: evidence that "
         "survives the typed guards); see section B.",
    "HS": "Baseline: headline sentiment (one request per stock-week; one Choice per headline, at most 20).",
}

CHANGES = [
    "Question text is identical to the linted bank of `proposals/jev_native_question_bank_v1.json` (the wording smoke "
    "test of jev_native section 5.3 applies). v1 adds `channel` and `role` to every question, two recall probes "
    "(`D.recall_beats_median`, `D.recall_market_up`), and the baseline readers H and HS.",
    "Relative to `question_bank_v0.md`: per-agent outlook Scores become forecast-type `*_view` questions used only in "
    "variant V-outlook; `priced_in` is replaced by code features plus `N3.reaction_fit`; new guards (N0 screening, "
    "`instructs_reader`, `promotional`); the sentinel is split into an absolute Noul, a relative Choice and a pending "
    "Noul; novelty uses per-record Nouls against M-epi; the policy taxonomy adds stage, uncertainty and 12 signed "
    "industry Scores; a 10-K profile (F-A) is added; the macro state is time-blinded.",
    "FF12 descriptions: health insurers are described under Money, matching the SIC ranges of the FF12 map "
    "(`data/raw/industry/Siccodes12.txt`).",
    "The emulator prompt `emu-v1` drops the v0 invitation to use general world knowledge (reviewer m4), asks the "
    "model to treat STATE as data, and puts the questions before the state so that provider prompt caching applies.",
    "v1.0-rc3 (design_v2_delta section 4; no existing question text changes): QB-1 adds the LAP probe "
    "`D.recall_direction` and its state; QB-2 adds `conventions.content_free` and the content-free states `CF`; "
    "QB-3 adds `conventions.note_text_spotlight`, `conventions.datamark` and the H-spot state `H_spot`; QB-4 adds "
    "`code_constants.corroboration` and the thresholds `sentinel.corroboration` and `S2.exclusion_citations`; QB-5 "
    "adds `parser_version` and `parse_rule` to `s2-v1`, `mono-cot-v1` and `ta-lite-v1` (`gen-parse-v1`) and documents "
    "the implemented rule of `verb-v1`; QB-9 puts a `reason` key first in the TA-style trader and risk JSON and "
    "raises their `max_tokens` from 150 to 220; QB-12 adds the H-guard state `H_guard`; QB-13 adds "
    "`code_constants.exclusion_anchor` and `code_constants.sizing`; QB-14 (decision D-1, approved) adds the prompt "
    "`ng-v1` of the local typed decision model.",
    "v1.0-rc4 (pending decisions of docs/impl/day2_integration.md section 4.2 and the P1 patches of design_v2_delta "
    "section 4; the reasons are in docs/impl/day3_bank.md; no question, option or level text changes): the price "
    "anchor's d0 clause becomes the implemented rule (every session of the week's news window, so no untrusted item "
    "time chooses d0), and `exclusion_anchor.status` records that the filing anchor is kept; the TA-style "
    "`risk_template` spells out the full JSON (the trader's keys plus `action`) and asks an approval to repeat the "
    "trader's fields; `verb-v1` keeps its implemented parser `llm-verbalized-v2`, and `code_constants.parsers` names "
    "the parser of every generative prompt; `code_constants.corroboration` documents the empty-byline key and how "
    "kappa is chosen, and `thresholds.sentinel.corroboration` says non-quarantined; QB-6 adds the LA-6 paraphrase "
    "variants (section G); QB-7 adds the prompt `emu-v1-text` (LA-4); QB-8 adds the prompt `emu-v1-probe` (LA-2 "
    "validation); QB-10 gives S2's evidence lines a per-escalation source index and asks for items from two "
    "different sources; QB-11 adds `code_constants.P1.stage.geopolitics`.",
    "v1.0-rc5 (audit response, docs/impl/day3_bank.md; no question, option, level or parse-rule text changes): the "
    "TA-style `risk_template` introduces its schema with 'plus \"action\". Format: {...}' instead of 'plus "
    "\"action\": {...}', which read as a nested object that the unchanged parse rule would reject; the paraphrase "
    "variants `N1.important~a` (false) and `N1.important~b` (true) restore the original's threshold phrases "
    "'routine or minor' and 'the same day'.",
    "v1.0-rc6 (stream n1, DEV #2; the experiment and its tables are in docs/impl/dev2_n1.md): DEV #1 found that the "
    "emulator answers the N1 gates differently when all 13 questions share one request (about_focal was 'no' on 47% "
    "of the items whose headline names the focal company). N1 is now asked in two stages of separate requests "
    "(new `code_constants.N1.requests`): stage 1 asks about_focal, instructs_reader and promotional each alone; stage "
    "2, only for items with P(about_focal) >= 0.5 and P(instructs_reader) < 0.5, asks new_development alone and the "
    "nine attribute questions jointly. The layout stays question first (emu-v1). Two criteria are reworded where the "
    "evidence implicated them: `N1.promotional` (true and false: promotion by the item's own author or publisher; "
    "reports of what named investors, traders or commentators said or did, and unusual trading activity, are news "
    "reporting) and `N1.unconfirmed` (stem, true and false: an unconfirmed report rests on unnamed sources, reported "
    "talks or plans, or speculation; statements of named parties, named analysts' actions, price moves, trading and "
    "opinion are on the record). The stems and criteria of about_focal, new_development and instructs_reader are "
    "unchanged: their rewordings did not help in the question-first layout. Decision ids are unchanged.",
    "v1.0-rc7 (DEV #2 audit response; docs/impl/dev2_n1.md section 9): the nine attribute questions are read from the "
    "unchanged rc5 request with all 13 N1 questions (stage 2 of `code_constants.N1.requests` is now new_development "
    "alone plus the group {ask: the 13 questions, use: the nine attributes}; the joint request's gate answers are "
    "logged as `N1-joint` diagnostics and never read). The rc6 rewordings of `N1.promotional` and `N1.unconfirmed` "
    "are reverted to their rc5 text, so every N1 question has its rc5 text and the attribute answers are the rc5 "
    "ones. Why: on the judged dev states rc6's nine-question request flagged 28 of 109 kept items as unconfirmed "
    "against the judge's 6 (the rc5 request: 5), and it moved event_type, effect and important away from the rc5 "
    "answers. The gates, their stage-1 requests and the stage-2 gate are rc6's. Decision ids are unchanged.",
    "v1.0 (freeze, 2026-09-28; docs/design/decisions.md, docs/impl/freeze_report.md): v1.0-rc7 with only the "
    "`version` label changed, so the frozen content is exactly what DEV #2 ran and validated (canonical SHA-256 of "
    "rc7: c44c1ec005cd14f081403fe021809851466054d934488e9d58859993050cbf4f; tests/test_bank_freeze.py). The `status` "
    "text is rc7's, kept byte-identical for that reason. `FROZEN_SHA256` in `src/jevtrader/programs/bank_v1.py` is "
    "the hash above; MAIN and PRE runs refuse any other bank (`bank_v1.FROZEN_RUN_TAGS`). Any later change bumps the "
    "version and is logged in docs/design/prereg_deviations.md.",
]


def _frozen_sha256() -> str | None:
    """`FROZEN_SHA256` of src/jevtrader/programs/bank_v1.py, read from the source text (no import of the package)."""
    import re

    m = re.search(r'^FROZEN_SHA256: str \| None = (?:"([0-9a-f]{64})"|None)', (ROOT / "src/jevtrader/programs/bank_v1.py")
                  .read_text(), re.M)
    return m.group(1) if m else None


def _q(x) -> str:
    """Render a JSON content value (string or object) exactly."""
    if isinstance(x, str):
        return f'"{x}"'
    return "`` " + json.dumps(x, ensure_ascii=False) + " ``"  # double backticks: the JSON contains backticks


def render_question(prog: str, qid: str, q: dict) -> list[str]:
    head = (f"- **`{prog}.{qid}`** — {q['kind'].capitalize()} · {q['judgment']} · taint `{q['taint']}` · "
            f"decision `{q['decision']}` · channel `{q['channel']}` · role `{q['role']}`")
    out = [head, f"  - instructions: {_q(q['instructions'])}"]
    if q["kind"] == "noul":
        if q.get("true") is not None:
            out.append(f"  - true: {_q(q['true'])}")
        if q.get("false") is not None:
            out.append(f"  - false: {_q(q['false'])}")
        if q.get("true") is None and q.get("false") is None:
            out.append("  - criteria: none (the label names yes/no are interpreted directly)")
    elif q["kind"] == "choice":
        for lab, desc in q["options"].items():
            out.append(f"  - option `{lab}`: " + ("(label only; filled by code)" if desc is None else _q(desc)))
    else:
        for k, lev in enumerate(q["levels"]):
            out.append(f"  - level {k}: {_q(lev)}")
    return out


def main() -> None:
    raw = SRC.read_bytes()
    b = json.loads(raw)
    sha = hashlib.sha256(json.dumps(b, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    frozen = _frozen_sha256()
    n_templates = sum(len(v) for v in b["programs"].values())
    variants = b.get("diagnostic_variants", {})
    n_variants = sum(len(blk["variants"]) for blk in variants.values())
    dm = b["conventions"]["datamark"]
    dm_char = chr(int(dm["char"].removeprefix("U+"), 16))
    L: list[str] = []
    L += [
        "# Question bank v1 (exact text of every question and prompt)",
        "",
        f"- **Version:** `{b['version']}`. **Status:** {b['status']}",
        f"- **Canonical SHA-256** (sorted-key JSON of `question_bank_v1.json`): `{sha}`",
        *([f"- **Frozen:** yes. `FROZEN_SHA256` in `src/jevtrader/programs/bank_v1.py` equals the hash above (set "
           f"2026-09-28; section F, v1.0)."] if frozen == sha else
          [f"- **Frozen:** no (`FROZEN_SHA256` is {'unset' if frozen is None else f'`{frozen}`, not this bank'})."]),
        "- **Generated** by `scripts/render_question_bank.py` from `docs/design/question_bank_v1.json`. Do not edit by "
        "hand; edit the JSON and re-render. The design that uses these questions is `docs/design/design_v1.md` "
        "(programs in section 5, features in section 6, prompts in section 11.2), as amended by "
        "`docs/design/design_v2_delta.md` (bank patches in its section 4).",
        f"- **Size:** {n_templates} question templates in {len(b['programs'])} programs; templates with suffixes "
        "expand per sentence, prior record, industry or headline (section A)."
        + (f" Plus {n_variants} diagnostic variants of existing questions (section G), never used in decisions."
           if n_variants else ""),
        "",
        "---",
        "",
        "## A. Conventions",
        "",
        f"- **Note field** (first field of every state): {_q(b['conventions']['note_text'])}",
        f"- **Spotlight note** (`note_text_spotlight`; the note of H-spot states instead): "
        f"{_q(b['conventions']['note_text_spotlight'])}",
        f"- **Datamark** (H-spot evidence): character {dm['char']} (\"{dm_char}\"); "
        f"rule: {_q(dm['rule'])}",
        "- **Content-free values** (CF probe states; `content_free`): "
        + "; ".join(f'"{v}"' for v in b["conventions"]["content_free"]["values"])
        + f"; code name `{b['conventions']['content_free']['code_name']}`. "
        + _q(b["conventions"]["content_free"]["rule"]),
        f"- **Evidence fence** (embedded as `judge_from` in evaluative raw-text questions): "
        f"{_q(b['conventions']['evidence_fence'])}",
        f"- **Template suffixes:** {b['conventions']['template_suffixes']}",
        f"- **Size buckets** of `focal_company.size`: " + "; ".join(f'"{s}"' for s in b["conventions"]["size_buckets"])
        + f" ({b['conventions']['size_rule']}).",
        "- **Probability bins** used when code renders a probability as a label (N3, P2, S2 inputs): "
        + "; ".join(f"≥ {t}: \"{s}\"" for t, s in b["conventions"]["probability_bins"]) + ".",
        "- **Relative times:** " + "; ".join(f'"{s}"' for s in b["conventions"]["relative_time"])
        + ". No absolute date appears in any decision state.",
        f"- **Stock reaction buckets (N3):** {b['conventions']['reaction_buckets']}.",
        "- **Masks:** " + "; ".join(f"{k} → `{v}`" for k, v in b["conventions"]["masks"].items()) + ".",
        "- **Emulator limits:** at most 20 options per question (letters A–T). Jev allows 255.",
        "",
        "**FF12 industry texts** (filled into `industry` in `P1.affects_industry_k`, `M.industry_tailwind_k` and "
        "`focal_company.industry`):",
        "",
        "| Code | Text |",
        "|---|---|",
    ]
    L += [f"| {k} | {v} |" for k, v in b["ff12_industries"].items()]
    L += ["", "---", "", "## B. State schemas (built by code; `programs/render.py`)", ""]
    for prog, sch in b["state_schemas"].items():
        L.append(f"**{prog}**")
        L.append("")
        for field, desc in sch.items():
            L.append(f"- `{field}`: {desc}")
        L.append("")
    L += ["---", "", "## C. Questions", ""]
    for prog, qs in b["programs"].items():
        L.append(f"### {prog}")
        L.append("")
        L.append(PROGRAMS[prog])
        L.append("")
        for qid, q in qs.items():
            L += render_question(prog, qid, q)
        L.append("")
    L += ["---", "", "## D. Code constants (used by the programs; fixed a priori)", "", "```json",
          json.dumps({"groups": b["groups"], **b["code_constants"]}, indent=1, ensure_ascii=False), "```", "",
          "---", "", "## E. Prompts (LLM backends and LLM baselines)", ""]
    for pid, pr in b["prompts"].items():
        L.append(f"### {pid}")
        L.append("")
        for k, v in pr.items():
            if isinstance(v, str):
                L += [f"**{k}**", "", "```text", v, "```", ""]
            else:
                L += [f"**{k}**", "", "```json", json.dumps(v, indent=1, ensure_ascii=False), "```", ""]
    L += ["---", "", "## F. Changes", ""] + [f"- {c}" for c in CHANGES] + [""]
    if variants:
        L += ["---", "", "## G. Diagnostic variants (never used in decisions or learning)", ""]
        for kind, blk in variants.items():
            L += [f"### {kind}", "", f"Rule: {_q(blk['rule'])}", ""]
            for vid, e in blk["variants"].items():
                prog, qid = e["of"].split(".", 1)
                base = b["programs"][prog][qid]
                L.append(f"- **`{vid}`** — {kind} of `{e['of']}` ({base['kind'].capitalize()}; decision "
                         f"`{base['decision']}`; diagnostic: {str(e['diagnostic']).lower()})")
                L.append(f"  - instructions: {_q(e['instructions'])}")
                for k in ("true", "false"):
                    if k in e:
                        L.append(f"  - {k}: {_q(e[k])}")
            L.append("")
    DST.write_text("\n".join(L))
    print(f"wrote {DST.relative_to(ROOT)}: {n_templates} templates, sha256 {sha[:16]}…")


if __name__ == "__main__":
    main()
