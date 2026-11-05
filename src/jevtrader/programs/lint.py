"""Question-bank linter, rules L1-L12 (jev_native §2.4; design_v1 §5.0). Run by tests/test_bank_lint.py.

The rules turn the documented jev-1.13 failure modes (docs/jev/pages/model-jaggedness__jev-1.13) into checks:
literal reading and indirection (L7 paths), math (L4 digits), contradictory criteria (L2 negation), no
structural invariants (L11 one phrasing per decision), context rot (L9 budgets), emulator limits (L8).
L9 is also enforced at run time on every request (`check_request`). Declared diagnostic variants (v1.0-rc4, QB-6:
paraphrases for LA-6) are the one exception to "one phrasing per decision"; `lint_variants` checks them instead.
L13 (v1.0-rc6) checks a program's request plan (`code_constants["N1.requests"]`: which questions share a request;
`lint_request_plan`), after DEV #1 showed that the emulator's gate answers depend on which questions share a request.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from jevtrader.programs.bank_v1 import BANK_MD_PATH, FROZEN_SHA256, Bank, md_sha256
from jevtrader.programs.base import QSpec

JUDGMENTS = {"descriptive", "evaluative", "forecast", "guard", "probe"}
TAINTS = {"raw_text", "typed_only", "numeric_buckets"}
CHANNELS = {"trusted", "semi_trusted", "untrusted", "diagnostic", "setup"}
NEGATION = re.compile(r"\b(not|no|never|none|without|free of|other than|isn't|doesn't|don't|fails?)\b", re.I)
CONJUNCTION = re.compile(r"\b(and|or)\b", re.I)
DIGIT_OK = re.compile(r"\b(50-day|200-day|52-week|RSI|10-K)\b")
CATCH_ALL = {"other", "none", "unclear", "unknown", "cannot_tell", "not_available", "not_stated", "not_policy",
             "other_policy", "no_clear_change", "says_nothing",
             "cannot_know"}  # v1.0-rc3 (QB-1): the abstain option of D.recall_direction (LAP = 1 - P(cannot_know))
# Choices whose options cover every case by construction (reviewed; L6 would otherwise flag them).
EXHAUSTIVE = {"T.trend", "M.regime", "M.growth", "M.inflation", "M.fed_direction", "P2.policy_regime"}
EMULATOR_MAX_OPTIONS, JEV_MAX_OPTIONS = 20, 255
JEV_SCORE_LEVELS = (2, 10)
# L9: soft per-program state budgets (jev_native §3.4, o200k proxy) and the hard request budget (F07).
STATE_BUDGET = {"N0": 400, "N1": 500, "N2": 500, "N3": 1500, "T": 600, "FA": 2000, "FB": 800, "P1": 500,
                "P2": 1500, "M": 1200, "V": 3000}
HARD_BUDGET = 24_000


class LintError(ValueError):
    pass


@dataclass(frozen=True)
class Issue:
    rule: str
    qid: str
    msg: str
    severity: Literal["error", "warning"] = "error"


def main_question(ins) -> str:
    return ins if isinstance(ins, str) else str(ins.get("question", "")) if isinstance(ins, dict) else ""


def _texts(q: QSpec) -> list[str]:
    out = [json.dumps(q.instructions, ensure_ascii=False)]
    out += [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in (q.true, q.false) if x is not None]
    out += [v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in (q.options or {}).values() if v]
    out += [v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in (q.levels or ())]
    return out


def allowed_paths(bank: Bank, program: str) -> set[str]:
    """State fields a question may point at by backticked path. A per-question schema (program D: keys are
    question names) is resolved from its descriptions: `{a, b: {c}}` sets and the programs it names."""
    schema = bank.state_schemas.get(program, {})
    if schema and set(schema) <= set(bank.programs.get(program, {})):
        out: set[str] = set()
        for desc in schema.values():
            for grp in re.findall(r"\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", desc):
                out |= set(re.findall(r"[A-Za-z_]+", grp))
            for other in bank.state_schemas:
                if other != program and re.search(rf"\b{re.escape(other)}\b", desc):
                    out |= set(bank.state_fields(other))
        return out
    return set(bank.state_fields(program)) | set(bank.instruction_fields(program))


def effective_decision(bank: Bank, q: QSpec) -> str:
    """A decision id naming a declared question group (bank `groups`, e.g. profile.customers for FA) is a
    group id; each member is then its own decision."""
    last = q.decision.rsplit(".", 1)[-1]
    if q.name in bank.groups.get(q.program, {}).get(last, []):
        return f"{q.decision}.{q.name}"
    return q.decision


def lint_question(bank: Bank, q: QSpec) -> list[Issue]:
    iss: list[Issue] = []
    mq = main_question(q.instructions)
    if len(mq.split()) > 40:
        iss.append(Issue("L1", q.qid, "main question > 40 words"))
    if q.kind == "noul" and (m := NEGATION.search(mq)):
        iss.append(Issue("L2", q.qid, f"negation in Noul question: {m.group(0)!r}"))
    if q.kind == "noul" and CONJUNCTION.search(mq):
        iss.append(Issue("L3", q.qid, "conjunction in Noul question (must be one category)", "warning"))
    if q.judgment != "probe":
        for t in _texts(q):
            if re.search(r"\d", DIGIT_OK.sub("", t)):
                iss.append(Issue("L4", q.qid, f"digit in text: {t[:60]!r}"))
    if q.kind == "score":
        lv = q.levels or ()
        if not 3 <= len(lv) <= 7:
            iss.append(Issue("L5", q.qid, f"{len(lv)} levels (need 3-7)"))
        if any(len(str(x).split()) < 5 for x in lv):
            iss.append(Issue("L5", q.qid, "a level has fewer than 5 words"))
        if not JEV_SCORE_LEVELS[0] <= len(lv) <= JEV_SCORE_LEVELS[1]:
            iss.append(Issue("L8", q.qid, f"{len(lv)} levels outside Jev's 2-10"))
    if q.kind == "choice":
        opts = q.options or {}
        if len(opts) > EMULATOR_MAX_OPTIONS:
            iss.append(Issue("L8", q.qid, f"{len(opts)} options > {EMULATOR_MAX_OPTIONS} (emulator)"))
        if len(opts) > JEV_MAX_OPTIONS:
            iss.append(Issue("L8", q.qid, f"{len(opts)} options > {JEV_MAX_OPTIONS} (Jev)"))
        # a diagnostic variant ("<qid>~<tag>") has its original's options, so it inherits the declaration
        if q.qid.split("~", 1)[0] not in EXHAUSTIVE and not set(opts) & CATCH_ALL:
            iss.append(Issue("L6", q.qid, "no catch-all option and not declared exhaustive"))
    ok = allowed_paths(bank, q.program) | (set(q.instructions) if isinstance(q.instructions, dict) else set())
    for path in re.findall(r"`([A-Za-z_]+)", " ".join(_texts(q))):
        if path not in ok:
            iss.append(Issue("L7", q.qid, f"unknown state path `{path}`"))
    if q.judgment not in JUDGMENTS:
        iss.append(Issue("L10", q.qid, f"judgment {q.judgment!r} not declared"))
    if q.taint not in TAINTS:
        iss.append(Issue("L10", q.qid, f"taint {q.taint!r} not declared"))
    if q.channel not in CHANNELS or not q.role:
        iss.append(Issue("L10", q.qid, "channel/role not declared"))
    if q.judgment == "forecast" and q.role not in ("ablation", "baseline"):
        iss.append(Issue("L10", q.qid, "forecast question outside ablation/baseline roles"))
    return iss


VARIANT_KEYS = {"of", "diagnostic", "instructions", "true", "false"}


def lint_variants(bank: Bank) -> list[Issue]:
    """Diagnostic variants (bank `diagnostic_variants`, v1.0-rc4 QB-6). A second phrasing of a decision is allowed
    only as a declared diagnostic rewrite of an existing question (L11): `diagnostic: true`, an id
    "<of>~<tag>", only the stem and (Noul) the criteria replaced, an object's other fields (the `judge_from` fence)
    verbatim, the stem and every replaced criterion actually reworded, no two variants of a question alike. The
    rewritten question must then pass L1-L10 like any bank question."""
    iss: list[Issue] = []
    for kind, block in bank.diagnostic_variants.items():
        entries = block.get("variants") if isinstance(block, dict) else None
        if not isinstance(block, dict) or not block.get("rule") or not entries:
            iss.append(Issue("L11", f"diagnostic_variants.{kind}", "a variant block needs a rule and variants"))
            continue
        seen: dict[str, set[str]] = defaultdict(set)
        for vid, e in entries.items():
            of = e.get("of") if isinstance(e, dict) else None
            try:
                base = bank.q(of) if isinstance(of, str) else None
            except KeyError:
                base = None
            if base is None:
                iss.append(Issue("L11", vid, f"`of` {of!r} is not a bank question"))
                continue
            bad = [msg for cond, msg in [
                (e.get("diagnostic") is not True, "not declared diagnostic"),
                (vid.split("~", 1)[0] != of or "~" not in vid, f"id must be '{of}~<tag>'"),
                (bool(set(e) - VARIANT_KEYS), f"unexpected keys {sorted(set(e) - VARIANT_KEYS)}"),
                ("instructions" not in e, "no rewritten stem"),
                (base.kind != "noul" and bool({"true", "false"} & set(e)), "criteria on a question that is not a Noul"),
                (base.kind == "noul" and base.true is not None and not {"true", "false"} <= set(e),
                 "a Noul variant rewrites both criteria"),
                (base.kind == "noul" and base.true is None and bool({"true", "false"} & set(e)),
                 "criteria on a Noul that has none"),
            ] if cond]
            if bad:
                iss += [Issue("L11", vid, m) for m in bad]
                continue
            ins, bins = e["instructions"], base.instructions
            if type(ins) is not type(bins) or (isinstance(bins, dict) and (
                    set(ins) != set(bins) or any(ins[k] != bins[k] for k in bins if k != "question"))):
                iss.append(Issue("L11", vid, "only the stem may change (an object's other fields stay verbatim)"))
                continue
            q = bank.variant(vid, kind)
            if main_question(q.instructions) == main_question(bins):
                iss.append(Issue("L11", vid, "stem identical to the original: not a rewrite"))
            if base.kind == "noul" and base.true is not None and (q.true == base.true or q.false == base.false):
                iss.append(Issue("L11", vid, "criteria identical to the original: not a rewrite"))
            text = json.dumps([q.instructions, q.true, q.false], sort_keys=True)
            if text in seen[of]:
                iss.append(Issue("L11", vid, f"duplicate variant of {of}"))
            seen[of].add(text)
            iss += lint_question(bank, q)
    return iss


REQUEST_LAYOUTS = {"question_first"}  # the layouts the programs can send (emu-v1); a state-first one needs S-2


def lint_request_plan(bank: Bank) -> list[Issue]:
    """L13 (v1.0-rc6, rc7): a program's request plan (`code_constants["<program>.requests"]`, today N1's; docs/impl/
    dev2_n1.md). A group is a list (asked and read) or {"ask": [...], "use": [...]} (rc7: the attributes are read from
    the rc5 13-question request). The `use` questions of `stage1` + `stage2` partition the program's questions exactly
    (every answer read from exactly one request); a group reads only questions it asks and asks each once; every
    question whose role is `gate` is read from a request that asks it alone, since the joint layout moved the gates'
    answers (DEV #1 §8.3); stage 1 reads the questions the stage-2 gate reads (the gate names them as P(<question>));
    the layout is one the programs can send."""
    iss: list[Issue] = []
    for key, spec in bank.code_constants.items():
        if not key.endswith(".requests"):
            continue
        prog = key.removesuffix(".requests")
        where = f"code_constants.{key}"
        if prog not in bank.programs or not isinstance(spec, dict) or not spec.get("rule"):
            iss.append(Issue("L13", where, "a request plan needs a bank program and a rule"))
            continue
        qs = bank.program(prog)

        def parts(g) -> tuple[list, list]:
            if isinstance(g, dict):
                return list(g.get("ask") or []), list(g.get("use") or [])
            return list(g), list(g)

        s1 = [parts(g) for g in (spec.get("stage1") or [])]
        s2 = [parts(g) for g in (spec.get("stage2") or [])]
        used = [q for _, use in s1 + s2 for q in use]
        if not s1 or sorted(used) != sorted(qs) or len(used) != len(set(used)):
            iss.append(Issue("L13", where, f"the groups must partition the {prog} questions exactly"))
        for ask, use in s1 + s2:
            if not use or not set(use) <= set(ask):
                iss.append(Issue("L13", where, f"group {ask} reads {sorted(set(use) - set(ask))} without asking them"
                                               if use else f"group {ask} reads nothing"))
            if len(set(ask)) != len(ask):
                iss.append(Issue("L13", where, f"group {ask} must ask each question once"))
            for q in use:
                if q in qs and qs[q].role == "gate" and len(ask) > 1:
                    iss.append(Issue("L13", where, f"gate {prog}.{q} shares a request with {sorted(set(ask) - {q})}"))
        if s2:
            read = set(re.findall(r"P\(([a-z_]+)\)", str(spec.get("stage2_gate", ""))))
            first = {q for _, use in s1 for q in use}
            if not read or not read <= first:
                iss.append(Issue("L13", where, f"the stage-2 gate must read stage-1 questions only, reads {sorted(read)}"))
        if spec.get("layout", "question_first") not in REQUEST_LAYOUTS:
            iss.append(Issue("L13", where, f"layout {spec.get('layout')!r} cannot be sent (only {sorted(REQUEST_LAYOUTS)})"))
    return iss


def lint_bank(bank: Bank, md_path=BANK_MD_PATH) -> list[Issue]:
    iss = [i for q in bank.questions() for i in lint_question(bank, q)]
    iss += lint_variants(bank)  # declared diagnostic variants: L1-L10 plus their own L11 checks
    iss += lint_request_plan(bank)  # L13: request plans (N1 since v1.0-rc6)
    by_decision: dict[str, list[QSpec]] = defaultdict(list)
    for q in bank.questions():  # one phrasing per decision: variants are not questions, so this skips them
        by_decision[effective_decision(bank, q)].append(q)
    for dec, qs in by_decision.items():
        phrasings = {json.dumps([q.kind, q.instructions, q.true, q.false, q.options, q.levels], sort_keys=True) for q in qs}
        if not dec:
            iss += [Issue("L11", q.qid, "no decision id") for q in qs]
        elif len(phrasings) > 1:
            iss.append(Issue("L11", ",".join(q.qid for q in qs), f"decision {dec!r} has {len(phrasings)} phrasings"))
    if md_path is not None and md_sha256(md_path) != bank.sha256:
        iss.append(Issue("L12", "*", f"bank hash {bank.sha256[:12]} != hash printed in {md_path}"))
    if FROZEN_SHA256 is not None and bank.sha256 != FROZEN_SHA256:
        iss.append(Issue("L12", "*", "bank differs from the frozen hash"))
    return iss


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == "error"]


def check_request(program: str, state, questions) -> list[str]:
    """L9 at run time: raise if state + longest question exceeds the hard budget; return soft warnings."""
    from jevtrader.s1.backends import estimate_tokens
    from jevtrader.s1.types import canonical_json, question_to_json

    st = estimate_tokens(canonical_json(state))
    longest = max(estimate_tokens(canonical_json(question_to_json(q))) for q in questions.values())
    if st + longest > HARD_BUDGET:
        raise LintError(f"L9: {program} request state+longest question ~{st + longest} tokens > {HARD_BUDGET}")
    soft = STATE_BUDGET.get(program)
    return [f"L9: {program} state ~{st} tokens > budget {soft}"] if soft is not None and st > soft else []
