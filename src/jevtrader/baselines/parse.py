"""gen-parse-v1: the one fixed parser for every generative output (design_v2_delta RP-1; bank QB-5).

It serves the TA-style pipeline (analyst stance lines; trader and risk JSON; the final decision), Mono-CoT's final
JSON and the S2 deliberator's final JSON. The rules are frozen in the bank (`prompts.<id>.parse_rule`, copied verbatim
into `BANK_RULES` below; `tests/test_generative_parser.py` checks that they still match the bank and that the code
does what they say):

- **Extraction.** The last well-formed JSON object of the output, found by balanced-brace scanning from the end that
  is aware of JSON strings: every "{" is a candidate start, taken from the right, and the strict JSON decoder reads
  the balanced object from it (braces inside string literals, with backslash escapes honoured, do not count, so a
  `reason` quoting "item {3" or ":-}" cannot hide the object). The object that ends last is returned; among objects
  ending at the same "}", the outermost. Strict JSON: NaN and Infinity are refused. No repair, no regex rescue.
- **Validation** is strict: numbers must be JSON numbers (not strings, not booleans, within float range), labels must
  match exactly, extra keys are ignored. An S2 claim is verifiable only if it cites at least one id and every cited id
  is an evidence id. An invalid output abstains (or takes the rule's default) and is counted by the caller; the parser
  is identical for clean and attacked runs.

`PARSER_SHA256` hashes this file, so any change to the rules changes the parser id that runs record.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

PARSER_VERSION = "gen-parse-v1"
PARSER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

# Verbatim copies of the bank's rules (question_bank_v1.json v1.0-rc3); tests compare them with the bank.
BANK_RULES = {
    "mono-cot-v1": "Take the last well-formed JSON object of the output, found by balanced-brace scanning from the "
                   "end; no repair and no regex rescue. Valid if it has a numeric `p_outperform` in [0, 1]; extra keys "
                   "are ignored. Otherwise invalid: abstain, counted.",
    "s2-v1": "Take the last well-formed JSON object of the output, found by balanced-brace scanning from the end; no "
             "repair and no regex rescue. Valid if it has `claims` (a list of at most 6 objects, each with a string "
             "`text` and a list-of-string `evidence_ids`) and `stance` in {keep, reduce, exclude}; extra keys are "
             "ignored; a claim that cites an id not among the evidence ids is unverifiable. Otherwise invalid: stance "
             "keep, no claims, counted.",
    "ta-lite-v1": {
        "json_extraction": "trader and risk outputs: the last well-formed JSON object of the output, found by "
                           "balanced-brace scanning from the end; no repair and no regex rescue",
        "analyst": "the last line matching ^Stance: (bullish|neutral|bearish)$ after trimming, case-insensitive; "
                   "otherwise neutral, counted",
        "trader": "valid if the object has `decision` in {BUY, HOLD, SELL}, a numeric `p_outperform` in [0, 1] and a "
                  "list `cited`; `reason` and other extra keys are ignored; otherwise invalid",
        "risk": "as trader, plus `action` in {approve, modify}",
        "final": "the risk output if valid, else the trader output; both invalid: HOLD with an abstaining p, counted",
    },
}

DECISIONS = ("BUY", "HOLD", "SELL")
ACTIONS = ("approve", "modify")
STANCES = ("bullish", "neutral", "bearish")
S2_STANCES = ("keep", "reduce", "exclude")
S2_MAX_CLAIMS = 6
_STANCE_LINE = re.compile(r"^Stance: (bullish|neutral|bearish)$", re.IGNORECASE)


def _no_constants(name: str):
    raise ValueError(f"non-JSON constant {name}")


_DECODER = json.JSONDecoder(parse_constant=_no_constants)


def last_json_object(text: str | None) -> dict | None:
    """The last well-formed JSON object in `text` (string-aware balanced-brace scanning from the end), else None."""
    if not isinstance(text, str):
        return None
    best_end, best = -1, None
    for s in reversed([i for i, ch in enumerate(text) if ch == "{"]):
        try:
            obj, end = _DECODER.raw_decode(text, s)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict) and end >= best_end:  # later end wins; same end: the outer (earlier) start
            best_end, best = end, obj
    return best


def _is_prob(x) -> bool:
    """A JSON number (not a bool, not a string) in [0, 1]. A JSON integer beyond float range is invalid (counted),
    never an OverflowError that would abort a whole batch."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return False
    try:
        v = float(x)
    except OverflowError:
        return False
    return math.isfinite(v) and 0.0 <= v <= 1.0


@dataclass(frozen=True)
class Parsed:
    valid: bool
    value: object  # the validated object / label, or the rule's default
    reason: str = "ok"  # "ok" | "no_json" | "schema:<field>" | "no_stance_line"


def parse_mono_cot(text: str | None) -> Parsed:
    """Mono-CoT: {"p_outperform": p}; invalid -> abstain (value None)."""
    obj = last_json_object(text)
    if obj is None:
        return Parsed(False, None, "no_json")
    if not _is_prob(obj.get("p_outperform")):
        return Parsed(False, None, "schema:p_outperform")
    return Parsed(True, {"p_outperform": float(obj["p_outperform"])})


def parse_analyst(text: str | None) -> Parsed:
    """TA analyst stance: the last line 'Stance: bullish|neutral|bearish' (trimmed, case-insensitive); else neutral."""
    stance = None
    for line in (text or "").splitlines():
        m = _STANCE_LINE.match(line.strip())
        if m:
            stance = m.group(1).lower()
    return Parsed(True, stance) if stance else Parsed(False, "neutral", "no_stance_line")


def _check_trader(obj: dict) -> str | None:
    if obj.get("decision") not in DECISIONS:
        return "schema:decision"
    if not _is_prob(obj.get("p_outperform")):
        return "schema:p_outperform"
    if not isinstance(obj.get("cited"), list):
        return "schema:cited"
    return None


def parse_trader(text: str | None) -> Parsed:
    obj = last_json_object(text)
    if obj is None:
        return Parsed(False, None, "no_json")
    err = _check_trader(obj)
    if err:
        return Parsed(False, None, err)
    return Parsed(True, {"decision": obj["decision"], "p_outperform": float(obj["p_outperform"]),
                         "cited": list(obj["cited"])})


def parse_risk(text: str | None) -> Parsed:
    obj = last_json_object(text)
    if obj is None:
        return Parsed(False, None, "no_json")
    err = _check_trader(obj) or (None if obj.get("action") in ACTIONS else "schema:action")
    if err:
        return Parsed(False, None, err)
    return Parsed(True, {"decision": obj["decision"], "p_outperform": float(obj["p_outperform"]),
                         "cited": list(obj["cited"]), "action": obj["action"]})


@dataclass(frozen=True)
class TaFinal:
    decision: str  # BUY | HOLD | SELL
    p_outperform: float | None  # None = abstain
    source: str  # "risk" | "trader" | "default"
    trader: Parsed
    risk: Parsed

    @property
    def invalid(self) -> bool:
        return self.source == "default"


def ta_final(trader_text: str | None, risk_text: str | None) -> TaFinal:
    """The risk output if valid, else the trader output; both invalid: HOLD with an abstaining p (counted)."""
    tr, rk = parse_trader(trader_text), parse_risk(risk_text)
    if rk.valid:
        return TaFinal(rk.value["decision"], rk.value["p_outperform"], "risk", tr, rk)
    if tr.valid:
        return TaFinal(tr.value["decision"], tr.value["p_outperform"], "trader", tr, rk)
    return TaFinal("HOLD", None, "default", tr, rk)


@dataclass(frozen=True)
class S2Result:
    valid: bool
    stance: str
    claims: list = field(default_factory=list)  # [{"text", "evidence_ids", "verifiable"}]
    reason: str = "ok"


def parse_s2(text: str | None, evidence_ids: set[str] | frozenset[str]) -> S2Result:
    """S2 deliberator: invalid -> stance keep, no claims (counted). A claim citing an id outside `evidence_ids` is
    kept but marked unverifiable; so is a claim that cites no id at all (the s2-v1 prompt requires each claim to be
    "stated by the evidence items it cites", so a claim without a citation cannot be verified)."""
    obj = last_json_object(text)
    if obj is None:
        return S2Result(False, "keep", [], "no_json")
    claims, stance = obj.get("claims"), obj.get("stance")
    if stance not in S2_STANCES:
        return S2Result(False, "keep", [], "schema:stance")
    if not isinstance(claims, list) or len(claims) > S2_MAX_CLAIMS:
        return S2Result(False, "keep", [], "schema:claims")
    out = []
    for c in claims:
        if not (isinstance(c, dict) and isinstance(c.get("text"), str) and isinstance(c.get("evidence_ids"), list)
                and all(isinstance(x, str) for x in c["evidence_ids"])):
            return S2Result(False, "keep", [], "schema:claim")
        out.append({"text": c["text"], "evidence_ids": list(c["evidence_ids"]),
                    "verifiable": bool(c["evidence_ids"]) and all(x in evidence_ids for x in c["evidence_ids"])})
    return S2Result(True, stance, out)
