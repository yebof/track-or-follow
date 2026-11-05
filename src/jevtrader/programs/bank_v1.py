"""Question bank v1, loaded from docs/design/question_bank_v1.json (the single source of truth).

The canonical SHA-256 is the one printed in question_bank_v1.md (sorted-key JSON, ensure_ascii=False); it is
the `bank_hash` in every request key and provenance record.

Freeze (design_v1 change control; docs/design/decisions.md 2026-09-28; docs/impl/freeze_report.md). v1.0 is v1.0-rc7
with only its `version` label changed (the content DEV #2 validated; tests/test_bank_freeze.py), and `FROZEN_SHA256`
is its canonical hash. After the freeze:
- a run whose ledger run tag is one of FROZEN_RUN_TAGS ("MAIN", "PRE", also as "MAIN/<program>") refuses to load a
  bank with any other hash (`load_bank(run_tag=...)`, `parse_bank(..., run_tag=...)`, `require_frozen`; the Layer-A
  runner passes its spec's run tag), and refuses to run at all while the bank is not frozen;
- lint L12 reports any bank whose hash differs from FROZEN_SHA256, so the test suite fails on an unlogged edit of the
  bank file;
- loading without a run tag (tests, diagnostics, the rc reconstructions of tests/test_bank_patches_rc*.py) is allowed.
Any later change bumps the version, sets FROZEN_SHA256 to the new hash and is logged in
docs/design/prereg_deviations.md.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from jevtrader.programs.base import QSpec

ROOT = Path(__file__).resolve().parents[3]
BANK_PATH = ROOT / "docs" / "design" / "question_bank_v1.json"
BANK_MD_PATH = ROOT / "docs" / "design" / "question_bank_v1.md"
# v1.0 = v1.0-rc7 (c44c1ec0...) with `version` relabelled; set at the freeze on 2026-09-28 (docs/impl/freeze_report.md)
FROZEN_SHA256: str | None = "d4f4e44cb7e03d4f1a6d1cb6e3e86cab2627eb302dd81ceee0e27bbe81012ccb"
# ledger run tags (the part before the first "/") whose runs must load exactly the frozen bank
FROZEN_RUN_TAGS: frozenset[str] = frozenset({"MAIN", "PRE", "TAL", "MCOT", "VERB", "PROBE", "FID", "FIDX", "MEM", "H", "HS", "HSPOT", "HGUARD", "NODEF", "RT-S", "RT-A", "FAITH", "LIVE", "CF", "S2", "XWEB", "LAP", "VERB-C1I", "MCOT-RT", "NG"})  # P-7 (2026-09-28): + post-freeze run tags

MARKET_PROGRAMS = ("P1", "P2", "M")  # their note field is conventions.note_text_market


class BankHashMismatch(RuntimeError):
    pass


def canonical_sha256(bank_json: dict) -> str:
    """SHA-256 of the sorted-key JSON (scripts/render_question_bank.py prints the same value)."""
    return hashlib.sha256(json.dumps(bank_json, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def md_sha256(md_path: Path = BANK_MD_PATH) -> str | None:
    m = re.search(r"Canonical SHA-256.*?`([0-9a-f]{64})`", Path(md_path).read_text())
    return m.group(1) if m else None


@dataclass(frozen=True)
class Bank:
    version: str
    sha256: str
    conventions: dict[str, Any]
    ff12: dict[str, str]  # FF12 code -> industry text
    state_schemas: dict[str, dict[str, str]]
    programs: dict[str, dict[str, QSpec]]  # program -> template name -> QSpec, in bank order
    groups: dict[str, dict[str, list[str]]]
    code_constants: dict[str, Any]
    prompts: dict[str, dict[str, Any]]
    # v1.0-rc4 (QB-6): diagnostic rewrites of existing questions, kind -> {"rule", "variants": {id: entry}}. They are
    # not programs: `questions()` never returns them, so no decision, learning or production request can use them.
    diagnostic_variants: dict[str, Any] = field(default_factory=dict)

    VARIANT_FIELDS = ("instructions", "true", "false")  # the only fields a variant may replace

    def program(self, name: str) -> dict[str, QSpec]:
        return self.programs[name]

    def q(self, qid: str) -> QSpec:
        prog, name = qid.split(".", 1)
        return self.programs[prog][name]

    def questions(self) -> list[QSpec]:
        return [q for qs in self.programs.values() for q in qs.values()]

    def note(self, program: str) -> str:
        """The note a state schema names (`conventions.<key>`; e.g. H_spot's spotlight note); otherwise the
        market note for P1/P2/M and the default note for every other program."""
        ref = self.state_schemas.get(program, {}).get("note", "")
        if ref.startswith("conventions.") and ref.removeprefix("conventions.") in self.conventions:
            return self.conventions[ref.removeprefix("conventions.")]
        key = "note_text_market" if program in MARKET_PROGRAMS else "note_text"
        return self.conventions[key]

    def state_fields(self, program: str) -> list[str]:
        """State keys in schema order, excluding instruction fields."""
        return [k for k in self.state_schemas[program] if not k.startswith("instruction field ")]

    def instruction_fields(self, program: str) -> list[str]:
        return [k.removeprefix("instruction field ") for k in self.state_schemas[program] if k.startswith("instruction field ")]

    def option_text(self, qid: str, label: str) -> str:
        """Description of a Choice option ("option text" in the bank schemas); the label if it has none."""
        desc = self.q(qid).options[label]
        return label if desc is None else (desc if isinstance(desc, str) else json.dumps(desc, ensure_ascii=False))

    def variant_ids(self, kind: str = "paraphrase") -> list[str]:
        """Ids ("<program>.<question>~<tag>") of the diagnostic variants of `kind`, in bank order."""
        return list(self.diagnostic_variants.get(kind, {}).get("variants", {}))

    def variant(self, vid: str, kind: str = "paraphrase") -> QSpec:
        """A diagnostic variant as a QSpec: the question named by its `of` with only `instructions` and (Noul)
        `true`/`false` replaced; kind, options, levels, decision, taint, channel and role are the original's. Its
        name is "<question>~<tag>", so it never collides with the original in a request. The linter checks the
        entries (L11); this only refuses an entry that is not declared diagnostic."""
        e = self.diagnostic_variants[kind]["variants"][vid]
        if e.get("diagnostic") is not True:
            raise ValueError(f"{vid}: a variant must be declared diagnostic")
        base = self.q(e["of"])
        tag = vid.split("~", 1)[1] if "~" in vid else vid
        new = {k: copy.deepcopy(e[k]) for k in self.VARIANT_FIELDS if k in e}
        return dataclasses.replace(base, name=f"{base.name}~{tag}", **new)

    def variants(self, kind: str = "paraphrase") -> dict[str, QSpec]:
        return {vid: self.variant(vid, kind) for vid in self.variant_ids(kind)}


def _qspec(program: str, name: str, d: dict) -> QSpec:
    return QSpec(
        program=program,
        name=name,
        kind=d["kind"],
        instructions=d["instructions"],
        true=d.get("true"),
        false=d.get("false"),
        options=d.get("options"),
        levels=tuple(d["levels"]) if d.get("levels") is not None else None,
        judgment=d.get("judgment", ""),
        taint=d.get("taint", ""),
        decision=d.get("decision", ""),
        channel=d.get("channel", ""),
        role=d.get("role", ""),
    )


def frozen_required(run_tag: str | None) -> bool:
    """Whether a run with this ledger run tag ("MAIN", "MAIN/T", "PRE/N01") must use exactly the frozen bank."""
    return bool(run_tag) and str(run_tag).split("/", 1)[0].strip() in FROZEN_RUN_TAGS


def require_frozen(bank_sha256: str, run_tag: str | None) -> None:
    """Refuse a bank whose canonical hash is not FROZEN_SHA256 for a run tag of FROZEN_RUN_TAGS (and any bank while
    FROZEN_SHA256 is unset). Other run tags and loads without a run tag pass."""
    if not frozen_required(run_tag):
        return
    if FROZEN_SHA256 is None:
        raise BankHashMismatch(f"run tag {run_tag!r} needs the frozen bank, but FROZEN_SHA256 is not set")
    if bank_sha256 != FROZEN_SHA256:
        raise BankHashMismatch(f"run tag {run_tag!r}: bank hash {bank_sha256} != frozen {FROZEN_SHA256}; a {run_tag} "
                               f"run uses only the frozen bank (bump the version, set FROZEN_SHA256, log the change)")


def parse_bank(raw: dict, *, run_tag: str | None = None) -> Bank:
    sha = canonical_sha256(raw)
    require_frozen(sha, run_tag)
    progs = {p: {n: _qspec(p, n, d) for n, d in qs.items()} for p, qs in raw["programs"].items()}
    return Bank(
        version=raw["version"],
        sha256=sha,
        conventions=raw["conventions"],
        ff12=raw["ff12_industries"],
        state_schemas=raw["state_schemas"],
        programs=progs,
        groups=raw.get("groups", {}),
        code_constants=raw.get("code_constants", {}),
        prompts=raw.get("prompts", {}),
        diagnostic_variants=raw.get("diagnostic_variants", {}),
    )


@lru_cache(maxsize=4)
def _load(path: str) -> Bank:
    return parse_bank(json.loads(Path(path).read_text()))


def load_bank(path: Path | str = BANK_PATH, *, run_tag: str | None = None) -> Bank:
    """The bank at `path` (default: the repository bank). With a run tag of FROZEN_RUN_TAGS its hash must be
    FROZEN_SHA256 (`require_frozen`), checked on every call, not only on the first (cached) load."""
    b = _load(str(Path(path).resolve()))
    require_frozen(b.sha256, run_tag)
    return b
