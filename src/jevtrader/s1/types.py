"""Backend-neutral System-One question/answer types.

They mirror the Jev wire contract (docs/jev/sdk_typesafe_0.7.1/typesafe_sdk/_schemas/models.py): a
request is a `state` plus named typed questions; every answer carries a probability distribution.
Downstream code only relies on `probs` (documented as calibrated by the vendor) and never on the
vendor-specific `confidence` statistic, whose formula is undocumented; see `entropy_confidence`.

An answer whose distribution could not be recovered (emulator parse failure after retries,
design_v1 §11.2) has `probs=None`: Layer B treats it as an abstention, never as a uniform guess.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Union

JSONContent = Union[str, dict, list]


@dataclass(frozen=True)
class Noul:
    """Yes/no question. `true`/`false` optionally describe what counts as each outcome."""

    instructions: JSONContent
    true: JSONContent | None = None
    false: JSONContent | None = None
    kind: Literal["noul"] = "noul"


@dataclass(frozen=True)
class Choice:
    """Pick one label. `options` maps label -> description (None = interpret by label name)."""

    instructions: JSONContent
    options: dict[str, JSONContent | None]
    kind: Literal["choice"] = "choice"


@dataclass(frozen=True)
class Score:
    """Rate on ordered levels; level i is described by `levels[i]` (scores start at 0)."""

    instructions: JSONContent
    levels: tuple[JSONContent, ...]
    kind: Literal["score"] = "score"


Question = Union[Noul, Choice, Score]


def labels(q: Question) -> list[str]:
    if q.kind == "noul":
        return ["yes", "no"]
    if q.kind == "choice":
        return list(q.options)
    return [str(i) for i in range(len(q.levels))]


@dataclass
class Answer:
    kind: str
    probs: dict[str, float] | None  # noul: {"yes","no"}; choice: label -> p; score: "0".."k" -> p; None = abstain
    vendor_confidence: float | None = None

    @property
    def degraded(self) -> bool:
        return self.probs is None

    @property
    def p_yes(self) -> float | None:
        assert self.kind == "noul"
        return None if self.probs is None else self.probs["yes"]

    @property
    def top(self) -> str | None:
        return None if self.probs is None else max(self.probs, key=self.probs.get)

    @property
    def expected_score(self) -> float | None:
        assert self.kind == "score"
        return None if self.probs is None else sum(int(k) * p for k, p in self.probs.items())

    @property
    def entropy_confidence(self) -> float | None:
        """1 - normalized Shannon entropy of the distribution (1 = certain, 0 = uniform)."""
        if self.probs is None:
            return None
        k = len(self.probs)
        if k < 2:
            return 1.0
        h = -sum(p * math.log(p) for p in self.probs.values() if p > 0)
        return 1.0 - h / math.log(k)


@dataclass
class S1Response:
    """One request's answers plus the provenance that `requests.parquet` needs (design_v1 §12.3).

    `model` is the pinned model we asked for; `response_model` is what the provider says answered
    (asserted equal by the backends, recorded for the audit log).
    """

    answers: dict[str, Answer]
    model: str
    backend: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0  # provider prompt-cache hits (priced lower), not our response cache
    latency_s: float = 0.0
    cost_usd: float = 0.0
    cached: bool = False  # served from our response cache (no call made in this run)
    response_model: str = ""
    prompt_version: str = ""
    request_key: str = ""
    degraded: list[str] = field(default_factory=list)  # question names answered with probs=None
    attempts: int = 1
    usage_estimated: bool = False  # provider sent no usage; tokens and cost are estimates (F07)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        d = asdict(self)
        d["answers"] = {k: asdict(v) for k, v in self.answers.items()}
        return d

    @staticmethod
    def from_json(d: dict) -> S1Response:
        names = {f.name for f in dataclasses.fields(S1Response)}
        d = {k: v for k, v in d.items() if k in names}
        d["answers"] = {k: Answer(**v) for k, v in d["answers"].items()}
        return S1Response(**d)


def normalize(probs: dict[str, float]) -> dict[str, float] | None:
    """Clip negatives and renormalize. No positive (or a non-finite) mass returns None, an abstention: a uniform
    distribution would be a confident-looking guess the model never made (§11.2)."""
    raw = [float(p) for p in probs.values()]
    if not raw or not all(math.isfinite(v) for v in raw):
        return None
    vals = [max(0.0, v) for v in raw]
    s = sum(vals)
    if s <= 0:
        return None
    return {k: v / s for k, v in zip(probs, vals)}


def question_to_json(q: Question) -> dict:
    return asdict(q)


def canonical_json(x: Any) -> str:
    """Order-preserving JSON: the emulator renders dicts in insertion order, so key order is content."""
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def state_sha256(state: JSONContent) -> str:
    return hashlib.sha256(canonical_json(state).encode()).hexdigest()


def request_key(
    backend_id: str,
    model: str,
    prompt_version: str,
    emu_cfg_hash: str,
    bank_hash: str,
    state: JSONContent,
    questions: Mapping[str, Question],
) -> str:
    """Content address of one request (design_v1 §11.2, validity M13).

    Covers everything that determines the prompt: the backend and its prompt version, the emulator
    configuration hash (prompt texts, decoding), the pinned model, the bank hash, the state with its key
    order, and the questions *in render order*. The emulator answers all questions in one completion,
    so a reordering is a different request and must not hit the cache.
    """
    payload = canonical_json(
        [
            backend_id,
            model,
            prompt_version,
            emu_cfg_hash,
            bank_hash,
            state,
            [[name, question_to_json(q)] for name, q in questions.items()],
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()
