"""System-One backends behind one async interface: `await backend.ask(state, questions, tag)`.

- `LogprobEmulator` (`llm-logprob`, prompt `emu-v1`): a Jev-API-compatible stand-in built from an LLM's
  next-token log-probabilities over option letters (one call answers all questions of a request).
- `VerbalizedLLM` (`llm-verbalized`, prompt `verb-v1`): the LLM *states* a probability per option (baseline).
- `JevBackend` (`jev`): TypeSafe Jev via the official SDK, pinned model id.
- `MockBackend` (`mock`): deterministic, offline, free; for dry runs and tests only.

Programs send identical `(state, questions)` to every backend; everything emulator-specific (prompt
layout, letters, log-prob parsing, retries of degraded parses) lives inside the emulator classes, so the
switch to Jev is a config change (`make_backend`, configs/backends.yaml; design_v1 §11.3).

Shared mechanics in `S1Backend.ask` (design_v1 §11.2): content-addressed cache (degraded answers never
cached), spend ledger with running total and per-run sub-caps, concurrency and request-rate limits, the
pinned-model assertion on every response, and degraded -> retry twice -> `probs=None` (abstention).

Provider prompt caching (the OpenAI emulators only; `s1/prompt_cache.py`): a routing-only, stable
`prompt_cache_key` per static-prefix family (partitioning by request rate is available but off by default), with a
warm-up gate. It never enters the request key or `cfg_hash`, so it cannot change an answer or a cache hit
(docs/impl/dev2_ops.md §1).
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import math
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from jevtrader.s1.budget import SpendLedger
from jevtrader.s1.cache import ResponseCache
from jevtrader.s1.pricing import cost_usd
from jevtrader.s1.prompt_cache import PromptCacheConfig, PromptCacheRouter, Route
from jevtrader.s1.types import (
    Answer,
    JSONContent,
    Question,
    S1Response,
    canonical_json,
    labels,
    normalize,
    question_to_json,
    request_key,
)

log = logging.getLogger("jevtrader.s1")

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "backends.yaml"
EMULATOR_MODEL = "gpt-4.1-mini-2025-04-14"  # pinned snapshot (validity M3, F16)
_LETTERS = "ABCDEFGHIJKLMNOPQRST"  # OpenAI returns at most 20 top log-probs per position
MIN_VALID_MASS = 0.5  # emulator: below this share of letter probability on valid options the answer is degraded
# the provider prompt-cache route of the request being sent (set by S1Backend._ask_uncached around each provider call,
# read by the OpenAI emulators' _ask); a context variable keeps concurrent requests apart without changing `_ask`
_ROUTE: contextvars.ContextVar[Route | None] = contextvars.ContextVar("s1_prompt_cache_route", default=None)
_STATE_SENTINEL = "\x00STATE-SENTINEL\x00"


class ModelPinError(RuntimeError):
    """The provider answered with a different model than the pinned one."""


def estimate_tokens(text: str) -> int:
    """Conservative o200k proxy (~3.5 bytes per token on JSON-heavy English). Used when a provider reports
    no usage (F07: an estimate, never $0) and for pre-call budget checks."""
    return math.ceil(len(text.encode("utf-8")) / 3.5)


def config_hash(cfg: Any) -> str:
    return hashlib.sha256(canonical_json(cfg).encode()).hexdigest()[:16]


def _bank_prompt(version: str) -> dict:
    from jevtrader.programs.bank_v1 import load_bank  # lazy: s1 must not depend on programs at import time

    return load_bank().prompts[version]


def _bank_sha256() -> str:
    from jevtrader.programs.bank_v1 import load_bank

    return load_bank().sha256


class S1Backend(ABC):
    backend_id: str = "abstract"
    prompt_version: str = ""

    def __init__(
        self,
        model: str,
        cache: ResponseCache | None = None,
        ledger: SpendLedger | None = None,
        max_concurrency: int = 8,
        rpm: float | None = None,
        bank_hash: str = "",
        degraded_retries: int = 2,
        assert_response_model: bool = True,
    ):
        self.model = model
        self.cache = cache
        self.ledger = ledger
        self.max_concurrency = max_concurrency
        self.bank_hash = bank_hash
        self.degraded_retries = degraded_retries
        self.assert_response_model = assert_response_model
        self.identity: dict[str, Any] | None = None  # provider-side identity, logged once (§11.3)
        self._min_gap = 60.0 / rpm if rpm else 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self.coalesced = 0  # duplicate in-flight requests served by the leader's call (single-flight)
        self._router: PromptCacheRouter | None = None  # provider prompt-cache routing (OpenAI emulators only)

    # ------------------------------------------------------------------ identity of the configuration
    @property
    def cfg_hash(self) -> str:
        """Hash of everything backend-specific that shapes the prompt (the emulator prompt texts, decoding)."""
        return ""

    def key(self, state: JSONContent, questions: Mapping[str, Question], bank_hash: str | None = None) -> str:
        return request_key(
            self.backend_id,
            self.model,
            self.prompt_version,
            self.cfg_hash,
            self.bank_hash if bank_hash is None else bank_hash,
            state,
            questions,
        )

    def describe(self) -> dict[str, Any]:
        """Provenance block for manifests and requests.parquet."""
        return {
            "backend": self.backend_id,
            "model_pinned": self.model,
            "prompt_version": self.prompt_version,
            "cfg_hash": self.cfg_hash,
            "bank_hash": self.bank_hash,
            "identity": self.identity,
        }

    # ------------------------------------------------------------------ loop-bound primitives
    def _prims(self) -> None:
        """asyncio primitives (and HTTP connection pools) bind to the running loop; rebuild them when the backend
        is reused on a new loop, e.g. by two consecutive `generate()` calls."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            if self._loop is not None:
                self._on_new_loop()
            self._loop = loop
            self._sem = asyncio.Semaphore(self.max_concurrency)
            self._slot_lock = asyncio.Lock()
            self._id_lock = asyncio.Lock()
            self._next_slot = 0.0
            self._inflight: dict[str, asyncio.Future] = {}

    def _on_new_loop(self) -> None:
        """Rebuild loop-bound clients this backend created itself (injected clients are the caller's)."""

    async def _rate_limit(self) -> None:
        if not self._min_gap:
            return
        async with self._slot_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_gap
        if wait > 0:
            await asyncio.sleep(wait)

    # ------------------------------------------------------------------ money
    def _cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        return cost_usd(self.model, input_tokens, output_tokens, cached_tokens)

    def estimate_cost(self, state: JSONContent, questions: Mapping[str, Question]) -> float:
        text = canonical_json(state) + canonical_json([question_to_json(q) for q in questions.values()])
        return self._cost(estimate_tokens(text) + 400, 8 * len(questions) + 16)

    # ------------------------------------------------------------------ the one entry point
    async def ask(
        self, state: JSONContent, questions: Mapping[str, Question], tag: str = "", *, bank_hash: str | None = None
    ) -> S1Response:
        """Cache, then single-flight: identical requests in flight at the same time (the runner starts every
        week at once, and F-A/F-B states repeat across weeks) share one provider call instead of each paying
        for a cache miss. A duplicate gets a copy marked `cached` (no call was made for it)."""
        if not questions:
            raise ValueError("at least one question is required")
        self._prims()
        key = self.key(state, questions, bank_hash)
        if self.cache is not None:
            hit = await self.cache.aget(key)
            if hit is not None:
                return hit
        leader = self._inflight.get(key)
        if leader is not None:
            self.coalesced += 1
            return _duplicate(await asyncio.shield(leader), key)
        fut = self._loop.create_future()
        fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # never "exception was never retrieved"
        self._inflight[key] = fut
        try:
            resp = await self._ask_uncached(key, state, questions, tag)
        except asyncio.CancelledError:
            fut.cancel()
            raise
        except BaseException as e:
            fut.set_exception(e)
            raise
        else:
            fut.set_result(resp)
            return resp
        finally:
            self._inflight.pop(key, None)

    def _route(self, key: str, state: JSONContent, questions: Mapping[str, Question]) -> Route | None:
        """Provider prompt-cache route of a request (routing only; None = the provider gets no cache key)."""
        return None

    async def _ask_uncached(self, key: str, state: JSONContent, questions: Mapping[str, Question], tag: str) -> S1Response:
        if self.ledger is not None:
            self.ledger.check(self.estimate_cost(state, questions), tag)
        route = self._route(key, state, questions)
        # warm-up gate: wait (outside the semaphore, so the wait holds no slot and is not latency) until the first
        # request of this cache key has returned; the leader releases its key when its call is over
        leader = await self._router.enter(route) if route is not None and self._router is not None else False
        token = _ROUTE.set(route)
        try:
            return await self._ask_attempts(key, state, questions, tag)
        finally:
            _ROUTE.reset(token)
            if leader:
                self._router.release(route)

    async def _ask_attempts(self, key: str, state: JSONContent, questions: Mapping[str, Question], tag: str) -> S1Response:
        best: S1Response | None = None
        tot = {"in": 0, "out": 0, "cached": 0, "usd": 0.0, "lat": 0.0, "est": False}
        attempts = 0
        for attempts in range(1, self.degraded_retries + 2):
            async with self._sem:
                if attempts == 1 and self.cache is not None:  # filled while we queued (e.g. by another process)
                    hit = await self.cache.aget(key)
                    if hit is not None:
                        return hit
                await self._rate_limit()
                t0 = time.perf_counter()
                resp = await self._ask(state, questions)
                tot["lat"] += time.perf_counter() - t0
            usd = self._cost(resp.input_tokens, resp.output_tokens, resp.cached_tokens)
            tot["in"] += resp.input_tokens
            tot["out"] += resp.output_tokens
            tot["cached"] += resp.cached_tokens
            tot["usd"] += usd
            tot["est"] |= resp.usage_estimated
            if self.ledger is not None:  # book the call before any check can raise: it has been paid for
                await self.ledger.acharge(self.backend_id, self.model, tag, resp.input_tokens, resp.output_tokens, usd)
            self._check_pin(resp)
            missing = set(questions) - set(resp.answers)
            if missing:
                raise RuntimeError(f"{self.backend_id} returned no answer for {sorted(missing)}")
            if best is None or len(resp.degraded) < len(best.degraded):
                best = resp
            if not best.degraded:
                break
        assert best is not None
        for name in best.degraded:  # still degraded after the retries: abstain, never a uniform guess
            best.answers[name] = Answer(questions[name].kind, None)
        best.answers = {name: best.answers[name] for name in questions}  # render order, no extras
        best.model, best.backend, best.prompt_version = self.model, self.backend_id, self.prompt_version
        best.request_key, best.attempts, best.cached = key, attempts, False
        best.input_tokens, best.output_tokens, best.cached_tokens = tot["in"], tot["out"], tot["cached"]
        best.cost_usd, best.latency_s, best.usage_estimated = tot["usd"], tot["lat"], tot["est"]
        if best.degraded:
            log.warning("%s: %d degraded answer(s) after %d attempts (%s)", self.backend_id, len(best.degraded), attempts, tag)
        elif self.cache is not None:
            await self.cache.aput(key, best)
        return best

    def _check_pin(self, resp: S1Response) -> None:
        ident = self.identity if self.identity is not None else {}
        if "response_model" not in ident:
            ident["response_model"] = resp.response_model
            self.identity = ident
            log.info("%s: first response served by %s (pinned %s)", self.backend_id, resp.response_model, self.model)
        if self.assert_response_model and resp.response_model != self.model:
            raise ModelPinError(f"{self.backend_id}: asked {self.model!r}, served by {resp.response_model!r}")

    @abstractmethod
    async def _ask(self, state: JSONContent, questions: Mapping[str, Question]) -> S1Response:
        """One provider call. Must set `answers` (probs=None for degraded ones), `degraded`, token counts
        (estimated, with `usage_estimated`, if the provider sent none) and `response_model`."""


def _duplicate(resp: S1Response, key: str) -> S1Response:
    """An independent copy of the leader's response for a coalesced duplicate (no call was made for it)."""
    r = S1Response.from_json(resp.to_json())
    r.cached, r.request_key = True, key
    return r


# ------------------------------------------------------------------------------ shared LLM prompting
def _text(x: JSONContent | None) -> str:
    """A string as is; an object as single-line JSON (bank §E question_block_rule)."""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def option_strings(q: Question) -> list[str]:
    """The text emu-v1 renders after each option letter, in `labels(q)` order: noul 'yes -- <true>' and
    'no -- <false>', choice '<name> -- <definition>', score 'level k -- <level text>' (a missing description renders
    the name alone). The typed-local backend (prompt ng-v1) passes exactly these strings as its options."""
    out = []
    for lab in labels(q):
        if q.kind == "noul":
            desc = q.true if lab == "yes" else q.false
        elif q.kind == "choice":
            desc = q.options[lab]
        else:
            desc = q.levels[int(lab)]
        shown = f"level {lab}" if q.kind == "score" else lab
        desc_s = "" if desc is None else _text(desc)
        out.append(shown + (f" -- {desc_s}" if desc_s else ""))
    return out


def render_questions(questions: Mapping[str, Question]) -> tuple[str, dict[str, tuple[str, dict[str, str]]]]:
    """Questions block of emu-v1 / verb-v1. Returns the text and qid -> (question name, {letter: label})."""
    lines: list[str] = []
    maps: dict[str, tuple[str, dict[str, str]]] = {}
    for n, (name, q) in enumerate(questions.items(), 1):
        qid = f"q{n}"
        lines.append(f"[{qid}] {_text(q.instructions)}")
        opts = labels(q)
        if len(opts) > len(_LETTERS):
            raise ValueError(f"question {name!r} has {len(opts)} options; emulator supports <= {len(_LETTERS)}")
        m = {}
        for letter, lab, shown in zip(_LETTERS, opts, option_strings(q)):
            lines.append(f"  {letter}: {shown}")
            m[letter] = lab
        maps[qid] = (name, m)
        lines.append("")
    return "\n".join(lines), maps


def state_text(state: JSONContent) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)


def _openai_client():
    import os

    from openai import AsyncOpenAI

    return AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=120.0, max_retries=4)


def _usage(r, prompt_text: str, max_out: int) -> tuple[int, int, int, bool]:
    """(input, output, cached, estimated) from an OpenAI response; estimates when usage is missing (F07)."""
    u = getattr(r, "usage", None)
    if u is None or u.prompt_tokens is None:
        return estimate_tokens(prompt_text) + 7, max_out, 0, True
    details = getattr(u, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return int(u.prompt_tokens), int(u.completion_tokens or 0), cached, False


class _PromptedLLM(S1Backend):
    """An OpenAI chat model driven by a bank prompt (`emu-v1` / `verb-v1`) whose decoding pins the model.

    `prompt_cache` (True = defaults, False = off, or a mapping of `PromptCacheConfig` fields; configs/backends.yaml)
    sets the provider prompt-cache routing. It is deliberately left out of `cfg_hash`: it changes where a request is
    served, never what is asked, so request keys and cached answers are identical with it on or off."""

    def __init__(self, model: str = EMULATOR_MODEL, *, prompt_version: str, prompt: dict | None = None, client=None,
                 prompt_cache: Any = True, **kw):
        super().__init__(model, **kw)
        self.prompt_version = prompt_version
        self.prompt = prompt if prompt is not None else _bank_prompt(prompt_version)
        pinned = self.prompt["decoding"]["model"]
        if pinned != model:
            raise ValueError(f"{prompt_version} pins {pinned!r}; got model {model!r}")
        self._own_client = client is None
        self._client = client if client is not None else _openai_client()
        self._cfg_hash = config_hash({"prompt_version": prompt_version, "prompt": self.prompt, "letters": _LETTERS,
                                      "parser": f"{self.backend_id}-v2", "degraded_retries": self.degraded_retries,
                                      "min_valid_mass": getattr(self, "min_valid_mass", None)})
        self.prompt_cache = PromptCacheConfig.from_value(prompt_cache)
        if self.prompt_cache.enabled:
            self._router = PromptCacheRouter(self.prompt_cache, namespace=prompt_version, token_estimate=estimate_tokens)

    @property
    def cfg_hash(self) -> str:
        return self._cfg_hash

    def _on_new_loop(self) -> None:
        if self._own_client:
            self._client = _openai_client()

    def messages(self, state: JSONContent, questions: Mapping[str, Question]) -> tuple[list[dict], dict]:
        raise NotImplementedError

    def static_prefix(self, questions: Mapping[str, Question]) -> str:
        """The part of the prompt that precedes STATE (system prompt, then the user text up to the state): identical
        for every request with these questions, so it is what provider prompt caching can reuse (F06)."""
        msgs, _ = self.messages(_STATE_SENTINEL, questions)
        user = msgs[-1]["content"]
        return "".join(m["content"] + "\x1e" for m in msgs[:-1]) + user[: user.index(_STATE_SENTINEL)]

    def _route(self, key: str, state: JSONContent, questions: Mapping[str, Question]) -> Route | None:
        if self._router is None:
            return None
        return self._router.route(self.static_prefix(questions), key)

    def _cache_kwargs(self) -> dict[str, Any]:
        """Routing-only request fields of the request being sent (see `s1/prompt_cache.py`)."""
        route = _ROUTE.get()
        if route is None:
            return {}
        kw: dict[str, Any] = {"prompt_cache_key": route.key}
        if self.prompt_cache.retention:
            kw["prompt_cache_retention"] = self.prompt_cache.retention
        return kw

    @staticmethod
    def _tag_route(resp: S1Response, kw: Mapping[str, Any]) -> S1Response:
        if "prompt_cache_key" in kw:  # logged in requests.parquet.backend_meta
            resp.meta = {**resp.meta, "prompt_cache_key": kw["prompt_cache_key"]}
        return resp


# ------------------------------------------------------------------------------- logprob emulator
class LogprobEmulator(_PromptedLLM):
    """Probabilities = renormalized next-token probabilities of the option letters at each answer slot.

    emu-v1 layout (bank §E): system prompt, then QUESTIONS in bank order with the answer-format lines, then
    STATE, then "Answer now." Everything before STATE is identical for every request of a program, which
    lets provider prompt caching apply (F06). No per-request shuffle, no chunking (§11.2).
    """

    backend_id = "llm-logprob"

    def __init__(self, model: str = EMULATOR_MODEL, *, prompt_version: str = "emu-v1",
                 min_valid_mass: float = MIN_VALID_MASS, **kw):
        self.min_valid_mass = min_valid_mass
        super().__init__(model, prompt_version=prompt_version, **kw)
        if self.prompt["decoding"]["max_completion_tokens"] != "8 * n_questions + 16":
            raise ValueError("emu decoding changed in the bank; update LogprobEmulator")

    def messages(self, state, questions):
        qtext, maps = render_questions(questions)
        fmt = "\n".join(f"{qid}=<letter>" for qid in maps)
        user = self.prompt["user_template"].format(questions_block=qtext, format_lines=fmt, state_json=state_text(state))
        return [{"role": "system", "content": self.prompt["system"]}, {"role": "user", "content": user}], maps

    async def _ask(self, state, questions):
        msgs, maps = self.messages(state, questions)
        dec = self.prompt["decoding"]
        max_out = 8 * len(maps) + 16
        route_kw = self._cache_kwargs()
        r = await self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=dec["temperature"],
            max_completion_tokens=max_out,
            logprobs=True,
            top_logprobs=dec["top_logprobs"],
            **route_kw,
        )
        choice = r.choices[0]
        text = choice.message.content or ""
        toks = (choice.logprobs.content or []) if choice.logprobs else []
        offsets, pos = [], 0  # char offset of each token in `text`
        for t in toks:
            offsets.append(pos)
            pos += len(t.token)
        answers, degraded = {}, []
        for qid, (name, letter_map) in maps.items():
            m = re.search(rf"(?m)^\s*{qid}\s*=\s*([A-T])", text)
            probs = None
            # the chosen letter must be one of this question's options, and most of the letter probability must
            # sit on valid options; otherwise renormalizing the scraps would fake a confident answer
            if m and toks and m.group(1) in letter_map:
                ci = m.start(1)
                ti = max(i for i, o in enumerate(offsets) if o <= ci)
                acc = {lab: 0.0 for lab in letter_map.values()}
                for cand in toks[ti].top_logprobs or []:
                    letter = cand.token.replace("=", "").strip()
                    if letter in letter_map:
                        acc[letter_map[letter]] += math.exp(cand.logprob)
                if sum(acc.values()) >= self.min_valid_mass:
                    probs = normalize(acc)
            if probs is None:
                degraded.append(name)
            answers[name] = Answer(questions[name].kind, probs)
        n_in, n_out, n_cached, est = _usage(r, msgs[0]["content"] + msgs[1]["content"], max_out)
        return self._tag_route(S1Response(
            answers=answers, model=self.model, backend=self.backend_id, input_tokens=n_in, output_tokens=n_out,
            cached_tokens=n_cached, response_model=r.model, degraded=degraded, usage_estimated=est,
        ), route_kw)


# ------------------------------------------------------------------------ verbalized-probability LLM
class VerbalizedLLM(_PromptedLLM):
    """The LLM states a probability for each option in JSON (verbalized confidence; prompt verb-v1)."""

    backend_id = "llm-verbalized"

    def __init__(self, model: str = EMULATOR_MODEL, *, prompt_version: str = "verb-v1", **kw):
        super().__init__(model, prompt_version=prompt_version, **kw)
        if self.prompt["decoding"]["max_completion_tokens"] != "12 * n_options_total + 64":
            raise ValueError("verb decoding changed in the bank; update VerbalizedLLM")

    def messages(self, state, questions):
        qtext, maps = render_questions(questions)
        hint = json.dumps({qid: {letter: "<probability>" for letter in m} for qid, (_, m) in maps.items()})
        user = self.prompt["user_template"].format(questions_block=qtext, schema_hint=hint, state_json=state_text(state))
        return [{"role": "system", "content": self.prompt["system"]}, {"role": "user", "content": user}], maps

    async def _ask(self, state, questions):
        msgs, maps = self.messages(state, questions)
        max_out = 12 * sum(len(m) for _, m in maps.values()) + 64
        route_kw = self._cache_kwargs()
        r = await self._client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=self.prompt["decoding"]["temperature"],
            response_format={"type": self.prompt["decoding"]["response_format"]},
            max_completion_tokens=max_out,
            **route_kw,
        )
        try:
            obj = json.loads(r.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            obj = {}
        answers, degraded = {}, []
        for qid, (name, letter_map) in maps.items():
            raw = obj.get(qid, {}) if isinstance(obj, dict) else {}
            acc = {}
            for letter, lab in letter_map.items():
                try:
                    acc[lab] = float(raw.get(letter, 0.0)) if isinstance(raw, dict) else 0.0
                except (TypeError, ValueError):
                    acc[lab] = 0.0
            probs = normalize(acc) if all(math.isfinite(v) for v in acc.values()) else None
            if probs is None:
                degraded.append(name)
            answers[name] = Answer(questions[name].kind, probs)
        n_in, n_out, n_cached, est = _usage(r, msgs[0]["content"] + msgs[1]["content"], max_out)
        return self._tag_route(S1Response(
            answers=answers, model=self.model, backend=self.backend_id, input_tokens=n_in, output_tokens=n_out,
            cached_tokens=n_cached, response_model=r.model, degraded=degraded, usage_estimated=est,
        ), route_kw)


# --------------------------------------------------------------------------------------------- Jev
def jev_questions(questions: Mapping[str, Question]) -> dict:
    """Our question types -> typesafe_sdk question objects (the Jev wire contract)."""
    from typesafe_sdk import Choice, Noul, Score

    out = {}
    for name, q in questions.items():
        if q.kind == "noul":
            crit = {k: v for k, v in (("true", q.true), ("false", q.false)) if v is not None} or None
            out[name] = Noul(instructions=q.instructions, criteria=crit)
        elif q.kind == "choice":
            out[name] = Choice(instructions=q.instructions, criteria=dict(q.options))
        else:
            out[name] = Score(instructions=q.instructions, criteria=list(q.levels))
    return out


class JevBackend(S1Backend):
    """TypeSafe Jev. Retries follow design_v1 §11.3: RetryPolicy(max_retries=8, backoff_max=60); the total
    retry budget per call is widened from the SDK's 30 s default so that the 8 retries can actually happen.
    Identity: GET /v1/models is logged once, and every `response.model` is checked against the pin (the
    docs say the response reports the versioned ID that answered, docs/jev/pages/models)."""

    backend_id = "jev"
    prompt_version = "jev-api-v1"

    def __init__(
        self,
        model: str = "jev-1.13.0",
        *,
        api_key: str | None = None,
        retry: Any = None,
        request_timeout: float = 30.0,
        transport: Any = None,
        client: Any = None,
        **kw,
    ):
        super().__init__(model, **kw)
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        if isinstance(retry, Mapping):
            retry = RetryPolicy(**retry)
        self.retry_policy = retry if retry is not None else RetryPolicy(max_retries=8, backoff_max=60.0, timeout=600.0)
        self._own_client = client is None and transport is None

        def build():
            return AsyncTypeSafeClient(api_key=api_key, model=model, retry=self.retry_policy, timeout=request_timeout,
                                       transport=transport)

        self._build_client = build
        self._client = client if client is not None else build()

    def _on_new_loop(self) -> None:
        if self._own_client:
            self._client = self._build_client()

    @property
    def cfg_hash(self) -> str:
        return config_hash({"wire": "systemone-v1", "noul": "criteria{true,false}", "score": "criteria list"})

    async def _identify(self) -> None:
        async with self._id_lock:
            if self.identity is not None and "models" in self.identity:
                return
            listed = await self._client.models.list()
            models = [{"name": m.name, "release_date": m.release_date, "description": m.description} for m in listed.models]
            self.identity = {**(self.identity or {}), "models": models}
            for m in models:
                log.info("jev: GET /v1/models -> %s (released %s)", m["name"], m["release_date"])

    async def _ask(self, state, questions):
        if self.identity is None or "models" not in self.identity:
            await self._identify()
        r = await self._client.system_one(state=state, questions=jev_questions(questions), model=self.model)
        answers, degraded = {}, []
        for name, a in r.answers.items():
            if a.type == "noul":
                p = float(a.noul)
                probs = {"yes": p, "no": 1.0 - p} if math.isfinite(p) and 0.0 <= p <= 1.0 else None
                answers[name] = Answer("noul", probs)
            elif a.type == "choice":
                answers[name] = Answer("choice", normalize(dict(a.probabilities)), float(a.confidence))
            else:
                probs = {str(k): float(v) for k, v in a.probabilities.items()}
                answers[name] = Answer("score", normalize(probs), float(a.confidence))
            if answers[name].probs is None:  # no usable distribution: abstain (retried like an emulator parse)
                degraded.append(name)
        n_in, n_out = r.usage.input_tokens, r.usage.output_tokens
        est = n_in is None
        if est:
            n_in = estimate_tokens(canonical_json({"state": state, "questions": [question_to_json(q) for q in questions.values()]}))
        return S1Response(
            answers=answers, model=self.model, backend=self.backend_id, input_tokens=int(n_in),
            output_tokens=int(n_out or 0), response_model=r.model, degraded=degraded, usage_estimated=est,
        )


# -------------------------------------------------------------------------------------------- mock
class MockBackend(S1Backend):
    """Deterministic offline stand-in: each answer is a pure function of (state, question), so whole-pipeline
    dry runs are reproducible and free. It models nothing; never use its outputs as results.

    `degrade(name, state, attempt) -> bool` lets tests simulate emulator parse failures.
    """

    backend_id = "mock"
    prompt_version = "mock-v1"

    def __init__(self, model: str = "mock-1", *, peak: float = 3.0,
                 degrade: Callable[[str, JSONContent, int], bool] | None = None, **kw):
        super().__init__(model, **kw)
        self.peak = peak
        self.degrade = degrade
        self.calls = 0
        self._attempts: dict[str, int] = {}

    @property
    def cfg_hash(self) -> str:
        return config_hash({"mock": 1, "peak": self.peak})

    def _cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        return 0.0

    async def _ask(self, state, questions):
        self.calls += 1
        k = self.key(state, questions)
        attempt = self._attempts[k] = self._attempts.get(k, 0) + 1
        answers, degraded = {}, []
        for name, q in questions.items():
            if self.degrade is not None and self.degrade(name, state, attempt):
                degraded.append(name)
                answers[name] = Answer(q.kind, None)
                continue
            h = hashlib.sha256(canonical_json([state, question_to_json(q)]).encode()).digest()
            labs = labels(q)
            z = [self.peak * (h[i % len(h)] / 255.0 - 0.5) * 2 for i in range(len(labs))]
            e = [math.exp(v) for v in z]
            answers[name] = Answer(q.kind, {lab: v / sum(e) for lab, v in zip(labs, e)})
        await asyncio.sleep(0)
        n_in = estimate_tokens(canonical_json(state) + canonical_json([question_to_json(q) for q in questions.values()]))
        return S1Response(answers=answers, model=self.model, backend=self.backend_id, input_tokens=n_in,
                          output_tokens=4 * len(questions), response_model=self.model, degraded=degraded)


# ------------------------------------------------------------------------------------------ factory
BACKENDS: dict[str, type[S1Backend] | str] = {
    "llm-logprob": LogprobEmulator,
    "llm-verbalized": VerbalizedLLM,
    "jev": JevBackend,
    "mock": MockBackend,
    # the open typed decision model in its own environment (design_v2_delta IF-1; imported lazily: it imports this
    # module)
    "typed-local": "jevtrader.s1.typed_local:TypedDecisionLocal",
    "jev-cf": "jevtrader.s1.cloudflare_jev:CloudflareJevBackend",  # Jev on Cloudflare Workers AI (lazy, as typed-local)
}


def backend_class(name: str) -> type[S1Backend]:
    cls = BACKENDS[name]
    if isinstance(cls, str):
        import importlib

        mod, attr = cls.split(":")
        cls = getattr(importlib.import_module(mod), attr)
    return cls


def load_backend_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def make_backend(
    cfg: Mapping | str | Path | None = None,
    *,
    backend: str | None = None,
    cache: ResponseCache | bool = True,
    ledger: SpendLedger | bool = True,
    root: Path = ROOT,
    **overrides,
) -> S1Backend:
    """Build the System-One backend named by `cfg['s1']['backend']` (or `backend`) from configs/backends.yaml.

    `cache`/`ledger`: True builds them from the config, False disables them, an instance is used as is.
    The bank hash defaults to the loaded bank's canonical SHA-256.
    """
    if cfg is None or isinstance(cfg, (str, Path)):
        cfg = load_backend_config(cfg or DEFAULT_CONFIG)
    name = backend or cfg["s1"]["backend"]
    if name not in BACKENDS:
        raise ValueError(f"unknown S1 backend {name!r}; one of {sorted(BACKENDS)}")
    kw = dict((cfg.get("backends") or {}).get(name) or {})
    kw.update(overrides)
    cls = backend_class(name)
    if not getattr(cls, "uses_ledger", True):
        ledger = False  # no API fees (the local typed model): no ledger entry (design_v2_delta §6)
    if cache is True:
        cache = ResponseCache(root / cfg["cache"]["path"]) if cfg.get("cache") else None
    if ledger is True:
        lc = cfg.get("ledger")
        ledger = SpendLedger(
            root / lc["path"], hard_cap_usd=float(lc.get("hard_cap_usd", 245.0)),
            sub_caps={k: float(v) for k, v in (lc.get("sub_caps_usd") or {}).items()},
            batch_size=int(lc.get("batch_size", 32)),
        ) if lc else None
    kw.setdefault("bank_hash", _bank_sha256())
    if name != "mock" and getattr(cls, "needs_api_key", True):
        try:
            from dotenv import load_dotenv

            load_dotenv(root / ".env", override=False)  # keys stay in the environment; never logged
        except ImportError:
            pass
    cache = cache if isinstance(cache, ResponseCache) else None  # (an empty cache is falsy: no `or`)
    ledger = ledger if isinstance(ledger, SpendLedger) else None
    return cls(cache=cache, ledger=ledger, **kw)
