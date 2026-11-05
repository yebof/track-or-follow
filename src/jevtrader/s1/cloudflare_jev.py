"""`jev-cf`: TypeSafe's Jev as served by Cloudflare Workers AI (model name `typesafe/jev`), behind the S1Backend contract.

Sources (archived 2026-09-29 in docs/jev/; every fact below comes from them):
- `cloudflare_jev.{html,txt,md}`: the model page (REST call, four request/response examples, "Context Window 32,000
  tokens", "Input (per 1M tokens) $0.042", "Output (per 1M tokens) $0.00", "Cached input (per 1M tokens) $0.00");
- `cloudflare_jev_schema-{input,output}.json`: the raw API schemas (input admits only `state` and `questions`);
- `cloudflare_get-started_rest-api.md`: the REST envelope `{"result": ..., "success": true, "errors": [], "messages": []}`;
- `cloudflare_platform_limits.md`: Text Generation, 300 requests per minute unless a model requires the Workers Paid
  plan (the Jev page does not say it does); `cloudflare_platform_errors.md`: the error codes used below.

Wire. POST {base}/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run with `Authorization: Bearer {CLOUDFLARE_API_TOKEN}` and body
{"model": "typesafe/jev", "input": {"state": ..., "questions": ...}}. `input.state` and `input.questions` are byte for
byte the TypeSafe API's `state` and `questions` as the official SDK serializes them (`wire_questions`), so programs
send identical (state, questions) on every backend (design_v1 §11.3). The response is TypeSafe's {model, answers,
usage}, either bare or wrapped in the Cloudflare envelope; both are accepted.

Pin. The input schema has no `model` field, so the version cannot be requested, only observed. `model` is the
versioned id every archived example response reports (`jev-1.13.0`); S1Backend asserts that every `response.model`
equals it (ModelPinError), so a move of the Cloudflare alias stops a run instead of mixing versions. This route has no
`GET /v1/models`: the identity block records the provider, the Cloudflare model name, the archived release date and
(S1Backend) the first `response.model`.

Retries (design_v1 §11.3, as JevBackend): HTTP 429, 408, 5xx, connection errors, timeouts and an envelope with
`success: false` and code 3040 (out of capacity) are retried with capped exponential backoff and full jitter,
honouring `retry-after` / `retry-after-ms`, at most `max_retries` = 8 times, `backoff_max` = 60 s, within `timeout` =
600 s per call. Code 3036 (daily free allocation used up) is a 429 that waiting cannot fix: it raises at once, as does
every other 4xx (`CloudflareAPIError`). HTTP 413 (request too large) abstains like the context guard.

Answers. noul -> {"yes": p, "no": 1 - p}; choice and score -> `probabilities` over exactly the question's labels
(label order), `confidence` kept as the vendor statistic. A missing answer, a type mismatch, a label set that differs
from the question's, or a non-finite value is degraded (S1Backend: two retries, then probs None, never cached). A 200
body that is not TypeSafe's shape is booked at the estimate (it may have been billed) and then stops the run.

Context guard. A request whose estimated input (the o200k proxy of the `input` JSON, bytes / 3.5, plus
REQUEST_OVERHEAD_TOKENS) exceeds `context_tokens` (32,000) is not sent: every question abstains (probs None, counted,
never cached, no charge) and `meta.context_overflow` records the estimate. States are never truncated.

Money. $0.042 per 1M input tokens; output and cached input free (the archived page; TypeSafe's own price is the same,
docs/jev/pages/models). Missing usage is estimated, never $0 (F07). Authorization: with a ledger, every call's tag must
belong to run JEV-CF and the ledger must hold a positive JEV-CF sub-cap (configs/backends.yaml
`ledger.sub_caps_usd.JEV-CF`, 0 until the user authorizes an amount); a real (not injected) transport refuses to run
without a ledger. Jev is billed to the user's Cloudflare account outside the $250 LLM budget (CLAUDE.md; decisions
2026-09-29 D-13): with `ledger_path` set (configs/backends.yaml), spend is booked in that separate ledger file, whose
hard cap and JEV-CF sub-cap are the configured JEV-CF sub-cap.

Credentials. CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN come from the environment (`make_backend` loads .env).
They never enter the request key, `describe()`, the identity block, a log line or an error message.
"""

from __future__ import annotations

import asyncio
import email.utils
import logging
import math
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx2

from jevtrader.s1.backends import ROOT, S1Backend, config_hash, estimate_tokens
from jevtrader.s1.budget import BudgetExceeded, SpendLedger, run_of
from jevtrader.s1.types import Answer, JSONContent, Question, S1Response, canonical_json, labels, normalize

log = logging.getLogger("jevtrader.s1")

PROVIDER = "cloudflare-workers-ai"
BASE_URL = "https://api.cloudflare.com/client/v4"
CF_MODEL = "typesafe/jev"  # docs/jev/cloudflare_jev.md: the Workers AI model name
JEV_MODEL = "jev-1.13.0"  # docs/jev/cloudflare_jev.md: `model` of every example response
# Release date of jev-1.13.0: docs/jev/wikipedia_jev.txt ("Release ... 2026-09-15", "Stable release jev-1.13.0") and
# the SDK's model-list example (docs/jev/sdk_typesafe_0.7.1/typesafe_sdk/_schemas/models.py, release_date).
RELEASE_DATE = "2026-09-15"
RELEASE_DATE_SOURCES = ("docs/jev/wikipedia_jev.txt", "docs/jev/sdk_typesafe_0.7.1/typesafe_sdk/_schemas/models.py")
CONTEXT_TOKENS = 32_000  # docs/jev/cloudflare_jev.md: "Context Window 32,000 tokens"
# USD per 1M tokens, docs/jev/cloudflare_jev.md (Pricing); tests/test_s1_cloudflare.py re-reads the archived page
PRICE_IN, PRICE_OUT, PRICE_CACHED = 0.042, 0.0, 0.0
# Tokens Jev reports beyond the o200k proxy of the `input` JSON: the maximum of (usage.input_tokens - bytes / 3.5) over
# the page's four request/response examples (264, 261, 255, 212; tests/test_s1_cloudflare.py recomputes it)
REQUEST_OVERHEAD_TOKENS = 264
RUN = "JEV-CF"  # ledger run of every Cloudflare Jev call (sub-cap key in configs/backends.yaml)
PARSER = "jev-cf-v1"
RETRY_CODES = frozenset({3040, 3007, 3008})  # out of capacity, timeout, aborted (docs/jev/cloudflare_platform_errors.md)
FATAL_429_CODES = frozenset({3036})  # daily free allocation used up: retrying cannot help


class _RedactAccount(logging.Filter):
    """httpx2 logs every request URL at INFO; the Workers AI URL carries the account id, which is never logged."""

    _PAT = re.compile(r"/accounts/[^/\s\"]+/")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "/accounts/" in msg:
            record.msg, record.args = self._PAT.sub("/accounts/<account>/", msg), ()
        return True


if not any(isinstance(f, _RedactAccount) for f in logging.getLogger("httpx2").filters):
    logging.getLogger("httpx2").addFilter(_RedactAccount())


class CloudflareAPIError(RuntimeError):
    """Cloudflare refused the call (non-retryable status or code, or the retry budget ran out). Carries the HTTP status
    and Cloudflare's error codes; never the URL (it holds the account id) or any header."""

    def __init__(self, message: str, status: int | None = None, codes: tuple[int, ...] = ()):
        super().__init__(message)
        self.status, self.codes = status, codes


class CloudflareCredentialsError(RuntimeError):
    """CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN is missing."""


class _ContextOverflow(Exception):
    """HTTP 413 from the provider (request too large)."""


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 8
    backoff_base: float = 0.5
    backoff_max: float = 60.0
    timeout: float = 600.0  # the total retry budget of one call, seconds

    def delay(self, retry: int, headers: Mapping[str, str] | None, rng: random.Random) -> float:
        """Seconds to wait before retry number `retry` (1-based): the server's retry-after when given (capped at
        backoff_max), else full jitter over the capped exponential."""
        hinted = retry_after(headers)
        if hinted is not None:
            return min(self.backoff_max, hinted)
        return rng.uniform(0.0, min(self.backoff_max, self.backoff_base * 2 ** (retry - 1)))


def retry_after(headers: Mapping[str, str] | None) -> float | None:
    """`retry-after-ms`, or `retry-after` in seconds or as an HTTP date; None if absent or unreadable."""
    if not headers:
        return None
    h = {k.lower(): v for k, v in headers.items()}
    try:
        if "retry-after-ms" in h:
            return max(0.0, float(h["retry-after-ms"]) / 1000.0)
        if "retry-after" in h:
            v = h["retry-after"].strip()
            try:
                return max(0.0, float(v))
            except ValueError:
                when = email.utils.parsedate_to_datetime(v)
                return max(0.0, when.timestamp() - time.time())
    except (TypeError, ValueError, OverflowError):
        return None
    return None


# ------------------------------------------------------------------------------------------ wire
def wire_question(q: Question) -> dict:
    """One question as the TypeSafe SDK puts it on the wire (typesafe_sdk 0.7.1; equality is tested against the SDK):
    noul `criteria` holds only the described outcomes and is omitted when neither is; choice `criteria` maps every
    label to its description (null = the label alone); score `criteria` lists the level texts (the Cloudflare input
    schema requires at least two)."""
    if q.kind == "noul":
        d: dict[str, Any] = {"type": "noul", "instructions": q.instructions}
        crit = {k: v for k, v in (("true", q.true), ("false", q.false)) if v is not None}
        if crit:
            d["criteria"] = crit
        return d
    if q.kind == "choice":
        return {"type": "choice", "instructions": q.instructions, "criteria": dict(q.options)}
    if len(q.levels) < 2:
        raise ValueError("a score question needs at least two levels (docs/jev/cloudflare_jev_schema-input.json)")
    return {"type": "score", "instructions": q.instructions, "criteria": list(q.levels)}


def wire_questions(questions: Mapping[str, Question]) -> dict[str, dict]:
    return {name: wire_question(q) for name, q in questions.items()}


def wire_input(state: JSONContent, questions: Mapping[str, Question]) -> dict:
    """The `input` object: TypeSafe's request body without `model`."""
    return {"state": state, "questions": wire_questions(questions)}


def input_bytes(state: JSONContent, questions: Mapping[str, Question]) -> int:
    return len(canonical_json(wire_input(state, questions)).encode("utf-8"))


def estimate_input_tokens(state: JSONContent, questions: Mapping[str, Question],
                          overhead: int = REQUEST_OVERHEAD_TOKENS) -> int:
    """Pre-call estimate of Jev's `usage.input_tokens` (the context guard and the ledger reservation use it)."""
    return estimate_tokens(canonical_json(wire_input(state, questions))) + int(overhead)


def unwrap(body: Any) -> dict:
    """TypeSafe's {model, answers, usage} from a bare body or from the Cloudflare envelope {"result": ..., "success": ...}.
    Raises ValueError for any other shape."""
    if not isinstance(body, dict):
        raise ValueError("response body is not a JSON object")  # noqa: TRY004 (the caller catches ValueError)
    if "result" in body and ("success" in body or "errors" in body):
        res = body["result"]
        if not isinstance(res, dict):
            raise ValueError("envelope without a result object")
        # observed 2026-10-09 for typesafe/jev: {"result": {"state": "Completed", "result": {model, answers, usage}}}
        if "state" in res and "answers" not in res:
            if res.get("state") != "Completed" or not isinstance(res.get("result"), dict):
                raise ValueError(f"job envelope in state {res.get('state')!r}")
            return res["result"]
        return res
    return body


def error_codes(body: Any) -> tuple[int, ...]:
    errs = body.get("errors") if isinstance(body, dict) else None
    out = []
    for e in errs or []:
        c = e.get("code") if isinstance(e, dict) else None
        if isinstance(c, int) and not isinstance(c, bool):
            out.append(c)
    return tuple(out)


def error_text(body: Any) -> str:
    errs = body.get("errors") if isinstance(body, dict) else None
    msgs = [str(e.get("message", ""))[:200] for e in errs or [] if isinstance(e, dict)]
    return "; ".join(m for m in msgs if m)


def _number(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def parse_answer(q: Question, a: Any) -> Answer:
    """One wire answer as our Answer; probs None (degraded) unless it matches the question exactly."""
    if not isinstance(a, dict) or a.get("type") != q.kind:
        return Answer(q.kind, None)
    if q.kind == "noul":
        p = _number(a.get("noul"))
        ok = p is not None and 0.0 <= p <= 1.0
        return Answer("noul", {"yes": p, "no": 1.0 - p} if ok else None)
    raw = a.get("probabilities")
    labs = labels(q)
    if not isinstance(raw, dict) or set(raw) != set(labs):
        return Answer(q.kind, None)
    vals = {lab: _number(raw[lab]) for lab in labs}
    if any(v is None or v < 0.0 for v in vals.values()):
        return Answer(q.kind, None)
    conf = _number(a.get("confidence"))
    return Answer(q.kind, normalize(vals), conf)


def jev_ledger(path: str | Path, cap_usd: float) -> SpendLedger:
    """The separate Jev spend ledger: hard cap = JEV-CF sub-cap = the authorized amount."""
    return SpendLedger(Path(path), hard_cap_usd=float(cap_usd), sub_caps={RUN: float(cap_usd)})


# ------------------------------------------------------------------------------------------ backend
class CloudflareJevBackend(S1Backend):
    backend_id = "jev-cf"
    prompt_version = "jev-api-v1"  # the TypeSafe wire contract (same `questions` JSON as `jev`)

    def __init__(
        self,
        model: str = JEV_MODEL,
        *,
        cf_model: str = CF_MODEL,
        account_id: str | None = None,
        api_token: str | None = None,
        base_url: str = BASE_URL,
        retry: RetryPolicy | Mapping[str, Any] | None = None,
        request_timeout: float = 60.0,
        context_tokens: int = CONTEXT_TOKENS,
        overhead_tokens: int = REQUEST_OVERHEAD_TOKENS,
        ledger_path: str | Path | None = None,
        transport: Any = None,
        client: Any = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        seed: int | None = None,
        root: Path = ROOT,
        **kw,
    ):
        super().__init__(model, **kw)
        self.cf_model = cf_model
        self.base_url = base_url.rstrip("/")
        self.retry_policy = RetryPolicy(**retry) if isinstance(retry, Mapping) else (retry or RetryPolicy())
        self.request_timeout = float(request_timeout)
        self.context_tokens = int(context_tokens)
        self.overhead_tokens = int(overhead_tokens)
        self.ledger_path = None if ledger_path is None else str(ledger_path)
        self._account_id, self._api_token = account_id, api_token
        self._transport = transport
        self._injected = client is not None or transport is not None  # a mock: no real network
        self._own_client = client is None
        self._client = client
        self._sleep = sleep or asyncio.sleep
        self._rng = random.Random(seed)
        self.http_retries = 0  # retried HTTP attempts (429/5xx/...), all calls
        self.overflows = 0  # requests abstained by the context guard or a 413
        self.identity = {"provider": PROVIDER, "cf_model": cf_model, "release_date_archived": RELEASE_DATE,
                         "release_date_sources": list(RELEASE_DATE_SOURCES)}
        if self.ledger_path is not None and self.ledger is not None:
            target = (Path(root) / self.ledger_path).resolve()
            if Path(self.ledger.path).resolve() != target:  # Jev spend never books into the LLM ledger
                self.ledger = jev_ledger(target, float(self.ledger.sub_caps.get(RUN, 0.0)))
                log.info("jev-cf: spend is booked in %s (outside the LLM ledger), cap %.2f USD", self.ledger_path,
                         self.ledger.hard_cap_usd)

    # ------------------------------------------------------------------ identity and money
    @property
    def cfg_hash(self) -> str:
        return config_hash({"wire": "systemone-v1", "provider": PROVIDER, "cf_model": self.cf_model,
                            "noul": "criteria{true,false}", "score": "criteria list", "parser": PARSER,
                            "degraded_retries": self.degraded_retries})

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "provider": PROVIDER, "cf_model": self.cf_model,
                "context_tokens": self.context_tokens, "overhead_tokens": self.overhead_tokens,
                "price_usd_per_m": {"input": PRICE_IN, "output": PRICE_OUT, "cached_input": PRICE_CACHED},
                "retry": asdict(self.retry_policy), "ledger_path": self.ledger_path}

    def _cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cached = max(0, min(int(cached_tokens), int(input_tokens)))
        return ((int(input_tokens) - cached) * PRICE_IN + cached * PRICE_CACHED + int(output_tokens) * PRICE_OUT) / 1e6

    def estimate_input_tokens(self, state: JSONContent, questions: Mapping[str, Question]) -> int:
        return estimate_input_tokens(state, questions, self.overhead_tokens)

    def estimate_cost(self, state: JSONContent, questions: Mapping[str, Question]) -> float:
        return self._cost(self.estimate_input_tokens(state, questions), 0)

    def request_body(self, state: JSONContent, questions: Mapping[str, Question]) -> dict:
        return {"model": self.cf_model, "input": wire_input(state, questions)}

    # ------------------------------------------------------------------ gates before any call
    def _authorize(self, tag: str) -> None:
        if self.ledger is None:
            if not self._injected:
                raise BudgetExceeded("jev-cf: no spend ledger; a paid Cloudflare call needs the JEV-CF ledger")
            return
        if run_of(tag) != RUN:
            raise BudgetExceeded(f"jev-cf: ledger tag {tag!r} is not under run {RUN!r}")
        cap = float(self.ledger.sub_caps.get(RUN, 0.0))
        if not cap > 0.0:
            raise BudgetExceeded(f"jev-cf: the {RUN} sub-cap is {cap:.2f} USD, i.e. not authorized "
                                 f"(configs/backends.yaml ledger.sub_caps_usd.{RUN})")

    def _credentials(self) -> tuple[str, str]:
        acct = self._account_id if self._account_id is not None else os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
        token = self._api_token if self._api_token is not None else os.environ.get("CLOUDFLARE_API_TOKEN", "")
        if not acct or not token:
            raise CloudflareCredentialsError("jev-cf: CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set "
                                             "(.env); refusing the call")
        return acct, token

    def _abstain(self, key: str, questions: Mapping[str, Question], est: int | None, why: str) -> S1Response:
        self.overflows += 1
        log.warning("jev-cf: request abstained (%s; estimated %s input tokens, context %d)", why, est,
                    self.context_tokens)
        return S1Response(
            answers={n: Answer(q.kind, None) for n, q in questions.items()}, model=self.model,
            backend=self.backend_id, prompt_version=self.prompt_version, request_key=key, attempts=0,
            degraded=list(questions), meta={"context_overflow": True, "why": why, "est_input_tokens": est,
                                            "context_tokens": self.context_tokens})

    async def _ask_uncached(self, key: str, state: JSONContent, questions: Mapping[str, Question], tag: str) -> S1Response:
        self._authorize(tag)
        est = self.estimate_input_tokens(state, questions)
        if est > self.context_tokens:
            return self._abstain(key, questions, est, "context guard")
        self._credentials()
        try:
            return await super()._ask_uncached(key, state, questions, tag)
        except _ContextOverflow:
            return self._abstain(key, questions, est, "HTTP 413")

    def _check_pin(self, resp: S1Response) -> None:
        """Runs after S1Backend has booked the call: an unreadable body stops the run with its own error (the provider
        broke its contract), then the pinned-model assertion."""
        if resp.meta.get("unreadable"):
            raise CloudflareAPIError(f"jev-cf: unreadable response ({resp.meta['unreadable']})")
        super()._check_pin(resp)

    # ------------------------------------------------------------------ HTTP
    def _on_new_loop(self) -> None:
        if self._own_client:
            self._client = None  # rebuilt on the running loop at the next call

    def _http(self):
        if self._client is None:
            self._client = httpx2.AsyncClient(transport=self._transport, timeout=self.request_timeout)
        return self._client

    async def _post(self, body: dict) -> tuple[Any, Mapping[str, str], int]:
        """POST with retries; returns (decoded JSON body or None, headers, retries used)."""
        acct, token = self._credentials()
        url = f"{self.base_url}/accounts/{acct}/ai/run"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        content = canonical_json(body).encode("utf-8")
        pol = self.retry_policy
        deadline = time.monotonic() + pol.timeout
        retries = 0
        while True:
            resp_headers: Mapping[str, str] | None = None
            try:
                r = await self._http().post(url, content=content, headers=headers)
            except (httpx2.TimeoutException, httpx2.TransportError) as e:
                what = f"{type(e).__name__}"
            else:
                resp_headers = r.headers
                try:
                    data = r.json()
                except (ValueError, UnicodeDecodeError):
                    data = None
                codes = error_codes(data)
                s = r.status_code
                if 200 <= s < 300:
                    if isinstance(data, dict) and data.get("success") is False:
                        if not set(codes) & RETRY_CODES:
                            raise CloudflareAPIError(f"jev-cf: success=false, codes {list(codes)}: {error_text(data)}",
                                                     s, codes)
                        what = f"success=false codes {list(codes)}"
                    else:
                        return data, r.headers, retries
                elif s == 413:
                    raise _ContextOverflow()
                elif s == 429 and set(codes) & FATAL_429_CODES:
                    raise CloudflareAPIError(f"jev-cf: HTTP 429 codes {list(codes)} (daily allocation used up): "
                                             f"{error_text(data)}", s, codes)
                elif s in (408, 429) or s >= 500:
                    what = f"HTTP {s}" + (f" codes {list(codes)}" if codes else "")
                else:
                    raise CloudflareAPIError(f"jev-cf: HTTP {s} codes {list(codes)}: {error_text(data)}", s, codes)
            retries += 1
            if retries > pol.max_retries:
                raise CloudflareAPIError(f"jev-cf: {what} after {pol.max_retries} retries")
            wait = pol.delay(retries, resp_headers, self._rng)
            if time.monotonic() + wait > deadline:
                raise CloudflareAPIError(f"jev-cf: {what}; retry budget of {pol.timeout:.0f} s exhausted")
            self.http_retries += 1
            log.info("jev-cf: %s, retry %d in %.2f s", what, retries, wait)
            await self._sleep(wait)

    async def _ask(self, state: JSONContent, questions: Mapping[str, Question]) -> S1Response:
        data, headers, retries = await self._post(self.request_body(state, questions))
        unreadable = None
        try:
            res = unwrap(data)
            raw_answers = res.get("answers")
            if not isinstance(raw_answers, dict):
                raise ValueError("no answers object")  # noqa: TRY004 (caught just below)
        except ValueError as e:  # possibly billed: booked at the estimate by S1Backend, then `_check_pin` raises
            unreadable = str(e)
            res, raw_answers = (data if isinstance(data, dict) else {}), {}
        answers, degraded = {}, []
        for name, q in questions.items():
            a = parse_answer(q, raw_answers.get(name))
            answers[name] = a
            if a.probs is None:
                degraded.append(name)
        usage = res.get("usage") if isinstance(res.get("usage"), dict) else {}
        n_in, n_out = usage.get("input_tokens"), usage.get("output_tokens")
        est = not (isinstance(n_in, int) and not isinstance(n_in, bool))
        if est:
            n_in = self.estimate_input_tokens(state, questions)
        n_out = n_out if isinstance(n_out, int) and not isinstance(n_out, bool) else 0
        served = res.get("model")
        meta: dict[str, Any] = {}
        if unreadable is not None:
            meta["unreadable"] = unreadable
        if retries:
            meta["http_retries"] = retries
        ray = headers.get("cf-ray") if headers is not None else None
        if ray:
            meta["cf_ray"] = ray
        return S1Response(answers=answers, model=self.model, backend=self.backend_id, input_tokens=int(n_in),
                          output_tokens=int(n_out), response_model=served if isinstance(served, str) else "",
                          degraded=degraded, usage_estimated=est, meta=meta)
