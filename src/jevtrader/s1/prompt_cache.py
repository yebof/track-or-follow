"""Provider prompt-cache routing for the OpenAI-backed emulators (emu-v1, verb-v1): a stable `prompt_cache_key` per
static-prefix family (optionally partitioned to about 15 requests per minute per key), plus a warm-up gate. Routing
only.

Why (docs/impl/dev1_report.md §4, P1-1). DEV #1 sent about 3,200 N1 requests in about 2 minutes, all with the same
static prefix and no `prompt_cache_key`; only 32.8% of N1's paid input tokens were provider-cached against an
85% ceiling (the 1,792-token cached prefix of a 2,102-token request). The archived OpenAI guide
(docs/pricing/openai_guide_prompt-caching_2026-09-26.txt, lines 743-763) says, for models before GPT-5.6 (the
emulator is pinned to gpt-4.1-mini-2025-04-14):
- cached prefixes live on individual machines, and traffic above about 15 requests per minute can overflow to
  machines that do not hold the entry;
- routing depends on a hash of the prompt's initial tokens and on a supplied `prompt_cache_key`;
- "use a stable prompt_cache_key for requests that share a reusable prefix ... aim for about 15 requests per minute
  in total across all prefixes using each key. Partition higher-volume traffic across multiple keys using a stable,
  deterministic mapping."

What this module does:
- **Family.** Requests whose static prefix (system prompt + the user text before the state) agrees on its first
  `family_chars` characters form one family. 3,072 characters is below the 1,024-token minimum cacheable prefix for
  ordinary text, so two requests that can share a cache entry (they share >= 1,024 prefix tokens) share a family, and
  requests of different programs (different question blocks) never do. A program with a variable question count
  (HS: one question per headline) stays one family, so its partial prefixes can still be reused on one machine.
  The family depends only on the prompt layout, so it follows any layout change (for example an N1 request split
  into a screening request and the rest: each part is its own family).
- **Shards (off by default: `max_shards` = 1).** A family's key count is the next power of two >= (requests of the
  family in the last `window_s` seconds) / `rpm_per_key`, at most `max_shards`, and it never shrinks within a
  process. A request's shard is its request key modulo the key count: a stable, deterministic mapping (doubling the
  count keeps half of the mapping). Counting only the last minute (no extrapolation) under-shards for the first
  minute of a burst and never over-shards; a fresh key costs one cold (uncached) request.
  **Why off:** the DEV #2 measurement (docs/impl/dev2_ops.md §1.4) found the guide's partitioning harmful for the
  emulator's model at our rates: one key paced at 14.6 requests per minute got 0 of 16 requests cached, and 497
  partitioned 9-question N1 requests of the concurrent n1 stream got 11.5% cached, while 17-request bursts on one key
  got 65-76% cached and DEV #1 (no key at all) 38%. Density per key, not the 15-rpm bound, is what the measurements
  reward, so the default is one stable key per family; `max_shards` > 1 turns partitioning back on.
- **Warm-up gate.** The first request of a key whose static prefix can be cached goes alone; the other requests of
  that key wait (outside the concurrency semaphore, so the wait is not counted as latency and holds no slot) until
  it has returned, so its prefix is written before the key fans out. A timeout releases the waiters.
- **Keys** look like `emu-v1:<family 16 hex>:<shard>`. They carry no state content (the family is a hash of the bank's
  question text, the shard a number). They are **never** part of the request key (`s1.types.request_key`) or of the
  backend's `cfg_hash`, so answers, the response cache and every result are unchanged by this module
  (tests/test_s1_prompt_cache.py). The key used for a paid call is logged in `requests.parquet.backend_meta`.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class PromptCacheConfig:
    enabled: bool = True
    rpm_per_key: float = 15.0  # the guide's "about 15 requests per minute" per key (used only if max_shards > 1)
    max_shards: int = 1  # power of two; 1 = one stable key per family (the measured default, dev2_ops §1.4)
    window_s: float = 60.0  # the rate window (requests per minute)
    family_chars: int = 3072  # characters of the static prefix that define a family
    warm: bool = True  # warm-up gate for families whose static prefix is cacheable
    warm_min_tokens: int = 1024  # the guide's minimum cacheable prefix, on the conservative byte proxy
    warm_timeout_s: float = 60.0  # waiters give up and proceed after this long
    retention: str | None = None  # prompt_cache_retention; None = the provider default (gpt-4.1-mini: in_memory)
    namespace: str = ""  # key prefix; "" = the backend's prompt version (e.g. "emu-v1")

    @classmethod
    def from_value(cls, value: Any) -> PromptCacheConfig:
        """True / None -> defaults; False -> disabled; a mapping -> those fields (unknown fields are an error)."""
        if value is None or value is True:
            return cls()
        if value is False:
            return cls(enabled=False)
        if isinstance(value, PromptCacheConfig):
            return value
        if isinstance(value, Mapping):
            known = {f.name for f in fields(cls)}
            bad = set(value) - known
            if bad:
                raise ValueError(f"unknown prompt_cache settings {sorted(bad)}; known: {sorted(known)}")
            return cls(**dict(value))
        raise TypeError(f"prompt_cache must be a bool or a mapping, got {type(value).__name__}")


@dataclass(frozen=True)
class Route:
    key: str  # the prompt_cache_key sent to the provider
    family: str  # sha256 of the family prefix (hex)
    shard: int
    shards: int  # the family's key count when this request was routed
    warm: bool  # the static prefix is long enough to be cached: the warm-up gate applies


def next_pow2(x: float) -> int:
    return 1 if x <= 1 else 1 << math.ceil(math.log2(x))


class PromptCacheRouter:
    """Per-backend routing state: family -> recent send times and key count; key -> warm-up state. Not thread-safe
    (one event loop per backend, as S1Backend already assumes)."""

    def __init__(self, cfg: PromptCacheConfig, namespace: str, token_estimate=None, clock=time.monotonic):
        self.cfg = cfg
        self.namespace = cfg.namespace or namespace
        self._estimate = token_estimate or (lambda s: math.ceil(len(s.encode("utf-8")) / 3.5))
        self._clock = clock
        self._sent: dict[str, deque] = {}
        self._shards: dict[str, int] = {}
        self._prefix_ok: dict[str, bool] = {}
        self._warmed: set[str] = set()
        self._pending: dict[str, asyncio.Event] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self.stats = {"routed": 0, "leaders": 0, "waited": 0, "wait_s": 0.0, "timeouts": 0}

    # ------------------------------------------------------------------ routing (pure, synchronous)
    def family(self, static_prefix: str) -> str:
        return hashlib.sha256(static_prefix[: self.cfg.family_chars].encode("utf-8")).hexdigest()

    def route(self, static_prefix: str, request_key: str) -> Route:
        fam = self.family(static_prefix)
        now = self._clock()
        q = self._sent.setdefault(fam, deque())
        q.append(now)
        while q and q[0] < now - self.cfg.window_s:
            q.popleft()
        n = min(self.cfg.max_shards, max(self._shards.get(fam, 1), next_pow2(len(q) / self.cfg.rpm_per_key)))
        self._shards[fam] = n
        shard = int(request_key[:15], 16) % n if request_key else 0
        ok = self._prefix_ok.get(fam)
        if ok is None:
            ok = self._prefix_ok[fam] = self._estimate(static_prefix) >= self.cfg.warm_min_tokens
        self.stats["routed"] += 1
        return Route(key=f"{self.namespace}:{fam[:16]}:{shard}", family=fam, shard=shard, shards=n,
                     warm=bool(self.cfg.warm and ok))

    def shards(self) -> dict[str, int]:
        return dict(self._shards)

    # ------------------------------------------------------------------ warm-up gate (loop-bound)
    def _bind(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:  # events bind to a loop; a new loop (a second generate() run) starts fresh
            self._loop, self._pending = loop, {}

    async def enter(self, route: Route) -> bool:
        """Wait until `route.key` is warm. Returns True if the caller is the key's warm-up leader: it must call
        `release(route)` once its provider call has returned (in a `finally`)."""
        if not route.warm or route.key in self._warmed:
            return False
        self._bind()
        ev = self._pending.get(route.key)
        if ev is None:
            self._pending[route.key] = asyncio.Event()
            self.stats["leaders"] += 1
            return True
        t0 = self._clock()
        try:
            await asyncio.wait_for(ev.wait(), timeout=self.cfg.warm_timeout_s)
        except TimeoutError:
            self.stats["timeouts"] += 1
        self.stats["waited"] += 1
        self.stats["wait_s"] += self._clock() - t0
        return False

    def release(self, route: Route) -> None:
        self._warmed.add(route.key)
        ev = self._pending.pop(route.key, None)
        if ev is not None:
            ev.set()


# ---------------------------------------------------------------------------------------------- the O-5 decision rule
# DEV #1 (no key): N1's paid input was 32.8% cached against a 1,792 / 2,102-token ceiling, so 0.385 of the cacheable
# prefix was realized (DEV1_REALIZATION). docs/impl/dev2_ops.md §1.5 (O-5) compared N1's cached SHARE with 32.8%; under a split N1 (bank
# v1.0-rc6/rc7: single-question requests of ~410 tokens that can never be cached, plus one cacheable request for about
# 77% of items) the share's ceiling is itself near 33-44%, so that rule would switch caching off exactly when it works
# (the rc6 smoke: 65.6% of the attribute requests' input cached, 29.7% of all N1 input; DEV #2 audit). The rule is
# re-based on REALIZATION per request family: the cached tokens of a family's paid requests over (requests x the
# family's cacheable prefix), compared with DEV #1's 0.385.
DEV1_REALIZATION = 0.3847  # DEV #1 N1: 2,233,344 cached tokens over 3,239 paid requests x 1,792 (0.38477)
MIN_CACHEABLE_TOKENS = 1024  # the guide's minimum cacheable prefix


def _family_of(meta: Any, program: str, n_questions: Any) -> str:
    """The family part of the logged `prompt_cache_key` ("emu-v1:<16 hex>:<shard>"), else program and question count
    (runs without keys, such as DEV #1)."""
    import json as _json

    try:
        key = (_json.loads(meta) if isinstance(meta, str) and meta else dict(meta or {})).get("prompt_cache_key")
    except (TypeError, ValueError):
        key = None
    if key and key.count(":") >= 2:
        return key.split(":")[-2]
    return f"{program}/q{n_questions}"


def family_realization(requests, programs: tuple[str, ...] = ("N1",)) -> dict[str, dict]:
    """Per request family of `programs` over the PAID rows of a requests.parquet frame (`cached == False`): n, mean
    input tokens, cached tokens, the cacheable prefix (the most frequent cached-token count over hits, the provider's
    128-token-rounded prefix; for a family without a hit, its mean input rounded down to 128 if that reaches the
    1,024-token minimum, else 0) and realization = cached / (n x prefix) (None for a family that cannot cache)."""
    paid = requests[(~requests["cached"].astype(bool)) & requests["program"].isin(programs)]
    meta = paid["backend_meta"] if "backend_meta" in paid else [None] * len(paid)
    fams = [_family_of(m, p, q) for m, p, q in zip(meta, paid["program"], paid["n_questions"])]
    out: dict[str, dict] = {}
    for fam, g in paid.assign(family=fams).groupby("family"):
        hits = g.loc[g["cached_tokens"] > 0, "cached_tokens"]
        mean_in = float(g["input_tokens"].mean())
        prefix = int(hits.mode().iloc[0]) if len(hits) else int(mean_in // 128 * 128)
        if prefix < MIN_CACHEABLE_TOKENS:
            prefix = 0
        cached = int(g["cached_tokens"].sum())
        out[fam] = {"n": len(g), "program": ",".join(sorted(set(g["program"]))),
                    "questions": sorted({int(x) for x in g["n_questions"]}), "input_tokens_mean": round(mean_in, 1),
                    "cached_tokens": cached, "cacheable_prefix": prefix,
                    "realization": (cached / (len(g) * prefix)) if prefix else None,
                    "cached_share": cached / float(g["input_tokens"].sum()) if len(g) else 0.0}
    return out


def caching_decision(requests, programs: tuple[str, ...] = ("N1",), baseline: float = DEV1_REALIZATION) -> dict:
    """Rule O-5, re-based (DEV #2 audit): pool the families of `programs` that can cache (a prefix of >= 1,024
    tokens), weight each by its cacheable tokens (n x prefix), and keep the default routing (one prompt_cache_key per
    family) if the pooled realization is at least DEV #1's (no key) `baseline`; otherwise set
    `backends.llm-logprob.prompt_cache.enabled: false` (DEV #1's regime) before MAIN. Families that cannot cache
    (single-question requests) do not enter: no routing can change them."""
    fam = family_realization(requests, programs)
    cacheable = {k: v for k, v in fam.items() if v["cacheable_prefix"]}
    denom = sum(v["n"] * v["cacheable_prefix"] for v in cacheable.values())
    pooled = (sum(v["cached_tokens"] for v in cacheable.values()) / denom) if denom else None
    keep = pooled is not None and pooled >= baseline
    return {"families": fam, "pooled_realization": pooled, "baseline": baseline,
            "decision": "keep one prompt_cache_key per family" if keep else
            ("no cacheable family in these requests" if pooled is None else
             "set backends.llm-logprob.prompt_cache.enabled: false before MAIN")}


def main(argv: list[str] | None = None) -> int:
    """`python -m jevtrader.s1.prompt_cache <generation dir or requests.parquet> [program ...]`: rule O-5's inputs and
    decision as JSON."""
    import json as _json
    import sys
    from pathlib import Path

    import pandas as pd

    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(main.__doc__)
        return 2
    p = Path(args[0])
    req = pd.read_parquet(p / "requests.parquet" if p.is_dir() else p)
    print(_json.dumps(caching_decision(req, tuple(args[1:]) or ("N1",)), indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
