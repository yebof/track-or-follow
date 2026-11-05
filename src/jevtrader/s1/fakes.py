"""Mock HTTP transports for the OpenAI and TypeSafe (Jev) SDKs, and the wire-level request parsers.

They back the mock-transport contract test (design_v1 §11.3, F07): programs send identical `(state,
questions)` to every backend, and what reaches the wire can be decoded back to the same pair for both the
emulator prompt and the Jev request body. No network, no keys (the dummy keys below are not secrets).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

import httpx2

from jevtrader.s1.types import Question

DUMMY_KEY = "test-key-not-a-secret"


# ------------------------------------------------------------------------------------ OpenAI side
def chat_completion(
    content_tokens: list[tuple[str, list[tuple[str, float]]]],
    model: str,
    usage: tuple[int, int, int] | None = (1000, 20, 0),
) -> dict:
    """A chat.completion body. `content_tokens`: (token, [(candidate, logprob), ...]); `usage`:
    (prompt_tokens, completion_tokens, cached_tokens) or None for a response without usage."""
    text = "".join(t for t, _ in content_tokens)
    lp = [{"token": t, "logprob": (c[0][1] if c else 0.0), "bytes": list(t.encode()),
           "top_logprobs": [{"token": k, "logprob": v, "bytes": list(k.encode())} for k, v in c]}
          for t, c in content_tokens]
    body = {
        "id": "chatcmpl-fake", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text},
                     "logprobs": {"content": lp}}],
    }
    if usage is not None:
        p, c, cached = usage
        body["usage"] = {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c,
                         "prompt_tokens_details": {"cached_tokens": cached}}
    return body


def letter_answers(dists: list[Mapping[str, float] | None]) -> list[tuple[str, list[tuple[str, float]]]]:
    """Content tokens for 'q1=<top>\\nq2=<top>...' with the given letter distributions at each answer slot.
    A None entry emits a malformed line (the emulator must treat it as degraded)."""
    toks: list[tuple[str, list[tuple[str, float]]]] = []
    for n, d in enumerate(dists, 1):
        if n > 1:
            toks.append(("\n", []))
        if d is None:
            toks.append((f"q{n}", []))
            toks.append((":?", []))
            continue
        top = max(d, key=d.get)
        toks += [(f"q{n}", []), ("=", []), (top, [(k, math.log(v)) for k, v in d.items() if v > 0])]
    return toks


def openai_client(handler: Callable[[dict], dict], calls: list | None = None):
    """AsyncOpenAI whose POST /chat/completions is answered by `handler(request_json) -> response_json`."""
    from openai import AsyncOpenAI

    def route(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append(body)
        return httpx2.Response(200, json=handler(body))

    return AsyncOpenAI(api_key=DUMMY_KEY, base_url="http://fake-openai/v1", max_retries=0,
                       http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(route)))


def parse_emu_messages(messages: list[dict]) -> tuple[Any, list[tuple]]:
    """Decode an emu-v1 / verb-v1 chat request back into (state, canonical questions in render order)."""
    user = messages[-1]["content"]
    q_part = user.split("QUESTIONS:\n", 1)[1]
    q_part = q_part.split("\nReply with", 1)[0].split("\nFor every question give", 1)[0]
    state_txt = user.split("\n\nSTATE:\n", 1)[1].rsplit("\n\nAnswer now.", 1)[0]
    state = json.loads(state_txt)
    out = []
    for block in [b for b in q_part.strip("\n").split("\n\n") if b.strip()]:
        lines = block.split("\n")
        ins_txt = lines[0].split("] ", 1)[1]
        ins = json.loads(ins_txt) if ins_txt.startswith(("{", "[")) else ins_txt
        opts = []
        for ln in lines[1:]:
            body = ln.strip().split(": ", 1)[1]
            label, _, desc = body.partition(" -- ")
            opts.append((label, json.loads(desc) if desc[:1] in "{[" and desc else (desc or None)))
        labels = [o[0] for o in opts]
        if labels == ["yes", "no"]:
            out.append(("noul", ins, {"true": opts[0][1], "false": opts[1][1]}))
        elif all(lab == f"level {i}" for i, lab in enumerate(labels)):
            out.append(("score", ins, [d for _, d in opts]))
        else:
            out.append(("choice", ins, dict(opts)))
    return state, out


# ------------------------------------------------------------------------------------ Jev side
def jev_answer(q: Mapping[str, Any], p: float = 0.8) -> dict:
    """A plausible wire answer for one Jev question dict (as sent by the SDK)."""
    if q["type"] == "noul":
        return {"type": "noul", "noul": p}
    crit = q["criteria"]
    if q["type"] == "choice":
        labs = list(crit)
        rest = (1 - p) / max(1, len(labs) - 1)
        probs = {lab: (p if i == 0 else rest) for i, lab in enumerate(labs)}
        return {"type": "choice", "choice": labs[0], "confidence": p, "probabilities": probs}
    n = len(crit)
    rest = (1 - p) / max(1, n - 1)
    probs = {str(i): (p if i == n - 1 else rest) for i in range(n)}
    return {"type": "score", "score": sum(i * v for i, v in enumerate(probs.values())), "confidence": p,
            "legend": {str(i): c for i, c in enumerate(crit)}, "probabilities": probs}


def jev_transport(
    model_served: str = "jev-1.13.0",
    calls: list | None = None,
    fail_first: int = 0,
    usage: dict | None = None,
    models: list[dict] | None = None,
) -> httpx2.MockTransport:
    """Fake TypeSafe API: GET /v1/models and POST /v1/systemone. The first `fail_first` POSTs get 429."""
    state = {"posts": 0}
    models = models or [{"name": "jev-latest", "description": "General-purpose system one model.", "release_date": "2026-09-15"}]

    def route(request: httpx2.Request) -> httpx2.Response:
        if calls is not None:
            calls.append((request.method, request.url.path, json.loads(request.content) if request.content else None))
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx2.Response(200, json={"models": models})
        if request.method == "POST" and request.url.path == "/v1/systemone":
            state["posts"] += 1
            if state["posts"] <= fail_first:
                return httpx2.Response(429, headers={"retry-after-ms": "1"}, json={"detail": "rate limited"})
            body = json.loads(request.content)
            answers = {name: jev_answer(q) for name, q in body["questions"].items()}
            return httpx2.Response(200, json={"model": model_served, "answers": answers,
                                              "usage": usage if usage is not None else {"input_tokens": 500, "output_tokens": 12}})
        return httpx2.Response(404, json={"detail": "not found"})

    return httpx2.MockTransport(route)


def parse_jev_body(body: Mapping[str, Any]) -> tuple[Any, list[tuple]]:
    """Decode a POST /v1/systemone body into (state, canonical questions in order)."""
    out = []
    for q in body["questions"].values():
        if q["type"] == "noul":
            crit = q.get("criteria") or {}
            out.append(("noul", q.get("instructions"), {"true": crit.get("true"), "false": crit.get("false")}))
        elif q["type"] == "choice":
            out.append(("choice", q.get("instructions"), dict(q["criteria"])))
        else:
            out.append(("score", q.get("instructions"), list(q["criteria"])))
    return body["state"], out


def canonical_questions(questions: Mapping[str, Question]) -> list[tuple]:
    """Our question objects in the same canonical form as the two parsers."""
    out = []
    for q in questions.values():
        if q.kind == "noul":
            out.append(("noul", q.instructions, {"true": q.true, "false": q.false}))
        elif q.kind == "choice":
            out.append(("choice", q.instructions, dict(q.options)))
        else:
            out.append(("score", q.instructions, list(q.levels)))
    return out
