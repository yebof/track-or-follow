"""USD per 1M tokens, standard tier.

Sources (archived): Jev -> docs/jev/pages/models (output tokens free); OpenAI -> docs/pricing/
openai_pricing_2026-09-25.html and the per-model pages docs/pricing/openai_model_<model>.html (cached input).
Pinned snapshots carry the prices of their alias (F16): the emulator is pinned to gpt-4.1-mini-2025-04-14.
"""

PRICES: dict[str, tuple[float, float]] = {  # (input, output)
    "jev-1.13.0": (0.042, 0.0),
    "jev-latest": (0.042, 0.0),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
}

# Provider prompt-cache hits (`usage.prompt_tokens_details.cached_tokens`), from the per-model pages.
CACHED_INPUT: dict[str, float] = {
    "gpt-4.1-mini": 0.10,
    "gpt-4.1-mini-2025-04-14": 0.10,
    "gpt-4.1-nano": 0.025,
    "gpt-5.4-mini": 0.075,
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """`input_tokens` includes the cached ones (OpenAI's `prompt_tokens` convention). A model without an
    archived cached price pays the full input price for cached tokens (conservative)."""
    if model not in PRICES:
        raise KeyError(f"no price for {model}; add it to jevtrader.s1.pricing with a source")
    pi, po = PRICES[model]
    cached = max(0, min(int(cached_tokens), int(input_tokens)))
    pc = CACHED_INPUT.get(model, pi)
    return ((input_tokens - cached) * pi + cached * pc + output_tokens * po) / 1e6
