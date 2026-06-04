"""GPT-5.2 rate card and per-call cost computation.

Rates as of 2026-06 for the standard (non-Flex, non-Batch, non-Priority)
tier. Update when OpenAI publishes new pricing; the calc_calls.py and
runner both call into here so a single update propagates.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RateCard:
    """USD per million tokens for one model + tier combination."""
    model: str
    input_per_million: float
    cached_input_per_million: float
    output_per_million: float


# Standard-tier rates, gpt-5.2, 2026-06.
GPT_5_2_STANDARD = RateCard(
    model="gpt-5.2",
    input_per_million=1.75,
    cached_input_per_million=0.175,   # 0.1x of input — OpenAI prompt cache read
    output_per_million=14.00,
)


def cost_for_call(
    rate: RateCard,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
) -> float:
    """Compute the USD cost of one chat.completions call.

    Splits prompt_tokens into the cached and fresh portions:
        fresh = prompt_tokens - cached_tokens
        bill  = fresh * input_rate + cached * cached_rate + output * output_rate
    All three rates are per million tokens.
    """
    fresh = max(0, prompt_tokens - cached_tokens)
    return (
        fresh * rate.input_per_million / 1_000_000
        + cached_tokens * rate.cached_input_per_million / 1_000_000
        + completion_tokens * rate.output_per_million / 1_000_000
    )


def monthly_extrapolation(per_cycle_usd: float, cycles_per_month: int) -> float:
    """Trivial helper: per-cycle × cycles → monthly. Kept here so the
    'how much would N users at M cycles/day cost' arithmetic lives in
    one place instead of being inlined in the CLI."""
    return per_cycle_usd * cycles_per_month
