from __future__ import annotations

from ._validation import validated_inputs

from typing import Sequence

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import EPSILON, FormulaResult, clamp, median, percentile, result_from_blockers, safe_div


@validated_inputs
def evaluate_liquidity(
    *,
    close: float,
    volume_history: Sequence[float],
    bid: float,
    ask: float,
    order_notional: float = 0.0,
    quote_age_seconds: float = 0.0,
    max_quote_age_seconds: float = 60.0,
    min_median_adv: float = 5_000_000.0,
    min_p20_adv: float = 2_000_000.0,
    max_entry_spread_bps: float = 35.0,
    max_participation_rate: float = 0.01,
) -> FormulaResult:
    price = finite_number(close, "liquidity.close")
    volumes = tuple(finite_number(value, "liquidity.volume") for value in volume_history)
    bid_value = finite_number(bid, "liquidity.bid")
    ask_value = finite_number(ask, "liquidity.ask")
    notional = finite_number(order_notional, "liquidity.order_notional")
    if notional < 0 or any(volume < 0 for volume in volumes):
        raise ValueError("liquidity notional and volume cannot be negative")
    if min(min_median_adv, min_p20_adv, max_entry_spread_bps, max_participation_rate) < 0:
        raise ValueError("liquidity thresholds cannot be negative")
    if price <= 0 or bid_value <= 0 or ask_value <= 0:
        raise ValueError("liquidity prices must be positive")
    inverted_quote = ask_value < bid_value
    dollar_volumes = tuple(price * max(0.0, volume) for volume in volumes)
    median_adv = median(dollar_volumes)
    p20_adv = percentile(dollar_volumes, 20)
    mid = (bid_value + ask_value) / 2.0
    spread_bps = safe_div(abs(ask_value - bid_value) * 10_000.0, mid, default=10_000.0)
    participation_rate = safe_div(notional, median_adv, default=0.0)
    quote_age = finite_number(quote_age_seconds, "liquidity.quote_age_seconds")
    max_age = finite_number(max_quote_age_seconds, "liquidity.max_quote_age_seconds")
    if quote_age < 0 or max_age < 0:
        raise ValueError("liquidity quote ages cannot be negative")
    liquidity_multiplier = min(1.0, safe_div(max_participation_rate, max(participation_rate, EPSILON), default=1.0))
    spread_multiplier = clamp(1.0 - safe_div(spread_bps, max_entry_spread_bps, default=1.0), 0.0, 1.0)
    fill_probability = clamp(
        1.0
        - 0.25 * safe_div(spread_bps, 100.0, default=0.0)
        - 2.0 * participation_rate
        - 0.25 * safe_div(quote_age, max_age, default=1.0),
        0.0,
        1.0,
    )
    blockers: list[str] = []
    if not volumes or median_adv <= 0:
        blockers.append("missing_liquidity_volume")
    if inverted_quote:
        blockers.append("inverted_quote")
    if median_adv < min_median_adv:
        blockers.append("median_adv_below_min")
    if p20_adv < min_p20_adv:
        blockers.append("p20_adv_below_min")
    if spread_bps > max_entry_spread_bps:
        blockers.append("spread_above_entry_limit")
    if participation_rate > max_participation_rate:
        blockers.append("participation_above_limit")
    if quote_age > max_age:
        blockers.append("quote_stale")
    score = liquidity_multiplier * spread_multiplier * fill_probability
    return result_from_blockers(
        score=score,
        confidence=1.0 if volumes else 0.0,
        blockers=blockers,
        evidence={
            "median_adv": median_adv,
            "p20_adv": p20_adv,
            "spread_bps": spread_bps,
            "inverted_quote": inverted_quote,
            "participation_rate": participation_rate,
            "liquidity_multiplier": liquidity_multiplier,
            "spread_multiplier": spread_multiplier,
            "fill_probability": fill_probability,
        },
    )


__all__ = ["evaluate_liquidity"]
