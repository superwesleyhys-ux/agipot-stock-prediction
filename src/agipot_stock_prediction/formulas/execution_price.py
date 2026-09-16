from __future__ import annotations

from ._validation import validated_inputs

from datetime import datetime, timedelta, timezone

from agipot_stock_prediction.research.contracts import finite_number

from .contracts import FormulaResult, result_from_blockers


@validated_inputs
def evaluate_execution_price(
    *,
    side: str,
    bid: float,
    ask: float,
    last_price: float,
    max_price_slippage_bps: float = 20.0,
    max_price_collar_pct: float = 0.02,
    max_order_lifetime_seconds: int = 60,
    now: datetime | None = None,
) -> FormulaResult:
    if max_price_slippage_bps < 0 or max_price_collar_pct < 0 or max_order_lifetime_seconds <= 0:
        raise ValueError("execution-price limits must be nonnegative and lifetime positive")
    bid_value = finite_number(bid, "execution_price.bid")
    ask_value = finite_number(ask, "execution_price.ask")
    last = finite_number(last_price, "execution_price.last_price")
    if bid_value <= 0 or ask_value <= 0 or last <= 0 or bid_value > ask_value:
        raise ValueError("execution prices must be positive and bid <= ask")
    side_key = side.upper()
    mid = (bid_value + ask_value) / 2.0
    if side_key == "BUY":
        limit_price = min(ask_value, mid * (1.0 + max_price_slippage_bps / 10_000.0))
    elif side_key == "SELL":
        limit_price = max(bid_value, mid * (1.0 - max_price_slippage_bps / 10_000.0))
    else:
        raise ValueError("side must be BUY or SELL")
    collar_distance = abs(limit_price / last - 1.0)
    blockers = [] if collar_distance <= max_price_collar_pct else ["price_collar_failed"]
    created_at = now or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    expires_at = created_at.astimezone(timezone.utc) + timedelta(seconds=max_order_lifetime_seconds)
    return result_from_blockers(
        score=1.0 if not blockers else 0.0,
        confidence=1.0,
        blockers=blockers,
        evidence={
            "side": side_key,
            "mid": mid,
            "limit_price": limit_price,
            "collar_distance": collar_distance,
            "order_type": "LIMIT",
            "expires_at": expires_at.isoformat(),
        },
    )


__all__ = ["evaluate_execution_price"]
