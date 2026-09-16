"""Deterministic ranking of complete supplied research metrics."""

from __future__ import annotations

from typing import Mapping, Sequence

from agipot_stock_prediction.research.contracts import finite_number


def rank_trials(trials: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    def key(item: Mapping[str, object]) -> tuple[bool, float, float, float]:
        if type(item.get("promotion_allowed")) is not bool:
            raise ValueError("promotion_allowed must be present and boolean")
        metrics = {}
        for name in ("geometric_daily_return", "max_drawdown_pct", "fill_rate"):
            if name not in item:
                raise ValueError(f"trial metric {name} is required")
            metrics[name] = finite_number(item[name], f"trial.{name}")
        if not 0 <= metrics["max_drawdown_pct"] <= 100 or not 0 <= metrics["fill_rate"] <= 1:
            raise ValueError("drawdown must be in [0, 100] percent and fill rate in [0, 1]")
        return (item["promotion_allowed"], metrics["geometric_daily_return"],
                -metrics["max_drawdown_pct"], metrics["fill_rate"])

    return tuple(sorted(trials, key=key, reverse=True))


__all__ = ["rank_trials"]
