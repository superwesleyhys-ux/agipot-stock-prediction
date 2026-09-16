"""Construct a real River Regressor around the existing AGIPOT edge learner."""

from __future__ import annotations

from typing import Any, Mapping

from ._optional import finite_values, require


def make_adaptive_edge_regressor(*, decay: float = 0.995, min_samples: int = 240,
                                 edge_floor_bps: float = 3.0) -> Any:
    """Return a River estimator with predict_one/learn_one, measured in basis points.

    ``learn_one`` has no clock; DelayedLabelReplayHarness must release matured
    labels. Calling learn_one directly is the caller's explicit update request.
    """
    base = require("river.base", "river")
    from agipot_stock_prediction.intraday.signal_library import AdaptiveEdgeModel

    class AdaptiveEdgeRegressor(base.Regressor):
        def __init__(self, decay: float = 0.995, min_samples: int = 240,
                     edge_floor_bps: float = 3.0) -> None:
            self.decay = decay
            self.min_samples = min_samples
            self.edge_floor_bps = edge_floor_bps
            self.model = AdaptiveEdgeModel(decay=decay, min_samples=min_samples, edge_floor_bps=edge_floor_bps)

        def predict_one(self, x: Mapping[str, float]) -> float:
            return self.model.edge_bps(x)

        def learn_one(self, x: Mapping[str, float], y: float) -> "AdaptiveEdgeRegressor":
            label = finite_values([y], name="matured return in basis points")[0]
            self.model.learn(x, realized_return_bps=label)
            return self

    return AdaptiveEdgeRegressor(decay=decay, min_samples=min_samples, edge_floor_bps=edge_floor_bps)
