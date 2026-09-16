from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping


def _validate_features(features: Mapping[str, float]) -> None:
    for key, value in features.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"feature {key!r} must be a finite number")


def zscore(value: float, scale: float = 0.0025) -> float:
    if scale <= 0 or not math.isfinite(value):
        return 0.0
    return max(-5.0, min(5.0, value / scale))


def intraday_momentum_score(features: Mapping[str, float]) -> float:
    return 0.25 * zscore(features.get("ret_5m", 0.0)) + 0.35 * zscore(features.get("ret_15m", 0.0)) + 0.40 * zscore(features.get("ret_60m", 0.0))


def vwap_mean_reversion_score(features: Mapping[str, float]) -> float:
    vol = max(features.get("rolling_vol_60m", 0.0), 0.0005)
    return -zscore(features.get("vwap_deviation", 0.0), vol)


def volume_confirmation_score(features: Mapping[str, float]) -> float:
    return math.log(1.0 + max(0.0, features.get("relative_volume", 0.0)))


def spread_penalty(features: Mapping[str, float], *, max_spread_bps: float) -> float:
    return min(1.0, max(0.0, features.get("spread_bps", 0.0) / max(max_spread_bps, 0.01)))


def long_short_alpha(features: Mapping[str, float], *, max_spread_bps: float = 10.0) -> dict[str, float]:
    _validate_features(features)
    momentum = intraday_momentum_score(features)
    mean_reversion = vwap_mean_reversion_score(features)
    volume_score = volume_confirmation_score(features)
    penalty = spread_penalty(features, max_spread_bps=max_spread_bps)
    long_alpha = 0.45 * momentum + 0.25 * mean_reversion + 0.20 * volume_score - 0.10 * penalty
    short_alpha = -0.45 * momentum - 0.25 * mean_reversion + 0.20 * volume_score - 0.10 * penalty
    return {
        "momentum_score": momentum,
        "mean_reversion_score": mean_reversion,
        "volume_score": volume_score,
        "spread_penalty": penalty,
        "long_alpha": long_alpha if math.isfinite(long_alpha) else 0.0,
        "short_alpha": short_alpha if math.isfinite(short_alpha) else 0.0,
    }


def family_alpha(strategy_family: str, features: Mapping[str, float], *, max_spread_bps: float = 10.0) -> dict[str, float]:
    _validate_features(features)
    momentum = intraday_momentum_score(features)
    mean_reversion = vwap_mean_reversion_score(features)
    volume_score = volume_confirmation_score(features)
    penalty = spread_penalty(features, max_spread_bps=max_spread_bps)
    gap = zscore(features.get("gap_from_open", 0.0))
    ret_5m = zscore(features.get("ret_5m", 0.0))
    ret_15m = zscore(features.get("ret_15m", 0.0))
    ret_30m = zscore(features.get("ret_30m", 0.0))
    vol_15m = max(0.0, min(3.0, features.get("rolling_vol_15m", 0.0) / 0.0025))
    distance_to_high = zscore(-features.get("distance_to_high", 0.0), 0.0025)
    distance_to_low = zscore(features.get("distance_to_low", 0.0), 0.0025)
    book_quality = max(0.0, min(1.0, features.get("order_book_proxy", 0.0)))
    news_score = max(-1.0, min(1.0, features.get("news_score", 0.0)))

    if strategy_family == "intraday_momentum":
        long_alpha = 0.70 * momentum + 0.25 * volume_score - 0.15 * penalty
        short_alpha = -0.70 * momentum + 0.25 * volume_score - 0.15 * penalty
    elif strategy_family == "vwap_mean_reversion":
        long_alpha = 0.75 * mean_reversion + 0.20 * volume_score - 0.10 * penalty
        short_alpha = -0.75 * mean_reversion + 0.20 * volume_score - 0.10 * penalty
    elif strategy_family == "opening_range_breakout":
        long_alpha = 0.40 * gap + 0.35 * distance_to_high + 0.25 * volume_score - 0.15 * penalty
        short_alpha = -0.40 * gap + 0.35 * distance_to_low + 0.25 * volume_score - 0.15 * penalty
    elif strategy_family == "volatility_breakout":
        long_alpha = 0.45 * ret_15m + 0.30 * vol_15m + 0.25 * volume_score - 0.20 * penalty
        short_alpha = -0.45 * ret_15m + 0.30 * vol_15m + 0.25 * volume_score - 0.20 * penalty
    elif strategy_family == "news_pool_momentum":
        long_alpha = 0.45 * news_score + 0.35 * momentum + 0.20 * volume_score - 0.15 * penalty
        short_alpha = -0.45 * news_score - 0.35 * momentum + 0.20 * volume_score - 0.15 * penalty
    elif strategy_family == "news_pool_reversal":
        long_alpha = -0.45 * news_score + 0.35 * mean_reversion + 0.20 * volume_score - 0.15 * penalty
        short_alpha = 0.45 * news_score - 0.35 * mean_reversion + 0.20 * volume_score - 0.15 * penalty
    elif strategy_family == "gap_fade":
        long_alpha = -0.55 * gap + 0.35 * mean_reversion + 0.20 * volume_score - 0.15 * penalty
        short_alpha = 0.55 * gap - 0.35 * mean_reversion + 0.20 * volume_score - 0.15 * penalty
    elif strategy_family == "liquidity_reversion":
        long_alpha = 0.45 * mean_reversion + 0.35 * book_quality + 0.20 * volume_score - 0.15 * penalty
        short_alpha = -0.45 * mean_reversion + 0.35 * book_quality + 0.20 * volume_score - 0.15 * penalty
    elif strategy_family == "micro_trend_pullback":
        long_alpha = 0.55 * ret_30m - 0.30 * ret_5m + 0.25 * volume_score - 0.15 * penalty
        short_alpha = -0.55 * ret_30m + 0.30 * ret_5m + 0.25 * volume_score - 0.15 * penalty
    elif strategy_family == "high_volume_breakout_retest":
        long_alpha = 0.35 * distance_to_high + 0.35 * volume_score + 0.20 * ret_5m - 0.15 * penalty
        short_alpha = 0.35 * distance_to_low + 0.35 * volume_score - 0.20 * ret_5m - 0.15 * penalty
    else:
        return long_short_alpha(features, max_spread_bps=max_spread_bps)

    return {
        "momentum_score": momentum,
        "mean_reversion_score": mean_reversion,
        "volume_score": volume_score,
        "spread_penalty": penalty,
        "long_alpha": long_alpha if math.isfinite(long_alpha) else 0.0,
        "short_alpha": short_alpha if math.isfinite(short_alpha) else 0.0,
    }


def cross_section_alpha(
    strategy_family: str,
    symbol: str,
    features: Mapping[str, float],
    peer_features: Mapping[str, Mapping[str, float]],
    *,
    max_spread_bps: float = 10.0,
) -> dict[str, float]:
    _validate_features(features)
    for peer in peer_features.values():
        _validate_features(peer)
    peers = {
        peer_symbol: values
        for peer_symbol, values in peer_features.items()
        if peer_symbol != symbol and values
    }
    if not peers:
        return family_alpha(strategy_family, features, max_spread_bps=max_spread_bps)

    own_ret_5m = features.get("ret_5m", 0.0)
    own_ret_15m = features.get("ret_15m", 0.0)
    own_ret_30m = features.get("ret_30m", 0.0)
    peer_ret_5m = sum(values.get("ret_5m", 0.0) for values in peers.values()) / len(peers)
    peer_ret_15m = sum(values.get("ret_15m", 0.0) for values in peers.values()) / len(peers)
    peer_ret_30m = sum(values.get("ret_30m", 0.0) for values in peers.values()) / len(peers)
    own_vwap = features.get("vwap_deviation", 0.0)
    peer_vwap = sum(values.get("vwap_deviation", 0.0) for values in peers.values()) / len(peers)
    volume_score = volume_confirmation_score(features)
    penalty = spread_penalty(features, max_spread_bps=max_spread_bps)
    relative_momentum = (
        0.50 * zscore(own_ret_5m - peer_ret_5m)
        + 0.35 * zscore(own_ret_15m - peer_ret_15m)
        + 0.15 * zscore(own_ret_30m - peer_ret_30m)
    )
    relative_vwap = zscore(own_vwap - peer_vwap)

    if strategy_family == "intraday_pair_stat_arb":
        long_alpha = -0.70 * relative_vwap - 0.20 * relative_momentum + 0.20 * volume_score - 0.15 * penalty
        short_alpha = 0.70 * relative_vwap + 0.20 * relative_momentum + 0.20 * volume_score - 0.15 * penalty
    elif strategy_family == "market_neutral_basket":
        long_alpha = 0.75 * relative_momentum + 0.20 * volume_score - 0.15 * penalty
        short_alpha = -0.75 * relative_momentum + 0.20 * volume_score - 0.15 * penalty
    else:
        return family_alpha(strategy_family, features, max_spread_bps=max_spread_bps)

    return {
        "momentum_score": relative_momentum,
        "mean_reversion_score": -relative_vwap,
        "volume_score": volume_score,
        "spread_penalty": penalty,
        "long_alpha": long_alpha if math.isfinite(long_alpha) else 0.0,
        "short_alpha": short_alpha if math.isfinite(short_alpha) else 0.0,
    }


def adaptive_edge_feature_vector(features: Mapping[str, float]) -> dict[str, float]:
    _validate_features(features)
    return {
        "ret_1m": zscore(features.get("ret_1m", 0.0)),
        "ret_5m": zscore(features.get("ret_5m", 0.0)),
        "ret_15m": zscore(features.get("ret_15m", 0.0)),
        "ret_30m": zscore(features.get("ret_30m", 0.0)),
        "ret_60m": zscore(features.get("ret_60m", 0.0)),
        "vwap_deviation": zscore(features.get("vwap_deviation", 0.0)),
        "gap_from_open": zscore(features.get("gap_from_open", 0.0)),
        "distance_to_high": zscore(-features.get("distance_to_high", 0.0)),
        "distance_to_low": zscore(features.get("distance_to_low", 0.0)),
        "intraday_drawdown": zscore(features.get("intraday_drawdown", 0.0)),
        "relative_volume": max(0.0, min(3.0, math.log1p(max(0.0, features.get("relative_volume", 0.0))))),
        "rolling_vol_15m": max(0.0, min(3.0, features.get("rolling_vol_15m", 0.0) / 0.0025)),
        "rolling_vol_60m": max(0.0, min(3.0, features.get("rolling_vol_60m", 0.0) / 0.0025)),
        "spread_penalty": -max(0.0, min(3.0, features.get("spread_bps", 0.0) / 5.0)),
        "news_score": max(-1.0, min(1.0, features.get("news_score", 0.0))),
    }


class AdaptiveEdgeModel:
    """Online feature-edge learner trained only from completed historical bars."""

    def __init__(self, *, decay: float = 0.995, min_samples: int = 240, edge_floor_bps: float = 3.0) -> None:
        if not math.isfinite(decay) or not 0.90 <= decay <= 0.9999:
            raise ValueError("decay must be finite and in [0.90, 0.9999]")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a positive integer")
        if not math.isfinite(edge_floor_bps) or edge_floor_bps < 0:
            raise ValueError("edge_floor_bps must be finite and nonnegative")
        self.decay = min(0.9999, max(0.90, decay))
        self.min_samples = max(1, int(min_samples))
        self.edge_floor_bps = max(0.0, float(edge_floor_bps))
        self.samples = 0
        self._normalizer = 0.0
        self._weights: dict[str, float] = defaultdict(float)

    def learn(self, features: Mapping[str, float], *, realized_return_bps: float) -> None:
        if not math.isfinite(realized_return_bps):
            return
        capped_return = max(-75.0, min(75.0, realized_return_bps))
        vector = adaptive_edge_feature_vector(features)
        self.samples += 1
        self._normalizer = self.decay * self._normalizer + 1.0
        for key in tuple(self._weights):
            self._weights[key] *= self.decay
        for key, value in vector.items():
            self._weights[key] += value * capped_return

    def edge_bps(self, features: Mapping[str, float]) -> float:
        _validate_features(features)
        if self.samples < self.min_samples or self._normalizer <= 0:
            return 0.0
        vector = adaptive_edge_feature_vector(features)
        raw_edge = sum((self._weights.get(key, 0.0) / self._normalizer) * value for key, value in vector.items())
        scale = sum(abs(self._weights.get(key, 0.0) / self._normalizer) for key in vector) or 1.0
        edge = raw_edge / max(1.0, math.sqrt(scale))
        if abs(edge) < self.edge_floor_bps:
            return 0.0
        return max(-50.0, min(50.0, edge if math.isfinite(edge) else 0.0))


def adaptive_edge_alpha(model: AdaptiveEdgeModel, features: Mapping[str, float]) -> dict[str, float]:
    edge = model.edge_bps(features)
    long_alpha = max(0.0, edge / 10.0)
    short_alpha = max(0.0, -edge / 10.0)
    return {
        "momentum_score": edge / 10.0,
        "mean_reversion_score": 0.0,
        "volume_score": volume_confirmation_score(features),
        "spread_penalty": spread_penalty(features, max_spread_bps=10.0),
        "long_alpha": long_alpha if math.isfinite(long_alpha) else 0.0,
        "short_alpha": short_alpha if math.isfinite(short_alpha) else 0.0,
    }


def objective_score(metrics: Mapping[str, float]) -> float:
    # This intentionally optimizes risk-adjusted out-of-sample quality, not a hit_2_percent flag.
    ret = metrics.get("geometric_daily_return", 0.0)
    dd = max(metrics.get("max_drawdown_pct", 100.0), 0.01)
    fill = metrics.get("fill_rate", 0.0)
    stability = metrics.get("regime_stability", 0.0)
    return ret / (dd / 100.0) + 0.1 * fill + 0.1 * stability


def reject_forbidden_objective(objective: str) -> tuple[str, ...]:
    forbidden = {"hit_2_percent_daily_return", "last_year_return", "mandate_passed_only"}
    return ("forbidden_optimizer_objective",) if objective in forbidden else ()
