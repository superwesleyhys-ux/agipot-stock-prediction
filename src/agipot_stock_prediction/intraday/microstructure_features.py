from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Sequence

from .contracts import IntradayBar


def _ret(bars: Sequence[IntradayBar], lookback: int) -> float:
    if len(bars) <= lookback or bars[-lookback - 1].close <= 0:
        return 0.0
    return bars[-1].close / bars[-lookback - 1].close - 1.0


def _rolling_vol(bars: Sequence[IntradayBar], lookback: int) -> float:
    if len(bars) <= lookback:
        return 0.0
    returns = [
        current.close / previous.close - 1.0
        for previous, current in zip(bars[-lookback - 1 : -1], bars[-lookback:])
        if previous.close > 0
    ]
    return statistics.pstdev(returns) if len(returns) > 1 else 0.0


def _rolling_vol_from_closes(closes: Sequence[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    window = list(closes)[-lookback - 1 :]
    returns = [
        current / previous - 1.0
        for previous, current in zip(window[:-1], window[1:])
        if previous > 0
    ]
    if len(returns) <= 1:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((item - mean) ** 2 for item in returns) / len(returns)
    return math.sqrt(max(variance, 0.0))


def _session_vwap(bars: Sequence[IntradayBar]) -> float:
    current_session = bars[-1].session
    session_bars: list[IntradayBar] = []
    for bar in reversed(bars):
        if bar.session != current_session:
            break
        session_bars.append(bar)
    session_bars.reverse()
    dollar_volume = sum(bar.close * bar.volume for bar in session_bars)
    volume = sum(bar.volume for bar in session_bars)
    return dollar_volume / volume if volume > 0 else bars[-1].close


def _spread_bps(bar: IntradayBar, synthetic_min_spread_bps: float = 1.0) -> float:
    return max(synthetic_min_spread_bps, float(bar.spread_bps)) if bar.spread_bps is not None else synthetic_min_spread_bps


def compute_microstructure_features(
    bars: Sequence[IntradayBar],
    *,
    same_time_median_volume_20d: float | None = None,
    synthetic_min_spread_bps: float = 1.0,
) -> dict[str, float]:
    if not bars:
        return {}
    if len({item.symbol for item in bars}) != 1:
        raise ValueError("feature history must contain one symbol")
    if any(b.timestamp <= a.timestamp for a, b in zip(bars, bars[1:])):
        raise ValueError("bars must be in strictly increasing timestamp order")
    bar = bars[-1]
    vwap = _session_vwap(bars)
    session_bars: list[IntradayBar] = []
    for item in reversed(bars):
        if item.session != bar.session:
            break
        session_bars.append(item)
    session_bars.reverse()
    high_today = max(item.high for item in session_bars)
    low_today = min(item.low for item in session_bars)
    max_close_today = max(item.close for item in session_bars)
    volume_last_15m = sum(item.volume for item in bars[-15:])
    baseline_volume = same_time_median_volume_20d or max(statistics.median([item.volume for item in bars[-60:]] or [bar.volume]), 1.0)
    features = {
        "ret_1m": _ret(bars, 1),
        "ret_5m": _ret(bars, 5),
        "ret_15m": _ret(bars, 15),
        "ret_30m": _ret(bars, 30),
        "ret_60m": _ret(bars, 60),
        "rolling_vol_15m": _rolling_vol(bars, 15),
        "rolling_vol_60m": _rolling_vol(bars, 60),
        "session_vwap": vwap,
        "vwap_deviation": bar.close / vwap - 1.0 if vwap > 0 else 0.0,
        "spread_bps": _spread_bps(bar, synthetic_min_spread_bps),
        "relative_volume": volume_last_15m / max(baseline_volume, 1.0),
        "order_book_proxy": max(0.0, 1.0 - _spread_bps(bar, synthetic_min_spread_bps) / 50.0),
        "intraday_drawdown": bar.close / max_close_today - 1.0 if max_close_today > 0 else 0.0,
        "gap_from_open": bar.close / session_bars[0].open - 1.0 if session_bars and session_bars[0].open > 0 else 0.0,
        "distance_to_high": bar.close / high_today - 1.0 if high_today > 0 else 0.0,
        "distance_to_low": bar.close / low_today - 1.0 if low_today > 0 else 0.0,
        "news_score": bar.news_score or 0.0,
    }
    return {key: value if math.isfinite(value) else 0.0 for key, value in features.items()}


class MicrostructureFeatureState:
    """Incremental equivalent of compute_microstructure_features for large runs."""

    def __init__(self) -> None:
        self._closes: deque[float] = deque(maxlen=61)
        self._returns: deque[float] = deque(maxlen=60)
        self._volumes: deque[float] = deque(maxlen=60)
        self._volume_sum = 0.0
        self._session = None
        self._session_open = 0.0
        self._session_high = 0.0
        self._session_low = math.inf
        self._session_max_close = 0.0
        self._session_dollar_volume = 0.0
        self._session_volume = 0.0
        self._last_timestamp = None
        self._symbol = None

    def update(
        self,
        bar: IntradayBar,
        *,
        same_time_median_volume_20d: float | None = None,
        synthetic_min_spread_bps: float = 1.0,
    ) -> dict[str, float]:
        if self._symbol is not None and bar.symbol != self._symbol:
            raise ValueError("feature state must contain one symbol")
        if self._last_timestamp is not None and bar.timestamp <= self._last_timestamp:
            raise ValueError("bars must be in strictly increasing timestamp order")
        self._symbol = bar.symbol
        self._last_timestamp = bar.timestamp
        if self._session != bar.session:
            self._session = bar.session
            self._session_open = bar.open
            self._session_high = bar.high
            self._session_low = bar.low
            self._session_max_close = bar.close
            self._session_dollar_volume = 0.0
            self._session_volume = 0.0
        else:
            self._session_high = max(self._session_high, bar.high)
            self._session_low = min(self._session_low, bar.low)
            self._session_max_close = max(self._session_max_close, bar.close)

        previous_close = self._closes[-1] if self._closes else None
        self._session_dollar_volume += bar.close * bar.volume
        self._session_volume += bar.volume
        if previous_close and previous_close > 0:
            self._returns.append(bar.close / previous_close - 1.0)
        if len(self._volumes) == self._volumes.maxlen:
            self._volume_sum -= self._volumes[0]
        self._closes.append(bar.close)
        self._volume_sum += bar.volume
        self._volumes.append(bar.volume)

        vwap = self._session_dollar_volume / self._session_volume if self._session_volume > 0 else bar.close
        volume_last_15m = sum(list(self._volumes)[-15:])
        baseline_volume = same_time_median_volume_20d or max(statistics.median(self._volumes), 1.0)
        spread = _spread_bps(bar, synthetic_min_spread_bps)

        def ret(lookback: int) -> float:
            if len(self._closes) <= lookback or self._closes[-lookback - 1] <= 0:
                return 0.0
            return self._closes[-1] / self._closes[-lookback - 1] - 1.0

        def rolling_vol(lookback: int) -> float:
            if len(self._returns) < lookback:
                return 0.0
            window = list(self._returns)[-lookback:]
            if len(window) <= 1:
                return 0.0
            mean = sum(window) / len(window)
            variance = sum((item - mean) ** 2 for item in window) / len(window)
            return math.sqrt(max(variance, 0.0))

        features = {
            "ret_1m": ret(1),
            "ret_5m": ret(5),
            "ret_15m": ret(15),
            "ret_30m": ret(30),
            "ret_60m": ret(60),
            "rolling_vol_15m": rolling_vol(15),
            "rolling_vol_60m": rolling_vol(60),
            "session_vwap": vwap,
            "vwap_deviation": bar.close / vwap - 1.0 if vwap > 0 else 0.0,
            "spread_bps": spread,
            "relative_volume": volume_last_15m / max(baseline_volume, 1.0),
            "order_book_proxy": max(0.0, 1.0 - spread / 50.0),
            "intraday_drawdown": bar.close / self._session_max_close - 1.0 if self._session_max_close > 0 else 0.0,
            "gap_from_open": bar.close / self._session_open - 1.0 if self._session_open > 0 else 0.0,
            "distance_to_high": bar.close / self._session_high - 1.0 if self._session_high > 0 else 0.0,
            "distance_to_low": bar.close / self._session_low - 1.0 if self._session_low > 0 else 0.0,
            "news_score": bar.news_score or 0.0,
        }
        return {key: value if math.isfinite(value) else 0.0 for key, value in features.items()}
