"""Deterministic fictional data; no real market dataset is redistributed."""
from datetime import date, timedelta
import math

from .calendar import is_us_equity_session


def demo_input() -> dict:
    day = date(2016, 1, 4)
    end = date(2025, 11, 28)
    rows = []
    price = 40.0
    while day <= end:
        if is_us_equity_session(day):
            i = len(rows)
            previous = price
            price *= math.exp(0.00035 + 0.003 * math.sin(i * 0.19) + 0.005 * math.sin(i * 0.037))
            rows.append({
                "date": day.isoformat(), "open": round(previous, 6),
                "high": round(max(previous, price) * 1.004, 6),
                "low": round(min(previous, price) * 0.996, 6),
                "close": round(price, 6), "adjusted_close": round(price, 6),
                "volume": 1_500_000 + (i % 13) * 50_000,
            })
        day += timedelta(days=1)
    return {
        "symbol": "SYNTH", "as_of": "2025-12-01T21:00:00+00:00",
        "data_source": "synthetic_demonstration_not_real_market_data",
        "daily_rows": rows,
        "factor_fundamentals": {"roic": 0.18, "roe": 0.22, "gross_margin": 0.45,
                                "operating_margin": 0.20, "fcf_margin": 0.14, "debt_to_equity": 0.3},
        "intraday_features": {"ret_5m": 0.002, "ret_15m": 0.003, "ret_60m": 0.004,
                              "relative_volume": 1.2, "spread_bps": 3.0},
    }
