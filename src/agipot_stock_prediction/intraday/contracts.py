"""Validated bars for locally supplied, completed intraday observations."""
from dataclasses import dataclass
from datetime import date, datetime
import math
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class IntradayBar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bid: float | None = None
    ask: float | None = None
    adjusted_close: float | None = None
    news_score: float | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("intraday timestamp must include a timezone")
        if not self.symbol:
            raise ValueError("symbol must not be empty")
        for key in ("open", "high", "low", "close", "volume", "bid", "ask", "adjusted_close", "news_score"):
            value = getattr(self, key)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{key} must be finite")
        if min(self.open, self.high, self.low, self.close) <= 0 or self.volume < 0:
            raise ValueError("prices must be positive and volume nonnegative")
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("OHLC bounds are inconsistent")
        if (self.bid is None) != (self.ask is None):
            raise ValueError("bid and ask must be provided together")
        if self.bid is not None and (self.bid <= 0 or self.ask < self.bid):
            raise ValueError("bid/ask must be positive and uncrossed")
        if self.adjusted_close is not None and self.adjusted_close <= 0:
            raise ValueError("adjusted_close must be positive")

    @property
    def session(self) -> date:
        return self.timestamp.astimezone(ZoneInfo("America/New_York")).date()

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2 if self.bid is not None else self.close

    @property
    def spread_bps(self) -> float | None:
        if self.bid is None:
            return None
        return 10_000 * (self.ask - self.bid) / self.mid
