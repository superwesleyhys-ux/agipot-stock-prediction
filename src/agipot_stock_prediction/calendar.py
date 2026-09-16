"""NYSE session dates and scheduled close times, including holidays and half-days."""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as exchange_calendars
import pandas as pd


ET = ZoneInfo("America/New_York")


@lru_cache(maxsize=8)
def _xnys_calendar(year: int):
    return exchange_calendars.get_calendar(
        "XNYS",
        start=f"{year - 2}-01-01",
        end=f"{year + 2}-12-31",
    )


def _session_label(day: date) -> pd.Timestamp:
    return pd.Timestamp(day.isoformat())


def is_us_equity_session(day: date) -> bool:
    return bool(_xnys_calendar(day.year).is_session(_session_label(day)))


def us_equity_session_bounds(day: date) -> tuple[datetime, datetime] | None:
    calendar = _xnys_calendar(day.year)
    label = _session_label(day)
    if not calendar.is_session(label):
        return None
    opened = calendar.session_open(label).to_pydatetime().astimezone(ET)
    closed = calendar.session_close(label).to_pydatetime().astimezone(ET)
    return opened, closed
