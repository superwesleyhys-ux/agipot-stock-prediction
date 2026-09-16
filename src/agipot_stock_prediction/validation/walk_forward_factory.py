"""Chronological calendar windows; this module does not fit or validate models."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


def _add_months(value: date, months: int) -> date:
    year, month = divmod(value.year * 12 + value.month - 1 + months, 12)
    return date(year, month + 1, min(value.day, monthrange(year, month + 1)[1]))


def build_walk_forward_windows(
    sessions: Sequence[date],
    *,
    train_years: int = 3,
    validation_months: int = 6,
    test_months: int = 6,
    step_months: int = 3,
) -> tuple[WalkForwardWindow, ...]:
    """Build disjoint train/validation/test intervals within each rolling window.

    Calendar boundaries are anchored to the first supplied session. Returned
    endpoints are actual available sessions, not fabricated market dates. Empty
    partitions and test intervals extending past the observed coverage are
    omitted. Coverage must reach the calendar end (exclusive) minus one day;
    this conservatively omits a final window if that day is a market holiday.
    Sessions may recur across *different* rolling windows. No purge/embargo or
    label-horizon treatment is implied; callers must add these before fitting.
    """
    for name, value in (("train_years", train_years), ("validation_months", validation_months),
                        ("test_months", test_months), ("step_months", step_months)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if any(type(session) is not date for session in sessions):
        raise ValueError("sessions must contain date values, not timestamps or strings")
    unique = tuple(sorted(set(sessions)))
    if not unique:
        return ()
    windows: list[WalkForwardWindow] = []
    first, coverage_end = unique[0], unique[-1] + timedelta(days=1)
    offset = 0
    while True:
        train_start = _add_months(first, offset)
        validation_start = _add_months(first, offset + train_years * 12)
        test_start = _add_months(first, offset + train_years * 12 + validation_months)
        test_end = _add_months(first, offset + train_years * 12 + validation_months + test_months)
        if test_end > coverage_end:
            break
        train = [session for session in unique if train_start <= session < validation_start]
        validation = [session for session in unique if validation_start <= session < test_start]
        test = [session for session in unique if test_start <= session < test_end]
        if train and validation and test:
            windows.append(WalkForwardWindow(train[0], train[-1], validation[0], validation[-1], test[0], test[-1]))
        offset += step_months
    return tuple(windows)


__all__ = ["WalkForwardWindow", "build_walk_forward_windows"]
