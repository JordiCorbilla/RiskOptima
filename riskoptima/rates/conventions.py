###############################################################################
#                              conventions.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Day-count, business-day, spot-date, and schedule conventions
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


_DAY_COUNT_ALIASES = {
    "ACT/360": "ACT/360",
    "ACT360": "ACT/360",
    "ACTUAL/360": "ACT/360",
    "ACT/365F": "ACT/365F",
    "ACT365F": "ACT/365F",
    "ACTUAL/365F": "ACT/365F",
    "30/360": "30/360",
    "30/360 US": "30/360",
    "30U/360": "30/360",
}


def normalize_day_count(convention: str) -> str:
    key = str(convention).upper().strip().replace("_", "/")
    try:
        return _DAY_COUNT_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_DAY_COUNT_ALIASES.values())))
        raise ValueError(f"day_count must be one of: {allowed}") from exc


def year_fraction(start_date, end_date, convention: str = "ACT/360") -> float:
    """Return the year fraction between two dates under a market convention."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    day_count = normalize_day_count(convention)
    if day_count == "ACT/360":
        return (end - start).days / 360.0
    if day_count == "ACT/365F":
        return (end - start).days / 365.0

    start_day = min(start.day, 30)
    end_day = end.day
    if start_day == 30 and end_day == 31:
        end_day = 30
    days_360 = 360 * (end.year - start.year) + 30 * (end.month - start.month) + end_day - start_day
    return days_360 / 360.0


@dataclass(frozen=True, init=False)
class BusinessCalendar:
    """Weekend/holiday calendar with standard business-day adjustments."""

    holidays: frozenset[pd.Timestamp]
    weekend: tuple[int, ...]
    name: str

    def __init__(
        self,
        holidays: Iterable[pd.Timestamp | str] = (),
        weekend: tuple[int, ...] = (5, 6),
        name: str = "USD",
    ):
        normalized_holidays = frozenset(pd.Timestamp(day).normalize() for day in holidays)
        if not weekend or any(day < 0 or day > 6 for day in weekend):
            raise ValueError("weekend must contain weekday numbers from 0 to 6")
        object.__setattr__(self, "holidays", normalized_holidays)
        object.__setattr__(self, "weekend", tuple(sorted(set(weekend))))
        object.__setattr__(self, "name", str(name))

    def is_business_day(self, value) -> bool:
        date = pd.Timestamp(value).normalize()
        return date.weekday() not in self.weekend and date not in self.holidays

    def adjust(self, value, convention: str = "modified_following") -> pd.Timestamp:
        """Adjust a date using following/preceding market conventions."""
        date = pd.Timestamp(value).normalize()
        normalized = convention.lower().strip().replace("-", "_").replace(" ", "_")
        allowed = {"unadjusted", "following", "modified_following", "preceding", "modified_preceding"}
        if normalized not in allowed:
            raise ValueError(f"business-day convention must be one of: {', '.join(sorted(allowed))}")
        if normalized == "unadjusted" or self.is_business_day(date):
            return date

        direction = 1 if "following" in normalized else -1
        adjusted = date
        while not self.is_business_day(adjusted):
            adjusted += pd.Timedelta(days=direction)

        if normalized.startswith("modified") and adjusted.month != date.month:
            direction *= -1
            adjusted = date
            while not self.is_business_day(adjusted):
                adjusted += pd.Timedelta(days=direction)
        return adjusted

    def advance_business_days(self, value, business_days: int) -> pd.Timestamp:
        date = pd.Timestamp(value).normalize()
        if business_days == 0:
            return self.adjust(date, "following")
        direction = 1 if business_days > 0 else -1
        remaining = abs(int(business_days))
        while remaining:
            date += pd.Timedelta(days=direction)
            if self.is_business_day(date):
                remaining -= 1
        return date


def spot_date(valuation_date, spot_lag: int = 2, calendar: BusinessCalendar | None = None) -> pd.Timestamp:
    """Return a business-day-adjusted spot date."""
    calendar = calendar or BusinessCalendar()
    if spot_lag < 0:
        raise ValueError("spot_lag must be non-negative")
    return calendar.advance_business_days(valuation_date, spot_lag)


def generate_schedule(
    start_date,
    end_date,
    *,
    payment_frequency: int = 2,
    day_count: str = "30/360",
    calendar: BusinessCalendar | None = None,
    business_day_convention: str = "modified_following",
) -> pd.DataFrame:
    """Generate an accrual/payment schedule with a possible short front stub."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        raise ValueError("end_date must be after start_date")
    if payment_frequency <= 0 or 12 % payment_frequency != 0:
        raise ValueError("payment_frequency must be a positive divisor of 12")
    normalize_day_count(day_count)
    calendar = calendar or BusinessCalendar()

    months = 12 // payment_frequency
    end_of_month = end == end + pd.offsets.MonthEnd(0)
    boundaries = [end]
    candidate = end
    while True:
        candidate = candidate - pd.DateOffset(months=months)
        if end_of_month:
            candidate = candidate + pd.offsets.MonthEnd(0)
        if candidate <= start:
            break
        boundaries.append(candidate.normalize())
    boundaries = [start, *sorted(boundaries)]

    rows = []
    for accrual_start, accrual_end in zip(boundaries[:-1], boundaries[1:]):
        rows.append(
            {
                "accrual_start": accrual_start,
                "accrual_end": accrual_end,
                "payment_date": calendar.adjust(accrual_end, business_day_convention),
                "accrual_factor": year_fraction(accrual_start, accrual_end, day_count),
            }
        )
    return pd.DataFrame(rows)

