###############################################################################
#                         test_rate_conventions.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Date, calendar, day-count, and schedule tests
###############################################################################

import pandas as pd
import pytest

from riskoptima.rates import BusinessCalendar, generate_schedule, spot_date, year_fraction


def test_market_day_count_conventions():
    assert year_fraction("2026-01-01", "2026-07-01", "ACT/360") == pytest.approx(181 / 360)
    assert year_fraction("2026-01-01", "2026-07-01", "ACT/365F") == pytest.approx(181 / 365)
    assert year_fraction("2026-01-31", "2026-07-31", "30/360") == pytest.approx(0.5)


def test_business_day_adjustments_and_spot_lag():
    calendar = BusinessCalendar(holidays=["2026-07-06"])

    assert calendar.adjust("2026-05-31", "modified_following") == pd.Timestamp("2026-05-29")
    assert calendar.adjust("2026-07-04", "following") == pd.Timestamp("2026-07-07")
    assert spot_date("2026-07-02", 2, calendar) == pd.Timestamp("2026-07-07")


def test_schedule_preserves_end_of_month_and_accruals():
    schedule = generate_schedule(
        "2026-01-31",
        "2027-01-31",
        payment_frequency=2,
        day_count="30/360",
    )

    assert list(schedule["accrual_end"]) == [pd.Timestamp("2026-07-31"), pd.Timestamp("2027-01-31")]
    assert schedule["accrual_factor"].sum() == pytest.approx(1.0)
    assert schedule.iloc[-1]["payment_date"] == pd.Timestamp("2027-01-29")

