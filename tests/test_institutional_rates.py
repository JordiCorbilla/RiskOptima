###############################################################################
#                       test_institutional_rates.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Dated SOFR, multi-curve, Jacobian, and key-rate tests
###############################################################################

import numpy as np
import pandas as pd
import pytest

from riskoptima.rates import (
    DiscountCurve,
    MultiCurveSet,
    bootstrap_discount_curve,
    bootstrap_projection_curve,
    bootstrap_sofr_curve,
    curve_calibration_jacobian,
    key_rate_dv01,
    price_fixed_rate_bond_dates,
    reprice_projection_instruments,
    reprice_sofr_instruments,
)


def _dated_sofr_quotes():
    return pd.DataFrame(
        [
            {
                "quote_id": "ON",
                "instrument_type": "deposit",
                "maturity_date": "2026-07-02",
                "rate": 0.0430,
            },
            {
                "quote_id": "FOMC-SEP",
                "instrument_type": "fomc",
                "start_date": "2026-07-02",
                "end_date": "2026-09-17",
                "rate": 0.0425,
            },
            {
                "quote_id": "FOMC-OCT",
                "instrument_type": "fomc",
                "start_date": "2026-09-17",
                "end_date": "2026-10-29",
                "rate": 0.0415,
            },
            {
                "quote_id": "SR3-H7",
                "instrument_type": "future",
                "start_date": "2026-10-29",
                "end_date": "2027-01-29",
                "price": 95.95,
                "convexity_adjustment_bps": 0.5,
            },
            {
                "quote_id": "OIS-2Y",
                "instrument_type": "swap",
                "maturity_date": "2028-07-03",
                "rate": 0.0390,
                "payment_frequency": 1,
            },
            {
                "quote_id": "OIS-5Y",
                "instrument_type": "swap",
                "maturity_date": "2031-07-03",
                "rate": 0.0375,
                "payment_frequency": 1,
            },
        ]
    )


def test_dated_sofr_curve_reprices_mixed_market_instruments():
    quotes = _dated_sofr_quotes()
    curve = bootstrap_sofr_curve(quotes, valuation_date="2026-07-01")
    calibration = reprice_sofr_instruments(curve, quotes)

    assert curve.valuation_date == pd.Timestamp("2026-07-01")
    assert curve.day_count == "ACT/365F"
    assert calibration["error_bps"].abs().max() < 1e-7
    assert 0 < curve.discount_factor_date("2031-07-03") < 1
    assert DiscountCurve.from_dict(curve.to_dict()).day_count == "ACT/365F"


def _projection_quotes():
    return pd.DataFrame(
        [
            {"quote_id": "DEP-3M", "instrument_type": "deposit", "maturity": 0.25, "rate": 0.0450},
            {"quote_id": "FRA-3X6", "instrument_type": "fra", "start": 0.25, "maturity": 0.50, "rate": 0.0440},
            {"quote_id": "SWAP-1Y", "instrument_type": "swap", "maturity": 1.0, "rate": 0.0430},
            {"quote_id": "SWAP-2Y", "instrument_type": "swap", "maturity": 2.0, "rate": 0.0420},
            {"quote_id": "SWAP-5Y", "instrument_type": "swap", "maturity": 5.0, "rate": 0.0410},
        ]
    )


def test_multi_curve_projection_calibration_and_serialization():
    discount_curve = DiscountCurve.from_zero_rates(
        [0.25, 0.5, 1.0, 2.0, 5.0],
        [0.040, 0.040, 0.039, 0.038, 0.037],
        name="USD SOFR OIS",
        valuation_date="2026-07-01",
    )
    quotes = _projection_quotes()
    projection_curve = bootstrap_projection_curve(quotes, discount_curve, name="SOFR 3M")
    calibration = reprice_projection_instruments(projection_curve, discount_curve, quotes)
    curves = MultiCurveSet(discount_curve, {"SOFR-3M": projection_curve})
    restored = MultiCurveSet.from_dict(curves.to_dict())

    assert calibration["error_bps"].abs().max() < 1e-7
    assert curves.par_swap_rate(5.0, "SOFR-3M") == pytest.approx(0.0410, abs=1e-11)
    assert restored.projection_curve("SOFR-3M").name == "SOFR 3M"

    mismatched = DiscountCurve.from_zero_rates(
        [1.0],
        [0.04],
        valuation_date="2026-07-02",
    )
    with pytest.raises(ValueError, match="valuation date"):
        MultiCurveSet(discount_curve, {"mismatched": mismatched})


def test_curve_jacobian_is_finite_and_quote_labelled():
    quotes = pd.DataFrame(
        [
            {"quote_id": "DEP-6M", "instrument_type": "deposit", "maturity": 0.5, "rate": 0.045},
            {"quote_id": "SWAP-1Y", "instrument_type": "swap", "maturity": 1.0, "rate": 0.043},
            {"quote_id": "SWAP-2Y", "instrument_type": "swap", "maturity": 2.0, "rate": 0.041},
            {"quote_id": "SWAP-5Y", "instrument_type": "swap", "maturity": 5.0, "rate": 0.039},
        ]
    )
    jacobian = curve_calibration_jacobian(quotes, curve_builder=bootstrap_discount_curve)

    assert jacobian.shape == (4, 4)
    assert list(jacobian.columns) == ["DEP-6M", "SWAP-1Y", "SWAP-2Y", "SWAP-5Y"]
    assert np.isfinite(jacobian.to_numpy()).all()
    assert (np.diag(jacobian) > 0).all()

    duplicate_labels = quotes.copy()
    duplicate_labels["quote_id"] = "duplicate"
    with pytest.raises(ValueError, match="unique"):
        curve_calibration_jacobian(duplicate_labels)


def test_dated_curve_jacobian_handles_futures_price_quotes():
    jacobian = curve_calibration_jacobian(
        _dated_sofr_quotes(),
        curve_builder=bootstrap_sofr_curve,
        builder_kwargs={"valuation_date": "2026-07-01"},
    )

    assert jacobian.shape == (6, 6)
    assert "SR3-H7" in jacobian.columns
    assert np.isfinite(jacobian.to_numpy()).all()


def test_key_rate_dv01_reconciles_to_parallel_bump():
    curve = DiscountCurve.from_zero_rates([1.0, 2.0, 5.0], [0.04, 0.04, 0.04])
    cash_flows = np.array([5.0, 5.0, 105.0])
    payment_times = np.array([1.0, 2.0, 5.0])
    key_dv01 = key_rate_dv01(curve, cash_flows, payment_times)

    parallel_curve = DiscountCurve.from_zero_rates([1.0, 2.0, 5.0], [0.0401, 0.0401, 0.0401])
    parallel_dv01 = curve.present_value(cash_flows, payment_times) - parallel_curve.present_value(
        cash_flows, payment_times
    )
    assert (key_dv01 > 0).all()
    assert key_dv01.sum() == pytest.approx(parallel_dv01, rel=2e-4)


def test_dated_bond_pricing_uses_generated_schedule():
    curve = DiscountCurve.from_zero_rates(
        [0.5, 1.0, 2.0, 5.0],
        [0.04, 0.04, 0.04, 0.04],
        valuation_date="2026-01-02",
    )
    price = price_fixed_rate_bond_dates(
        curve,
        "2026-01-02",
        "2028-01-02",
        0.05,
        face_value=100,
        payment_frequency=2,
    )

    assert price > 100
