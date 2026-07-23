###############################################################################
#                       test_interest_rate_curves.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Interest-rate curve and bootstrap tests
###############################################################################

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from riskoptima import DiscountCurve, bootstrap_discount_curve
from riskoptima.branding import riskoptima_signature
from riskoptima.rates import plot_yield_curve, reprice_curve_instruments


def _sofr_quotes():
    return pd.DataFrame(
        {
            "instrument_type": ["deposit", "deposit", "swap", "swap", "swap", "swap"],
            "maturity": [0.25, 0.50, 1.0, 2.0, 5.0, 10.0],
            "rate": [0.0520, 0.0510, 0.0490, 0.0460, 0.0425, 0.0415],
            "payment_frequency": [np.nan, np.nan, 2, 2, 2, 2],
        }
    )


def test_zero_rates_discount_factors_and_forward_identity():
    tenors = np.array([0.5, 1.0, 2.0, 5.0])
    rates = np.array([0.030, 0.032, 0.035, 0.040])
    curve = DiscountCurve.from_zero_rates(tenors, rates)

    np.testing.assert_allclose(curve.discount_factor(tenors), np.exp(-rates * tenors))
    np.testing.assert_allclose(curve.zero_rate(tenors), rates)

    forward = curve.forward_rate(1.0, 2.0)
    implied_df = curve.discount_factor(1.0) * np.exp(-forward)
    assert implied_df == pytest.approx(curve.discount_factor(2.0))


def test_curve_serialization_round_trip_is_platform_safe():
    curve = DiscountCurve.from_zero_rates(
        [1.0, 2.0, 5.0],
        [0.03, 0.035, 0.04],
        name="USD SOFR",
        valuation_date="2026-07-22",
    )
    restored = DiscountCurve.from_dict(curve.to_dict())

    assert restored.name == curve.name
    assert restored.valuation_date == pd.Timestamp("2026-07-22")
    np.testing.assert_allclose(restored.discount_factors, curve.discount_factors)


def test_bootstrap_reprices_all_input_quotes():
    quotes = _sofr_quotes()
    curve = bootstrap_discount_curve(quotes, valuation_date="2026-07-22")
    calibration = reprice_curve_instruments(curve, quotes)

    assert np.isfinite(curve.discount_factors).all()
    assert (curve.discount_factors > 0).all()
    assert calibration["error_bps"].abs().max() < 1e-7
    assert curve.par_swap_rate(5.0, payment_frequency=2) == pytest.approx(0.0425, abs=1e-11)


def test_curve_validation_rejects_ambiguous_inputs():
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscountCurve([1.0, 0.5], [0.97, 0.98])

    duplicate_quotes = _sofr_quotes().iloc[:2].copy()
    duplicate_quotes.loc[1, "maturity"] = duplicate_quotes.loc[0, "maturity"]
    with pytest.raises(ValueError, match="unique"):
        bootstrap_discount_curve(duplicate_quotes)


def test_yield_curve_chart_has_single_signature():
    curve = bootstrap_discount_curve(_sofr_quotes())
    ax = plot_yield_curve(curve)

    assert sum(riskoptima_signature() in text.get_text() for text in ax.texts) == 1
    plt.close(ax.figure)

