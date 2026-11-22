"""Numerical contracts for options, rates and realized frequencies."""

from datetime import date
from math import exp, sqrt

import numpy as np
import pandas as pd
import pytest
import QuantLib as ql

from riskoptima import RiskOptima
from riskoptima.options import OptionContract, black_scholes_greeks, black_scholes_price, value_option_contract
from riskoptima.rates import DiscountCurve, year_fraction
from riskoptima.volatility import historical_volatility


def _quantlib_black(S, K, T, r, sigma, option_type, q=0.0):
    side = ql.Option.Call if option_type == "call" else ql.Option.Put
    return ql.BlackCalculator(
        ql.PlainVanillaPayoff(side, K), S * exp((r - q) * T), sigma * sqrt(T), exp(-r * T),
    )


def _reference_greeks(model, spot, maturity):
    return {
        "delta": model.delta(spot),
        "gamma": model.gamma(spot),
        "theta": model.theta(spot, maturity),
        "vega": model.vega(maturity),
        "rho": model.rho(maturity),
    }


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_option_greeks_against_quantlib_with_negative_rate_and_dividend(option_type):
    S, K, T, r, sigma, q = 102.5, 108., 1.3, -.015, .28, .023
    reference = _quantlib_black(S, K, T, r, sigma, option_type, q)
    assert black_scholes_price(S, K, T, r, sigma, option_type, q) == pytest.approx(reference.value(), abs=1e-12)
    assert black_scholes_greeks(S, K, T, r, sigma, option_type, q) == pytest.approx(
        _reference_greeks(reference, S, T), abs=1e-12, rel=0,
    )


def test_platform_library_example_and_legacy_notebook_greek_units():
    S, r, sigma = 100., .04, .20
    # The Platform library and the options notebook service use two interfaces.
    pricing_term = 45 / 365
    price = black_scholes_price(S=100., K=102., T=pricing_term, r=.04, sigma=.22, option_type="call")
    assert type(price) is float
    assert price == pytest.approx(_quantlib_black(100., 102., pricing_term, .04, .22, "call").value(), abs=1e-12)

    T = 30 / 365
    expected = _reference_greeks(_quantlib_black(S, S, T, r, sigma, "call"), S, T)
    annual = black_scholes_greeks(S=S, K=S, T=T, r=r, sigma=sigma, option_type="call")
    assert all(type(value) is float for value in annual.values())
    assert annual.keys() == expected.keys()
    assert annual == pytest.approx(expected, abs=1e-12, rel=0)
    legacy = RiskOptima.black_scholes_greeks(S, S, T, r, sigma, "call")
    assert isinstance(legacy, tuple)
    assert legacy == pytest.approx(
        (expected["delta"], expected["gamma"], expected["theta"] / 365, expected["vega"] / 100), abs=1e-12, rel=0,
    )


def test_legacy_option_price_broadcasts_and_handles_mixed_boundary_terms():
    spots = np.array([[100.], [110.]])
    strikes = np.array([100., 105., 110.])
    maturities = np.array([0., 1., 1.25])
    volatility = np.array([.25, 0., .2])
    prices = RiskOptima.black_scholes(spots, strikes, maturities, .04, volatility)
    expected = [[
        max(S - strikes[0], 0),
        max(S - strikes[1] * exp(-.04), 0),
        _quantlib_black(S, strikes[2], maturities[2], .04, volatility[2], "call").value(),
    ] for S in spots.ravel()]
    np.testing.assert_allclose(prices, expected, rtol=0, atol=1e-12)
    assert RiskOptima.black_scholes(100., 100., 0., .04, .2) == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        RiskOptima.black_scholes(spots, strikes, -.25, .04, .2)


def test_legacy_put_greeks_keep_broadcast_shape_and_deprecate_unknown_side():
    strikes = np.array([[80., 100., 120.]])
    maturities = np.array([[.25], [1.]])
    value = RiskOptima.black_scholes_greeks(100., strikes, maturities, .05, .22, "put")
    assert all(component.shape == (2, 3) for component in value)
    for row, T in enumerate(maturities.ravel()):
        for col, K in enumerate(strikes.ravel()):
            expected = _reference_greeks(_quantlib_black(100., K, T, .05, .22, "put"), 100., T)
            np.testing.assert_allclose(
                [component[row, col] for component in value],
                [expected["delta"], expected["gamma"], expected["theta"] / 365, expected["vega"] / 100],
                rtol=0, atol=1e-12,
            )
    with pytest.warns(FutureWarning, match="unrecognized legacy option_type"):
        actual = RiskOptima.black_scholes_greeks(100., 100., 1., .05, .22, "not-a-side")
    assert actual == RiskOptima.black_scholes_greeks(100., 100., 1., .05, .22, "put")
    with pytest.raises(ValueError, match="positive"):
        RiskOptima.black_scholes_greeks(100., 100., 0., .05, .22, "put")


def test_option_valuation_uses_actual_365_fixed_across_leap_day():
    valuation, expiry = date(2024, 2, 28), date(2024, 3, 30)
    T = ql.Actual365Fixed().yearFraction(ql.Date(28, 2, 2024), ql.Date(30, 3, 2024))
    option = OptionContract("SPY", 102., expiry, "call", quantity=-3, multiplier=100, implied_volatility=.22)
    actual = value_option_contract(option, 100., valuation, .04)
    expected = _quantlib_black(100., 102., T, .04, .22, "call").value()
    assert actual.time_to_expiry == T
    assert actual.market_value == pytest.approx(-300 * expected, abs=1e-10, rel=0)


@pytest.mark.parametrize(("start", "end"), [
    (date(2023, 2, 28), date(2023, 3, 31)),
    (date(2024, 2, 29), date(2025, 2, 28)),
    (date(2024, 2, 28), date(2024, 3, 31)),
    (date(2024, 1, 31), date(2024, 2, 29)),
])
def test_explicit_us_30_360_and_bond_basis_against_quantlib(start, end):
    qstart = ql.Date(start.day, start.month, start.year)
    qend = ql.Date(end.day, end.month, end.year)
    expected_us = ql.Thirty360(ql.Thirty360.USA).yearFraction(qstart, qend)
    expected_bond = ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(qstart, qend)
    assert year_fraction(start, end, "30/360 US") == expected_us
    assert year_fraction(start, end, "30U/360") == expected_us
    assert year_fraction(start, end, "30/360") == expected_bond


@pytest.mark.parametrize(("convention", "counter"), [("ACT/360", ql.Actual360()), ("ACT/365F", ql.Actual365Fixed())])
def test_actual_day_counts_use_calendar_days_across_dst(convention, counter):
    start = pd.Timestamp("2026-03-29", tz="Europe/London")
    end = pd.Timestamp("2026-03-30", tz="Europe/London")
    expected = counter.yearFraction(ql.Date(29, 3, 2026), ql.Date(30, 3, 2026))
    assert year_fraction(start, end, convention) == expected
    with pytest.raises(ValueError, match="valid calendar dates"):
        year_fraction(pd.NaT, end, convention)


def test_continuous_and_nominal_discount_rates_preserve_separate_contracts():
    nominal_rate, term = .06, 2.5
    day_counter = ql.Actual365Fixed()
    nominal = ql.InterestRate(nominal_rate, day_counter, ql.Compounded, ql.Semiannual).discountFactor(term)
    continuous = ql.InterestRate(nominal_rate, day_counter, ql.Continuous, ql.Annual).discountFactor(term)
    equivalent = 2 * np.log1p(nominal_rate / 2)
    assert RiskOptima.discount_v2([term], nominal_rate, freq=2).iloc[0, 0] == pytest.approx(nominal, abs=1e-12)
    assert RiskOptima.discount([term], nominal_rate / 2).iloc[0, 0] == pytest.approx(
        ql.InterestRate(nominal_rate / 2, day_counter, ql.Compounded, ql.Annual).discountFactor(term), abs=1e-12,
    )
    assert DiscountCurve.from_zero_rates([term], [nominal_rate]).discount_factor(term) == pytest.approx(
        continuous, abs=1e-12,
    )
    assert DiscountCurve.from_zero_rates([term], [equivalent]).discount_factor(term) == pytest.approx(nominal, abs=1e-12)


def test_legacy_and_modular_annualization_preserve_frequency_and_std_conventions():
    returns = np.array([.005, -.009, .013, .003, -.008])
    rate = .02
    periods = 260
    assert RiskOptima.get_trading_days() == periods
    per_period = (1 + rate)**(1 / periods) - 1
    excess = (np.prod(1 + returns - per_period)**(periods / len(returns)) - 1)
    sample_std = np.std(returns, ddof=1)
    population_std = np.std(returns, ddof=0)
    assert RiskOptima.sharpe_ratio(pd.Series(returns), rate) == pytest.approx(excess / (sample_std * sqrt(periods)))
    assert RiskOptima.annualize_volatility(returns, periods) == pytest.approx(population_std * sqrt(periods))
    assert historical_volatility(returns) == pytest.approx(population_std * sqrt(252))
    assert historical_volatility(returns, periods_per_year=260, ddof=1) == pytest.approx(sample_std * sqrt(periods))
