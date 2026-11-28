"""Independent pricing references and path conventions for Heston and SABR."""

from math import exp, expm1, log, sqrt

import numpy as np
import pytest
import QuantLib as ql
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.special import ndtr

from riskoptima import RiskOptima


def test_heston_deterministic_variance_call():
    spot, strike, maturity, rate, kappa, theta, v0 = 100, 105, 1.4, -0.02, 1.5, 0.04, 0.07
    total_variance = theta * maturity + (theta - v0) * expm1(-kappa * maturity) / kappa
    d1 = (log(spot / strike) + rate * maturity + total_variance / 2) / sqrt(total_variance)
    expected = spot * ndtr(d1) - strike * exp(-rate * maturity) * ndtr(d1 - sqrt(total_variance))
    actual = RiskOptima.heston(spot, strike, maturity, rate, kappa, theta, 0, -0.6, v0, 1, 1)
    assert actual == pytest.approx(expected, abs=1e-12, rel=0)
    assert RiskOptima.heston(spot, 95, 0, rate, kappa, theta, 0.3, 0, v0) == 5.0


@pytest.mark.parametrize("eta", [0.3, 0.65])
def test_stochastic_heston_call_against_quantlib(eta):
    spot, strike, rate, kappa, theta, rho, v0 = 100, 105, -0.02, 1.5, 0.04, -0.65, 0.07
    today = ql.Date(3, 1, 2026)
    basis = ql.Actual365Fixed()
    risk_free = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, basis))
    dividend = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, basis))
    process = ql.HestonProcess(risk_free, dividend, ql.QuoteHandle(ql.SimpleQuote(spot)), v0, kappa, theta, eta, rho)
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike), ql.EuropeanExercise(today + 365))
    option.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(process), 144))
    with ql.SavedSettings():
        ql.Settings.instance().evaluationDate = today
        expected = option.NPV()
    estimates = np.array([
        RiskOptima.heston(spot, strike, 1, rate, kappa, theta, eta, rho, v0, 20_000, 512, random_state=seed)
        for seed in range(105, 111)
    ])
    # Independent batches estimate Monte Carlo noise. The larger eta violates the Feller condition.
    standard_error = estimates.std(ddof=1) / sqrt(len(estimates))
    assert estimates.mean() == pytest.approx(expected, abs=5 * standard_error, rel=0)


def test_heston_path_full_truncation_and_endpoint():
    seed, rate, variance, theta = 1, 0.05, 0.04, 0.02
    z1 = np.random.default_rng(seed).standard_normal(2)[0]
    prices, variances = RiskOptima.simulate_heston(
        100, variance, rate, 2, 3, -1, 1, theta, 2, random_state=seed,
    )
    expected_first = 100 * exp(rate - variance / 2 + sqrt(variance) * z1)
    np.testing.assert_allclose(prices, [100, expected_first, expected_first * exp(rate)], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(variances, [variance, 0, 0])
    deterministic, _ = RiskOptima.simulate_heston(sigma0=0, theta=0, r=rate, T=1, N=2)
    assert deterministic[-1] == pytest.approx(100 * exp(rate), abs=1e-12, rel=0)
    stressed, _ = RiskOptima.simulate_heston(sigma0=4, theta=4, eta=0, r=0, N=2, random_state=5)
    z1 = np.random.default_rng(5).standard_normal(2)[0]
    assert stressed[-1] == pytest.approx(100 * exp(-2 + 2 * z1), abs=1e-12, rel=0)


@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_sabr_exact_constant_level_limits(beta):
    seed, maturity, points, level = 42, 1.5, 8, 0.4
    forward = -0.5 if beta == 0 else 1.5
    dt = maturity / (points - 1)
    brownian = np.r_[0, np.random.default_rng(seed).standard_normal((points - 1, 2))[:, 0].cumsum()] * sqrt(dt)
    if beta == 0:
        expected = forward + level * brownian
    else:
        expected = forward * np.exp(level * brownian - level**2 * np.linspace(0, maturity, points) / 2)
    forwards, levels = RiskOptima.simulate_sabr(forward, level, maturity, points, 0, beta, 0, random_state=seed)
    np.testing.assert_allclose(forwards, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(levels, np.full(points, level))


def test_sabr_correlated_level_and_fractional_forward_step():
    seed, rho, nu = 42, -0.4, 0.3
    forward_shock, independent = np.random.default_rng(seed).standard_normal(2)
    expected_forward = 100 + 0.2 * sqrt(100) * forward_shock
    expected_level = 0.2 * exp(-nu**2 / 2 + nu * (rho * forward_shock + sqrt(1 - rho**2) * independent))
    forwards, levels = RiskOptima.simulate_sabr(100, 0.2, 1, 2, rho, 0.5, nu, random_state=seed)
    assert forwards[-1] == pytest.approx(expected_forward, abs=1e-12, rel=0)
    assert levels[-1] == pytest.approx(expected_level, abs=1e-12, rel=0)
    absorbed, _ = RiskOptima.simulate_sabr(0.01, 2, 1, 3, beta=0.5, random_state=5)
    np.testing.assert_array_equal(absorbed, [0.01, 0.0, 0.0])


def test_fractional_sabr_call_against_quantlib_finite_differences():
    today = ql.Date(3, 1, 2026)
    days, forward, strike, level, beta, nu, rho = 183, 100, 102, 2, 0.5, 0.4, -0.35
    basis = ql.Actual365Fixed()
    discount = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, basis))
    option = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, strike), ql.EuropeanExercise(today + days))
    with ql.SavedSettings():
        ql.Settings.instance().evaluationDate = today
        option.setPricingEngine(ql.FdSabrVanillaEngine(forward, level, beta, nu, rho, discount, 100, 400, 50))
        expected = option.NPV()
    rng = np.random.default_rng(774)
    terminal = np.array([
        RiskOptima.simulate_sabr(forward, level, days / 365, 65, rho, beta, nu, random_state=rng)[0][-1]
        for _ in range(6_000)
    ])
    payoffs = np.maximum(terminal - strike, 0)
    standard_error = payoffs.std(ddof=1) / sqrt(len(payoffs))
    assert payoffs.mean() == pytest.approx(expected, abs=5 * standard_error, rel=0)


@given(
    factor=st.floats(min_value=0.01, max_value=10),
    beta=st.floats(min_value=0, max_value=1),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=40, deadline=None)
def test_sabr_forward_units_scaling(factor, beta, seed):
    original, levels = RiskOptima.simulate_sabr(15, 0.3, N=9, beta=beta, random_state=seed)
    level_factor = factor ** (1 - beta)
    rescaled, scaled_levels = RiskOptima.simulate_sabr(
        15 * factor, 0.3 * level_factor, N=9, beta=beta, random_state=seed,
    )
    np.testing.assert_allclose(rescaled, original * factor, atol=1e-12, rtol=1e-13)
    np.testing.assert_allclose(scaled_levels, levels * level_factor, atol=1e-12, rtol=1e-13)


@pytest.mark.parametrize("path", [RiskOptima.simulate_heston, RiskOptima.simulate_sabr])
def test_path_stationary_boundaries_and_legacy_seed(path):
    for boundary in ({"N": 1}, {"T": 0, "N": 3}):
        prices, states = path(100, 0.2, **boundary)
        np.testing.assert_array_equal(prices, np.full(boundary["N"], 100))
        np.testing.assert_array_equal(states, np.full(boundary["N"], 0.2))
    state = np.random.get_state()
    try:
        np.random.seed(19)
        expected = path(N=5)
        np.random.seed(19)
        np.testing.assert_array_equal(path(N=5), expected)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(path(N=5, random_state=19), path(N=5, random_state=19))


@pytest.mark.parametrize("path, kwargs", [
    (RiskOptima.simulate_heston, {"sigma0": np.nan}),
    (RiskOptima.simulate_heston, {"theta": -0.01}),
    (RiskOptima.simulate_heston, {"N": 0}),
    (RiskOptima.simulate_sabr, {"F0": -1}),
    (RiskOptima.simulate_sabr, {"beta": 1.01}),
    (RiskOptima.simulate_sabr, {"rho": 1.01}),
    (RiskOptima.simulate_sabr, {"N": True}),
])
def test_invalid_model_inputs_raise_value_error(path, kwargs):
    with pytest.raises(ValueError):
        path(**kwargs)
