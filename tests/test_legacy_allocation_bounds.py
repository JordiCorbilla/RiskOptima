"""Independent constraints and compatibility tests for the legacy allocation entry point."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from riskoptima import RiskOptima
from riskoptima.optim.mean_variance import _bounded_allocation_candidates


@pytest.fixture
def allocation_inputs(monkeypatch):
    # Distinct shocks give zero cross-product and the specified sample variances.
    annual_variances = np.array([.04, .09])
    shocks = np.sqrt(annual_variances * 3 / (2 * 260))
    periodic_returns = np.array([[shocks[0], 0.], [-shocks[0], 0.], [0., shocks[1]], [0., -shocks[1]]])
    values = 100. * np.vstack([np.ones(2), np.cumprod(1 + periodic_returns, axis=0)])
    closes = pd.DataFrame(values, columns=["A", "B"])
    source = pd.concat({"Close": closes[["B", "A"]]}, axis=1)
    monkeypatch.setattr(RiskOptima, "fetch_historical_stock_prices", lambda *args: source)
    return annual_variances, closes


def test_hard_bounds_hit_analytic_frontier_and_align_predefined_returns(allocation_inputs):
    variances, _ = allocation_inputs
    bounds = .35, .65
    expected_returns = pd.Series({"B": .12, "A": 0.})
    kwargs = dict(enforce_allocation_bounds=True, random_state=73)
    args = ["A", "B"], "2024-01-01", "2024-02-01", .30, expected_returns, *bounds, 5000
    weights = RiskOptima.perform_mean_variance_optimization(*args, **kwargs)
    # B-only return makes Sharpe strictly increasing in B over this interval.
    np.testing.assert_allclose(weights, [.35, .65], rtol=0.0, atol=.001)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all((weights >= bounds[0]) & (weights <= bounds[1]))
    assert np.sqrt(variances @ weights**2) < .30
    np.testing.assert_array_equal(weights, RiskOptima.perform_mean_variance_optimization(*args, **kwargs))
    with pytest.raises(ValueError, match="one finite annual return"):
        RiskOptima.perform_mean_variance_optimization(*args[:4], pd.Series({"A": .01}))


def test_hard_bounds_include_exact_risk_feasibility_candidate(allocation_inputs):
    variances, _ = allocation_inputs
    inverse_variances = 1 / variances
    expected = inverse_variances / inverse_variances.sum()
    minimum_volatility = np.sqrt(1 / inverse_variances.sum())
    tickers = ["A", "B"]
    args = tickers, "2024-01-01", "2024-02-01"
    settings_ = dict(min_allocation=.20, max_allocation=.80, enforce_allocation_bounds=True, random_state=1)
    weights = RiskOptima.perform_mean_variance_optimization(
        *args, minimum_volatility + 1e-10, num_simulations=1, **settings_,
    )
    np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-7)
    assert np.sqrt(variances @ weights**2) <= minimum_volatility + 1e-10
    with pytest.raises(ValueError, match="maximum volatility"):
        RiskOptima.perform_mean_variance_optimization(*args, minimum_volatility - .001, **settings_)


def test_equal_weight_boundary_and_legacy_sampling_reproducibility(allocation_inputs):
    variances, _ = allocation_inputs
    tickers = ["A", "B"]
    args = tickers, "2024-01-01", "2024-02-01", .4
    annual = np.array([.02, .13])
    # An upper bound of 50% forces both weights to 50% for two assets.
    exact = RiskOptima.perform_mean_variance_optimization(
        *args, annual, .1, .50, 3, enforce_allocation_bounds=True, random_state=9,
    )
    np.testing.assert_array_equal(exact, [.5, .5])
    rng = np.random.RandomState(9)
    raw = rng.uniform(.01, .35, size=(40, 2))
    reference = raw / raw.sum(axis=1)[:, None]
    score = (reference @ annual) / np.sqrt(reference**2 @ variances)
    expected = reference[np.argmax(score)]
    state = np.random.get_state()
    try:
        np.random.seed(9)
        with pytest.warns(FutureWarning, match="before normalization"):
            actual = RiskOptima.perform_mean_variance_optimization(*args, annual, .01, .35, 40)
    finally:
        np.random.set_state(state)
    np.testing.assert_array_equal(actual, expected)


@given(n=st.integers(2, 30), fraction=st.floats(.01, .99), seed=st.integers(0, 100_000))
@settings(max_examples=25)
def test_feasible_sampling_preserves_budget_at_arbitrary_bounds(n, fraction, seed):
    lower = fraction / n
    upper = min(1., 1 / (fraction * n))
    weights = _bounded_allocation_candidates(n, 19, lower, upper, np.random.default_rng(seed))
    assert np.all((weights >= lower) & (weights <= upper))
    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-12)


def test_strict_bounds_reject_impossible_budgets_before_downloading(monkeypatch):
    def missing_download(*args):
        pytest.fail("Rejected input should not download prices")

    monkeypatch.setattr(RiskOptima, "fetch_historical_stock_prices", missing_download)
    args = ["A", "B"], "2024-01-01", "2024-02-01", .3
    with pytest.raises(ValueError, match="fully invested"):
        RiskOptima.perform_mean_variance_optimization(*args, enforce_allocation_bounds=True)
    with pytest.raises(ValueError, match="finite and non-negative"):
        RiskOptima.perform_mean_variance_optimization(*args[:3], np.nan)
    with pytest.raises(ValueError, match="positive integer"):
        RiskOptima.perform_mean_variance_optimization(*args, num_simulations=0)


def test_legacy_ml_entry_point_passes_sampling_budget_and_bounds_to_both_allocations(monkeypatch, allocation_inputs):
    _, closes = allocation_inputs
    source = pd.concat({"Close": closes}, axis=1)
    benchmark = pd.concat({"Close": closes[["A"]]}, axis=1)
    monkeypatch.setattr(
        RiskOptima, "fetch_historical_stock_prices", lambda tickers, *_: benchmark if tickers == ["SPY"] else source,
    )
    monkeypatch.setattr(
        RiskOptima, "generate_predictions_tickers", lambda *args: ({"A": .001, "B": .001}, {"A": 0., "B": 0.}),
    )
    calls = []
    implementation = RiskOptima.perform_mean_variance_optimization

    class ReachedSecondAllocation(Exception):
        pass

    def track_call(*args, **kwargs):
        allocation = implementation(*args, **kwargs)
        calls.append((kwargs, allocation))
        if len(calls) == 2:
            raise ReachedSecondAllocation
        return allocation

    monkeypatch.setattr(RiskOptima, "perform_mean_variance_optimization", track_call)
    table = pd.DataFrame({
        "Asset": ["A", "B"], "Weight": [.5, .5], "Label": ["A", "B"], "MarketCap": [1., 2.],
    })
    with pytest.raises(ReachedSecondAllocation):
        RiskOptima.run_portfolio_optimization_mv_ml(
            table, "2024-01-01", "2024-02-01", num_portfolios=4, min_weight=.5, max_weight=.5,
            max_volatility=.3, enforce_allocation_bounds=True,
        )
    assert len(calls) == 2
    for kwargs, weights in calls:
        assert kwargs["num_simulations"] == 4
        assert kwargs["enforce_allocation_bounds"] is True
        np.testing.assert_array_equal(weights, [.5, .5])
