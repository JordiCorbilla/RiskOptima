"""Offline regressions for the legacy portfolio probability workflow."""

import importlib
from math import prod, sqrt
from statistics import mean, pstdev, stdev
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from riskoptima import RiskOptima

ASSET_RETURNS = [[0.01, 0.05], [0.03, -0.01], [0.02, 0.03], [-0.02, 0.04]]
BENCHMARK_RETURNS = [0.01, -0.01, 0.02, 0.0]
DATES = pd.bdate_range("2025-01-01", periods=5)
ALLOCATION = {"Z": 75, "A": 25}


def _prices():
    gross = 1 + np.array(ASSET_RETURNS)
    stocks = pd.DataFrame(100 * np.vstack([np.ones(2), gross.cumprod(axis=0)]), index=DATES, columns=ALLOCATION)
    benchmark = pd.Series(100 * np.r_[1.0, (1 + np.array(BENCHMARK_RETURNS)).cumprod()], index=DATES, name="SPY")
    return stocks, benchmark


def _analyze(stocks, benchmark, allocation=ALLOCATION, **overrides):
    defaults = dict(benchmark_index="SPY", risk_free_rate=0.01, number_of_portfolio_weights=4,
                    trading_days_per_year=3, number_of_monte_carlo_runs=2, random_state=41)
    with patch.object(RiskOptima, "download_data_yfinance", side_effect=[stocks, benchmark]) as download:
        result = RiskOptima.run_portfolio_analysis(allocation, "2025-01-01", "2025-01-08", **(defaults | overrides))
    assert [call.args[0] for call in download.call_args_list] == [list(allocation), ["SPY"]]
    return result


def test_rankings_and_compounded_paths_match_a_scalar_reference():
    stocks, benchmark = _prices()
    state = np.random.get_state()
    try:
        np.random.seed(41)
        result = _analyze(stocks[["A", "Z"]].iloc[::-1], benchmark.to_frame(), random_state=None)
    finally:
        np.random.set_state(state)

    first, second = zip(*ASSET_RETURNS, strict=True)
    mean_first, mean_second = mean(first), mean(second)
    sd_first, sd_second = stdev(first), stdev(second)
    cross = sum((x - mean_first) * (y - mean_second) for x, y in zip(first, second, strict=True)) / (len(first) - 1)
    expected_cov = [[sd_first**2, cross], [cross, sd_second**2]]
    assert list(result["covariance_matrix"]) == list(result["covariance_matrix"].index) == ["Z", "A"]
    assert result["covariance_matrix"].attrs["covariance_unit"] == "per_period"
    np.testing.assert_allclose(result["covariance_matrix"], expected_cov, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["stock_daily_returns"], ASSET_RETURNS, atol=1e-12, rtol=0)

    rng = np.random.RandomState(41)
    weights = [tuple(row / sum(row)) for row in (rng.random_sample(2) for _ in range(4))]
    rows = []
    for index, (a, b) in enumerate(weights):
        annual_return = 3 * (a * mean_first + b * mean_second)
        annual_vol = sqrt(3 * (a * a * sd_first**2 + 2 * a * b * cross + b * b * sd_second**2))
        rows.append((annual_return, annual_vol, (annual_return - 0.01) / annual_vol, index))
    np.testing.assert_allclose(result["recorded_weights"], np.asarray(weights).T, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["simulated_portfolios"], rows, atol=1e-12, rtol=0)
    best = max(range(len(rows)), key=lambda index: rows[index][2])
    median = sorted(range(len(rows)), key=lambda index: rows[index][1])[len(rows) // 2]
    assert result["optimal_sharpe_idx"] == best
    np.testing.assert_allclose(result["optimal_weights"], weights[best], atol=1e-12, rtol=0)
    np.testing.assert_allclose(result["median_volatility_weights"], weights[median], atol=1e-12, rtol=0)

    def path(a, b):
        factors = []
        for _ in range(3):
            z1, z2 = rng.normal(size=2)
            r1 = mean_first + sd_first * z1
            r2 = mean_second + (cross / sd_first) * z1 + sqrt(sd_second**2 - (cross / sd_first) ** 2) * z2
            factors.append(1 + a * r1 + b * r2)
        return 10_000 * prod(factors)

    for name, (a, b) in (("Optimized Portfolio", weights[best]), ("Current Portfolio", (0.75, 0.25)),
                         ("Median Portfolio", weights[median])):
        expected = [path(a, b) for _ in range(2)]
        terminal = [value / 10_000 - 1 for value in expected]
        np.testing.assert_allclose(result["portfolio_results"][name], expected, atol=2e-10, rtol=0)
        np.testing.assert_allclose(result["portfolio_metrics"][name], [mean(terminal), pstdev(terminal)], atol=1e-12, rtol=0)
    benchmark_mean, benchmark_sd = mean(BENCHMARK_RETURNS), stdev(BENCHMARK_RETURNS)
    expected_market = [10_000 * prod(1 + rng.normal(benchmark_mean, benchmark_sd) for _ in range(3)) for _ in range(2)]
    market_net = [value / 10_000 - 1 for value in expected_market]
    np.testing.assert_allclose(result["market_final_values"], expected_market, atol=2e-10, rtol=0)
    np.testing.assert_allclose(result["market_final_values_percent"], np.array(market_net) * 100, atol=1e-12, rtol=0)
    assert result["market_expected_return"] == pytest.approx(mean(market_net), abs=1e-12, rel=0)
    assert result["market_volatility"] == pytest.approx(pstdev(market_net), abs=1e-12, rel=0)
    assert result["market_sharpe_ratio"] == pytest.approx((mean(market_net) - 0.01) / pstdev(market_net), abs=1e-12, rel=0)


@given(seed=st.integers(min_value=0, max_value=2**32 - 1), amount=st.integers(min_value=1, max_value=300))
@settings(max_examples=30, deadline=None)
def test_provider_label_and_date_permutations_leave_seeded_results_unchanged(seed, amount):
    stocks, benchmark = _prices()
    allocation = {"Z": amount, "A": 100}
    original = _analyze(stocks, benchmark, allocation, random_state=seed)
    permuted = _analyze(stocks[["A", "Z"]].iloc[::-1], benchmark.iloc[::-1], allocation, random_state=seed)
    assert original.keys() == permuted.keys()
    pd.testing.assert_frame_equal(original["covariance_matrix"], permuted["covariance_matrix"])
    pd.testing.assert_frame_equal(original["simulated_portfolios"], permuted["simulated_portfolios"])
    assert original["portfolio_results"] == permuted["portfolio_results"]
    assert original["market_final_values"] == permuted["market_final_values"]


def test_perfectly_correlated_assets_match_analytical_wealth_moments():
    stocks, benchmark = _prices()
    stocks["A"] = stocks["Z"]
    paths, horizon = 10_000, 2
    result = _analyze(stocks, benchmark, trading_days_per_year=horizon, number_of_monte_carlo_runs=paths)
    one_asset = [row[0] for row in ASSET_RETURNS]
    mu, variance = mean(one_asset), stdev(one_asset)**2
    cov = result["covariance_matrix"]
    np.testing.assert_allclose(cov, np.full((2, 2), variance), atol=1e-12, rtol=0)
    assert np.linalg.matrix_rank(cov) == 1
    expected = (1 + mu) ** horizon - 1
    expected_sd = sqrt(((1 + mu)**2 + variance) ** horizon - (1 + mu) ** (2 * horizon))
    actual_mean, actual_sd = result["portfolio_metrics"]["Current Portfolio"]
    assert actual_mean == pytest.approx(expected, abs=5 * expected_sd / sqrt(paths), rel=0)
    assert actual_sd == pytest.approx(expected_sd, rel=0.05)


def test_missing_quotes_do_not_fill_or_bridge_gaps():
    stocks, benchmark = _prices()
    stocks.loc[DATES[2], "Z"] = np.nan
    result = _analyze(stocks, benchmark)
    assert list(result["stock_daily_returns"].index) == [DATES[1], DATES[4]]
    np.testing.assert_allclose(result["stock_daily_returns"], [ASSET_RETURNS[0], ASSET_RETURNS[-1]], atol=1e-12, rtol=0)
    assert len(result["benchmark_daily_returns"]) == 4


def test_deterministic_single_asset_and_benchmark_have_zero_dispersion():
    flat = pd.Series(100.0, index=DATES, name="Close")
    result = _analyze(flat, flat, {"Z": 100}, risk_free_rate=0.0, number_of_monte_carlo_runs=5)
    assert result["optimal_sharpe_idx"] == 0
    assert result["simulated_portfolios"]["Sharpe Ratio"].isna().all()
    for name, values in result["portfolio_results"].items():
        assert values == [10_000.0] * 5
        assert result["portfolio_metrics"][name] == (0.0, 0.0)
    assert result["market_volatility"] == 0.0 and np.isnan(result["market_sharpe_ratio"])


def test_gross_losses_cannot_revive_a_zero_fund(monkeypatch):
    price = pd.Series([100., 50., 87.5, 87.5], index=DATES[:4])
    monkeypatch.setattr(np.random, "normal", lambda *, size: np.full(size, -3.0))
    result = _analyze(price, price, {"Z": 100}, trading_days_per_year=2, random_state=None)
    assert 1 + mean([-0.5, 0.75, 0]) - 3 * stdev([-0.5, 0.75, 0]) < 0
    assert all(values == [0.0, 0.0] for values in result["portfolio_results"].values())
    assert result["market_final_values"] == [0.0, 0.0]
    assert result["market_expected_return"] == -1.0


def test_ruin_still_sets_terminal_wealth_to_zero_after_extreme_growth(monkeypatch):
    price = pd.Series([100., 50., 87.5, 87.5], index=DATES[:4])

    def shocks(*, size):
        return np.broadcast_to(np.array([1e156, 1e156, -3.0, -3.0]).reshape(1, 4, 1), size)

    monkeypatch.setattr(np.random, "normal", shocks)
    result = _analyze(price, price, {"Z": 100}, trading_days_per_year=4, random_state=None)
    assert result["market_final_values"] == [0.0, 0.0]
    assert all(values == [0.0, 0.0] for values in result["portfolio_results"].values())


def test_unrepresentable_surviving_wealth_raises(monkeypatch):
    price = pd.Series([100., 50., 87.5, 87.5], index=DATES[:4])
    monkeypatch.setattr(np.random, "normal", lambda *, size: np.full(size, 1e156))
    with pytest.raises(OverflowError, match="outside the representable range"):
        _analyze(price, price, {"Z": 100}, trading_days_per_year=2, random_state=None)


@pytest.mark.parametrize("change, message", [
    (lambda prices: prices.iloc[:2], "two complete"),
    (lambda prices: prices[["Z"]], "unique columns"),
    (lambda prices: prices.set_axis(["Z", "Z"], axis=1), "unique columns"),
    (lambda prices: prices.set_axis([*DATES[:-1], DATES[-2]]), "unique and non-missing"),
    (lambda prices: prices.assign(Z=0), "must be positive"),
])
def test_invalid_source_data_is_rejected(change, message):
    stocks, benchmark = _prices()
    with pytest.raises(ValueError, match=message):
        _analyze(change(stocks), benchmark)


@pytest.mark.parametrize("options, message", [
    ({"number_of_portfolio_weights": 0}, "number_of_portfolio_weights"),
    ({"trading_days_per_year": True}, "trading_days_per_year"),
    ({"number_of_monte_carlo_runs": 2.5}, "number_of_monte_carlo_runs"),
    ({"risk_free_rate": np.nan}, "risk_free_rate"),
])
def test_invalid_simulation_arguments_fail_before_downloading(options, message):
    with patch.object(RiskOptima, "download_data_yfinance") as download:
        with pytest.raises(ValueError, match=message):
            RiskOptima.run_portfolio_analysis(ALLOCATION, "2025-01-01", "2025-01-08", **options)
        download.assert_not_called()


@pytest.mark.parametrize("allocation", [{}, {"Z": 0}, {"Z": np.inf}])
def test_invalid_allocations_fail_before_downloading(allocation):
    with patch.object(RiskOptima, "download_data_yfinance") as download:
        with pytest.raises(ValueError, match="positive total"):
            RiskOptima.run_portfolio_analysis(allocation, "2025-01-01", "2025-01-08")
        download.assert_not_called()


def test_notebook_probability_workflow_preserves_the_global_rng(tmp_path, monkeypatch):
    module = importlib.import_module("riskoptima.riskoptima")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.plt, "show", lambda: None)
    stocks, benchmark = _prices()
    assets = pd.DataFrame({"Asset": list(ALLOCATION), "Portfolio": list(ALLOCATION.values()), "Label": ["Z fund", "A fund"]})
    before = np.random.get_state()
    try:
        with patch.object(RiskOptima, "download_data_yfinance", side_effect=(stocks[["A", "Z"]], benchmark)):
            RiskOptima.run_portfolio_probability_analysis(assets, "2025-01-01", "2025-01-08", "SPY", 0.01, 12, 3, 20,
                                                          random_state=41)
        assert any("Probability Distributions" in axes.get_title() for axes in module.plt.gcf().axes)
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
    finally:
        module.plt.close("all")
