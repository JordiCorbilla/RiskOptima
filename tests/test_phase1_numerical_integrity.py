"""Independent numerical and point-in-time regressions for the 3.0 audit."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
import QuantLib as ql
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import norm, poisson
from sklearn.linear_model import LinearRegression

from riskoptima import RiskOptima
from riskoptima.backtest import Strategy, run_backtest
from riskoptima.core import BacktestConfig
from riskoptima.credit import credit_cvar
from riskoptima.optim import SimpleCostModel, optimize_max_sharpe, optimize_min_variance, sample_covariance
from riskoptima.options import implied_volatility, merton_jump_diffusion_call
from riskoptima.reporting import build_market_risk_report
from riskoptima.risk import FactorRiskModel, component_expected_shortfall, fit_markov_regime_model


def test_merton_matches_quantlib_poisson_mixture_and_discounted_stock_mean():
    spot, strike, term, rate, sigma = 100., 105., 1.25, .04, .2
    intensity, mean, std = 4., -.14, .28
    discount = np.exp(-rate * term)
    compensated_forward = spot * np.exp(rate * term - intensity * term * np.expm1(mean + std**2 / 2))
    reference = sum(
        poisson.pmf(count, term * intensity)
        * ql.blackFormula(
            ql.Option.Call, strike,
            compensated_forward * np.exp(count * (mean + std**2 / 2)),
            np.sqrt(term * sigma**2 + count * std**2), discount,
        )
        for count in range(40)
    )
    inputs = spot, strike, term, rate, sigma, intensity, mean, std
    actual = RiskOptima.merton_jump_diffusion(*inputs, 500_000, 100, random_state=240)
    assert actual == pytest.approx(reference, abs=0.30)
    assert actual == RiskOptima.merton_jump_diffusion(*inputs, 500_000, 1, random_state=240)
    # At a near-zero strike the risk-neutral call should approximate spot.
    martingale = merton_jump_diffusion_call(spot, 1e-8, term, rate, sigma, intensity, mean, std, 500_000, 240)
    assert martingale == pytest.approx(spot, abs=0.50)


def test_merton_conditional_jump_log_standard_deviation(monkeypatch):
    class FixedRng:
        def poisson(self, rate, size):
            assert rate == 2 and size == 2
            return np.array([0, 4])

        def standard_normal(self, size):
            return np.ones(size)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedRng())
    expected = (
        max(100 * np.exp(-2 * np.expm1(.1 + .2**2/2)) - 100, 0)
        + max(100 * np.exp(-2 * np.expm1(.1 + .2**2/2) + 4*.1 + 2*.2) - 100, 0)
    ) / 2
    assert RiskOptima.merton_jump_diffusion(100, 100, 1, 0, 0, 2, .1, .2, 2, random_state=42) == pytest.approx(expected)


def test_merton_expiry_no_jumps_and_invalid_inputs():
    assert RiskOptima.merton_jump_diffusion(110, 100, 0, .05, .2, 3, .1, .3) == 10.
    call = RiskOptima.merton_jump_diffusion(100, 100, 1, .05, 0, 0, .1, .3, random_state=1)
    assert call == pytest.approx(100 * (1 - np.exp(-.05)), abs=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        RiskOptima.merton_jump_diffusion(100, 100, 1, .05, .2, -1, .1, .3)
    with pytest.raises(ValueError, match="num_steps"):
        RiskOptima.merton_jump_diffusion(100, 100, 1, .05, .2, 1, .1, .3, num_steps=0)


class LastKnownClose(Strategy):
    def generate_target_weights(self, date, prices, state=None):
        assert (prices.index < date).all()
        weights = pd.Series(0., index=prices.columns)
        if not prices.empty:
            weights.loc[prices.iloc[-1].idxmax()] = 1.
        # Strategies can manipulate their input snapshot without corrupting later runs.
        prices.loc[:, :] = -1.
        state.positions.loc[:] = 999.
        state.cash = -100.
        return weights


@given(replacement=st.lists(st.floats(10, 200), min_size=4, max_size=4))
@settings(max_examples=20)
def test_backtest_has_prefix_invariance_and_isolates_the_strategy(replacement):
    dates = pd.date_range("2024-01-01", periods=6)
    prices = pd.DataFrame({"A": [110., 99., 115., 111., 114., 116.], "B": [90., 101., 100., 99., 102., 103.]}, index=dates)
    modified = prices.copy()
    modified.iloc[4:] = np.reshape(replacement, (2, 2))
    no_cost = SimpleCostModel(spread_bps=0)
    before, weights_before = run_backtest(prices, LastKnownClose(), cost_model=no_cost)
    after, weights_after = run_backtest(modified, LastKnownClose(), cost_model=no_cost)
    pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4])
    pd.testing.assert_frame_equal(weights_before.iloc[:4], weights_after.iloc[:4])
    assert weights_before.iloc[0].sum() == 0
    assert weights_before.iloc[1]["A"] == pytest.approx(1.)


def test_backtest_adv_is_lagged_and_does_not_trade_missing_quotes():
    class Switch(Strategy):
        def generate_target_weights(self, date, prices, state=None):
            return pd.Series({"A": float(prices.empty), "B": float(not prices.empty)})

    class RecordingCost:
        def __init__(self):
            self.inputs = []

        def estimate_cost(self, notional, adv=None):
            if notional:
                self.inputs.append(adv)
            return 0.

    dates = pd.date_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"A": [100., np.nan, 120.], "B": [np.nan, 79., 80.]}, index=dates)
    adv = pd.DataFrame({"A": [1000., 1100., 1200.], "B": [2000., 2100., 2200.]}, index=dates)
    costs = RecordingCost()
    equity, weights = run_backtest(prices, Switch(), BacktestConfig(initial_cash=1000.), costs, adv)
    assert costs.inputs == [None, 1100., 2100.]
    assert equity.iloc[1]["Turnover"] == 0.
    assert equity.iloc[2]["PortfolioValue"] == pytest.approx(1200.)
    assert weights.iloc[2]["B"] == 1.
    with pytest.raises(ValueError, match="unique"):
        run_backtest(prices.iloc[[0, 0]], Switch())


def test_legacy_performance_table_handles_benchmark_frame_and_series_equally():
    returns = pd.Series([np.nan, .03, -.01, .02], name="SPY")
    as_frame = RiskOptima.calculate_performance_metrics(returns.to_frame(), returns.to_frame(), .02, 4.0, True)
    as_series = RiskOptima.calculate_performance_metrics(returns, returns, .02, 4.0)
    assert as_frame == as_series
    assert "Info Ratio: 0.00" in as_frame


def test_ml_metrics_match_independent_expanding_window_predictions():
    x = np.arange(36, dtype=float)
    X = x.reshape(-1, 1)
    y = np.sin(x / 3) + x*x/100
    scores, squared_errors = [], []
    for cutoff in range(6, 36, 6):
        design = np.column_stack([np.ones(cutoff), x[:cutoff]])
        intercept, slope = np.linalg.lstsq(design, y[:cutoff], rcond=None)[0]
        future = y[cutoff:cutoff+6]
        is_squared_error = (future - intercept - slope*x[cutoff:cutoff+6])**2
        scores.append(1 - is_squared_error.sum() / ((future - future.mean())**2).sum())
        squared_errors.append(is_squared_error.mean())
    model = LinearRegression()
    r2, mse = RiskOptima.evaluate_model(model, X, y)
    assert r2 == pytest.approx(np.mean(scores), abs=1e-12)
    assert mse == pytest.approx(np.mean(squared_errors), abs=1e-12)
    assert RiskOptima.predict_with_model(model, X[-1:]) == RiskOptima.predict_with_model(model, X)
    with pytest.raises(ValueError, match="chronological"):
        RiskOptima.evaluate_model(model, pd.DataFrame(X, index=x[::-1]), y)


def test_ml_targets_future_returns_and_fit_preprocessing_inside_gapped_folds(monkeypatch):
    from sklearn.impute import SimpleImputer

    changes = np.tile([.02, -.005, .012, .001], 15)
    close = pd.Series(100 * np.cumprod(1 + changes), index=pd.date_range("2024-01-01", periods=60), name="A")
    close.iloc[25] = np.nan
    complete = close.iloc[9:]
    targets = (complete.shift(-1) / complete - 1).dropna()
    fit_inputs = []
    original_fit = SimpleImputer.fit

    def record_fit(self, X, *args, **kwargs):
        fit_inputs.append(X.copy())
        return original_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", record_fit)
    monkeypatch.setattr(RiskOptima, "download_data_yfinance", lambda *args: close.iloc[::-1].to_frame())
    value, rating = RiskOptima.generate_stock_predictions("A", "2024-01-01", "2024-04-01")
    assert np.isfinite([value, rating]).all()
    assert abs(value) < .03
    assert len(fit_inputs) == 6
    assert fit_inputs[-1].index.equals(targets.index)
    assert fit_inputs[0].index.equals(targets.index[:len(targets)-5*(len(targets)//6)-1])
    assert fit_inputs[-1].loc[targets.index[0], "return_1"] == pytest.approx(changes[9], abs=1e-12)
    assert fit_inputs[-1].isna().any().any()
    # Independent one-step target: a deterministic daily return should forecast itself.
    constant = 100 * 1.02**np.arange(len(close))
    monkeypatch.setattr(RiskOptima, "download_data_yfinance", lambda *args: pd.Series(constant, index=close.index))
    value, _ = RiskOptima.generate_stock_predictions("A", "2024-01-01", "2024-04-01")
    assert value == pytest.approx(.02, abs=1e-12)


def test_black_litterman_aligns_views_and_ignores_nonpositive_cv_scores():
    closes = pd.DataFrame({"B": [100., 102., 101., 104., 106.], "A": [100., 101., 103., 102., 110.]})
    historical = pd.concat({"Close": closes}, axis=1)
    prior = {"A": .05, "B": .04}
    views = {"B": .10, "A": .08}
    confidence = {"B": -.5, "A": .6}
    cov = np.cov(closes[["A", "B"]].to_numpy()[1:] / closes[["A", "B"]].to_numpy()[:-1] - 1, rowvar=False) * 260
    expected = np.array([.05, .04]) + cov[:, 0] * (.08 - .05) / (cov[0, 0] + 1/.6)
    assert RiskOptima.black_litterman_adjust_returns(prior, views, confidence, historical) == pytest.approx(expected)
    assert RiskOptima.black_litterman_adjust_returns(prior, views, {"A": 0, "B": -.1}, historical) == pytest.approx([.05, .04])


def test_legacy_optimizer_identifies_an_asset_with_missing_prices(monkeypatch):
    closes = pd.DataFrame({"B": [100., 102., 101.], "A": [np.nan, np.nan, np.nan]})
    historical = pd.concat({"Close": closes}, axis=1)
    monkeypatch.setattr(RiskOptima, "fetch_historical_stock_prices", lambda *args: historical)
    with pytest.raises(ValueError, match="Historical prices.*A"):
        RiskOptima.perform_mean_variance_optimization(["A", "B"], "2022-01-01", "2023-01-01", .25)


def test_minimum_variance_accuracy_at_small_covariance_scale():
    covariance = np.diag([.125, .125, .0625]) * 1e-7
    weights = optimize_min_variance(covariance)
    np.testing.assert_allclose(weights, [.25, .25, .5], rtol=0, atol=1e-6)


@given(variances=st.lists(st.floats(.01, .20), min_size=3, max_size=3), order=st.permutations(["A", "B", "C"]))
@settings(max_examples=18)
def test_min_variance_matches_analytic_weights_under_asset_label_permutations(variances, order):
    original = ["A", "B", "C"]
    covariance = pd.DataFrame(np.diag(variances), index=original, columns=original)
    reordered = covariance.loc[list(order), original[::-1]]
    weights = optimize_min_variance(reordered).reindex(original)
    precision = 1 / np.asarray(variances)
    np.testing.assert_allclose(weights, precision/precision.sum(), rtol=0, atol=.003)


@pytest.mark.parametrize(("covariance", "message"), [
    (pd.DataFrame([[.04, .01], [.02, .09]]), "symmetric"),
    (pd.DataFrame([[.04, .07], [.07, .09]]), "semidefinite"),
    (pd.DataFrame([[.04, 0], [0, .09]], columns=["A", "B"]), "must match"),
    (pd.DataFrame([[.04, 0], [0, .09]], columns=["A", "A"]), "unique"),
    (pd.DataFrame([[.04, np.nan], [np.nan, .09]]), "NaN"),
])
def test_optimizer_rejects_invalid_covariance(covariance, message):
    with pytest.raises(ValueError, match=message):
        optimize_min_variance(covariance)


def test_factor_covariance_units_and_factor_alignment():
    factors = pd.DataFrame({"X": [.01, -.02, .01, -.03, .02, -.01], "Y": [-.02, .01, .01, .04, -.03, .02]})
    beta = np.array([[.4, .8], [.2, -.6]])
    assets = pd.DataFrame(factors.to_numpy() @ beta.T + .005, columns=["A", "B"])
    fit = FactorRiskModel(factors).fit(assets)
    per_period = fit.covariance_matrix()
    expected = 252 * beta @ np.cov(factors, rowvar=False) @ beta.T
    np.testing.assert_allclose(fit.covariance_matrix(periods_per_year=252), expected, atol=1e-12)
    fit.factor_cov = fit.factor_cov.loc[["Y", "X"], ["X", "Y"]]
    np.testing.assert_allclose(fit.covariance_matrix(periods_per_year=252), expected, atol=1e-12)
    expected_returns = assets.mean() * 252
    with pytest.raises(ValueError, match="annualized"):
        optimize_max_sharpe(expected_returns, per_period)
    with pytest.raises(ValueError, match="periods_per_year"):
        fit.covariance_matrix(periods_per_year=0)
    with pytest.raises(ValueError, match="at least two"):
        sample_covariance(assets.iloc[:1])


@given(volatility=st.floats(.10, .65), expiry=st.floats(.35, 3.0), option=st.sampled_from(["call", "put"]))
@settings(max_examples=25)
def test_implied_volatility_round_trips_quantlib_and_exposes_residual(volatility, expiry, option):
    option_type = ql.Option.Call if option == "call" else ql.Option.Put
    spot, strike, rate, dividend = 100., 105., -.01, .02
    forward = spot * np.exp((rate - dividend) * expiry)
    discount = np.exp(-rate * expiry)
    market = ql.blackFormula(option_type, strike, forward, volatility * np.sqrt(expiry), discount)
    result = implied_volatility(market, spot, strike, expiry, rate, option, dividend, return_result=True)
    assert result.converged and abs(result.price_error) <= 1e-8
    assert result.volatility == pytest.approx(volatility, abs=3e-8)


def test_iv_budget_exhaustion_has_an_explicit_status_and_float_warning():
    market = ql.blackFormula(ql.Option.Call, 100., 100., .2)
    result = implied_volatility(market, 100, 100, 1., 0., max_iter=1, return_result=True)
    assert result.iterations == 1
    assert not result.converged and result.price_error > 1e-8
    with pytest.warns(FutureWarning, match="non-converged"):
        assert implied_volatility(market, 100, 100, 1., 0., max_iter=1) == result.volatility
    with pytest.raises(ValueError, match="not bracketed"):
        implied_volatility(market, 100, 100, 1., 0., high=.1, return_result=True)


def test_markov_outputs_match_enumerated_paths_under_final_parameters():
    values = np.array([-.03, -.015, -.01, .01, .018, .012, -.008, -.02, -.01, .011])
    model = fit_markov_regime_model(values, n_regimes=2, n_iter=1, random_state=4)
    emissions = norm.pdf(values[:, None], loc=model.means, scale=np.sqrt(model.variances))
    initial = model.initial_probabilities.to_numpy()
    transitions = model.transition_matrix.to_numpy()
    filtered, smoothed = [], np.zeros_like(emissions)
    for end in range(1, len(values)+1):
        mass = np.zeros(2)
        for path in product(range(2), repeat=end):
            prob = initial[path[0]] * emissions[0, path[0]]
            for i in range(1, end):
                prob *= transitions[path[i-1], path[i]] * emissions[i, path[i]]
            mass[path[-1]] += prob
            if end == len(values):
                smoothed[np.arange(end), path] += prob
        filtered.append(mass / mass.sum())
    assert model.log_likelihood == pytest.approx(np.log(mass.sum()), abs=1e-12)
    np.testing.assert_allclose(model.filtered_regime_probabilities, filtered, atol=1e-12)
    np.testing.assert_allclose(model.smoothed_regime_probabilities, smoothed/mass.sum(), atol=1e-12)
    np.testing.assert_array_equal(model.regimes, smoothed.argmax(axis=1))
    assert np.max(np.abs(np.asarray(filtered)[:-1] - smoothed[:-1]/mass.sum())) > .01


@given(losses=st.lists(st.integers(-5, 15), min_size=1, max_size=30),
       confidence=st.floats(.02, .998))
@settings(max_examples=30)
def test_expected_shortfall_matches_quantile_integral(losses, confidence):
    count = len(losses)
    expected = sum(
        amount * max(0, (i+1)/count - max(i/count, confidence))
        for i, amount in enumerate(sorted(losses))
    ) / (1-confidence)
    assert credit_cvar(losses, confidence) == pytest.approx(expected, abs=1e-11)


def test_expected_shortfall_ties_missing_data_and_attribution_reconcile():
    returns = pd.DataFrame({"A": [-.2, -.2, 0., 0., np.nan], "B": [-.2, 0., -.2, 0., -.9]})
    weights = pd.Series({"A": .5, "B": .5})
    confidence = .625
    expected = (.2 + .5*.1) / 1.5
    components = component_expected_shortfall(returns, weights, confidence)
    assert components.to_dict() == pytest.approx({"A": expected/2, "B": expected/2})
    permuted = component_expected_shortfall(returns.iloc[[2, 4, 1, 0, 3], ::-1], weights, confidence)
    pd.testing.assert_series_equal(components, permuted.reindex(components.index))
    portfolio = returns.dropna().dot(weights)
    assert RiskOptima.cvar_historic(portfolio, 37.5) == pytest.approx(expected)
    assert build_market_risk_report(portfolio, confidence_levels=[confidence]).metrics["expected_shortfall"][confidence] == pytest.approx(expected)
    assert credit_cvar([0, 0, 0, 10], .6) == pytest.approx(6.25)
    with pytest.raises(ValueError, match="finite"):
        credit_cvar([np.nan, np.inf], .95)
