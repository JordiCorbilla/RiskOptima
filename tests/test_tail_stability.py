"""Independent references for HMM likelihoods and Merton pricing in the tails."""

from decimal import Decimal, localcontext
from itertools import product
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.stats import norm, poisson

from riskoptima.options import merton_jump_diffusion_call
from riskoptima.risk import fit_markov_regime_model
from riskoptima.risk.markov_regime import _forward_backward, _gaussian_log_density


def _enumerated_hmm(values, means, variances, initial, transition):
    density = norm.logpdf(values[:, None], loc=means, scale=np.sqrt(variances))

    def path_weights(length):
        paths = np.asarray(list(product(range(len(initial)), repeat=length)))
        with np.errstate(divide="ignore"):
            logs = np.log(initial[paths[:, 0]]) + density[np.arange(length), paths].sum(axis=1)
            logs += np.log(transition[paths[:, :-1], paths[:, 1:]]).sum(axis=1)
        offset = logs.max()
        weights = np.exp(logs - offset)
        return paths, weights / weights.sum(), offset + np.log(weights.sum())

    filtered = []
    for end in range(1, len(values) + 1):
        paths, weights, likelihood = path_weights(end)
        filtered.append(np.bincount(paths[:, -1], weights=weights, minlength=len(initial)))
    smoothed = np.array([np.bincount(states, weights=weights, minlength=len(initial)) for states in paths.T])
    counts = np.zeros((len(initial), len(initial)))
    for path, weight in zip(paths, weights, strict=True):
        np.add.at(counts, (path[:-1], path[1:]), weight)
    return np.array(filtered), smoothed, counts, likelihood


@pytest.mark.parametrize("transition", [np.array([[.9, .1], [.2, .8]]), np.eye(2)])
def test_hmm_e_step_matches_enumerated_paths_after_both_emissions_underflow(transition):
    initial = np.array([.6, .4])
    means, variances = np.array([0., 0.]), np.array([1., 4.])
    values = np.array([0., 100., 0., -110.])
    assert np.all(norm.pdf(values[1], loc=means, scale=np.sqrt(variances)) == 0)
    expected = _enumerated_hmm(values, means, variances, initial, transition)
    actual = _forward_backward(_gaussian_log_density(values, means, variances), initial, transition)

    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=0, atol=1e-10)
    assert actual[0][1, 1] == 1.
    np.testing.assert_allclose(actual[1].sum(axis=1), 1., rtol=0, atol=1e-12)
    assert actual[2].sum() == pytest.approx(len(values) - 1, abs=1e-12)


def test_hmm_fit_retains_e_step_consistency_for_widely_separated_regimes():
    values = np.array([0., .0011, -.0009, .0002, .0001, -.0004, 100., 105., 108., 99., 107.])
    model = fit_markov_regime_model(values, n_regimes=2, n_iter=25, random_state=18)
    expected = _enumerated_hmm(
        values, model.means.to_numpy(), model.variances.to_numpy(),
        model.initial_probabilities.to_numpy(), model.transition_matrix.to_numpy(),
    )
    np.testing.assert_allclose(model.filtered_regime_probabilities, expected[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.smoothed_regime_probabilities, expected[1], atol=1e-12, rtol=0)
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1., atol=1e-12, rtol=0)
    assert model.log_likelihood == pytest.approx(expected[3], abs=1e-10, rel=0)
    first_update = fit_markov_regime_model(values, n_regimes=2, n_iter=1, random_state=18)
    assert model.log_likelihood >= first_update.log_likelihood


def test_hmm_rejects_unrepresentable_fit_without_nan_results():
    with pytest.raises(ValueError, match="too extreme"):
        fit_markov_regime_model([-1e200, 1e200] * 5, n_regimes=2, random_state=1)


@pytest.mark.parametrize("rate", [-750., 1e16])
def test_merton_discounting_handles_extreme_rates_without_erasing_diffusion(rate, monkeypatch):
    class PairedShocks:
        def standard_normal(self, size):
            assert size == 2
            return np.array([-1., 1.])

    monkeypatch.setattr(np.random, "default_rng", lambda seed: PairedShocks())
    # At -750 the discounted strike dominates both paths. At 1e16 its
    # contribution is below a float ulp, and both lognormal shocks survive.
    expected = 0. if rate < 0 else 50. * (exp(-.02 - .2) + exp(-.02 + .2))
    with np.errstate(over="raise", invalid="raise"):
        actual = merton_jump_diffusion_call(100., 100., 1., rate, .2, 0., 0., 0., 2, 7)
    assert actual == pytest.approx(expected, abs=1e-12, rel=0)


def test_merton_scales_rare_jump_compensation_before_exponentiating():
    intensity, log_multiplier = 1e-310, 720.
    with localcontext() as ctx:
        ctx.prec = 70
        # Only the no-jump paths are sampled at this intensity. The drift
        # still includes the expected jump multiplier.
        compensation = Decimal.from_float(intensity) * (Decimal(720).exp() - 1)
        expected_sample = float((-compensation).exp() - Decimal.from_float(1e-250))
    actual = merton_jump_diffusion_call(1., 1e-250, 1., 0., 0., intensity, log_multiplier, 0., 8, 3)
    assert actual > 0.
    assert actual == pytest.approx(expected_sample, abs=0., rel=2e-10)


def test_merton_preserves_small_deterministic_payoff_just_above_strike():
    strike = 100.
    spot = np.nextafter(strike, np.inf)
    price = merton_jump_diffusion_call(spot, strike, 1., 0., 0., 0., 0., 0., 2, 7)
    assert price == pytest.approx(spot - strike, abs=0., rel=2e-14)


def test_merton_averages_representable_price_with_an_overflowing_individual_payoff(monkeypatch):
    class OneExtremeShock:
        def standard_normal(self, size):
            return np.array([710.5] + [-1.] * (size - 1))

    monkeypatch.setattr(np.random, "default_rng", lambda seed: OneExtremeShock())
    with localcontext() as ctx:
        ctx.prec = 70
        expected = float((Decimal(710).exp() - 1) / 10)
    actual = merton_jump_diffusion_call(1., 1., 1., 0., 1., 0., 0., 0., 10, 3)
    assert np.isfinite(actual)
    assert actual == pytest.approx(expected, abs=0., rel=2e-13)
    with pytest.raises(OverflowError, match="outside the representable range"):
        merton_jump_diffusion_call(1., 1., 1., 0., 1., 0., 0., 0., 1, 3)


def test_merton_deep_out_of_money_at_high_jump_counts_matches_analytic_mixture():
    spot, strike, term, rate, sigma = 100., 350., 2., -.035, .16
    intensity, jump_mean, jump_std, paths = 18., .04, .14, 500_000
    counts = np.arange(201)
    mass = poisson.pmf(counts, intensity * term)
    discount = exp(-rate * term)
    drift = (rate - intensity * np.expm1(jump_mean + jump_std**2 / 2)) * term
    variances = term * sigma**2 + counts * jump_std**2
    log_mean = log(spot) + drift - 0.5 * term * sigma**2 + counts * jump_mean
    forward = np.exp(log_mean + 0.5 * variances)
    threshold = (log_mean - log(strike)) / np.sqrt(variances)
    # Conditional first moment above the strike minus the strike times
    # the probability of finishing in the money, integrated over counts.
    conditional_call = forward * norm.cdf(threshold + np.sqrt(variances)) - strike * norm.cdf(threshold)
    expected = float(discount * (mass @ conditional_call))
    second_payoff = (
        np.exp(2 * log_mean + 2 * variances) * norm.cdf(threshold + 2 * np.sqrt(variances))
        - 2 * strike * forward * norm.cdf(threshold + np.sqrt(variances))
        + strike**2 * norm.cdf(threshold)
    )
    standard_error = sqrt((mass @ second_payoff * discount**2 - expected**2) / paths)
    actual = merton_jump_diffusion_call(
        spot, strike, term, rate, sigma, intensity, jump_mean, jump_std, paths, 320,
    )
    assert poisson.sf(counts[-1], intensity * term * exp(2 * jump_mean + 2 * jump_std**2)) < 1e-50
    assert actual == pytest.approx(expected, abs=5 * standard_error, rel=0)
