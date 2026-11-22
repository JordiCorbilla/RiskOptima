"""Independent forward-filter checks with a frozen train/validation split."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from riskoptima.risk import fit_markov_regime_model
from riskoptima.risk.markov_regime import MarkovRegimeModel


@pytest.fixture
def dated_fit():
    dates = pd.bdate_range("2024-01-01", periods=14)
    training = pd.Series([-.03, -.02, -.01, .03, .035, .02, -.018, -.012, -.025, .028], index=dates[:10])
    future = pd.Series([-.01, np.nan, .025, .03], index=dates[10:])
    model = fit_markov_regime_model(training, n_regimes=2, n_iter=2, random_state=4)
    return model, training, future


def _direct_reference(model, series, continued):
    transition = model.transition_matrix.to_numpy()
    if continued:
        prior = model.filtered_regime_probabilities.iloc[-1].to_numpy() @ transition
    else:
        prior = model.initial_probabilities.to_numpy()
    expected = []
    for observation in series.dropna():
        density = norm.pdf(observation, loc=model.means.to_numpy(), scale=np.sqrt(model.variances.to_numpy()))
        posterior = prior * density
        posterior /= posterior.sum()
        expected.append(posterior)
        prior = posterior @ transition
    return np.array(expected)


@pytest.mark.parametrize("continued", [True, False])
def test_frozen_probabilities_match_direct_forward_recursion_and_preserve_prefix(dated_fit, continued):
    model, _, future = dated_fit
    before = model.filtered_regime_probabilities.copy(deep=True)
    actual = model.filter_probabilities(future, continue_from_fit=continued)
    np.testing.assert_allclose(actual, _direct_reference(model, future, continued), rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual.sum(axis=1), 1., rtol=0, atol=1e-12)
    assert actual.index.equals(future.dropna().index)
    # Dropping an intervening missing observation removes pandas' inferred
    # frequency from the longer result, while the prefix values and dates agree.
    pd.testing.assert_frame_equal(
        actual.iloc[:1], model.filter_probabilities(future.iloc[:1], continue_from_fit=continued),
        check_freq=False, check_exact=True,
    )
    pd.testing.assert_frame_equal(before, model.filtered_regime_probabilities)


def test_filter_rejects_temporal_overlap_out_of_order_and_missing_observations(dated_fit):
    model, training, future = dated_fit
    with pytest.raises(ValueError, match="after the training cutoff"):
        model.filter_probabilities(training.iloc[-2:])
    with pytest.raises(ValueError, match="chronological"):
        model.filter_probabilities(future.iloc[::-1])
    with pytest.raises(ValueError, match="finite"):
        model.filter_probabilities([np.nan, np.inf])
    # A restart may intentionally evaluate an independent series on the old dates.
    restarted = model.filter_probabilities(training.iloc[:1], continue_from_fit=False)
    assert restarted.index.equals(training.index[:1])


def test_log_filter_distinguishes_extreme_values_after_both_densities_underflow():
    states = pd.Index(["narrow", "wide"])
    model = MarkovRegimeModel(
        means=pd.Series([0., 0.], index=states),
        variances=pd.Series([1., 4.], index=states),
        transition_matrix=pd.DataFrame([[.9, .1], [.2, .8]], index=states, columns=states),
        initial_probabilities=pd.Series([.6, .4], index=states),
        regime_probabilities=pd.DataFrame(columns=states),
        regimes=pd.Series(dtype=int),
        log_likelihood=0.,
    )
    assert np.all(norm.pdf(100, loc=0, scale=[1., 2.]) == 0.0)
    probabilities = model.filter_probabilities([100, 0], continue_from_fit=False)
    # At 100 the log likelihood ratio favoring the wide state exceeds 3,749.
    # At zero the narrow state's likelihood is twice that of the wide state.
    np.testing.assert_allclose(probabilities, [[0., 1.], [1/3, 2/3]], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="terminal filtered"):
        model.filter_probabilities([1.])
