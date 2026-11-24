###############################################################################
#                              markov_regime.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Gaussian Hidden Markov Model market regime detection
###############################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp


@dataclass
class MarkovRegimeModel:
    """Fitted univariate Gaussian Hidden Markov Model.

    Filtered probabilities use observations through each row under fixed
    parameters. Fitting those parameters on later data still creates
    look-ahead if the results are used for historical trading.
    """

    means: pd.Series
    variances: pd.Series
    transition_matrix: pd.DataFrame
    initial_probabilities: pd.Series
    regime_probabilities: pd.DataFrame
    regimes: pd.Series
    log_likelihood: float
    filtered_regime_probabilities: pd.DataFrame | None = None

    @property
    def smoothed_regime_probabilities(self) -> pd.DataFrame:
        """Full-sample state probabilities, also available as regime_probabilities."""
        return self.regime_probabilities

    def filter_probabilities(self, observations, *, continue_from_fit: bool = True) -> pd.DataFrame:
        """Filter a new ordered series with fixed parameters.

        By default, advance the fit's terminal filtered posterior one step
        for the first new observation. Set continue_from_fit to False to
        start an independent sequence at the model's initial probabilities.
        Dated continuations must begin strictly after the last training row.
        Missing observations are skipped without advancing the state.
        """
        series = _finite_observations(observations)
        index = series.index
        if isinstance(index, pd.MultiIndex) or not index.is_unique or not index.is_monotonic_increasing or index.hasnans:
            raise ValueError("observations must have a unique chronological index without missing labels")

        states = self.means.index
        means = self.means.to_numpy(dtype=float)
        if not len(states) or not states.is_unique or not np.isfinite(means).all():
            raise ValueError("the model must have unique states with finite means")
        transition = self.transition_matrix.reindex(index=states, columns=states).to_numpy(dtype=float)
        variances = self.variances.reindex(states).to_numpy(dtype=float)
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError("the model must have positive finite variances")
        if not np.isfinite(transition).all() or np.any(transition < 0):
            raise ValueError("the transition matrix must contain non-negative finite probabilities")
        if not np.allclose(transition.sum(axis=1), 1.0, rtol=0, atol=1e-10):
            raise ValueError("transition rows must sum to one")

        if continue_from_fit:
            training = self.filtered_regime_probabilities
            if training is None or training.empty:
                raise ValueError("continuation requires a terminal filtered training posterior")
            date_indexes = (pd.DatetimeIndex, pd.PeriodIndex)
            if isinstance(index, date_indexes) and isinstance(training.index, date_indexes):
                if not training.index.is_unique or not training.index.is_monotonic_increasing or training.index.hasnans:
                    raise ValueError("training observations must have a unique chronological index")
                try:
                    overlaps = index[0] <= training.index[-1]
                except TypeError as exc:
                    raise ValueError("training and continuation timestamps must be comparable") from exc
                if overlaps:
                    raise ValueError("continuation observations must be strictly after the training cutoff")
            prior = training.iloc[-1].reindex(states).to_numpy(dtype=float)
        else:
            prior = self.initial_probabilities.reindex(states).to_numpy(dtype=float)
        if not np.isfinite(prior).all() or np.any(prior < 0) or not np.isclose(prior.sum(), 1.0, rtol=0, atol=1e-10):
            raise ValueError("the starting probabilities must be non-negative and sum to one")
        if continue_from_fit:
            prior = prior @ transition

        filtered = _filter_observations(series.to_numpy(dtype=float), means, variances, prior, transition)
        return pd.DataFrame(filtered, index=index, columns=states)


def _finite_observations(observations) -> pd.Series:
    series = pd.Series(observations, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("observations must contain at least one finite value")
    return series


def _clean_series(observations) -> pd.Series:
    series = _finite_observations(observations)
    if len(series) < 10:
        raise ValueError("at least 10 observations are required to fit a Markov regime model")
    return series


def _gaussian_log_density(values: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        standardized = (values[:, None] - means) / np.sqrt(variances)
        return -0.5 * (standardized**2 + np.log(variances) + np.log(2.0 * np.pi))


def _log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


def _log_forward(log_emissions, initial, log_transition):
    log_prior = _log_probabilities(initial)
    log_alpha = np.empty_like(log_emissions)
    centered = np.full_like(log_emissions, -np.inf)
    log_scales = np.empty(len(log_emissions))
    for row, log_emission in enumerate(log_emissions):
        reachable = np.isfinite(log_prior)
        offset = np.max(log_emission, where=reachable, initial=-np.inf)
        if not np.isfinite(offset):
            raise ValueError("observations are too extreme for the fitted Gaussian variances")

        # Center before including the prior so a large common negative
        # likelihood cannot round away the relative state probabilities.
        np.subtract(log_emission, offset, out=centered[row], where=reachable)
        log_joint = log_prior + centered[row]
        normalizer = np.logaddexp.reduce(log_joint)
        log_alpha[row] = log_joint - normalizer
        log_scales[row] = offset + normalizer
        log_prior = np.logaddexp.reduce(log_alpha[row, :, None] + log_transition, axis=0)
    return log_alpha, centered, log_scales


def _filter_observations(values, means, variances, initial, transition):
    emissions = _gaussian_log_density(values, means, variances)
    log_alpha, _, _ = _log_forward(emissions, initial, _log_probabilities(transition))
    return np.exp(log_alpha)


def _forward_backward(log_emissions: np.ndarray, initial: np.ndarray, transition: np.ndarray):
    log_transition = _log_probabilities(transition)
    log_alpha, centered, log_scales = _log_forward(log_emissions, initial, log_transition)
    log_beta = np.full_like(log_alpha, -np.inf)
    log_beta[-1, np.isfinite(log_alpha[-1])] = 0.0
    transition_counts = np.zeros_like(transition)
    for row in range(len(log_alpha) - 2, -1, -1):
        reachable = np.isfinite(log_alpha[row])
        next_weights = centered[row + 1] + log_beta[row + 1]
        next_states = log_transition + next_weights
        log_weights = np.logaddexp.reduce(next_states, axis=1)
        offset = np.max(log_weights, where=reachable, initial=-np.inf)
        np.subtract(log_weights, offset, out=log_beta[row], where=reachable)
        # Accumulate pair probabilities in place without a time-by-state-pair tensor.
        log_pairs = log_alpha[row, :, None] + next_states
        transition_counts += np.exp(log_pairs - np.logaddexp.reduce(log_pairs, axis=None))

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    with np.errstate(over="ignore"):
        log_likelihood = float(log_scales.sum())
    if not np.isfinite(log_likelihood):
        raise ValueError("log likelihood is outside the representable range")
    return np.exp(log_alpha), np.exp(log_gamma), transition_counts, log_likelihood


def fit_markov_regime_model(
    observations,
    n_regimes: int = 3,
    n_iter: int = 100,
    tol: float = 1e-5,
    random_state: int | None = None,
) -> MarkovRegimeModel:
    """
    Fit a univariate Gaussian Hidden Markov Model to returns or features.

    Regimes are sorted by fitted mean from lowest to highest, so regime 0 is the
    weakest/most defensive state and the final regime is the strongest state.
    """
    series = _clean_series(observations)
    if n_regimes < 2:
        raise ValueError("n_regimes must be at least 2")
    if n_regimes > min(8, len(series) // 3):
        raise ValueError("n_regimes is too high for the number of observations")
    if n_iter < 1:
        raise ValueError("n_iter must be positive")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be non-negative and finite")

    values = series.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    quantiles = np.linspace(0.1, 0.9, n_regimes)
    with np.errstate(over="ignore", invalid="ignore"):
        variance = values.var(ddof=0)
        means = np.quantile(values, quantiles)
        means = means + rng.normal(0.0, max(np.sqrt(variance), 1e-6) * 1e-3, size=n_regimes)
    if not np.isfinite(variance) or not np.isfinite(means).all():
        raise ValueError("observations are too extreme for a finite Gaussian fit")
    variances = np.repeat(max(variance, 1e-8), n_regimes)
    transition = np.full((n_regimes, n_regimes), 0.05 / max(n_regimes - 1, 1))
    np.fill_diagonal(transition, 0.95)
    initial = np.repeat(1.0 / n_regimes, n_regimes)

    emissions = _gaussian_log_density(values, means, variances)
    alpha, gamma, xi_sum, log_likelihood = _forward_backward(emissions, initial, transition)
    for _ in range(n_iter):
        weights = gamma.sum(axis=0)
        visited = weights > 0
        state_weights = gamma[:, visited] / weights[visited]
        initial = gamma[0]
        row_counts = xi_sum.sum(axis=1, keepdims=True)
        transition = np.divide(xi_sum, row_counts, out=transition.copy(), where=row_counts > 0)
        means[visited] = (state_weights * values[:, None]).sum(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            residuals = np.sqrt(state_weights) * (values[:, None] - means[visited])
            variances[visited] = np.maximum((residuals**2).sum(axis=0), 1e-10)
        if not np.isfinite(means).all() or not np.isfinite(variances).all():
            raise ValueError("observations are too extreme for a finite Gaussian fit")

        previous_ll = log_likelihood
        emissions = _gaussian_log_density(values, means, variances)
        alpha, gamma, xi_sum, log_likelihood = _forward_backward(emissions, initial, transition)
        if abs(log_likelihood - previous_ll) < tol:
            break

    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    transition = transition[np.ix_(order, order)]
    initial = initial[order]
    gamma = gamma[:, order]
    labels = [f"Regime {i}" for i in range(n_regimes)]

    probabilities = pd.DataFrame(gamma, index=series.index, columns=labels)
    regimes = pd.Series(np.argmax(gamma, axis=1), index=series.index, name="regime")
    return MarkovRegimeModel(
        means=pd.Series(means, index=labels, name="mean"),
        variances=pd.Series(variances, index=labels, name="variance"),
        transition_matrix=pd.DataFrame(transition, index=labels, columns=labels),
        initial_probabilities=pd.Series(initial, index=labels, name="initial_probability"),
        regime_probabilities=probabilities,
        regimes=regimes,
        log_likelihood=log_likelihood,
        filtered_regime_probabilities=pd.DataFrame(alpha[:, order], index=series.index, columns=labels),
    )


def classify_markov_regimes(model: MarkovRegimeModel) -> pd.Series:
    """Return the per-row most probable smoothed regime from a fitted model."""
    return model.regimes.copy()


def regime_summary(observations, regimes) -> pd.DataFrame:
    """Summarize observations by inferred regime."""
    series = _clean_series(observations)
    states = pd.Series(regimes, index=series.index).reindex(series.index)
    data = pd.concat([series.rename("observation"), states.rename("regime")], axis=1).dropna()
    summary = data.groupby("regime")["observation"].agg(["count", "mean", "std", "min", "max"])
    return summary

