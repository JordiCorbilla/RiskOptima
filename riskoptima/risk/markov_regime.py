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


def _filter_observations(values, means, variances, initial, transition):
    # Relative log likelihoods retain information when all densities underflow.
    with np.errstate(divide="ignore", over="ignore"):
        log_prior = np.log(initial)
        log_transition = np.log(transition)
        standardized = (values[:, None] - means) / np.sqrt(variances)
        log_emissions = -0.5 * (standardized**2 + np.log(variances))

    probabilities = np.empty_like(log_emissions)
    for row, log_emission in enumerate(log_emissions):
        reachable = np.isfinite(log_prior)
        offset = np.max(log_emission[reachable])
        if not np.isfinite(offset):
            raise ValueError("observations are too extreme for the fitted Gaussian variances")
        log_joint = np.full_like(log_prior, -np.inf)
        log_joint[reachable] = log_prior[reachable] + (log_emission[reachable] - offset)
        log_posterior = log_joint - logsumexp(log_joint)
        probabilities[row] = np.exp(log_posterior)
        log_prior = logsumexp(log_posterior[:, None] + log_transition, axis=0)
    return probabilities



def _gaussian_density(values: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    variances = np.maximum(variances, 1e-10)
    diff = values[:, None] - means[None, :]
    density = np.exp(-0.5 * diff**2 / variances[None, :]) / np.sqrt(2.0 * np.pi * variances[None, :])
    return np.maximum(density, 1e-300)


def _forward_backward(emissions: np.ndarray, initial: np.ndarray, transition: np.ndarray):
    n_obs, n_states = emissions.shape
    alpha = np.zeros((n_obs, n_states))
    beta = np.zeros((n_obs, n_states))
    scales = np.zeros(n_obs)

    alpha[0] = initial * emissions[0]
    scales[0] = max(alpha[0].sum(), 1e-300)
    alpha[0] /= scales[0]

    for t in range(1, n_obs):
        alpha[t] = emissions[t] * (alpha[t - 1] @ transition)
        scales[t] = max(alpha[t].sum(), 1e-300)
        alpha[t] /= scales[t]

    beta[-1] = 1.0
    for t in range(n_obs - 2, -1, -1):
        beta[t] = transition @ (emissions[t + 1] * beta[t + 1])
        beta[t] /= max(scales[t + 1], 1e-300)

    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi_sum = np.zeros((n_states, n_states))
    for t in range(n_obs - 1):
        xi = (
            alpha[t, :, None]
            * transition
            * emissions[t + 1, None, :]
            * beta[t + 1, None, :]
        )
        xi_sum += xi / max(xi.sum(), 1e-300)

    log_likelihood = float(np.log(scales).sum())
    return alpha, gamma, xi_sum, log_likelihood


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
    means = np.quantile(values, quantiles)
    means = means + rng.normal(0.0, max(values.std(ddof=0), 1e-6) * 1e-3, size=n_regimes)
    variances = np.repeat(max(values.var(ddof=0), 1e-8), n_regimes)
    transition = np.full((n_regimes, n_regimes), 0.05 / max(n_regimes - 1, 1))
    np.fill_diagonal(transition, 0.95)
    initial = np.repeat(1.0 / n_regimes, n_regimes)

    emissions = _gaussian_density(values, means, variances)
    alpha, gamma, xi_sum, log_likelihood = _forward_backward(emissions, initial, transition)
    for _ in range(n_iter):
        weights = np.maximum(gamma.sum(axis=0), 1e-300)

        initial = gamma[0] / gamma[0].sum()
        transition = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
        means = (gamma * values[:, None]).sum(axis=0) / weights
        variances = (gamma * (values[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
        variances = np.maximum(variances, 1e-10)

        previous_ll = log_likelihood
        emissions = _gaussian_density(values, means, variances)
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

