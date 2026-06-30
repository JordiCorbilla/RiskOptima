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


@dataclass
class MarkovRegimeModel:
    """Fitted univariate Gaussian Hidden Markov Model."""

    means: pd.Series
    variances: pd.Series
    transition_matrix: pd.DataFrame
    initial_probabilities: pd.Series
    regime_probabilities: pd.DataFrame
    regimes: pd.Series
    log_likelihood: float


def _clean_series(observations) -> pd.Series:
    series = pd.Series(observations, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("observations must contain at least one finite value")
    if len(series) < 10:
        raise ValueError("at least 10 observations are required to fit a Markov regime model")
    return series


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
    return gamma, xi_sum, log_likelihood


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

    values = series.to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    quantiles = np.linspace(0.1, 0.9, n_regimes)
    means = np.quantile(values, quantiles)
    means = means + rng.normal(0.0, max(values.std(ddof=0), 1e-6) * 1e-3, size=n_regimes)
    variances = np.repeat(max(values.var(ddof=0), 1e-8), n_regimes)
    transition = np.full((n_regimes, n_regimes), 0.05 / max(n_regimes - 1, 1))
    np.fill_diagonal(transition, 0.95)
    initial = np.repeat(1.0 / n_regimes, n_regimes)

    previous_ll = -np.inf
    gamma = np.zeros((len(values), n_regimes))
    log_likelihood = -np.inf
    for _ in range(n_iter):
        emissions = _gaussian_density(values, means, variances)
        gamma, xi_sum, log_likelihood = _forward_backward(emissions, initial, transition)
        weights = np.maximum(gamma.sum(axis=0), 1e-300)

        initial = gamma[0] / gamma[0].sum()
        transition = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
        means = (gamma * values[:, None]).sum(axis=0) / weights
        variances = (gamma * (values[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
        variances = np.maximum(variances, 1e-10)

        if abs(log_likelihood - previous_ll) < tol:
            break
        previous_ll = log_likelihood

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
    )


def classify_markov_regimes(model: MarkovRegimeModel) -> pd.Series:
    """Return the most likely regime sequence from a fitted model."""
    return model.regimes.copy()


def regime_summary(observations, regimes) -> pd.DataFrame:
    """Summarize observations by inferred regime."""
    series = _clean_series(observations)
    states = pd.Series(regimes, index=series.index).reindex(series.index)
    data = pd.concat([series.rename("observation"), states.rename("regime")], axis=1).dropna()
    summary = data.groupby("regime")["observation"].agg(["count", "mean", "std", "min", "max"])
    return summary

