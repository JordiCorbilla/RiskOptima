###############################################################################
#                                simulation.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Credit migration and loss simulation
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_transition_matrix(matrix):
    """
    Validates that a rating transition matrix is square, non-negative, and row-stochastic.
    """
    mat = pd.DataFrame(matrix, copy=True)
    if mat.empty:
        raise ValueError("transition matrix cannot be empty")
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("transition matrix must be square")
    values = mat.to_numpy(dtype=float)
    if np.any(values < 0):
        raise ValueError("transition probabilities cannot be negative")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("transition matrix rows must sum to 1")
    return True


def simulate_rating_migration(initial_ratings, transition_matrix, periods, random_state=None):
    """
    Simulates credit rating migration paths.

    Returns a DataFrame with one row per period, including period 0.
    """
    if periods < 0:
        raise ValueError("periods must be non-negative")

    mat = pd.DataFrame(transition_matrix, copy=True)
    validate_transition_matrix(mat)
    states = list(mat.columns)
    row_labels = list(mat.index)
    if set(row_labels) != set(states):
        raise ValueError("transition matrix index and columns must contain the same ratings")

    ratings = pd.Series(initial_ratings, dtype=object).reset_index(drop=True)
    unknown = set(ratings.unique()) - set(states)
    if unknown:
        raise ValueError(f"unknown initial ratings: {sorted(unknown)}")

    rng = np.random.default_rng(random_state)
    paths = [ratings.copy()]
    current = ratings.copy()

    for _ in range(periods):
        next_ratings = []
        for rating in current:
            probs = mat.loc[rating, states].to_numpy(dtype=float)
            next_ratings.append(rng.choice(states, p=probs))
        current = pd.Series(next_ratings, dtype=object)
        paths.append(current.copy())

    result = pd.DataFrame(paths)
    result.index.name = "period"
    result.columns = [f"obligor_{i}" for i in range(len(ratings))]
    return result


def _column(df: pd.DataFrame, *names: str) -> str:
    lookup = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(f"portfolio must contain one of: {', '.join(names)}")


def simulate_credit_losses(portfolio_df, n_sims=10000, random_state=None):
    """
    Simulates default losses for a PD/LGD/EAD portfolio.
    """
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if not isinstance(portfolio_df, pd.DataFrame):
        raise TypeError("portfolio_df must be a pandas DataFrame")

    pd_col = _column(portfolio_df, "pd", "probability_of_default")
    lgd_col = _column(portfolio_df, "lgd", "loss_given_default")
    ead_col = _column(portfolio_df, "ead", "exposure_at_default")

    pd_values = portfolio_df[pd_col].astype(float).to_numpy()
    lgd_values = portfolio_df[lgd_col].astype(float).to_numpy()
    ead_values = portfolio_df[ead_col].astype(float).to_numpy()

    if np.any((pd_values < 0) | (pd_values > 1)):
        raise ValueError("PD values must be between 0 and 1")
    if np.any((lgd_values < 0) | (lgd_values > 1)):
        raise ValueError("LGD values must be between 0 and 1")
    if np.any(ead_values < 0):
        raise ValueError("EAD values must be non-negative")

    rng = np.random.default_rng(random_state)
    defaults = rng.binomial(1, pd_values, size=(int(n_sims), len(pd_values)))
    losses = defaults * (lgd_values * ead_values)
    return losses.sum(axis=1)
