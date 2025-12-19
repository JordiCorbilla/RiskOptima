from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import Constraints, factor_constraint_func


def _sum_to_one_constraint():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1}


def optimize_max_sharpe(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    constraints: Optional[Constraints] = None,
    risk_free_rate: float = 0.0,
    factor_exposures: Optional[pd.DataFrame] = None,
):
    if constraints is None:
        constraints = Constraints()

    n = expected_returns.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = constraints.bounds(n)

    def neg_sharpe(weights):
        port_ret = np.dot(weights, expected_returns.values)
        port_vol = np.sqrt(weights.T @ cov.values @ weights)
        return -(port_ret - risk_free_rate) / port_vol

    cons = [_sum_to_one_constraint()]
    cons += factor_constraint_func(init_guess, factor_exposures, constraints.factor_bounds)

    result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(result.x, index=expected_returns.index)


def optimize_min_variance(
    cov: pd.DataFrame,
    expected_returns: Optional[pd.Series] = None,
    target_return: Optional[float] = None,
    constraints: Optional[Constraints] = None,
    factor_exposures: Optional[pd.DataFrame] = None,
):
    if constraints is None:
        constraints = Constraints()

    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = constraints.bounds(n)

    def portfolio_vol(weights):
        return np.sqrt(weights.T @ cov.values @ weights)

    cons = [_sum_to_one_constraint()]
    if target_return is not None and expected_returns is not None:
        cons.append({"type": "eq", "fun": lambda w: np.dot(w, expected_returns.values) - target_return})
    cons += factor_constraint_func(init_guess, factor_exposures, constraints.factor_bounds)

    result = minimize(portfolio_vol, init_guess, method="SLSQP", bounds=bounds, constraints=cons)
    index = expected_returns.index if expected_returns is not None else cov.index
    return pd.Series(result.x, index=index)
