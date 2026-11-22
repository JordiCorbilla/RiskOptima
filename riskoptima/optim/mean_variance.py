###############################################################################
#                               mean_variance.py                               
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .constraints import Constraints, category_constraint_func, exposure_constraint_func
from .covariance import _validate_covariance


@dataclass
class OptimizationResult:
    weights: pd.Series
    objective: str
    expected_return: float
    volatility: float
    sharpe: Optional[float]
    success: bool
    message: str
    diagnostics: dict = field(default_factory=dict)
    active_constraints: dict = field(default_factory=dict)


def _sum_to_one_constraint():
    return {"type": "eq", "fun": lambda w: np.sum(w) - 1}


def _validate_inputs(expected_returns: Optional[pd.Series], cov: pd.DataFrame):
    cov = _validate_covariance(cov, require_annualized=True)
    if expected_returns is not None:
        expected_returns = pd.Series(expected_returns, dtype=float)
        if expected_returns.empty or not expected_returns.index.is_unique:
            raise ValueError("expected_returns must have unique asset labels and be non-empty")
        missing = [asset for asset in expected_returns.index if asset not in cov.index or asset not in cov.columns]
        if missing:
            raise ValueError(f"covariance matrix is missing assets: {missing}")
        cov = cov.loc[expected_returns.index, expected_returns.index]
        if not np.isfinite(expected_returns.values).all():
            raise ValueError("expected_returns contains NaN or infinite values")
    else:
        expected_returns = pd.Series(np.zeros(cov.shape[0]), index=cov.index, dtype=float)
    return expected_returns, cov


def _aligned_optional_frame(frame: Optional[pd.DataFrame], index: pd.Index, name: str):
    if frame is None:
        return None
    missing = [asset for asset in index if asset not in frame.index]
    if missing:
        raise ValueError(f"{name} is missing assets: {missing}")
    return frame.reindex(index)


def _aligned_optional_series(series, index: pd.Index, name: str):
    if series is None:
        return None
    series = pd.Series(series, dtype=float)
    missing = [asset for asset in index if asset not in series.index]
    if missing:
        raise ValueError(f"{name} is missing assets: {missing}")
    return series.reindex(index)


def _portfolio_metrics(weights, expected_returns, cov, risk_free_rate):
    ret = float(np.dot(weights, expected_returns.values))
    variance = float(weights.T @ cov.values @ weights)
    vol = float(np.sqrt(max(variance, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
    return ret, vol, float(sharpe) if np.isfinite(sharpe) else np.nan


def _build_constraints(
    constraints: Constraints,
    previous_weights: Optional[pd.Series],
    factor_exposures: Optional[pd.DataFrame],
    metadata: Optional[pd.DataFrame],
    allow_missing_factors: bool,
):
    cons = [_sum_to_one_constraint()]
    leverage_limit = constraints.leverage_limit
    # Gross leverage equals the budget for long-only portfolios. Adding the
    # same active constraint twice can produce an inaccurate SLSQP optimum.
    if leverage_limit is not None and (constraints.weight_bounds[0] < 0 or leverage_limit < 1):
        cons.append({"type": "ineq", "fun": lambda w: leverage_limit - np.sum(np.abs(w))})
    if constraints.turnover_limit is not None:
        if previous_weights is None:
            raise ValueError("previous_weights is required when turnover_limit is set")
        prev = previous_weights.values.astype(float)
        cons.append({"type": "ineq", "fun": lambda w, p=prev, lim=constraints.turnover_limit: lim - np.sum(np.abs(w - p))})
    cons += exposure_constraint_func(factor_exposures, constraints.factor_bounds, allow_missing_factors)
    cons += category_constraint_func(metadata, "sector", constraints.sector_bounds)
    cons += category_constraint_func(metadata, "asset_class", constraints.asset_class_bounds)
    return cons


def _active_constraints(weights, constraints: Constraints, previous_weights=None, factor_exposures=None, metadata=None):
    active: dict[str, object] = {
        "gross_leverage": float(np.sum(np.abs(weights))),
    }
    if previous_weights is not None:
        active["turnover"] = float(np.sum(np.abs(weights - previous_weights.values)))
    if factor_exposures is not None and constraints.factor_bounds:
        active["factor_exposures"] = {
            factor: float(np.dot(weights, factor_exposures[factor].values))
            for factor in constraints.factor_bounds
            if factor in factor_exposures.columns
        }
    if metadata is not None and constraints.sector_bounds:
        active["sector_weights"] = {
            sector: float(weights[metadata["sector"].values == sector].sum())
            for sector in constraints.sector_bounds
        }
    if metadata is not None and constraints.asset_class_bounds:
        active["asset_class_weights"] = {
            asset_class: float(weights[metadata["asset_class"].values == asset_class].sum())
            for asset_class in constraints.asset_class_bounds
        }
    return active


def _format_result(result, objective, expected_returns, cov, risk_free_rate, constraints, previous_weights, factor_exposures, metadata):
    weights = pd.Series(result.x, index=expected_returns.index)
    ret, vol, sharpe = _portfolio_metrics(weights.values, expected_returns, cov, risk_free_rate)
    return OptimizationResult(
        weights=weights,
        objective=objective,
        expected_return=ret,
        volatility=vol,
        sharpe=sharpe,
        success=bool(result.success),
        message=str(result.message),
        diagnostics={"nit": getattr(result, "nit", None), "fun": float(result.fun) if np.isfinite(result.fun) else result.fun},
        active_constraints=_active_constraints(weights.values, constraints, previous_weights, factor_exposures, metadata),
    )


def optimize_max_sharpe(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    constraints: Optional[Constraints] = None,
    risk_free_rate: float = 0.0,
    factor_exposures: Optional[pd.DataFrame] = None,
    previous_weights: Optional[pd.Series] = None,
    metadata: Optional[pd.DataFrame] = None,
    allow_missing_factors: bool = False,
    return_result: bool = False,
):
    """Optimize with annual decimal expected returns, rates and covariance.

    Untagged covariance inputs are assumed annualized for compatibility.
    """
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    if constraints is None:
        constraints = Constraints()

    expected_returns, cov = _validate_inputs(expected_returns, cov)
    factor_exposures = _aligned_optional_frame(factor_exposures, expected_returns.index, "factor_exposures")
    metadata = _aligned_optional_frame(metadata, expected_returns.index, "metadata")
    previous_weights = _aligned_optional_series(previous_weights, expected_returns.index, "previous_weights")
    n = expected_returns.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = constraints.bounds(n)

    def neg_sharpe(weights):
        port_ret = np.dot(weights, expected_returns.values)
        port_vol = np.sqrt(max(float(weights.T @ cov.values @ weights), 0.0))
        if port_vol <= 0:
            return 1e12
        return -(port_ret - risk_free_rate) / port_vol

    cons = _build_constraints(constraints, previous_weights, factor_exposures, metadata, allow_missing_factors)

    result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=cons)
    if not result.success:
        raise ValueError(f"Max Sharpe optimization failed: {result.message}")
    opt_result = _format_result(result, "max_sharpe", expected_returns, cov, risk_free_rate, constraints, previous_weights, factor_exposures, metadata)
    return opt_result if return_result else opt_result.weights


def optimize_min_variance(
    cov: pd.DataFrame,
    expected_returns: Optional[pd.Series] = None,
    target_return: Optional[float] = None,
    constraints: Optional[Constraints] = None,
    factor_exposures: Optional[pd.DataFrame] = None,
    previous_weights: Optional[pd.Series] = None,
    metadata: Optional[pd.DataFrame] = None,
    allow_missing_factors: bool = False,
    return_result: bool = False,
):
    """Minimize annualized volatility using covariance of decimal returns.

    Untagged covariance inputs are assumed annualized for compatibility.
    """
    if target_return is not None and not np.isfinite(target_return):
        raise ValueError("target_return must be finite")
    if constraints is None:
        constraints = Constraints()

    expected_returns, cov = _validate_inputs(expected_returns, cov)
    factor_exposures = _aligned_optional_frame(factor_exposures, expected_returns.index, "factor_exposures")
    metadata = _aligned_optional_frame(metadata, expected_returns.index, "metadata")
    previous_weights = _aligned_optional_series(previous_weights, expected_returns.index, "previous_weights")
    n = cov.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = constraints.bounds(n)

    values = cov.to_numpy()
    scale = np.max(np.abs(values))
    normalized = values / scale if scale else values

    def portfolio_variance(weights):
        return float(weights @ normalized @ weights)

    def variance_gradient(weights):
        return 2.0 * (normalized @ weights)

    cons = _build_constraints(constraints, previous_weights, factor_exposures, metadata, allow_missing_factors)
    if target_return is not None and expected_returns is not None:
        cons.append({"type": "eq", "fun": lambda w: np.dot(w, expected_returns.values) - target_return})

    result = minimize(
        portfolio_variance, init_guess, method="SLSQP", jac=variance_gradient,
        bounds=bounds, constraints=cons, options={"ftol": 1e-12},
    )
    if not result.success:
        raise ValueError(f"Minimum variance optimization failed: {result.message}")
    opt_result = _format_result(result, "min_variance", expected_returns, cov, 0.0, constraints, previous_weights, factor_exposures, metadata)
    return opt_result if return_result else opt_result.weights
