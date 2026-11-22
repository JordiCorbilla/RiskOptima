###############################################################################
#                              attribution.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Portfolio risk attribution analytics
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskoptima.optim.covariance import _validate_covariance
from ._tail import _empirical_tail_weights


def _weights_cov(weights, cov):
    w = pd.Series(weights, dtype=float)
    if not w.index.is_unique or not np.isfinite(w).all():
        raise ValueError("weights must have unique labels and be finite")
    if not isinstance(cov, pd.DataFrame):
        cov = pd.DataFrame(np.asarray(cov), index=w.index, columns=w.index)
    c = _validate_covariance(cov)
    if not w.index.isin(c.index).all():
        raise ValueError("covariance must contain all weight labels")
    return w, c.loc[w.index, w.index]


def _marginal_volatility(w: pd.Series, c: pd.DataFrame) -> pd.Series:
    weighted_covariance = c.to_numpy() @ w.to_numpy()
    variance = float(w.to_numpy() @ weighted_covariance)
    if variance <= 0:
        raise ValueError("portfolio volatility must be positive")
    return pd.Series(weighted_covariance / np.sqrt(variance), index=w.index, name="marginal_volatility_contribution")


def marginal_volatility_contribution(weights, cov) -> pd.Series:
    return _marginal_volatility(*_weights_cov(weights, cov))


def component_volatility_contribution(weights, cov, percent: bool = False) -> pd.Series:
    w, c = _weights_cov(weights, cov)
    contrib = (w * _marginal_volatility(w, c)).rename("component_volatility_contribution")
    return contrib / contrib.sum() if percent else contrib


def marginal_var(weights, cov, confidence: float = 0.95) -> pd.Series:
    z = norm.ppf(confidence)
    return (z * marginal_volatility_contribution(weights, cov)).rename("marginal_var")


def component_var(weights, cov, confidence: float = 0.95, percent: bool = False) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    contrib = (w * marginal_var(w, cov, confidence)).rename("component_var")
    return contrib / contrib.sum() if percent else contrib


def component_expected_shortfall(returns, weights, confidence: float = 0.95) -> pd.Series:
    """Attribute empirical ES by allocating fractional tail mass equally across ties."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    data = pd.DataFrame(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if not data.columns.is_unique:
        raise ValueError("return asset labels must be unique")
    w = pd.Series(weights, dtype=float)
    if not w.index.is_unique:
        raise ValueError("weight asset labels must be unique")
    w = w.reindex(data.columns)
    if not np.isfinite(w).all():
        raise ValueError("finite weights must align with return columns")
    component_losses = -data.mul(w, axis=1)
    tail_weights = _empirical_tail_weights(component_losses.sum(axis=1).to_numpy(), confidence)
    return pd.Series(tail_weights @ component_losses.to_numpy(), index=w.index, name="component_expected_shortfall")


def component_cvar(returns, weights, confidence: float = 0.95) -> pd.Series:
    return component_expected_shortfall(returns, weights, confidence).rename("component_cvar")


def factor_risk_contribution(weights, factor_exposures, factor_cov) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    exposures = pd.DataFrame(factor_exposures).reindex(w.index).astype(float)
    if not np.isfinite(exposures).all().all() or not exposures.index.is_unique or not exposures.columns.is_unique:
        raise ValueError("factor exposures must be finite and uniquely labeled")
    factor_load, f_cov = _weights_cov(exposures.T.dot(w), factor_cov)
    factor_variance = float(factor_load.T @ f_cov @ factor_load)
    if factor_variance <= 0:
        raise ValueError("factor variance must be positive")
    marginal = f_cov.dot(factor_load) / np.sqrt(factor_variance)
    return (factor_load * marginal).rename("factor_risk_contribution")


def tracking_error_contribution(weights, benchmark_weights, cov, percent: bool = False) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    b = pd.Series(benchmark_weights, dtype=float).reindex(w.index).fillna(0.0)
    active = w - b
    contrib = component_volatility_contribution(active, cov, percent=False).rename("tracking_error_contribution")
    return contrib / contrib.sum() if percent else contrib

