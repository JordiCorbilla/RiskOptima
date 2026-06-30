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


def _weights_cov(weights, cov):
    w = pd.Series(weights, dtype=float)
    c = pd.DataFrame(cov, index=w.index, columns=w.index).astype(float) if not isinstance(cov, pd.DataFrame) else cov.reindex(index=w.index, columns=w.index).astype(float)
    if not np.isfinite(c.values).all():
        raise ValueError("cov contains NaN or infinite values")
    return w, c


def marginal_volatility_contribution(weights, cov) -> pd.Series:
    w, c = _weights_cov(weights, cov)
    vol = float(np.sqrt(w.values.T @ c.values @ w.values))
    if vol <= 0:
        raise ValueError("portfolio volatility must be positive")
    return pd.Series(c.values @ w.values / vol, index=w.index, name="marginal_volatility_contribution")


def component_volatility_contribution(weights, cov, percent: bool = False) -> pd.Series:
    w, _ = _weights_cov(weights, cov)
    mvc = marginal_volatility_contribution(w, cov)
    contrib = (w * mvc).rename("component_volatility_contribution")
    return contrib / contrib.sum() if percent else contrib


def marginal_var(weights, cov, confidence: float = 0.95) -> pd.Series:
    z = norm.ppf(confidence)
    return (z * marginal_volatility_contribution(weights, cov)).rename("marginal_var")


def component_var(weights, cov, confidence: float = 0.95, percent: bool = False) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    contrib = (w * marginal_var(w, cov, confidence)).rename("component_var")
    return contrib / contrib.sum() if percent else contrib


def component_expected_shortfall(returns, weights, confidence: float = 0.95) -> pd.Series:
    data = pd.DataFrame(returns).astype(float).dropna(how="any")
    w = pd.Series(weights, dtype=float).reindex(data.columns)
    if w.isna().any():
        raise ValueError("weights must align with return columns")
    portfolio = data.dot(w)
    threshold = portfolio.quantile(1.0 - confidence)
    tail = data.loc[portfolio <= threshold]
    if tail.empty:
        return pd.Series(np.zeros(len(w)), index=w.index, name="component_expected_shortfall")
    asset_tail_losses = -tail.mean()
    contrib = (w * asset_tail_losses).rename("component_expected_shortfall")
    return contrib


def component_cvar(returns, weights, confidence: float = 0.95) -> pd.Series:
    return component_expected_shortfall(returns, weights, confidence).rename("component_cvar")


def factor_risk_contribution(weights, factor_exposures, factor_cov) -> pd.Series:
    w = pd.Series(weights, dtype=float)
    exposures = pd.DataFrame(factor_exposures).reindex(w.index).astype(float)
    f_cov = pd.DataFrame(factor_cov, index=exposures.columns, columns=exposures.columns).astype(float)
    factor_load = exposures.T.dot(w)
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

