###############################################################################
#                               estimators.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Historical and realized volatility estimators
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(data, name: str = "value") -> pd.Series:
    series = pd.Series(data, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError(f"{name} must contain at least one finite value")
    return series


def _returns(data, input_type: str = "returns", log_returns: bool = False) -> pd.Series:
    input_type = input_type.lower()
    if input_type not in {"returns", "prices"}:
        raise ValueError("input_type must be 'returns' or 'prices'")
    series = _as_series(data, "data")
    if input_type == "prices":
        if (series <= 0).any():
            raise ValueError("prices must be positive")
        if log_returns:
            return np.log(series / series.shift(1)).dropna().rename("returns")
        return series.pct_change().dropna().rename("returns")
    return series.rename("returns")


def historical_volatility(
    data,
    input_type: str = "returns",
    periods_per_year: int = 252,
    log_returns: bool = False,
    ddof: int = 0,
) -> float:
    """Annualized close-to-close historical volatility."""
    returns = _returns(data, input_type=input_type, log_returns=log_returns)
    if len(returns) <= ddof:
        raise ValueError("not enough observations for the requested ddof")
    return float(returns.std(ddof=ddof) * np.sqrt(periods_per_year))


def rolling_volatility(
    data,
    window: int = 21,
    input_type: str = "returns",
    periods_per_year: int = 252,
    log_returns: bool = False,
    ddof: int = 0,
) -> pd.Series:
    """Annualized rolling close-to-close volatility."""
    if window < 1:
        raise ValueError("window must be positive")
    returns = _returns(data, input_type=input_type, log_returns=log_returns)
    return (returns.rolling(window).std(ddof=ddof) * np.sqrt(periods_per_year)).rename("rolling_volatility")


def realized_volatility(
    intraday_data,
    input_type: str = "prices",
    periods_per_year: int = 252,
    log_returns: bool = True,
) -> float:
    """
    Annualized realized volatility from intraday returns.

    For intraday prices, the estimator is sqrt(sum(r_t^2) * periods_per_year).
    """
    returns = _returns(intraday_data, input_type=input_type, log_returns=log_returns)
    return float(np.sqrt(np.square(returns).sum() * periods_per_year))


def ewma_volatility(
    data,
    input_type: str = "returns",
    lambda_: float = 0.94,
    periods_per_year: int = 252,
    log_returns: bool = False,
) -> float:
    """RiskMetrics-style exponentially weighted volatility."""
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    returns = _returns(data, input_type=input_type, log_returns=log_returns)
    squared = np.square(returns.to_numpy(dtype=float))
    weights = (1.0 - lambda_) * lambda_ ** np.arange(len(squared) - 1, -1, -1)
    weights = weights / weights.sum()
    variance = float(np.dot(weights, squared))
    return float(np.sqrt(variance * periods_per_year))

