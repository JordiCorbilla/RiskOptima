###############################################################################
#                               covariance.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Covariance estimators and matrix utilities
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd


def _returns_frame(returns) -> pd.DataFrame:
    data = pd.DataFrame(returns).astype(float).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if data.empty:
        raise ValueError("returns must contain at least one complete finite row")
    return data


def sample_covariance(returns, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized sample covariance matrix."""
    data = _returns_frame(returns)
    return data.cov() * periods_per_year


def ewma_covariance(returns, lambda_: float = 0.94, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized exponentially weighted covariance matrix."""
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    data = _returns_frame(returns)
    centered = data - data.mean()
    weights = (1.0 - lambda_) * lambda_ ** np.arange(len(data) - 1, -1, -1)
    weights = weights / weights.sum()
    cov = centered.to_numpy().T @ (centered.to_numpy() * weights[:, None])
    return pd.DataFrame(cov * periods_per_year, index=data.columns, columns=data.columns)


def ledoit_wolf_covariance(returns, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized Ledoit-Wolf covariance when scikit-learn is installed."""
    data = _returns_frame(returns)
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as exc:
        raise ImportError("scikit-learn is required for ledoit_wolf_covariance") from exc
    estimator = LedoitWolf().fit(data.to_numpy())
    return pd.DataFrame(estimator.covariance_ * periods_per_year, index=data.columns, columns=data.columns)


def nearest_psd(matrix, epsilon: float = 1e-10) -> pd.DataFrame | np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite approximation."""
    is_frame = isinstance(matrix, pd.DataFrame)
    values = pd.DataFrame(matrix).to_numpy(dtype=float) if is_frame else np.asarray(matrix, dtype=float)
    if values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be square")
    sym = 0.5 * (values + values.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, epsilon)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    psd = 0.5 * (psd + psd.T)
    if is_frame:
        return pd.DataFrame(psd, index=matrix.index, columns=matrix.columns)
    return psd

