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


_PER_PERIOD = "per_period"
_ANNUALIZED = "annualized"
_UNIT_ATTRIBUTE = "covariance_unit"


def _tag_covariance(cov: pd.DataFrame, periods_per_year: int | None) -> pd.DataFrame:
    cov.attrs[_UNIT_ATTRIBUTE] = _PER_PERIOD if periods_per_year is None else _ANNUALIZED
    if periods_per_year is not None:
        cov.attrs["periods_per_year"] = periods_per_year
    return cov


def _validate_periods_per_year(periods_per_year: int) -> None:
    if not isinstance(periods_per_year, (int, np.integer)) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")


def _labeled_covariance(matrix) -> pd.DataFrame:
    cov = pd.DataFrame(matrix, dtype=float)
    if cov.empty or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance matrix must be non-empty and square")
    if not cov.index.is_unique or not cov.columns.is_unique:
        raise ValueError("covariance asset labels must be unique")
    if not cov.index.isin(cov.columns).all():
        raise ValueError("covariance row and column labels must match")
    cov = cov.reindex(columns=cov.index)
    if not np.isfinite(cov.to_numpy()).all():
        raise ValueError("covariance matrix contains NaN or infinite values")
    if isinstance(matrix, pd.DataFrame):
        cov.attrs = matrix.attrs.copy()
    return cov


def _validate_covariance(matrix, *, require_annualized: bool = False) -> pd.DataFrame:
    cov = _labeled_covariance(matrix)
    unit = cov.attrs.get(_UNIT_ATTRIBUTE)
    if unit not in (None, _PER_PERIOD, _ANNUALIZED):
        raise ValueError("covariance_unit must be 'annualized' or 'per_period'")
    periods = cov.attrs.get("periods_per_year")
    if periods is not None:
        _validate_periods_per_year(periods)
        if unit != _ANNUALIZED:
            raise ValueError("periods_per_year metadata requires annualized covariance")
    if require_annualized and unit == _PER_PERIOD:
        raise ValueError("optimization requires annualized covariance (decimal returns squared)")

    values = cov.to_numpy()
    scale = np.max(np.abs(values))
    # Relative tolerance retains the same checks for daily or annual units.
    if scale:
        normalized = values / scale
        if np.max(np.abs(normalized - normalized.T)) > 1e-10:
            raise ValueError("covariance matrix must be symmetric")
        symmetric = 0.5 * normalized + 0.5 * normalized.T
        if np.linalg.eigvalsh(symmetric)[0] < -1e-10 * len(cov):
            raise ValueError("covariance matrix must be positive semidefinite")
        result = pd.DataFrame(symmetric * scale, index=cov.index, columns=cov.columns)
        result.attrs = cov.attrs.copy()
        return result
    return cov


def _returns_frame(returns) -> pd.DataFrame:
    data = pd.DataFrame(returns, dtype=float)
    if not data.columns.is_unique:
        raise ValueError("return asset labels must be unique")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if data.empty:
        raise ValueError("returns must contain at least one complete finite row")
    return data


def sample_covariance(returns, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized covariance of complete decimal return rows, with sample ddof=1."""
    _validate_periods_per_year(periods_per_year)
    data = _returns_frame(returns)
    if len(data) < 2:
        raise ValueError("sample covariance requires at least two complete finite rows")
    return _tag_covariance(data.cov() * periods_per_year, periods_per_year)


def ewma_covariance(returns, lambda_: float = 0.94, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized exponentially weighted covariance of chronological decimal returns."""
    _validate_periods_per_year(periods_per_year)
    if not 0.0 < lambda_ < 1.0:
        raise ValueError("lambda_ must be between 0 and 1")
    data = _returns_frame(returns)
    centered = data - data.mean()
    weights = (1.0 - lambda_) * lambda_ ** np.arange(len(data) - 1, -1, -1)
    weights = weights / weights.sum()
    cov = centered.to_numpy().T @ (centered.to_numpy() * weights[:, None])
    result = pd.DataFrame(cov * periods_per_year, index=data.columns, columns=data.columns)
    return _tag_covariance(result, periods_per_year)


def ledoit_wolf_covariance(returns, periods_per_year: int = 252) -> pd.DataFrame:
    """Annualized Ledoit-Wolf covariance of complete decimal returns."""
    _validate_periods_per_year(periods_per_year)
    data = _returns_frame(returns)
    try:
        from sklearn.covariance import LedoitWolf
    except ImportError as exc:
        raise ImportError("scikit-learn is required for ledoit_wolf_covariance") from exc
    estimator = LedoitWolf().fit(data.to_numpy())
    result = pd.DataFrame(estimator.covariance_ * periods_per_year, index=data.columns, columns=data.columns)
    return _tag_covariance(result, periods_per_year)


def nearest_psd(matrix, epsilon: float = 1e-10) -> pd.DataFrame | np.ndarray:
    """Explicitly symmetrize and clip eigenvalues to the non-negative floor."""
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be non-negative and finite")
    cov = _labeled_covariance(matrix)
    values = cov.to_numpy()
    sym = 0.5 * values + 0.5 * values.T
    eigvals, eigvecs = np.linalg.eigh(sym)
    psd = (eigvecs * np.maximum(eigvals, epsilon)) @ eigvecs.T
    psd = 0.5 * psd + 0.5 * psd.T
    if isinstance(matrix, pd.DataFrame):
        result = pd.DataFrame(psd, index=cov.index, columns=cov.columns)
        result.attrs = cov.attrs.copy()
        return result.reindex(columns=matrix.columns)
    return psd
