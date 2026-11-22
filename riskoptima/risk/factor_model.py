###############################################################################
#                               factor_model.py                                
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from riskoptima.optim.covariance import _tag_covariance, _validate_covariance, _validate_periods_per_year


@dataclass
class FactorRiskModel:
    factor_returns: pd.DataFrame
    exposures: Optional[pd.DataFrame] = None
    factor_cov: Optional[pd.DataFrame] = None
    specific_var: Optional[pd.Series] = None

    def fit(self, asset_returns: pd.DataFrame):
        import statsmodels.api as sm

        assets = pd.DataFrame(asset_returns).apply(pd.to_numeric, errors="coerce")
        factors = pd.DataFrame(self.factor_returns).apply(pd.to_numeric, errors="coerce")
        assets = assets.replace([np.inf, -np.inf], np.nan)
        factors = factors.replace([np.inf, -np.inf], np.nan)
        if assets.empty or factors.empty:
            raise ValueError("asset_returns and factor_returns must be non-empty")

        for name, frame in (("asset_returns", assets), ("factor_returns", factors)):
            if not frame.index.is_unique or not frame.columns.is_unique:
                raise ValueError(f"{name} must have unique dates and column labels")

        exposures = {}
        specific = {}

        for asset in assets.columns:
            aligned = pd.concat([assets[asset], factors], axis=1, join="inner").dropna()
            minimum_observations = len(factors.columns) + 2
            if len(aligned) < minimum_observations:
                raise ValueError(
                    f"asset {asset!r} requires at least {minimum_observations} complete observations"
                )
            y = aligned.iloc[:, 0].to_numpy()
            x = sm.add_constant(aligned.iloc[:, 1:].to_numpy(), has_constant="add")
            if np.linalg.matrix_rank(x) != x.shape[1]:
                raise ValueError(f"factor exposures for asset {asset!r} require a full-rank design")
            model = sm.OLS(y, x).fit()
            exposures[asset] = model.params[1:]
            resid = model.resid
            specific[asset] = np.var(resid, ddof=1)

        complete_factors = factors.dropna(how="any")
        if len(complete_factors) < 2:
            raise ValueError("factor_returns must contain at least two complete observations")
        self.exposures = pd.DataFrame(exposures, index=factors.columns).T
        self.factor_cov = _tag_covariance(complete_factors.cov(), None)
        self.specific_var = pd.Series(specific)
        return self

    def covariance_matrix(self, *, periods_per_year: int | None = None) -> pd.DataFrame:
        """Return covariance in input-period units, or annualize with an explicit frequency."""
        if self.exposures is None or self.factor_cov is None or self.specific_var is None:
            raise ValueError("Call fit() before computing covariance matrix.")
        if periods_per_year is not None:
            _validate_periods_per_year(periods_per_year)
        exposures = self.exposures
        if not exposures.index.is_unique or not exposures.columns.is_unique or not np.isfinite(exposures).all().all():
            raise ValueError("factor exposures must have unique labels and finite values")
        factor_cov = _validate_covariance(self.factor_cov)
        if factor_cov.attrs.get("covariance_unit") not in (None, "per_period"):
            raise ValueError("factor covariance must be in input-period units")
        if len(factor_cov) != exposures.shape[1] or not exposures.columns.isin(factor_cov.index).all():
            raise ValueError("factor covariance labels must match exposure columns")
        specific = self.specific_var.reindex(exposures.index).to_numpy()
        if not np.isfinite(specific).all() or (specific < 0).any():
            raise ValueError("specific variances must be finite, non-negative and aligned with assets")
        b = exposures.to_numpy()
        f = factor_cov.loc[exposures.columns, exposures.columns].to_numpy()
        covariance = (b @ f @ b.T + np.diag(specific)) * (periods_per_year or 1)
        return _tag_covariance(
            pd.DataFrame(covariance, index=exposures.index, columns=exposures.index), periods_per_year
        )
