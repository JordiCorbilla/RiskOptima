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
import statsmodels.api as sm


@dataclass
class FactorRiskModel:
    factor_returns: pd.DataFrame
    exposures: Optional[pd.DataFrame] = None
    factor_cov: Optional[pd.DataFrame] = None
    specific_var: Optional[pd.Series] = None

    def fit(self, asset_returns: pd.DataFrame):
        assets = pd.DataFrame(asset_returns).apply(pd.to_numeric, errors="coerce")
        factors = pd.DataFrame(self.factor_returns).apply(pd.to_numeric, errors="coerce")
        assets = assets.replace([np.inf, -np.inf], np.nan)
        factors = factors.replace([np.inf, -np.inf], np.nan)
        if assets.empty or factors.empty:
            raise ValueError("asset_returns and factor_returns must be non-empty")

        exposures = {}
        specific = {}

        for asset in assets.columns:
            aligned = pd.concat([assets[asset].rename("asset_return"), factors], axis=1, join="inner").dropna()
            minimum_observations = len(factors.columns) + 2
            if len(aligned) < minimum_observations:
                raise ValueError(
                    f"asset {asset!r} requires at least {minimum_observations} complete observations"
                )
            y = aligned["asset_return"]
            x = sm.add_constant(aligned[factors.columns])
            model = sm.OLS(y.to_numpy(), x.to_numpy()).fit()
            exposures[asset] = model.params[1:]
            resid = model.resid
            specific[asset] = np.var(resid, ddof=1)

        complete_factors = factors.dropna(how="any")
        if len(complete_factors) < 2:
            raise ValueError("factor_returns must contain at least two complete observations")
        self.exposures = pd.DataFrame(exposures, index=factors.columns).T
        self.factor_cov = complete_factors.cov()
        self.specific_var = pd.Series(specific)
        return self

    def covariance_matrix(self) -> pd.DataFrame:
        if self.exposures is None or self.factor_cov is None or self.specific_var is None:
            raise ValueError("Call fit() before computing covariance matrix.")
        b = self.exposures.values
        f = self.factor_cov.values
        d = np.diag(self.specific_var.reindex(self.exposures.index).values)
        cov = b @ f @ b.T + d
        return pd.DataFrame(cov, index=self.exposures.index, columns=self.exposures.index)
