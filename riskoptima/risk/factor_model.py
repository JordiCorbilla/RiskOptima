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
        aligned = asset_returns.join(self.factor_returns, how="inner")
        factors = self.factor_returns.loc[aligned.index]
        exposures = {}
        specific = {}

        for asset in asset_returns.columns:
            y = aligned[asset].dropna()
            x = factors.loc[y.index]
            x = sm.add_constant(x)
            model = sm.OLS(y.values, x.values).fit()
            exposures[asset] = model.params[1:]
            resid = model.resid
            specific[asset] = np.var(resid, ddof=1)

        self.exposures = pd.DataFrame(exposures, index=self.factor_returns.columns).T
        self.factor_cov = self.factor_returns.cov()
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
