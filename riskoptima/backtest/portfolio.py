###############################################################################
#                                 portfolio.py                                 
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    positions: pd.Series
    cash: float

    def value(self, prices: pd.Series) -> float:
        aligned = pd.Series(prices, dtype=float).reindex(self.positions.index)
        missing_held_assets = aligned.isna() & self.positions.ne(0.0)
        if missing_held_assets.any():
            missing = aligned.index[missing_held_assets].tolist()
            raise ValueError(f"prices are missing held assets: {missing}")
        if not np.isfinite(aligned.dropna()).all():
            raise ValueError("prices must be finite")
        aligned = aligned.fillna(0.0)
        return float(self.cash + (self.positions * aligned).sum())
