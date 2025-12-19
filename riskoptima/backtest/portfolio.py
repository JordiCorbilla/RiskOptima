###############################################################################
#                                 portfolio.py                                 
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class PortfolioState:
    positions: pd.Series
    cash: float

    def value(self, prices: pd.Series) -> float:
        aligned = prices.reindex(self.positions.index).fillna(0.0)
        return float(self.cash + (self.positions * aligned).sum())
