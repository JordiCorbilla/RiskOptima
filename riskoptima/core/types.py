###############################################################################
#                                   types.py                                   
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Any

import pandas as pd


@dataclass
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    calendar: str = "D"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Portfolio:
    weights: pd.Series
    benchmark: Optional[pd.Series] = None
    constraints: Optional[Any] = None


@dataclass
class BacktestConfig:
    start: Optional[pd.Timestamp] = None
    end: Optional[pd.Timestamp] = None
    initial_cash: float = 1_000_000.0
    rebalance_rule: str = "D"
    slippage_bps: float = 0.0


@dataclass
class RiskReport:
    metrics: Dict[str, Any] = field(default_factory=dict)
    factor_exposures: Optional[pd.DataFrame] = None
    attribution: Optional[pd.DataFrame] = None
