###############################################################################
#                                constraints.py                                
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class Constraints:
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
    leverage_limit: Optional[float] = 1.0
    turnover_limit: Optional[float] = None
    factor_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def bounds(self, n_assets: int):
        return (self.weight_bounds,) * n_assets


def enforce_turnover(prev_weights: np.ndarray, new_weights: np.ndarray, limit: Optional[float]) -> bool:
    if limit is None:
        return True
    turnover = np.sum(np.abs(new_weights - prev_weights))
    return turnover <= limit


def factor_constraint_func(weights: np.ndarray, exposures: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]):
    if exposures is None or not bounds:
        return []
    constraints = []
    for factor, (lower, upper) in bounds.items():
        if factor not in exposures.columns:
            continue
        f_exp = exposures[factor].values
        constraints.append({"type": "ineq", "fun": lambda w, f=f_exp, lb=lower: np.dot(w, f) - lb})
        constraints.append({"type": "ineq", "fun": lambda w, f=f_exp, ub=upper: ub - np.dot(w, f)})
    return constraints
