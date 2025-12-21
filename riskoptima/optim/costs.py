###############################################################################
#                                   costs.py                                   
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleCostModel:
    spread_bps: float = 2.0
    impact_coeff: float = 0.0

    def estimate_cost(self, order_notional: float, adv: Optional[float] = None) -> float:
        if order_notional <= 0:
            return 0.0
        spread_cost = order_notional * (self.spread_bps * 1e-4)
        if adv is None or adv <= 0 or self.impact_coeff <= 0:
            return spread_cost
        impact_cost = order_notional * (self.impact_coeff * math.sqrt(order_notional / adv))
        return spread_cost + impact_cost
