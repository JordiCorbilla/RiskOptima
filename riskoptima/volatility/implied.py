###############################################################################
#                                implied.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Implied volatility API wrapper
###############################################################################

from __future__ import annotations

from riskoptima.options import implied_volatility as _option_implied_volatility


def implied_volatility(*args, **kwargs) -> float:
    """Black-Scholes implied volatility delegated to the options engine."""
    return _option_implied_volatility(*args, **kwargs)

