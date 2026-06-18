###############################################################################
#                              black_scholes.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Black-Scholes option pricing
###############################################################################

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _option_type(option_type: str) -> str:
    value = option_type.lower()
    if value not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return value


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Prices a European option using the Black-Scholes-Merton formula.
    """
    option_type = _option_type(option_type)
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if T == 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)
    if sigma == 0:
        forward_intrinsic = (
            max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
            if option_type == "call"
            else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)
        )
        return float(forward_intrinsic)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return float(price)
