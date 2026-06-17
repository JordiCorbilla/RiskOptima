###############################################################################
#                                  greeks.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Option Greeks
###############################################################################

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Computes Black-Scholes Greeks for a European option.

    Theta is annualized. Vega is per 1.00 volatility point.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("S, K, T, and sigma must be positive")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    discount_q = np.exp(-q * T)
    discount_r = np.exp(-r * T)

    gamma = discount_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * discount_q * norm.pdf(d1) * np.sqrt(T)

    if option_type == "call":
        delta = discount_q * norm.cdf(d1)
        theta = (
            -S * discount_q * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * discount_r * norm.cdf(d2)
            + q * S * discount_q * norm.cdf(d1)
        )
        rho = K * T * discount_r * norm.cdf(d2)
    else:
        delta = -discount_q * norm.cdf(-d1)
        theta = (
            -S * discount_q * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * discount_r * norm.cdf(-d2)
            - q * S * discount_q * norm.cdf(-d1)
        )
        rho = -K * T * discount_r * norm.cdf(-d2)

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),
        "vega": float(vega),
        "rho": float(rho),
    }
