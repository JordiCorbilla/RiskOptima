###############################################################################
#                                  greeks.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Option Greeks
###############################################################################

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

from .black_scholes import _d1_d2, _option_type, _scalar_option_inputs


def _black_scholes_greeks(S, K, T, r, sigma, q, *, option_type):
    """Compute Greeks of validated scalar or compatible array inputs."""
    sqrt_t = np.sqrt(T)
    stddev = sigma * sqrt_t
    d1, d2 = _d1_d2(S, K, T, r, q, stddev)
    discount_q = np.exp(-q * T)
    discount_r = np.exp(-r * T)

    density = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    sign = 1 if option_type == "call" else -1
    nd1, nd2 = ndtr(sign * d1), ndtr(sign * d2)
    return {
        "delta": sign * discount_q * nd1,
        "gamma": discount_q * density / (S * stddev),
        "theta": (
            -S * discount_q * density * sigma / (2 * sqrt_t)
            + sign * (q * S * discount_q * nd1 - r * K * discount_r * nd2)
        ),
        "vega": S * discount_q * density * sqrt_t,
        "rho": sign * K * T * discount_r * nd2,
    }


def black_scholes_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Compute European option Greeks for scalar inputs.

    Theta is per year of calendar-time passage. Vega and rho are per
    unit change in decimal volatility and continuous decimal rate.
    """
    option_type = _option_type(option_type)
    values = _scalar_option_inputs(S, K, T, r, sigma, q, greeks=True)
    greeks = _black_scholes_greeks(*values, option_type=option_type)
    return {name: float(value) for name, value in greeks.items()}
