###############################################################################
#                              black_scholes.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Black-Scholes option pricing
###############################################################################

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def _option_type(option_type: str) -> str:
    value = option_type.lower()
    if value not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return value


def _validate_option_inputs(values, *, greeks=False):
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("S, K, T, r, sigma, and q must be finite")
    S, K, T, _, sigma, _ = values
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be positive")
    if greeks:
        if np.any(T <= 0) or np.any(sigma <= 0):
            raise ValueError("T and sigma must be positive for Black-Scholes Greeks")
    elif np.any(T < 0) or np.any(sigma < 0):
        raise ValueError("T and sigma must be non-negative")


def _scalar_option_inputs(S, K, T, r, sigma, q, *, greeks=False):
    values = np.asarray([S, K, T, r, sigma, q], dtype=float)
    if values.shape != (6,):
        raise ValueError("Black-Scholes option inputs must be scalars")
    _validate_option_inputs(values, greeks=greeks)
    return values


def _d1_d2(S, K, T, r, q, stddev):
    d1 = (np.log(S / K) + (r - q) * T) / stddev + 0.5 * stddev
    return d1, d1 - stddev


def _black_scholes_price(S, K, T, r, sigma, q, *, option_type):
    """Price validated scalar or compatible array inputs."""
    stddev = sigma * np.sqrt(T)
    sign = 1 if option_type == "call" else -1
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    deterministic = stddev == 0
    if np.all(deterministic):
        return np.maximum(sign * (discounted_spot - discounted_strike), 0.0)

    has_deterministic = np.any(deterministic)
    safe_stddev = np.where(deterministic, 1.0, stddev) if has_deterministic else stddev
    d1, d2 = _d1_d2(S, K, T, r, q, safe_stddev)
    price = sign * (discounted_spot * ndtr(sign * d1) - discounted_strike * ndtr(sign * d2))
    if has_deterministic:
        intrinsic = np.maximum(sign * (discounted_spot - discounted_strike), 0.0)
        return np.where(deterministic, intrinsic, price)
    return price


def black_scholes_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """Price a European option with scalar inputs and continuously compounded rates."""
    option_type = _option_type(option_type)
    values = _scalar_option_inputs(S, K, T, r, sigma, q)
    return float(_black_scholes_price(*values, option_type=option_type))
