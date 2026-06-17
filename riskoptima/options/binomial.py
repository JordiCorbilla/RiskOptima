###############################################################################
#                                 binomial.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Binomial option pricing
###############################################################################

from __future__ import annotations

import numpy as np


def binomial_tree_price(S, K, T, r, sigma, steps=100, option_type="call", q=0.0, american=False):
    """
    Prices an option using a Cox-Ross-Rubinstein binomial tree.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if steps <= 0:
        raise ValueError("steps must be positive")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0 <= p <= 1:
        raise ValueError("invalid tree parameters; risk-neutral probability is outside [0, 1]")

    j = np.arange(steps + 1)
    prices = S * (u ** (steps - j)) * (d ** j)
    if option_type == "call":
        values = np.maximum(prices - K, 0.0)
    else:
        values = np.maximum(K - prices, 0.0)

    for step in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])
        if american:
            prices = S * (u ** (step - np.arange(step + 1))) * (d ** np.arange(step + 1))
            exercise = np.maximum(prices - K, 0.0) if option_type == "call" else np.maximum(K - prices, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])
