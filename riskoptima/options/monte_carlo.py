###############################################################################
#                               monte_carlo.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Monte Carlo option pricing
###############################################################################

from __future__ import annotations

import numpy as np


def monte_carlo_european_option(
    S,
    K,
    T,
    r,
    sigma,
    option_type="call",
    q=0.0,
    n_sims=100000,
    random_state=None,
):
    """
    Prices a European option with one-step geometric Brownian motion simulation.
    """
    option_type = option_type.lower()
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")

    rng = np.random.default_rng(random_state)
    z = rng.standard_normal(int(n_sims))
    terminal = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(terminal - K, 0.0) if option_type == "call" else np.maximum(K - terminal, 0.0)
    return float(np.exp(-r * T) * payoff.mean())
