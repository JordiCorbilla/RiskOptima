###############################################################################
#                               implied_vol.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Implied volatility solver
###############################################################################

from __future__ import annotations

from .black_scholes import black_scholes_price


def implied_volatility(
    market_price,
    S,
    K,
    T,
    r,
    option_type="call",
    q=0.0,
    tol=1e-8,
    max_iter=100,
    low=1e-6,
    high=5.0,
):
    """
    Solves Black-Scholes implied volatility using bisection.
    """
    if market_price <= 0:
        raise ValueError("market_price must be positive")

    lo = float(low)
    hi = float(high)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = black_scholes_price(S, K, T, r, mid, option_type=option_type, q=q)
        if abs(price - market_price) < tol:
            return float(mid)
        if price > market_price:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))
