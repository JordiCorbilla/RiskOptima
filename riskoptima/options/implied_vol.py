###############################################################################
#                               implied_vol.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Implied volatility solver
###############################################################################

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from operator import index
from typing import Literal, overload

import numpy as np

from .black_scholes import black_scholes_price


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    """Bisection result with a signed model-minus-market residual in price units."""

    volatility: float
    converged: bool
    iterations: int
    price_error: float


@overload
def implied_volatility(
    market_price, S, K, T, r, option_type=..., q=..., tol=..., max_iter=..., low=..., high=...,
    *, return_result: Literal[False] = False,
) -> float: ...


@overload
def implied_volatility(
    market_price, S, K, T, r, option_type=..., q=..., tol=..., max_iter=..., low=..., high=...,
    *, return_result: Literal[True],
) -> ImpliedVolatilityResult: ...


@overload
def implied_volatility(
    market_price, S, K, T, r, option_type=..., q=..., tol=..., max_iter=..., low=..., high=...,
    *, return_result: bool,
) -> float | ImpliedVolatilityResult: ...


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
    *,
    return_result: bool = False,
) -> float | ImpliedVolatilityResult:
    """Solve Black-Scholes implied volatility to an absolute price tolerance.

    Request a result object to inspect convergence. The default returns a
    float and warns if it is only an estimate after exhausting the budget.
    Invalid or unbracketed prices raise ValueError in either mode.
    """
    values = (market_price, S, K, T, r, q, tol, low, high)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("option inputs must be finite")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0:
        raise ValueError("T must be positive when solving implied volatility")
    if market_price <= 0:
        raise ValueError("market_price must be positive")
    try:
        max_iter = index(max_iter)
    except TypeError as exc:
        raise ValueError("max_iter must be a positive integer") from exc
    if tol <= 0 or max_iter <= 0:
        raise ValueError("tol and max_iter must be positive")
    if low <= 0 or high <= low:
        raise ValueError("volatility bounds must satisfy 0 < low < high")

    option = option_type.lower()
    if option not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)
    lower_bound = (
        max(discounted_spot - discounted_strike, 0.0)
        if option == "call"
        else max(discounted_strike - discounted_spot, 0.0)
    )
    upper_bound = discounted_spot if option == "call" else discounted_strike
    if market_price < lower_bound - tol or market_price > upper_bound + tol:
        raise ValueError("market_price violates no-arbitrage option price bounds")

    lo = float(low)
    hi = float(high)
    low_price = black_scholes_price(S, K, T, r, lo, option_type=option, q=q)
    high_price = black_scholes_price(S, K, T, r, hi, option_type=option, q=q)
    if market_price < low_price - tol or market_price > high_price + tol:
        raise ValueError("market_price is not bracketed by the volatility bounds")
    for iteration in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        error = black_scholes_price(S, K, T, r, mid, option_type=option, q=q) - market_price
        if abs(error) <= tol:
            result = ImpliedVolatilityResult(float(mid), True, iteration, float(error))
            return result if return_result else result.volatility
        if error > 0:
            hi = mid
        else:
            lo = mid

    # Preserve the historical final estimate and check its residual too.
    mid = 0.5 * (lo + hi)
    error = black_scholes_price(S, K, T, r, mid, option_type=option, q=q) - market_price
    result = ImpliedVolatilityResult(float(mid), abs(error) <= tol, max_iter, float(error))
    if return_result:
        return result
    if not result.converged:
        warnings.warn(
            "Returning a float for non-converged implied volatility is deprecated. "
            "Pass return_result=True and inspect converged.",
            FutureWarning,
            stacklevel=2,
        )
    return result.volatility
