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
    inputs = np.asarray([S, K, T, r, sigma, q], dtype=float)
    if not np.isfinite(inputs).all():
        raise ValueError("option inputs must be finite")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T < 0:
        raise ValueError("T must be non-negative")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if n_sims <= 0:
        raise ValueError("n_sims must be positive")
    if T == 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)
    if sigma == 0:
        discounted_spot = S * np.exp(-q * T)
        discounted_strike = K * np.exp(-r * T)
        deterministic = (
            max(discounted_spot - discounted_strike, 0.0)
            if option_type == "call"
            else max(discounted_strike - discounted_spot, 0.0)
        )
        return float(deterministic)

    rng = np.random.default_rng(random_state)
    z = rng.standard_normal(int(n_sims))
    terminal = S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    payoff = np.maximum(terminal - K, 0.0) if option_type == "call" else np.maximum(K - terminal, 0.0)
    return float(np.exp(-r * T) * payoff.mean())


def merton_jump_diffusion_call(
    S, K, T, r, sigma, jump_intensity, log_jump_mean, log_jump_std,
    n_sims=10000, random_state=None,
) -> float:
    """Price a European call by sampling its exact Merton terminal distribution.

    Rates and jump intensity are annual, maturity is in years, and jump
    parameters describe the normal distribution of individual log multipliers.
    With no seed, use NumPy's legacy global RNG for existing RiskOptima callers.
    The estimate can miss rare, large payoffs even when the arithmetic is stable.
    """
    inputs = np.asarray([S, K, T, r, sigma, jump_intensity, log_jump_mean, log_jump_std], dtype=float)
    if not np.isfinite(inputs).all():
        raise ValueError("Merton inputs must be finite")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if min(T, sigma, jump_intensity, log_jump_std) < 0:
        raise ValueError("maturity, volatility, jump intensity and jump standard deviation must be non-negative")
    if not isinstance(n_sims, (int, np.integer)) or n_sims <= 0:
        raise ValueError("n_sims must be a positive integer")
    if T == 0:
        return float(max(S - K, 0.0))

    S, K, T, r, sigma, jump_intensity, log_jump_mean, log_jump_std = inputs
    rng = np.random if random_state is None else np.random.default_rng(random_state)
    with np.errstate(over="ignore", invalid="ignore"):
        diffusion_scale = sigma * np.sqrt(T)
        log_return = -(0.5 * diffusion_scale) * diffusion_scale
        log_return = log_return + diffusion_scale * rng.standard_normal(n_sims)
        log_ratio = np.log(S) - np.log(K)
        if abs(S - K) <= 0.5 * K:
            log_ratio = np.log1p((S - K) / K)
        forward_moneyness = log_ratio + r * T
        if jump_intensity > 0:
            jump_count_mean = jump_intensity * T
            jump_log_expectation = log_jump_mean + (0.5 * log_jump_std) * log_jump_std
            if jump_log_expectation > 0:
                # Scale in log units before exponentiating a large jump mean.
                log_compensator = jump_log_expectation + np.log(-np.expm1(-jump_log_expectation))
                compensation = np.exp(np.log(jump_intensity) + np.log(T) + log_compensator)
            else:
                compensation = jump_count_mean * np.expm1(jump_log_expectation)
            if not np.isfinite(compensation):
                raise ValueError("Merton jump compensation is outside the representable range")
            counts = rng.poisson(jump_count_mean, size=n_sims)
            log_return += (
                -compensation + counts * log_jump_mean
                + np.sqrt(counts) * log_jump_std * rng.standard_normal(n_sims)
            )
    if not np.isfinite(log_return).all():
        raise ValueError("Merton terminal log prices are outside the representable range")

    with np.errstate(over="ignore"):
        moneyness = forward_moneyness + log_return
    in_the_money = moneyness > 0
    if not in_the_money.any():
        return 0.0
    with np.errstate(over="ignore"):
        log_payoff = np.log(S) + log_return[in_the_money] + np.log(-np.expm1(-moneyness[in_the_money]))
        peak = log_payoff.max()
        scaled_mean = np.exp(log_payoff - peak).sum() / n_sims
        price = float(np.exp(peak + np.log(scaled_mean)))
    if not np.isfinite(price):
        raise OverflowError("Merton call price is outside the representable range")
    return price
