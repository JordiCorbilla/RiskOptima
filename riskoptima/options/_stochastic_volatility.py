"""Numerical kernels for the legacy Heston and SABR entry points."""

import numpy as np

from .black_scholes import black_scholes_price


def _positive_count(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _heston_inputs(spot, maturity, rate, kappa, theta, eta, rho, initial_variance):
    values = np.asarray([spot, maturity, rate, kappa, theta, eta, rho, initial_variance], dtype=float)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("Heston inputs must be finite scalars")
    if spot <= 0 or min(maturity, kappa, theta, eta, initial_variance) < 0 or abs(rho) > 1:
        raise ValueError("Heston requires positive spot, non-negative maturity and variance parameters, and |rho| <= 1")


def _brownian_increments(rng, dt, rho, shape=()):
    spot, independent = rng.standard_normal((2, *shape)) * np.sqrt(dt)
    return spot, rho * spot + np.sqrt((1 - rho) * (1 + rho)) * independent


def _heston_step(variance, kappa, theta, eta, dt, spot_shock, variance_shock):
    # Full truncation uses v+ in both coefficients and retains the raw state for the next step.
    positive_variance = np.maximum(variance, 0.0)
    volatility = np.sqrt(positive_variance)
    log_move = -0.5 * positive_variance * dt + volatility * spot_shock
    variance = variance + kappa * (theta - positive_variance) * dt + eta * volatility * variance_shock
    return log_move, variance


def heston_call(spot, strike, maturity, rate, kappa, theta, eta, rho, v0, paths, steps, random_state):
    _heston_inputs(spot, maturity, rate, kappa, theta, eta, rho, v0)
    if not np.isscalar(strike) or not np.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be a finite positive scalar")
    _positive_count(paths, "num_simulations")
    _positive_count(steps, "num_steps")
    if maturity == 0:
        return float(max(spot - strike, 0.0))
    if eta == 0:
        # The deterministic CIR trajectory gives the time-averaged variance exactly.
        x = kappa * maturity
        if x < 1e-4:
            reversion_weight = x * (0.5 - x / 6 + x * x / 24)
        else:
            reversion_weight = 1 + np.expm1(-x) / x
        average_variance = (1 - reversion_weight) * v0 + reversion_weight * theta
        return black_scholes_price(spot, strike, maturity, rate, np.sqrt(average_variance))

    rng = np.random if random_state is None else np.random.default_rng(random_state)
    dt = maturity / steps
    log_moves = np.zeros(paths)
    variance = np.full(paths, v0, dtype=float)
    for _ in range(steps):
        spot_shock, variance_shock = _brownian_increments(rng, dt, rho, (paths,))
        log_move, variance = _heston_step(variance, kappa, theta, eta, dt, spot_shock, variance_shock)
        log_moves += log_move
    discounted_spots = np.exp(np.log(spot) + log_moves)
    discounted_strike = np.exp(np.log(strike) - rate * maturity)
    return float(np.maximum(discounted_spots - discounted_strike, 0.0).mean())


def simulate_heston_path(spot, v0, rate, maturity, points, rho, kappa, theta, eta, random_state):
    _heston_inputs(spot, maturity, rate, kappa, theta, eta, rho, v0)
    _positive_count(points, "N")
    prices = np.full(points, spot, dtype=float)
    variances = np.full(points, v0, dtype=float)
    if points == 1 or maturity == 0:
        return prices, variances

    rng = np.random if random_state is None else np.random.default_rng(random_state)
    dt = maturity / (points - 1)
    variance = v0
    for t in range(1, points):
        spot_shock, variance_shock = _brownian_increments(rng, dt, rho)
        log_move, variance = _heston_step(variance, kappa, theta, eta, dt, spot_shock, variance_shock)
        prices[t] = prices[t - 1] * np.exp(rate * dt + log_move)
        variances[t] = max(variance, 0.0)
    return prices, variances


def simulate_sabr_path(forward, level, maturity, points, rho, beta, vol_of_vol, random_state):
    values = np.asarray([forward, level, maturity, rho, beta, vol_of_vol], dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("SABR inputs must be finite scalars")
    if min(level, maturity, vol_of_vol) < 0 or not 0 <= beta <= 1 or abs(rho) > 1:
        raise ValueError("SABR requires non-negative maturity and volatility parameters, 0 <= beta <= 1, and |rho| <= 1")
    if beta > 0 and forward < 0:
        raise ValueError("SABR requires a non-negative forward when beta > 0")
    _positive_count(points, "N")
    forwards = np.full(points, forward, dtype=float)
    levels = np.full(points, level, dtype=float)
    if points == 1 or maturity == 0 or level == 0:
        return forwards, levels

    rng = np.random if random_state is None else np.random.default_rng(random_state)
    dt = maturity / (points - 1)
    level_drift = -0.5 * vol_of_vol**2 * dt
    for t in range(1, points):
        forward_shock, level_shock = _brownian_increments(rng, dt, rho)
        levels[t] = level * np.exp(level_drift + vol_of_vol * level_shock)
        if beta == 0:
            forward += level * forward_shock
        elif beta == 1:
            forward *= np.exp(-0.5 * level**2 * dt + level * forward_shock)
        else:
            forward = max(forward + level * forward**beta * forward_shock, 0.0)
        forwards[t], level = forward, levels[t]
    return forwards, levels
