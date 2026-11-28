"""Numerical implementation of the legacy portfolio probability analysis."""

import numpy as np
import pandas as pd

from .covariance import _tag_covariance, _validate_covariance

_INITIAL_VALUE = 10_000.0


def validate_portfolio_request(allocation, risk_free_rate, weight_count, days, runs):
    amounts = np.asarray(list(allocation.values()), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        total = amounts.sum()
    if amounts.shape != (len(allocation),) or not np.isfinite(amounts).all() or not np.isfinite(total) or total <= 0:
        raise ValueError("investment_allocation must contain finite amounts with a positive total")
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")
    for value, name in ((weight_count, "number_of_portfolio_weights"), (days, "trading_days_per_year"),
                        (runs, "number_of_monte_carlo_runs")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def _daily_prices(data, labels):
    if isinstance(data, pd.Series) and len(labels) == 1:
        data = data.to_frame(name=labels[0])
    prices = pd.DataFrame(data, dtype=float)
    if not prices.columns.is_unique or not pd.Index(labels).isin(prices.columns).all():
        raise ValueError(f"Historical prices must have unique columns for the requested assets: {labels}")
    if not prices.index.is_unique or prices.index.hasnans:
        raise ValueError("Historical price timestamps must be unique and non-missing")
    prices = prices.loc[:, labels].sort_index().replace([np.inf, -np.inf], np.nan)
    if (prices <= 0).any().any():
        raise ValueError("Historical prices must be positive where available")
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 2:
        raise ValueError(f"At least two complete daily return observations are required for: {labels}")
    return prices, returns


def _compound_values(daily_returns):
    # Once daily gross wealth reaches zero, the fund cannot recover.
    gross = 1 + daily_returns
    ruined = (gross <= 0).any(axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        values = _INITIAL_VALUE * np.prod(gross, axis=1, where=~ruined[:, None])
    values[ruined] = 0.0
    if not np.isfinite(values).all():
        raise OverflowError("Portfolio terminal values are outside the representable range")
    return values


def _portfolio_values(mean, loading, days, runs, rng):
    values = np.empty(runs)
    batch_size = max(1, 1_000_000 // (days * len(loading)))
    for start in range(0, runs, batch_size):
        stop = min(start + batch_size, runs)
        shocks = rng.normal(size=(stop - start, days, len(loading)))
        values[start:stop] = _compound_values(mean + shocks @ loading)
    return values


def _terminal_metrics(returns):
    volatility = 0.0 if np.all(returns == returns[0]) else float(returns.std())
    return float(returns.mean()), volatility


def analyze_portfolio(tickers, initial_weights, prices, benchmark, benchmark_index, risk_free_rate,
                      weight_count, days, runs, random_state):
    stock_data, stock_returns = _daily_prices(prices, tickers)
    benchmark_data, benchmark_returns = _daily_prices(benchmark, [benchmark_index])
    benchmark_close = benchmark_data[benchmark_index]
    market_returns = benchmark_returns[benchmark_index]
    covariance = _validate_covariance(_tag_covariance(stock_returns.cov(), None))
    cov_values = covariance.to_numpy()
    mean = stock_returns.mean().to_numpy()
    try:
        factor = np.linalg.cholesky(cov_values)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_values)
        factor = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))

    rng = np.random if random_state is None else np.random.default_rng(random_state)
    raw_weights = rng.random((weight_count, len(tickers)))
    totals = raw_weights.sum(axis=1, keepdims=True)
    weights = np.divide(raw_weights, totals, out=np.full_like(raw_weights, 1 / len(tickers)), where=totals > 0)
    annual_return = (weights @ mean) * days
    annual_volatility = np.sqrt(np.maximum(((weights @ cov_values) * weights).sum(axis=1), 0.0)) * np.sqrt(days)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (annual_return - risk_free_rate) / annual_volatility
    candidates = pd.DataFrame({
        "Annualized Return": annual_return,
        "Annualized Volatility": annual_volatility,
        "Sharpe Ratio": sharpe,
        "Simulation Index": np.arange(weight_count, dtype=float),
    })
    best = candidates["Sharpe Ratio"].idxmax() if not np.isnan(sharpe).all() else 0
    median = candidates["Annualized Volatility"].sort_values(kind="stable").index[weight_count // 2]
    optimal_weights = weights[best]
    median_weights = weights[median]

    portfolio_results = {}
    portfolio_metrics = {}
    for name, allocation in (("Optimized Portfolio", optimal_weights), ("Current Portfolio", initial_weights),
                             ("Median Portfolio", median_weights)):
        values = _portfolio_values(allocation @ mean, allocation @ factor, days, runs, rng)
        portfolio_results[name] = values.tolist()
        portfolio_metrics[name] = _terminal_metrics(values / _INITIAL_VALUE - 1)

    market_values = _portfolio_values(market_returns.mean(), np.array([market_returns.std()]), days, runs, rng)
    market_net = market_values / _INITIAL_VALUE - 1
    market_return, market_vol = _terminal_metrics(market_net)
    return {
        "stock_tickers": tickers,
        "initial_weights": initial_weights,
        "stock_data": stock_data,
        "benchmark_data": benchmark_close,
        "stock_daily_returns": stock_returns,
        "benchmark_daily_returns": market_returns,
        "covariance_matrix": covariance,
        "simulated_portfolios": candidates,
        "recorded_weights": weights.T,
        "optimal_weights": optimal_weights,
        "optimal_sharpe_idx": best,
        "median_volatility_weights": median_weights,
        "portfolio_results": portfolio_results,
        "portfolio_metrics": portfolio_metrics,
        "market_final_values": market_values.tolist(),
        "market_final_values_percent": (100 * market_net).tolist(),
        "market_expected_return": market_return,
        "market_volatility": market_vol,
        "market_sharpe_ratio": (market_return - risk_free_rate) / market_vol if market_vol else np.nan,
    }
