###############################################################################
#                              market_risk.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Dashboard-ready market risk reporting
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from riskoptima.core import RiskReport


def _validated_weights(weights, columns) -> pd.Series:
    if weights is None:
        return pd.Series(np.repeat(1.0 / len(columns), len(columns)), index=columns, dtype=float)

    if isinstance(weights, pd.Series):
        missing = [col for col in columns if col not in weights.index]
        if missing:
            raise ValueError(f"weights are missing assets: {missing}")
        w = weights.reindex(columns).astype(float)
    else:
        values = np.asarray(weights, dtype=float)
        if values.ndim != 1 or len(values) != len(columns):
            raise ValueError("weights must be a 1D array-like with one value per return column")
        w = pd.Series(values, index=columns, dtype=float)

    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    if np.isclose(w.sum(), 0.0):
        raise ValueError("weights must not sum to zero")
    return w / w.sum()


def _portfolio_returns(returns, weights=None) -> pd.Series:
    data = pd.DataFrame(returns).copy() if not isinstance(returns, pd.Series) else returns.to_frame("portfolio")
    data = data.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if isinstance(returns, pd.Series) or data.shape[1] == 1:
        return data.iloc[:, 0].dropna().rename("portfolio")

    clean = data.dropna(how="any")
    if clean.empty:
        raise ValueError("returns must contain at least one complete finite row")
    w = _validated_weights(weights, clean.columns)
    return clean.dot(w).rename("portfolio")


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return float(drawdown.min())


def _rolling_drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    return (wealth / wealth.cummax() - 1.0).rename("rolling_drawdown")


def build_market_risk_report(
    returns,
    weights=None,
    benchmark_returns=None,
    confidence_levels=(0.95, 0.99),
    periods_per_year=252,
    rolling_window=21,
    risk_free_rate=0.0,
):
    """
    Builds a dashboard-ready market risk report from return data.
    """
    portfolio = _portfolio_returns(returns, weights=weights)
    if portfolio.empty:
        raise ValueError("returns must contain at least one finite observation")

    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = portfolio - rf_per_period
    annualized_return = float((1.0 + portfolio).prod() ** (periods_per_year / len(portfolio)) - 1.0)
    annualized_volatility = float(portfolio.std(ddof=0) * np.sqrt(periods_per_year))
    downside = portfolio[portfolio < 0].std(ddof=0) * np.sqrt(periods_per_year)
    sharpe = float((excess.mean() * periods_per_year) / annualized_volatility) if annualized_volatility else np.nan
    sortino = float((excess.mean() * periods_per_year) / downside) if downside and not np.isnan(downside) else np.nan

    historical_var = {}
    gaussian_var = {}
    cvar = {}
    for confidence in confidence_levels:
        if not 0 < confidence < 1:
            raise ValueError("confidence levels must be between 0 and 1")
        alpha = 1.0 - confidence
        var_level = float(-portfolio.quantile(alpha))
        historical_var[confidence] = var_level
        gaussian_var[confidence] = float(-(portfolio.mean() + norm.ppf(alpha) * portfolio.std(ddof=0)))
        tail = portfolio[portfolio <= -var_level]
        cvar[confidence] = float(-tail.mean()) if not tail.empty else var_level

    beta = np.nan
    tracking_error = np.nan
    information_ratio = np.nan
    if benchmark_returns is not None:
        benchmark = pd.Series(benchmark_returns, dtype=float).dropna().rename("benchmark")
        aligned = pd.concat([portfolio, benchmark], axis=1).dropna()
        if not aligned.empty and aligned["benchmark"].var(ddof=0) > 0:
            cov = np.cov(aligned["portfolio"], aligned["benchmark"], ddof=0)[0, 1]
            beta = float(cov / aligned["benchmark"].var(ddof=0))
            active = aligned["portfolio"] - aligned["benchmark"]
            tracking_error = float(active.std(ddof=0) * np.sqrt(periods_per_year))
            information_ratio = (
                float(active.mean() * periods_per_year / tracking_error)
                if tracking_error
                else np.nan
            )

    rolling_volatility = (portfolio.rolling(rolling_window).std(ddof=0) * np.sqrt(periods_per_year)).rename(
        "rolling_volatility"
    )
    rolling_drawdown = _rolling_drawdown(portfolio)

    return RiskReport(
        metrics={
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": _max_drawdown(portfolio),
            "historical_var": historical_var,
            "parametric_gaussian_var": gaussian_var,
            "cvar": cvar,
            "expected_shortfall": cvar,
            "beta": beta,
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "rolling_volatility": rolling_volatility,
            "rolling_drawdown": rolling_drawdown,
            "portfolio_returns": portfolio,
        }
    )


def plot_drawdown_curve(returns, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    series = _portfolio_returns(returns)
    drawdown = _rolling_drawdown(series)
    ax = ax or plt.gca()
    drawdown.plot(ax=ax, **kwargs)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    return ax


def plot_rolling_volatility(returns, window=21, periods_per_year=252, ax=None, **kwargs):
    import matplotlib.pyplot as plt

    series = _portfolio_returns(returns)
    rolling_vol = series.rolling(window).std(ddof=0) * np.sqrt(periods_per_year)
    ax = ax or plt.gca()
    rolling_vol.plot(ax=ax, **kwargs)
    ax.set_title("Rolling Volatility")
    ax.set_ylabel("Annualized Volatility")
    return ax


def plot_var_cvar_distribution(returns, confidence=0.99, ax=None, bins=40, **kwargs):
    import matplotlib.pyplot as plt

    series = _portfolio_returns(returns)
    losses = -series.dropna()
    var = float(np.quantile(losses, confidence))
    cvar = float(losses[losses >= var].mean())
    ax = ax or plt.gca()
    ax.hist(losses, bins=bins, alpha=0.75, **kwargs)
    ax.axvline(var, color="red", linestyle="--", label=f"VaR {confidence:.0%}")
    ax.axvline(cvar, color="black", linestyle="-", label=f"CVaR {confidence:.0%}")
    ax.set_title("Loss Distribution")
    ax.legend()
    return ax


def plot_correlation_heatmap(returns, ax=None, **kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = pd.DataFrame(returns).apply(pd.to_numeric, errors="coerce").dropna(how="all")
    corr = data.corr()
    ax = ax or plt.gca()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax, **kwargs)
    ax.set_title("Correlation Heatmap")
    return ax
