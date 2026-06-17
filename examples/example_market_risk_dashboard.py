###############################################################################
#                     example_market_risk_dashboard.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Market risk dashboard example
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.reporting import (
    build_market_risk_report,
    plot_correlation_heatmap,
    plot_drawdown_curve,
    plot_rolling_volatility,
    plot_var_cvar_distribution,
)


def _sample_returns():
    try:
        import yfinance as yf

        prices = yf.download(
            ["SPY", "QQQ", "TLT", "GLD"],
            period="2y",
            progress=False,
            auto_adjust=True,
        )["Close"]
        returns = prices.pct_change().dropna()
        if not returns.empty:
            return returns
    except Exception:
        pass

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=504, freq="B")
    return pd.DataFrame(
        rng.normal([0.0004, 0.0005, 0.0001, 0.0002], [0.011, 0.014, 0.008, 0.010], size=(504, 4)),
        index=dates,
        columns=["SPY", "QQQ", "TLT", "GLD"],
    )


if __name__ == "__main__":
    returns = _sample_returns()
    weights = pd.Series({"SPY": 0.45, "QQQ": 0.25, "TLT": 0.20, "GLD": 0.10})
    benchmark = returns["SPY"]

    report = build_market_risk_report(returns, weights=weights, benchmark_returns=benchmark)
    metrics = report.metrics
    print("Annualized return:", round(metrics["annualized_return"], 4))
    print("Annualized volatility:", round(metrics["annualized_volatility"], 4))
    print("VaR 99%:", round(metrics["historical_var"][0.99], 4))
    print("CVaR 99%:", round(metrics["cvar"][0.99], 4))

    portfolio_returns = metrics["portfolio_returns"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_drawdown_curve(portfolio_returns, ax=axes[0, 0])
    plot_rolling_volatility(portfolio_returns, ax=axes[0, 1])
    plot_var_cvar_distribution(portfolio_returns, confidence=0.99, ax=axes[1, 0])
    plot_correlation_heatmap(returns, ax=axes[1, 1])
    fig.suptitle("RiskOptima Market Risk Dashboard")
    fig.tight_layout()
    plt.show()
