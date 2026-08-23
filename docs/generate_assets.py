"""Generate the deterministic charts embedded in the project README."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from riskoptima import BacktestConfig, SMACrossStrategy, load_sample_market_returns, run_backtest
from riskoptima.branding import add_riskoptima_signature
from riskoptima.credit import credit_cvar, credit_var, simulate_credit_losses
from riskoptima.optim import Constraints, optimize_max_sharpe
from riskoptima.options import black_scholes_greeks, black_scholes_price
from riskoptima.reporting import (
    build_market_risk_report,
    plot_correlation_heatmap,
    plot_drawdown_curve,
    plot_rolling_volatility,
    plot_var_cvar_distribution,
)


ASSET_DIR = ROOT / "docs" / "assets"


def _save(fig, filename: str, *, rect=(0.0, 0.04, 1.0, 0.96)) -> None:
    add_riskoptima_signature(fig, y=0.008)
    fig.tight_layout(rect=rect)
    fig.savefig(ASSET_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _market_risk_dashboard(returns: pd.DataFrame) -> None:
    weights = pd.Series({"Equity": 0.35, "Quality": 0.30, "Duration": 0.20, "Gold": 0.15})
    report = build_market_risk_report(returns, weights=weights, benchmark_returns=returns["Equity"])
    portfolio_returns = report.metrics["portfolio_returns"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_drawdown_curve(portfolio_returns, ax=axes[0, 0], add_signature=False)
    plot_rolling_volatility(portfolio_returns, ax=axes[0, 1], add_signature=False)
    plot_var_cvar_distribution(portfolio_returns, confidence=0.99, ax=axes[1, 0], add_signature=False)
    plot_correlation_heatmap(returns, ax=axes[1, 1], add_signature=False)
    fig.suptitle("RiskOptima Market Risk Dashboard", fontsize=18)
    _save(fig, "market_risk_dashboard.png")


def _algorithmic_backtest(returns: pd.DataFrame) -> None:
    prices = 100.0 * (1.0 + returns).cumprod()
    equity, _ = run_backtest(
        prices,
        SMACrossStrategy(short_window=20, long_window=60),
        BacktestConfig(initial_cash=1_000_000, rebalance_rule="D", slippage_bps=1.0),
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    normalized = equity["PortfolioValue"] / equity["PortfolioValue"].iloc[0] * 100.0
    ax.plot(normalized.index, normalized, linewidth=2.0, color="#1f77b4")
    ax.set_title("Algorithmic Backtesting: SMA Portfolio Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (start = 100)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(alpha=0.3)
    _save(fig, "algorithmic_backtesting.png")


def _portfolio_optimization(returns: pd.DataFrame) -> None:
    expected_returns = returns.mean() * 252.0
    covariance = returns.cov() * 252.0
    rng = np.random.default_rng(271)
    weights = rng.dirichlet(np.ones(returns.shape[1]), size=2_500)
    simulated_returns = weights @ expected_returns.to_numpy()
    simulated_volatility = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance.to_numpy(), weights))
    sharpe = (simulated_returns - 0.03) / simulated_volatility
    optimal = optimize_max_sharpe(
        expected_returns,
        covariance,
        constraints=Constraints(weight_bounds=(0.0, 1.0)),
        risk_free_rate=0.03,
    )
    optimal_return = float(optimal @ expected_returns)
    optimal_volatility = float(np.sqrt(optimal @ covariance @ optimal))

    fig, ax = plt.subplots(figsize=(11, 7))
    points = ax.scatter(simulated_volatility, simulated_returns, c=sharpe, cmap="viridis", s=14, alpha=0.55)
    ax.scatter(optimal_volatility, optimal_return, marker="*", s=240, color="#d62728", label="Max Sharpe")
    ax.set_title("Portfolio Optimization: Simulated Opportunity Set")
    ax.set_xlabel("Annualized volatility")
    ax.set_ylabel("Expected annual return")
    ax.legend(loc="best")
    fig.colorbar(points, ax=ax, label="Sharpe ratio")
    _save(fig, "portfolio_optimization.png")


def _credit_risk_model() -> None:
    rng = np.random.default_rng(271)
    obligors = pd.DataFrame(
        {
            "PD": np.clip(rng.beta(1.4, 45.0, 250), 0.001, 0.20),
            "LGD": np.clip(rng.beta(4.0, 5.0, 250), 0.15, 0.80),
            "EAD": rng.lognormal(mean=12.0, sigma=0.65, size=250),
        }
    )
    losses = simulate_credit_losses(obligors, n_sims=25_000, random_state=271)
    var_99 = credit_var(losses, confidence=0.99)
    cvar_99 = credit_cvar(losses, confidence=0.99)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(losses, bins=55, color="#4c78a8", alpha=0.82)
    ax.axvline(var_99, color="#d62728", linestyle="--", linewidth=2, label="VaR 99%")
    ax.axvline(cvar_99, color="#111111", linewidth=2, label="CVaR 99%")
    ax.set_title("Credit Risk Model: Simulated Portfolio Loss Distribution")
    ax.set_xlabel("Loss")
    ax.set_ylabel("Simulations")
    ax.legend()
    _save(fig, "credit_risk_model.png")


def _option_pricing_engine() -> None:
    strikes = np.linspace(80.0, 120.0, 17)
    prices = [black_scholes_price(100.0, strike, 1.0, 0.05, 0.20) for strike in strikes]
    deltas = [black_scholes_greeks(100.0, strike, 1.0, 0.05, 0.20)["delta"] for strike in strikes]

    fig, ax = plt.subplots(figsize=(11, 6))
    delta_ax = ax.twinx()
    price_line = ax.plot(strikes, prices, marker="o", linewidth=2, label="Call price")
    delta_line = delta_ax.plot(strikes, deltas, marker="s", linewidth=2, color="#ff7f0e", label="Call delta")
    ax.set_title("Option Pricing Engine: Black-Scholes Price and Delta")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Option price")
    delta_ax.set_ylabel("Delta")
    ax.legend(price_line + delta_line, [line.get_label() for line in price_line + delta_line], loc="best")
    ax.grid(alpha=0.3)
    _save(fig, "option_pricing_engine.png")


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    returns = load_sample_market_returns()
    _algorithmic_backtest(returns)
    _portfolio_optimization(returns)
    _market_risk_dashboard(returns)
    _option_pricing_engine()
    _credit_risk_model()


if __name__ == "__main__":
    main()
