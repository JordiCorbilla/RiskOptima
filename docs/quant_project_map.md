# Quant Project Map

This page maps RiskOptima to the five canonical projects commonly expected in a quant developer portfolio.

## 1. Algorithmic Trading Backtester

![Algorithmic backtesting](assets/algorithmic_backtesting.png)

RiskOptima includes a modular backtesting engine with strategy interfaces, portfolio state tracking, transaction cost estimation, and weight history output.

Core files:

- `riskoptima/backtest/engine.py`
- `riskoptima/backtest/strategy.py`
- `riskoptima/backtest/portfolio.py`
- `tests/test_backtest_engine.py`

## 2. Portfolio Optimization

![Portfolio optimization](assets/portfolio_optimization.png)

The optimizer layer supports max-Sharpe and minimum-variance portfolios with bounds and factor exposure constraints.

Core files:

- `riskoptima/optim/mean_variance.py`
- `riskoptima/optim/constraints.py`
- `tests/test_optimizer_constraints.py`
- `02-portfolio_optimization_riskoptima.ipynb`

## 3. Market Risk Dashboard

![Market risk dashboard](assets/market_risk_dashboard.png)

The reporting layer builds a dashboard-ready `RiskReport` with volatility, Sharpe, Sortino, drawdown, historical VaR, Gaussian VaR, CVaR, beta, tracking error, information ratio, rolling volatility, and rolling drawdown.

Core files:

- `riskoptima/reporting/market_risk.py`
- `examples/example_market_risk_dashboard.py`
- `examples/streamlit_market_risk_dashboard.py`
- `tests/test_market_risk_report.py`

## 4. Option Pricing Engine

![Option pricing engine](assets/option_pricing_engine.png)

The options package exposes Black-Scholes pricing, Greeks, implied volatility, binomial trees, and Monte Carlo European option pricing.

Core files:

- `riskoptima/options/black_scholes.py`
- `riskoptima/options/greeks.py`
- `riskoptima/options/implied_vol.py`
- `riskoptima/options/binomial.py`
- `riskoptima/options/monte_carlo.py`
- `tests/test_options_engine.py`

## 5. Credit Risk Model

![Credit risk model](assets/credit_risk_model.png)

The credit package covers PD/LGD/EAD expected loss, unexpected loss, transition matrices, rating migration, Merton structural default probability, Credit VaR, and Credit CVaR.

Core files:

- `riskoptima/credit/models.py`
- `riskoptima/credit/metrics.py`
- `riskoptima/credit/portfolio.py`
- `riskoptima/credit/simulation.py`
- `08-credit_risk_model_demo.ipynb`
- `tests/test_credit_risk.py`

