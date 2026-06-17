# RiskOptima Documentation

RiskOptima is a Python toolkit for portfolio risk, optimization, backtesting, options analytics, market risk reporting, fixed income, stochastic models, and credit risk.

The project is organized around five quant portfolio projects:

| Project | What it proves | Main entry points |
|---|---|---|
| Algorithmic Trading Backtester | Strategy lifecycle, rebalancing, transaction costs, equity curves | `riskoptima.backtest`, `SMACrossStrategy`, `run_backtest` |
| Portfolio Optimization | Efficient frontier, constrained optimization, factor-aware portfolios | `riskoptima.optim`, `optimize_max_sharpe`, `optimize_min_variance` |
| Market Risk Dashboard | VaR, CVaR, drawdowns, tracking error, dashboard-ready reporting | `riskoptima.reporting` |
| Option Pricing Engine | Black-Scholes, Greeks, implied volatility, binomial trees, Monte Carlo | `riskoptima.options` |
| Credit Risk Model | PD/LGD/EAD, expected loss, migration, Merton PD, Credit VaR | `riskoptima.credit` |

## Quick Start

```python
import pandas as pd
from riskoptima.reporting import build_market_risk_report
from riskoptima.credit import portfolio_expected_loss
from riskoptima.options import black_scholes_price

returns = pd.read_csv("data/synthetic_market_returns.csv", index_col=0, parse_dates=True)
report = build_market_risk_report(returns, weights=[0.35, 0.30, 0.20, 0.15])

credit = pd.read_csv("data/synthetic_credit_portfolio.csv")
expected_loss = portfolio_expected_loss(credit)

call = black_scholes_price(100, 100, 1.0, 0.05, 0.20, option_type="call")
```

See [Quant Project Map](quant_project_map.md) for the recruiter/interviewer view.

