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
from riskoptima import load_sample_credit_portfolio, load_sample_market_returns
from riskoptima.reporting import build_market_risk_report
from riskoptima.credit import portfolio_expected_loss
from riskoptima.options import black_scholes_price

returns = load_sample_market_returns()
report = build_market_risk_report(returns, weights=[0.35, 0.30, 0.20, 0.15])

credit = load_sample_credit_portfolio()
expected_loss = portfolio_expected_loss(credit)

call = black_scholes_price(100, 100, 1.0, 0.05, 0.20, option_type="call")
```

See [Quant Project Map](quant_project_map.md) for the recruiter/interviewer view.

For the institutional workflow, see `13-institutional_risk_engine_demo.ipynb`, which combines option book analytics, constrained optimization, risk attribution, and stress scenarios.

Interest-rate curve construction and conventions are documented in [Interest-Rate Curves](interest_rate_curves.md). The `DiscountCurve.to_dict()` payload is suitable for RiskOptima platform APIs.

For new applications, prefer the modular package APIs. The legacy `RiskOptima` class remains supported for existing consumers and notebooks. Keep the RiskOptima Platform dependency pinned to the same release reported by `riskoptima.__version__`.

## SMA Strategy Helpers

The SMA helpers in `riskoptima.backtest.sma` provide a transparent baseline strategy:

- calculate short and long moving averages
- detect bullish and bearish crossover events
- generate a trade log
- apply optional stop-loss and take-profit exits
- run the workflow across one ticker or a weighted asset table
