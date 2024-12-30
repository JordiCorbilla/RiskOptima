# RiskOptima

RiskOptima is a comprehensive Python toolkit for evaluating, managing, and optimizing investment portfolios. This package is designed to empower investors and data scientists by combining financial risk analysis, backtesting, mean-variance optimization, and machine learning capabilities into a single, cohesive package.

## Key Features

- Portfolio Optimization: Includes mean-variance optimization, efficient frontier calculation, and maximum Sharpe ratio portfolio construction.
- Risk Management: Compute key financial risk metrics such as Value at Risk (VaR), Conditional Value at Risk (CVaR), volatility, and drawdowns.
- Backtesting Framework: Simulate historical performance of investment strategies and analyze portfolio dynamics over time.
- Machine Learning Integration: Future-ready for implementing machine learning models for predictive analytics and advanced portfolio insights.
- Monte Carlo Simulations: Perform extensive simulations to analyze potential portfolio outcomes.
- Comprehensive Financial Metrics: Calculate returns, Sharpe ratios, covariance matrices, and more.

## Installation

```
pip install riskoptima
```
## Usage

Example 1: Efficient Frontier
```python
from riskoptima import RiskOptima
import pandas as pd

# Download market data
data = RiskOptima.download_data_yfinance(['AAPL', 'MSFT', 'GOOG'], '2022-01-01', '2022-12-31')
daily_returns, cov_matrix = RiskOptima.calculate_statistics(data)

# Calculate Efficient Frontier
mean_returns = daily_returns.mean()
vols, rets, weights = RiskOptima.efficient_frontier(mean_returns, cov_matrix)

# Plot Efficient Frontier
RiskOptima.plot_ef_ax(50, mean_returns, cov_matrix)
```
Example 2: Monte Carlo Simulation
```python
simulated_portfolios, weights_record = RiskOptima.run_monte_carlo_simulation(daily_returns, cov_matrix)
```

Example 3: Macaulay Duration
```
Navigate to -> https://github.com/JordiCorbilla/portfolio_risk_kit/blob/main/portfolio_risk_kit.ipynb
```

## Documentation

For complete documentation and usage examples, visit the GitHub repository:

RiskOptima GitHub

## Contributing

We welcome contributions! If you'd like to improve the package or report issues, please visit the GitHub repository.

## License

RiskOptima is licensed under the MIT License.

