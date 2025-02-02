# RiskOptima

![image](https://github.com/user-attachments/assets/b9bc3bd0-d8fa-4f01-97e6-44bf4b886bcb)


RiskOptima is a comprehensive Python toolkit for evaluating, managing, and optimizing investment portfolios. This package is designed to empower investors and data scientists by combining financial risk analysis, backtesting, mean-variance optimization, and machine learning capabilities into a single, cohesive package.

## Stats
https://pypistats.org/packages/riskoptima

## Key Features

- Portfolio Optimization: Includes mean-variance optimization, efficient frontier calculation, and maximum Sharpe ratio portfolio construction.
- Risk Management: Compute key financial risk metrics such as Value at Risk (VaR), Conditional Value at Risk (CVaR), volatility, and drawdowns.
- Backtesting Framework: Simulate historical performance of investment strategies and analyze portfolio dynamics over time.
- Machine Learning Integration: Future-ready for implementing machine learning models for predictive analytics and advanced portfolio insights.
- Monte Carlo Simulations: Perform extensive simulations to analyze potential portfolio outcomes. See example here https://github.com/JordiCorbilla/efficient-frontier-monte-carlo-portfolio-optimization
- Comprehensive Financial Metrics: Calculate returns, Sharpe ratios, covariance matrices, and more.

## Installation

See the project here: https://pypi.org/project/riskoptima/

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

[RiskOptima GitHub](https://github.com/JordiCorbilla/RiskOptima)

## Contributing

We welcome contributions! If you'd like to improve the package or report issues, please visit the GitHub repository.

## License

RiskOptima is licensed under the MIT License.

### Support me

<a href="https://www.buymeacoffee.com/jordicorbilla" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>
