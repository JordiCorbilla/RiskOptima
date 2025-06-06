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

### Example 1: Setting up your portfolio

Create your portfolio table similar to the below:

| Asset | Weight | Label                         | MarketCap |
|-------|--------|-------------------------------|-----------|
| MO    | 0.04   | Altria Group Inc.             | 110.0e9   |
| NWN   | 0.14   | Northwest Natural Gas         | 1.8e9     |
| BKH   | 0.01   | Black Hills Corp.             | 4.5e9     |
| ED    | 0.01   | Con Edison                    | 30.0e9    |
| PEP   | 0.09   | PepsiCo Inc.                  | 255.0e9   |
| NFG   | 0.16   | National Fuel Gas             | 5.6e9     |
| KO    | 0.06   | Coca-Cola Company             | 275.0e9   |
| FRT   | 0.28   | Federal Realty Inv. Trust     | 9.8e9     |
| GPC   | 0.16   | Genuine Parts Co.             | 25.3e9    |
| MSEX  | 0.05   | Middlesex Water Co.           | 2.4e9     |

```python
import pandas as pd
from riskoptima import RiskOptima

import warnings
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*DataFrame.std with axis=None is deprecated.*"
)

# Define your current porfolio with your weights and company names
asset_data = [
    {"Asset": "MO",    "Weight": 0.04, "Label": "Altria Group Inc.",       "MarketCap": 110.0e9},
    {"Asset": "NWN",   "Weight": 0.14, "Label": "Northwest Natural Gas",   "MarketCap": 1.8e9},
    {"Asset": "BKH",   "Weight": 0.01, "Label": "Black Hills Corp.",         "MarketCap": 4.5e9},
    {"Asset": "ED",    "Weight": 0.01, "Label": "Con Edison",                "MarketCap": 30.0e9},
    {"Asset": "PEP",   "Weight": 0.09, "Label": "PepsiCo Inc.",              "MarketCap": 255.0e9},
    {"Asset": "NFG",   "Weight": 0.16, "Label": "National Fuel Gas",         "MarketCap": 5.6e9},
    {"Asset": "KO",    "Weight": 0.06, "Label": "Coca-Cola Company",         "MarketCap": 275.0e9},
    {"Asset": "FRT",   "Weight": 0.28, "Label": "Federal Realty Inv. Trust", "MarketCap": 9.8e9},
    {"Asset": "GPC",   "Weight": 0.16, "Label": "Genuine Parts Co.",         "MarketCap": 25.3e9},
    {"Asset": "MSEX",  "Weight": 0.05, "Label": "Middlesex Water Co.",       "MarketCap": 2.4e9}
]
asset_table = pd.DataFrame(asset_data)

capital = 100_000

asset_table['Portfolio'] = asset_table['Weight'] * capital

ANALYSIS_START_DATE = RiskOptima.get_previous_year_date(RiskOptima.get_previous_working_day(), 1)
ANALYSIS_END_DATE   = RiskOptima.get_previous_working_day()
BENCHMARK_INDEX     = 'SPY'
RISK_FREE_RATE      = 0.05
NUMBER_OF_WEIGHTS   = 10_000
NUMBER_OF_MC_RUNS   = 1_000
```

### Example 1: Creating a Portfolio Area Chart

If you want to know visually how's your portfolio doing right now

```python
RiskOptima.create_portfolio_area_chart(
    asset_table,
    end_date=ANALYSIS_END_DATE,
    lookback_days=2,
    title="Portfolio Area Chart"
)
```
![portfolio_area_chart_20250212_095626](https://github.com/user-attachments/assets/e54899e2-8592-48bb-906b-53bdd774d367)

### Example 2: Efficient Frontier - Monte Carlo Portfolio Optimization
```python
RiskOptima.plot_efficient_frontier_monte_carlo(
    asset_table,
    start_date=ANALYSIS_START_DATE,
    end_date=ANALYSIS_END_DATE,
    risk_free_rate=RISK_FREE_RATE,
    num_portfolios=NUMBER_OF_WEIGHTS,
    market_benchmark=BENCHMARK_INDEX,
    set_ticks=False,
    x_pos_table=1.15,    # Position for the weight table on the plot
    y_pos_table=0.52,    # Position for the weight table on the plot
    title=f'Efficient Frontier - Monte Carlo Simulation {ANALYSIS_START_DATE} to {ANALYSIS_END_DATE}'
)
```
![efficient_frontier_monter_carlo_20250203_205339](https://github.com/user-attachments/assets/f48f9f44-38cd-4d4c-96f2-48e767d7316e)

### Example 3: Portfolio Optimization using Mean Variance and Machine Learning
```python
RiskOptima.run_portfolio_optimization_mv_ml(
    asset_table=asset_table,
    training_start_date='2022-01-01',
    training_end_date='2023-11-27',
    model_type='Linear Regression',    
    risk_free_rate=RISK_FREE_RATE,
    num_portfolios=100000,
    market_benchmark=[BENCHMARK_INDEX],
    max_volatility=0.25,
    min_weight=0.03,
    max_weight=0.2
)
```
![machine_learning_optimization_20250203_210953](https://github.com/user-attachments/assets/0fae24a6-8d1d-45e7-b3d2-16939a1aadf7)

### Example 4: Portfolio Optimization using Probability Analysis
```python
RiskOptima.run_portfolio_probability_analysis(
    asset_table=asset_table,
    analysis_start_date=ANALYSIS_START_DATE,
    analysis_end_date=ANALYSIS_END_DATE,
    benchmark_index=BENCHMARK_INDEX,
    risk_free_rate=RISK_FREE_RATE,
    number_of_portfolio_weights=NUMBER_OF_WEIGHTS,
    trading_days_per_year=RiskOptima.get_trading_days(),
    number_of_monte_carlo_runs=NUMBER_OF_MC_RUNS
)
```
![probability_distributions_of_final_fund_returns20250205_212501](https://github.com/user-attachments/assets/8ea20d1f-e74f-4559-b66f-41ee657dd63b)

### Example 5: Macaulay Duration

```python
from riskoptima import RiskOptima
cf = RiskOptima.bond_cash_flows_v2(4, 1000, 0.06, 2)  # 2 years, semi-annual, hence 4 periods
md_2 = RiskOptima.macaulay_duration_v3(cf, 0.05, 2)
md_2
```
![image](https://github.com/user-attachments/assets/8bf54461-7256-4162-9230-f29aeeef4a10)

### Example 6:  Market Turns with SPY & VIX Divergence

```python
ANALYSIS_START_DATE = RiskOptima.get_previous_year_date(RiskOptima.get_previous_working_day(), 1)
ANALYSIS_END_DATE   = RiskOptima.get_previous_working_day()

df_signals, df_exits, returns = RiskOptima.run_index_vol_divergence_signals(start_date=ANALYSIS_START_DATE, 
                                                                            end_date=ANALYSIS_END_DATE)
```
![riskoptima_index_vol_divergence_signals_entry_20250316_200414](https://github.com/user-attachments/assets/fe9b0e73-34f2-4175-bad6-80c75b514fe5)


## Documentation

For complete documentation and usage examples, visit the GitHub repository:

[RiskOptima GitHub](https://github.com/JordiCorbilla/RiskOptima)

## Contributing

We welcome contributions! If you'd like to improve the package or report issues, please visit the GitHub repository.

## License

RiskOptima is licensed under the MIT License.

### Support me

<a href="https://www.buymeacoffee.com/jordicorbilla" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>
