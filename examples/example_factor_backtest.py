###############################################################################
#                          example_factor_backtest.py                          
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

import numpy as np
import pandas as pd

from riskoptima.risk import FactorRiskModel
from riskoptima.optim import Constraints, optimize_max_sharpe, SimpleCostModel
from riskoptima.backtest import SMACrossStrategy, run_backtest
from riskoptima.core import BacktestConfig


def make_synthetic_prices(n_assets=3, n_days=252, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, size=(n_days, n_assets))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    return pd.DataFrame(prices, index=dates, columns=[f"Asset{i+1}" for i in range(n_assets)])


def main():
    prices = make_synthetic_prices()
    asset_returns = prices.pct_change().dropna()

    # Example Fama-French style factors (replace with RiskOptima.get_fff_returns data)
    factors = pd.DataFrame(
        np.random.normal(0.0002, 0.008, size=(asset_returns.shape[0], 4)),
        index=asset_returns.index,
        columns=["MKT", "SMB", "HML", "UMD"],
    )

    factor_model = FactorRiskModel(factor_returns=factors).fit(asset_returns)
    factor_cov = factor_model.covariance_matrix()

    exp_returns = asset_returns.mean() * 252
    constraints = Constraints(factor_bounds={"MKT": (-0.2, 0.8)})
    weights = optimize_max_sharpe(
        expected_returns=exp_returns,
        cov=factor_cov,
        constraints=constraints,
        factor_exposures=factor_model.exposures,
        risk_free_rate=0.02,
    )

    strategy = SMACrossStrategy(short_window=20, long_window=50)
    config = BacktestConfig(initial_cash=1_000_000, rebalance_rule="D")
    cost_model = SimpleCostModel(spread_bps=2.0, impact_coeff=0.0)
    equity_curve, weights_history = run_backtest(
        prices=prices, strategy=strategy, config=config, cost_model=cost_model
    )

    print("Optimized weights:")
    print(weights.round(4))
    print(equity_curve.tail())


if __name__ == "__main__":
    main()
