###############################################################################
#                           test_backtest_engine.py                            
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

import unittest
import numpy as np
import pandas as pd

from riskoptima.backtest import SMACrossStrategy, run_backtest
from riskoptima.core import BacktestConfig
from riskoptima.optim import SimpleCostModel


class TestBacktestEngine(unittest.TestCase):
    def test_run_backtest_smoke(self):
        dates = pd.date_range("2023-01-02", periods=120, freq="B")
        prices = pd.DataFrame(
            {
                "A": np.linspace(100, 130, len(dates)),
                "B": np.linspace(90, 120, len(dates)),
            },
            index=dates,
        )

        strategy = SMACrossStrategy(short_window=5, long_window=20)
        config = BacktestConfig(initial_cash=100_000, rebalance_rule="D")
        cost_model = SimpleCostModel(spread_bps=1.0, impact_coeff=0.0)

        equity_curve, weights_history = run_backtest(
            prices=prices, strategy=strategy, config=config, cost_model=cost_model
        )

        self.assertFalse(equity_curve.empty)
        self.assertIn("PortfolioValue", equity_curve.columns)
        self.assertEqual(weights_history.shape[1], prices.shape[1])

        last_weights = weights_history.iloc[-1].fillna(0.0)
        self.assertLessEqual(float(last_weights.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
