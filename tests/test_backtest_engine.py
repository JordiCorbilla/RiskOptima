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

from riskoptima.backtest import SMACrossStrategy, Strategy, run_backtest
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
        self.assertLessEqual(float(last_weights.sum()), 1.0 + 1e-6)

    def test_costs_are_reflected_immediately_and_monthly_rebalance_executes(self):
        class AlwaysLong(Strategy):
            def generate_target_weights(self, date, prices, state=None):
                return pd.Series({"A": 1.0})

        dates = pd.to_datetime(["2026-01-02", "2026-02-02"])
        prices = pd.DataFrame({"A": [100.0, 100.0]}, index=dates)
        equity, weights = run_backtest(
            prices,
            AlwaysLong(),
            BacktestConfig(initial_cash=1_000.0, rebalance_rule="M"),
            SimpleCostModel(spread_bps=100.0),
        )

        self.assertAlmostEqual(equity.iloc[0]["PortfolioValue"], 990.0)
        self.assertGreater(weights.iloc[0]["A"], 1.0)
        self.assertGreater(weights.iloc[1]["A"], 0.0)

    def test_missing_quote_uses_last_finite_price(self):
        class AlwaysLong(Strategy):
            def generate_target_weights(self, date, prices, state=None):
                return pd.Series({"A": 1.0})

        dates = pd.bdate_range("2026-01-01", periods=3)
        prices = pd.DataFrame({"A": [100.0, np.nan, 105.0]}, index=dates)
        equity, _ = run_backtest(
            prices,
            AlwaysLong(),
            BacktestConfig(initial_cash=1_000.0, rebalance_rule="once"),
            SimpleCostModel(spread_bps=0.0),
        )

        self.assertAlmostEqual(equity.iloc[1]["PortfolioValue"], 1_000.0)
        self.assertAlmostEqual(equity.iloc[2]["PortfolioValue"], 1_050.0)


if __name__ == "__main__":
    unittest.main()
