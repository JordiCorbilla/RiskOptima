###############################################################################
#                         test_optim_backtest_helpers.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Optimizer and strategy helper tests
###############################################################################

import unittest

import numpy as np
import pandas as pd

from riskoptima.backtest import SMACrossStrategy
from riskoptima.optim.constraints import Constraints, enforce_turnover, factor_constraint_func
from riskoptima.optim import optimize_min_variance


class TestOptimBacktestHelpers(unittest.TestCase):
    def test_enforce_turnover(self):
        prev = np.array([0.5, 0.5])
        new = np.array([0.6, 0.4])
        self.assertTrue(enforce_turnover(prev, new, limit=0.25))
        self.assertFalse(enforce_turnover(prev, new, limit=0.10))
        self.assertTrue(enforce_turnover(prev, new, limit=None))

    def test_factor_constraint_builder(self):
        exposures = pd.DataFrame({"MKT": [1.0, 0.5]}, index=["A", "B"])
        constraints = factor_constraint_func(np.array([0.5, 0.5]), exposures, {"MKT": (0.0, 1.0)})
        self.assertEqual(len(constraints), 2)
        self.assertGreaterEqual(constraints[0]["fun"](np.array([0.5, 0.5])), 0.0)

    def test_min_variance_respects_bounds(self):
        cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
        weights = optimize_min_variance(cov, constraints=Constraints(weight_bounds=(0.0, 0.8)))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertLessEqual(float(weights.max()), 0.8 + 1e-6)

    def test_sma_strategy_waits_for_history(self):
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        prices = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=dates)
        strategy = SMACrossStrategy(short_window=2, long_window=4)
        early_weights = strategy.generate_target_weights(dates[2], prices)
        later_weights = strategy.generate_target_weights(dates[4], prices)

        self.assertAlmostEqual(float(early_weights.sum()), 0.0)
        self.assertAlmostEqual(float(later_weights.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
