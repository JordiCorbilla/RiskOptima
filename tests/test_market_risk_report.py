###############################################################################
#                         test_market_risk_report.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Market risk report tests
###############################################################################

import unittest

import numpy as np
import pandas as pd

from riskoptima.core import RiskReport
from riskoptima.reporting import build_market_risk_report


class TestMarketRiskReport(unittest.TestCase):
    def test_report_from_series(self):
        returns = pd.Series([0.01, -0.02, 0.015, np.nan, 0.004, -0.005])
        report = build_market_risk_report(returns, confidence_levels=(0.95,))
        self.assertIsInstance(report, RiskReport)
        self.assertIn("annualized_volatility", report.metrics)
        self.assertIn(0.95, report.metrics["historical_var"])

    def test_report_from_dataframe_with_weights_and_benchmark(self):
        rng = np.random.default_rng(42)
        returns = pd.DataFrame(
            rng.normal(0.001, 0.01, size=(100, 3)),
            columns=["A", "B", "C"],
        )
        returns.iloc[3, 1] = np.nan
        report = build_market_risk_report(
            returns,
            weights=pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}),
            benchmark_returns=returns["A"],
        )
        self.assertIn("beta", report.metrics)
        self.assertIn("tracking_error", report.metrics)
        self.assertFalse(report.metrics["rolling_drawdown"].empty)

    def test_invalid_empty_returns(self):
        with self.assertRaises(ValueError):
            build_market_risk_report(pd.Series([np.nan, np.nan]))


if __name__ == "__main__":
    unittest.main()
