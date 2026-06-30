###############################################################################
#                   test_portfolio_sophistication_report.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Portfolio sophistication report tests
###############################################################################

import unittest

import matplotlib
import numpy as np
import pandas as pd

from riskoptima import build_portfolio_sophistication_report
from riskoptima.core import RiskReport
from riskoptima.reporting import (
    available_sophistication_methods,
    plot_portfolio_sophistication_report,
)

matplotlib.use("Agg")


class TestPortfolioSophisticationReport(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        dates = pd.bdate_range("2020-01-01", periods=180)
        factors = rng.normal(0.0003, 0.007, size=(180, 1))
        noise = rng.normal(0.0, 0.006, size=(180, 5))
        loadings = np.array([[0.8, 1.1, 0.6, 1.3, 0.4]])
        self.returns = pd.DataFrame(
            factors @ loadings + noise + np.array([0.0001, 0.0002, 0.0000, -0.0001, 0.00015]),
            index=dates,
            columns=["MO", "NWN", "PEP", "KO", "FRT"],
        )

    def test_report_contains_expected_outputs(self):
        report = build_portfolio_sophistication_report(
            self.returns,
            methods=("MV", "CVaR", "WR", "1N"),
            confidence=0.95,
        )

        self.assertIsInstance(report, RiskReport)
        self.assertEqual(list(report.metrics["wealth"].columns), ["MV", "CVaR", "WR", "1N"])
        self.assertIn("Sharpe Ratio", report.metrics["performance_table"].index)
        self.assertIn("Value at Risk", report.metrics["performance_table"].index)
        self.assertFalse(report.metrics["performance_table"].isna().all().any())

    def test_weights_are_long_only_and_sum_to_one(self):
        report = build_portfolio_sophistication_report(
            self.returns,
            methods=("MV", "CDaR", "MDD", "1N"),
        )
        weights = report.metrics["weights"]

        self.assertTrue(np.all(weights.to_numpy() >= -1e-10))
        np.testing.assert_allclose(weights.sum(axis=0).to_numpy(), np.ones(weights.shape[1]), atol=1e-8)

    def test_equal_weight_returns_match_one_over_n(self):
        report = build_portfolio_sophistication_report(self.returns, methods=("1N",))
        expected = self.returns.mean(axis=1)
        actual = report.metrics["returns"]["1N"]

        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_plot_returns_matplotlib_figure(self):
        report = build_portfolio_sophistication_report(self.returns, methods=("MV", "WR", "1N"))
        fig = plot_portfolio_sophistication_report(report)

        self.assertEqual(len(fig.axes), 2)
        fig.clf()

    def test_available_methods_are_documented(self):
        methods = available_sophistication_methods()

        self.assertIn("method", methods.columns)
        self.assertIn("CVaR", methods["method"].values)

    def test_invalid_inputs_raise_clear_errors(self):
        with self.assertRaises(ValueError):
            build_portfolio_sophistication_report(pd.DataFrame({"A": [0.01, 0.02]}))
        with self.assertRaises(ValueError):
            build_portfolio_sophistication_report(self.returns, confidence=1.2)


if __name__ == "__main__":
    unittest.main()

