###############################################################################
#                         test_synthetic_data_examples.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Synthetic example data tests
###############################################################################

import unittest
from pathlib import Path

import pandas as pd

from riskoptima.credit import portfolio_expected_loss
from riskoptima.reporting import build_market_risk_report


ROOT = Path(__file__).resolve().parents[1]


class TestSyntheticDataExamples(unittest.TestCase):
    def test_synthetic_market_returns_drive_risk_report(self):
        returns = pd.read_csv(ROOT / "data" / "synthetic_market_returns.csv", index_col=0, parse_dates=True)
        report = build_market_risk_report(returns, weights=[0.35, 0.30, 0.20, 0.15])

        self.assertFalse(returns.empty)
        self.assertIn("annualized_volatility", report.metrics)
        self.assertIn(0.99, report.metrics["historical_var"])

    def test_synthetic_credit_portfolio_expected_loss(self):
        portfolio = pd.read_csv(ROOT / "data" / "synthetic_credit_portfolio.csv")
        self.assertGreater(portfolio_expected_loss(portfolio), 0.0)


if __name__ == "__main__":
    unittest.main()

