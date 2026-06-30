###############################################################################
#                            test_core_public_api.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Public API and core type tests
###############################################################################

import unittest

import pandas as pd

import riskoptima
from riskoptima import BacktestConfig, MarketData, Portfolio, RiskReport
from riskoptima.backtest import PortfolioState
from riskoptima.optim import SimpleCostModel


class TestCorePublicApi(unittest.TestCase):
    def test_public_project_exports_importable(self):
        expected_exports = [
            "RiskOptima",
            "FactorRiskModel",
            "build_market_risk_report",
            "build_markov_regime_report",
            "fit_markov_regime_model",
            "expected_loss",
            "credit_var",
            "merton_pd",
            "black_scholes_price",
            "black_scholes_greeks",
            "implied_volatility",
            "historical_volatility",
            "realized_volatility",
            "ewma_volatility",
        ]
        for name in expected_exports:
            self.assertTrue(hasattr(riskoptima, name), name)

    def test_core_dataclasses_hold_expected_data(self):
        prices = pd.DataFrame({"A": [100.0, 101.0]})
        returns = prices.pct_change().dropna()
        market_data = MarketData(prices=prices, returns=returns, calendar="B", metadata={"source": "synthetic"})
        portfolio = Portfolio(weights=pd.Series({"A": 1.0}))
        config = BacktestConfig(initial_cash=50_000, rebalance_rule="M")
        report = RiskReport(metrics={"vol": 0.2})

        self.assertEqual(market_data.metadata["source"], "synthetic")
        self.assertAlmostEqual(float(portfolio.weights.sum()), 1.0)
        self.assertEqual(config.rebalance_rule, "M")
        self.assertEqual(report.metrics["vol"], 0.2)

    def test_portfolio_state_and_cost_model(self):
        state = PortfolioState(positions=pd.Series({"A": 10.0, "B": 5.0}), cash=100.0)
        prices = pd.Series({"A": 20.0, "B": 30.0})
        self.assertAlmostEqual(state.value(prices), 450.0)

        cost_model = SimpleCostModel(spread_bps=2.0, impact_coeff=0.1)
        self.assertEqual(cost_model.estimate_cost(0.0), 0.0)
        self.assertGreater(cost_model.estimate_cost(10_000.0, adv=1_000_000.0), 0.0)


if __name__ == "__main__":
    unittest.main()
