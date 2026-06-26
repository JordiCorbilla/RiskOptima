###############################################################################
#                       test_riskoptima_legacy_analytics.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Legacy RiskOptima analytics tests
###############################################################################

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from riskoptima import RiskOptima


class TestRiskOptimaLegacyAnalytics(unittest.TestCase):
    def test_return_risk_metrics(self):
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])

        self.assertAlmostEqual(RiskOptima.compound(returns), (1 + returns).prod() - 1)
        self.assertGreater(RiskOptima.annualize_volatility(returns, 252), 0.0)
        self.assertLess(RiskOptima.drawdown(returns)["Drawdown"].min(), 0.0)
        self.assertGreater(RiskOptima.var_historic(returns, level=20), 0.0)
        self.assertGreater(RiskOptima.cvar_historic(returns, level=20), 0.0)
        self.assertGreater(RiskOptima.var_gaussian(returns, level=5), 0.0)
        self.assertGreater(RiskOptima.semideviation(returns), 0.0)

    def test_portfolio_math(self):
        weights = np.array([0.6, 0.4])
        returns = np.array([0.10, 0.05])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])

        self.assertAlmostEqual(RiskOptima.portfolio_return(weights, returns), 0.08)
        self.assertAlmostEqual(RiskOptima.portfolio_volatility(weights, cov), np.sqrt(weights.T @ cov @ weights))
        self.assertAlmostEqual(RiskOptima.tracking_error(pd.Series([0.01, 0.02]), pd.Series([0.00, 0.01])), np.sqrt(0.0002))

    def test_fixed_income_helpers(self):
        cash_flows = RiskOptima.bond_cash_flows_v2(n_periods=4, par=1000, coupon_rate=0.06, freq=2)
        self.assertEqual(len(cash_flows), 4)
        self.assertAlmostEqual(cash_flows[-1], 1030.0)

        duration_table, metrics = RiskOptima.macaulay_duration_v3(cash_flows, yield_rate=0.05, freq=2)
        self.assertIn("pv", duration_table.columns)
        self.assertGreater(float(metrics["Macaulay Duration"].iloc[0]), 0.0)
        self.assertGreater(float(metrics["Convexity"].iloc[0]), 0.0)

    def test_legacy_black_scholes_call(self):
        self.assertAlmostEqual(RiskOptima.black_scholes(100, 100, 1.0, 0.05, 0.2), 10.4506, places=4)

    def test_sma_strategy_handles_no_closed_trades(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        prices = pd.DataFrame({"Close": np.linspace(100, 110, len(dates))}, index=dates)

        with patch("riskoptima.backtest.sma.download_close_prices", return_value=prices):
            trades = RiskOptima.run_sma_strategy_with_risk("SPY", "2024-01-01", "2024-04-30")
            self.assertIn("Return", trades.columns)
            self.assertTrue(trades.empty)

            portfolio_trades = RiskOptima.run_strategy_on_portfolio(
                pd.DataFrame([{"Asset": "SPY", "Weight": 1.0}]),
                "2024-01-01",
                "2024-04-30",
            )
            self.assertIn("Weighted Return", portfolio_trades.columns)
            self.assertTrue(portfolio_trades.empty)
            with patch("matplotlib.pyplot.show"):
                RiskOptima.plot_sma_strategy_cumulative_return(portfolio_trades)
            plt.close("all")

    def test_efficient_frontier_accepts_supplied_prices_and_returns_output(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        prices = pd.DataFrame(
            {
                "AAA": np.linspace(100, 118, len(dates)),
                "BBB": np.linspace(50, 54, len(dates)) + np.sin(np.arange(len(dates))) * 0.5,
                "CCC": np.linspace(80, 84, len(dates)) + np.cos(np.arange(len(dates))) * 0.4,
            },
            index=dates,
        )
        asset_table = pd.DataFrame(
            [
                {"Asset": "AAA", "Weight": 0.5, "Label": "Alpha"},
                {"Asset": "BBB", "Weight": 0.3, "Label": "Beta"},
                {"Asset": "CCC", "Weight": 0.2, "Label": "Gamma"},
            ]
        )

        output = RiskOptima.plot_efficient_frontier_monte_carlo(
            asset_table,
            price_data=prices,
            benchmark_statistics=(0.08, 0.12, 0.67),
            num_portfolios=100,
            show_plot=False,
            show_tables=False,
            save_data=False,
            save_plot=False,
            return_output=True,
        )

        self.assertIn("simulated_portfolios", output)
        self.assertIn("optimal_weights", output)
        self.assertEqual(list(output["optimal_weights"].index), ["AAA", "BBB", "CCC"])
        self.assertEqual(len(output["simulated_portfolios"]), 100)
        plt.close(output["figure"])


if __name__ == "__main__":
    unittest.main()
