###############################################################################
#                              test_sma_backtest.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: SMA helper tests
###############################################################################

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from riskoptima.backtest import (
    build_sma_signal_frame,
    run_sma_strategy_with_risk,
    trades_from_sma_signals,
)


class TestSMABacktestHelpers(unittest.TestCase):
    def test_build_sma_signal_frame(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        prices = pd.DataFrame({"Close": [10, 10, 10, 11, 12, 11, 10, 9]}, index=dates)

        signals = build_sma_signal_frame(prices, short_window=2, long_window=3)

        self.assertIn("SMA2", signals.columns)
        self.assertIn("SMA3", signals.columns)
        self.assertIn("Signal", signals.columns)
        self.assertTrue(set(signals["Signal"].unique()).issubset({-1, 0, 1}))

    def test_build_sma_signal_frame_handles_yfinance_multiindex_close(self):
        dates = pd.date_range("2024-01-01", periods=8, freq="B")
        columns = pd.MultiIndex.from_tuples([("Close", "SPY")])
        prices = pd.DataFrame([10, 10, 10, 11, 12, 11, 10, 9], index=dates, columns=columns)

        signals = build_sma_signal_frame(prices, short_window=2, long_window=3)

        self.assertEqual(list(signals.columns[:1]), ["Close"])
        self.assertIn("Signal", signals.columns)

    def test_trades_from_sma_signals_with_stop_loss(self):
        dates = pd.date_range("2024-01-01", periods=4, freq="B")
        signals = pd.DataFrame(
            {
                "Close": [100.0, 105.0, 96.0, 95.0],
                "Signal": [1, 0, 0, 0],
            },
            index=dates,
        )

        trades = trades_from_sma_signals(signals, ticker="TEST", stop_loss=0.05)

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Exit Reason"], "Stop Loss")
        self.assertLess(trades.iloc[0]["Return"], 0.0)

    def test_run_sma_strategy_accepts_supplied_prices(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        prices = pd.DataFrame({"Close": np.linspace(100, 120, len(dates))}, index=dates)

        trades = run_sma_strategy_with_risk(
            "TEST",
            "2024-01-01",
            "2024-04-30",
            prices=prices,
            short_window=5,
            long_window=20,
        )

        self.assertIn("Return", trades.columns)

    def test_run_sma_strategy_download_boundary(self):
        dates = pd.date_range("2024-01-01", periods=80, freq="B")
        prices = pd.DataFrame({"Close": np.linspace(100, 120, len(dates))}, index=dates)

        with patch("riskoptima.backtest.sma.download_close_prices", return_value=prices):
            trades = run_sma_strategy_with_risk("TEST", "2024-01-01", "2024-04-30")

        self.assertIn("Return", trades.columns)


if __name__ == "__main__":
    unittest.main()
