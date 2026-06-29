###############################################################################
#                         test_volatility_toolkit.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Volatility toolkit tests
###############################################################################

import unittest

import numpy as np
import pandas as pd

import riskoptima
from riskoptima.options import black_scholes_price
from riskoptima.volatility import (
    ewma_volatility,
    garman_klass_volatility,
    historical_volatility,
    implied_volatility,
    parkinson_volatility,
    realized_volatility,
    rogers_satchell_volatility,
    rolling_volatility,
    yang_zhang_volatility,
)


class TestVolatilityToolkit(unittest.TestCase):
    def setUp(self):
        self.returns = pd.Series([0.01, -0.02, 0.015, 0.004, -0.006])
        self.prices = pd.Series([100.0, 101.0, 99.0, 100.5, 100.9, 100.3])
        self.ohlc = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 100.5, 102.0, 101.2],
                "High": [102.0, 102.5, 103.0, 103.2, 102.0],
                "Low": [99.5, 100.2, 99.8, 100.8, 100.5],
                "Close": [101.0, 100.5, 102.0, 101.2, 101.8],
            }
        )

    def test_historical_volatility_from_returns(self):
        expected = self.returns.std(ddof=0) * np.sqrt(252)

        self.assertAlmostEqual(historical_volatility(self.returns), expected)

    def test_historical_volatility_from_prices_with_log_returns(self):
        log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        expected = log_returns.std(ddof=0) * np.sqrt(252)

        self.assertAlmostEqual(
            historical_volatility(self.prices, input_type="prices", log_returns=True),
            expected,
        )

    def test_rolling_realized_and_ewma_volatility(self):
        rolling = rolling_volatility(self.returns, window=3)

        self.assertTrue(rolling.iloc[:2].isna().all())
        self.assertGreater(realized_volatility(self.prices, input_type="prices"), 0.0)
        self.assertGreater(ewma_volatility(self.returns), 0.0)

    def test_ohlc_estimators_are_positive(self):
        self.assertGreater(parkinson_volatility(self.ohlc), 0.0)
        self.assertGreater(garman_klass_volatility(self.ohlc), 0.0)
        self.assertGreater(rogers_satchell_volatility(self.ohlc), 0.0)
        self.assertGreater(yang_zhang_volatility(self.ohlc), 0.0)

    def test_implied_volatility_delegates_to_options_engine(self):
        price = black_scholes_price(100, 100, 1.0, 0.05, 0.2)

        self.assertAlmostEqual(implied_volatility(price, 100, 100, 1.0, 0.05), 0.2, places=5)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            historical_volatility([100, -1], input_type="prices")
        with self.assertRaises(ValueError):
            ewma_volatility(self.returns, lambda_=1.5)
        with self.assertRaises(ValueError):
            parkinson_volatility(pd.DataFrame({"Open": [1.0]}))

    def test_public_exports_importable(self):
        self.assertTrue(hasattr(riskoptima, "historical_volatility"))
        self.assertTrue(hasattr(riskoptima, "realized_volatility"))
        self.assertTrue(hasattr(riskoptima, "ewma_volatility"))


if __name__ == "__main__":
    unittest.main()

