###############################################################################
#                           test_options_engine.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Option pricing engine tests
###############################################################################

import unittest

import numpy as np

from riskoptima.options import (
    binomial_tree_price,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    monte_carlo_european_option,
)


class TestOptionsEngine(unittest.TestCase):
    def test_black_scholes_known_values(self):
        call = black_scholes_price(100, 100, 1, 0.05, 0.2, option_type="call")
        put = black_scholes_price(100, 100, 1, 0.05, 0.2, option_type="put")
        self.assertAlmostEqual(call, 10.4506, places=4)
        self.assertAlmostEqual(put, 5.5735, places=4)

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 95, 0.75, 0.03, 0.22
        call = black_scholes_price(S, K, T, r, sigma, "call")
        put = black_scholes_price(S, K, T, r, sigma, "put")
        self.assertAlmostEqual(call - put, S - K * np.exp(-r * T), places=8)

    def test_greeks_and_implied_volatility(self):
        greeks = black_scholes_greeks(100, 100, 1, 0.05, 0.2)
        self.assertGreater(greeks["delta"], 0)
        self.assertGreater(greeks["gamma"], 0)
        price = black_scholes_price(100, 100, 1, 0.05, 0.2)
        self.assertAlmostEqual(implied_volatility(price, 100, 100, 1, 0.05), 0.2, places=5)

    def test_binomial_and_monte_carlo(self):
        bs = black_scholes_price(100, 100, 1, 0.05, 0.2)
        tree = binomial_tree_price(100, 100, 1, 0.05, 0.2, steps=500)
        mc = monte_carlo_european_option(100, 100, 1, 0.05, 0.2, n_sims=200000, random_state=1)
        self.assertAlmostEqual(tree, bs, places=2)
        self.assertAlmostEqual(mc, bs, delta=0.15)


if __name__ == "__main__":
    unittest.main()
