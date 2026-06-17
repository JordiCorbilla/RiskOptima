###############################################################################
#                            test_credit_risk.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Credit risk tests
###############################################################################

import unittest

import numpy as np
import pandas as pd

from riskoptima.credit import (
    credit_var,
    expected_loss,
    merton_pd,
    portfolio_expected_loss,
    simulate_credit_losses,
    simulate_rating_migration,
    validate_transition_matrix,
)


class TestCreditRisk(unittest.TestCase):
    def test_expected_loss(self):
        self.assertAlmostEqual(float(expected_loss(0.02, 0.45, 1_000_000)), 9_000.0)
        df = pd.DataFrame({"PD": [0.01, 0.02], "LGD": [0.4, 0.5], "EAD": [100, 200]})
        self.assertAlmostEqual(portfolio_expected_loss(df), 2.4)

    def test_transition_matrix_validation(self):
        matrix = pd.DataFrame(
            [[0.9, 0.1], [0.0, 1.0]],
            index=["A", "D"],
            columns=["A", "D"],
        )
        self.assertTrue(validate_transition_matrix(matrix))
        with self.assertRaises(ValueError):
            validate_transition_matrix([[0.8, 0.3], [0.1, 0.9]])

    def test_credit_var_monotonicity(self):
        losses = np.array([0, 1, 2, 5, 10, 20, 50], dtype=float)
        self.assertLessEqual(credit_var(losses, 0.95), credit_var(losses, 0.99))

    def test_deterministic_random_seed_behaviour(self):
        portfolio = pd.DataFrame({"PD": [0.1, 0.2], "LGD": [0.5, 0.4], "EAD": [1000, 2000]})
        a = simulate_credit_losses(portfolio, n_sims=1000, random_state=7)
        b = simulate_credit_losses(portfolio, n_sims=1000, random_state=7)
        np.testing.assert_array_equal(a, b)

        matrix = pd.DataFrame(
            [[0.8, 0.2], [0.0, 1.0]],
            index=["A", "D"],
            columns=["A", "D"],
        )
        path_a = simulate_rating_migration(["A", "A"], matrix, periods=3, random_state=11)
        path_b = simulate_rating_migration(["A", "A"], matrix, periods=3, random_state=11)
        pd.testing.assert_frame_equal(path_a, path_b)

    def test_merton_pd_sanity(self):
        low_leverage_pd = merton_pd(200, 100, 0.25, 0.03, 1.0)
        high_leverage_pd = merton_pd(105, 100, 0.25, 0.03, 1.0)
        self.assertGreater(high_leverage_pd, low_leverage_pd)
        self.assertGreaterEqual(low_leverage_pd, 0.0)
        self.assertLessEqual(high_leverage_pd, 1.0)


if __name__ == "__main__":
    unittest.main()
