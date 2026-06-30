###############################################################################
#                         test_markov_regime_report.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Markov regime model and report tests
###############################################################################

import unittest

import matplotlib
import numpy as np
import pandas as pd

import riskoptima
from riskoptima.reporting import (
    build_markov_regime_report,
    plot_markov_regime_chart,
    plot_markov_regime_probabilities,
)
from riskoptima.risk import fit_markov_regime_model

matplotlib.use("Agg")


class TestMarkovRegimeReport(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2021-01-01", periods=180)
        low = rng.normal(0.0004, 0.004, size=60)
        stress = rng.normal(-0.0012, 0.018, size=60)
        recovery = rng.normal(0.0010, 0.007, size=60)
        self.returns = pd.Series(np.r_[low, stress, recovery], index=dates, name="returns")
        self.prices = (1.0 + self.returns).cumprod() * 100.0

    def test_model_fit_is_deterministic_with_seed(self):
        first = fit_markov_regime_model(self.returns, n_regimes=3, random_state=42, n_iter=50)
        second = fit_markov_regime_model(self.returns, n_regimes=3, random_state=42, n_iter=50)

        pd.testing.assert_series_equal(first.regimes, second.regimes)
        np.testing.assert_allclose(first.transition_matrix.values, second.transition_matrix.values)

    def test_transition_matrix_rows_sum_to_one(self):
        model = fit_markov_regime_model(self.returns, n_regimes=3, random_state=42, n_iter=50)

        np.testing.assert_allclose(model.transition_matrix.sum(axis=1).to_numpy(), np.ones(3), atol=1e-8)
        self.assertEqual(model.regime_probabilities.shape, (len(self.returns), 3))

    def test_report_from_prices_contains_expected_fields(self):
        report = build_markov_regime_report(
            self.prices,
            input_type="prices",
            n_regimes=3,
            random_state=42,
            n_iter=50,
        )

        self.assertIn("wealth", report.metrics)
        self.assertIn("transition_matrix", report.metrics)
        self.assertIn("regime_summary", report.metrics)
        self.assertEqual(len(report.metrics["wealth"]), len(self.prices) - 1)

    def test_plot_functions_return_axes(self):
        report = build_markov_regime_report(self.returns, n_regimes=3, random_state=42, n_iter=50)

        ax_chart = plot_markov_regime_chart(report)
        self.assertEqual(ax_chart.get_ylabel(), "Cumulative value")
        ax_chart.figure.clf()

        ax_probs = plot_markov_regime_probabilities(report)
        self.assertEqual(ax_probs.get_ylabel(), "Probability")
        ax_probs.figure.clf()

    def test_public_exports_importable(self):
        self.assertTrue(hasattr(riskoptima, "fit_markov_regime_model"))
        self.assertTrue(hasattr(riskoptima, "build_markov_regime_report"))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            fit_markov_regime_model([0.01, 0.02], n_regimes=2)
        with self.assertRaises(ValueError):
            build_markov_regime_report(self.returns, input_type="bad")


if __name__ == "__main__":
    unittest.main()

