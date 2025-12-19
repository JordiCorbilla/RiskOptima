###############################################################################
#                             test_factor_model.py                             
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

import unittest
import numpy as np
import pandas as pd

from riskoptima.risk import FactorRiskModel


class TestFactorRiskModel(unittest.TestCase):
    def test_covariance_matrix_shape(self):
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=120, freq="B")
        asset_returns = pd.DataFrame(
            rng.normal(0.0002, 0.01, size=(len(dates), 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        factor_returns = pd.DataFrame(
            rng.normal(0.0001, 0.008, size=(len(dates), 4)),
            index=dates,
            columns=["MKT", "SMB", "HML", "UMD"],
        )

        model = FactorRiskModel(factor_returns=factor_returns).fit(asset_returns)
        cov = model.covariance_matrix()

        self.assertEqual(cov.shape, (3, 3))
        self.assertTrue((np.diag(cov.values) > 0).all())


if __name__ == "__main__":
    unittest.main()
