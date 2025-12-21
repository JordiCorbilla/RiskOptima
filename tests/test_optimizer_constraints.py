###############################################################################
#                        test_optimizer_constraints.py                         
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

import unittest
import numpy as np
import pandas as pd

from riskoptima.optim import Constraints, optimize_max_sharpe


class TestOptimizerConstraints(unittest.TestCase):
    def test_factor_exposure_bounds(self):
        assets = ["A", "B", "C"]
        expected_returns = pd.Series([0.08, 0.06, 0.05], index=assets)
        cov = pd.DataFrame(np.eye(3) * 0.04, index=assets, columns=assets)

        exposures = pd.DataFrame(
            {"MKT": [0.1, 0.1, 0.1]},
            index=assets,
        )
        constraints = Constraints(factor_bounds={"MKT": (0.0, 0.2)})

        weights = optimize_max_sharpe(
            expected_returns=expected_returns,
            cov=cov,
            constraints=constraints,
            factor_exposures=exposures,
            risk_free_rate=0.0,
        )

        mkt_exp = float((weights * exposures["MKT"]).sum())
        self.assertGreaterEqual(mkt_exp, 0.0)
        self.assertLessEqual(mkt_exp, 0.2)


if __name__ == "__main__":
    unittest.main()
