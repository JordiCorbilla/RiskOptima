from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.optim import Constraints, optimize_max_sharpe


def main():
    assets = ["Equity", "Quality", "Duration", "Gold"]
    expected_returns = pd.Series([0.09, 0.07, 0.035, 0.04], index=assets)
    cov = pd.DataFrame(
        [
            [0.040, 0.018, -0.004, 0.006],
            [0.018, 0.030, -0.002, 0.004],
            [-0.004, -0.002, 0.010, 0.001],
            [0.006, 0.004, 0.001, 0.025],
        ],
        index=assets,
        columns=assets,
    )
    previous = pd.Series(np.repeat(0.25, 4), index=assets)
    exposures = pd.DataFrame({"growth": [1.0, 0.6, -0.2, 0.1]}, index=assets)
    metadata = pd.DataFrame(
        {"sector": ["Cyclical", "Defensive", "Rates", "Commodity"], "asset_class": ["Equity", "Equity", "Bond", "Commodity"]},
        index=assets,
    )
    constraints = Constraints(
        weight_bounds=(0.0, 0.55),
        leverage_limit=1.0,
        turnover_limit=0.35,
        factor_bounds={"growth": (0.0, 0.65)},
        asset_class_bounds={"Bond": (0.10, 0.45)},
    )
    result = optimize_max_sharpe(
        expected_returns,
        cov,
        constraints=constraints,
        previous_weights=previous,
        factor_exposures=exposures,
        metadata=metadata,
        return_result=True,
    )

    print(result.weights.round(4))
    print(result.active_constraints)
    print(result.expected_return, result.volatility, result.sharpe)


if __name__ == "__main__":
    main()
