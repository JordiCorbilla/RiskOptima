from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.risk import (
    StressScenario,
    component_volatility_contribution,
    factor_risk_contribution,
    run_stress_scenario,
)


def main():
    weights = pd.Series({"Equity": 0.55, "Duration": 0.25, "Gold": 0.20})
    cov = pd.DataFrame(
        [[0.040, -0.006, 0.004], [-0.006, 0.012, 0.002], [0.004, 0.002, 0.030]],
        index=weights.index,
        columns=weights.index,
    )
    exposures = pd.DataFrame({"equity": [1.0, -0.2, 0.1], "duration": [0.0, 1.0, 0.1]}, index=weights.index)
    factor_cov = pd.DataFrame([[0.035, -0.005], [-0.005, 0.010]], index=["equity", "duration"], columns=["equity", "duration"])

    print(component_volatility_contribution(weights, cov))
    print(factor_risk_contribution(weights, exposures, factor_cov))

    holdings = pd.Series({"Equity": 100.0, "Duration": 80.0, "Gold": 50.0})
    prices = pd.Series({"Equity": 100.0, "Duration": 95.0, "Gold": 120.0})
    scenario = StressScenario("risk_off", price_shocks={"Equity": -0.15, "Gold": 0.06})
    result = run_stress_scenario(holdings, prices, scenario)
    print(result)


if __name__ == "__main__":
    main()
