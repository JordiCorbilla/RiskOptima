###############################################################################
#                     test_risk_attribution_scenarios.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Risk attribution and scenario tests
###############################################################################

import numpy as np
import pandas as pd

from riskoptima.risk import (
    StressScenario,
    component_cvar,
    component_var,
    component_volatility_contribution,
    default_scenarios,
    factor_risk_contribution,
    run_scenario_set,
    run_stress_scenario,
)


def test_volatility_contributions_sum_to_total_volatility():
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.00], [0.00, 0.00, 0.02]],
        index=weights.index,
        columns=weights.index,
    )
    contrib = component_volatility_contribution(weights, cov)
    total_vol = np.sqrt(weights.values.T @ cov.values @ weights.values)

    assert np.isclose(contrib.sum(), total_vol)
    assert np.isclose(component_var(weights, cov).sum(), component_var(weights, cov).sum())


def test_component_cvar_returns_asset_contributions():
    returns = pd.DataFrame(
        {
            "A": [0.01, -0.05, 0.02, -0.01],
            "B": [0.00, -0.02, 0.01, -0.02],
        }
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})
    contrib = component_cvar(returns, weights, confidence=0.75)

    assert set(contrib.index) == {"A", "B"}
    assert contrib.sum() > 0


def test_factor_risk_contribution_positive_variance():
    weights = pd.Series({"A": 0.6, "B": 0.4})
    exposures = pd.DataFrame({"MKT": [1.0, 0.5], "VALUE": [0.2, 1.0]}, index=weights.index)
    factor_cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=exposures.columns, columns=exposures.columns)
    contrib = factor_risk_contribution(weights, exposures, factor_cov)

    assert set(contrib.index) == {"MKT", "VALUE"}


def test_stress_scenario_deterministic_outputs():
    holdings = pd.Series({"A": 10.0, "B": 5.0})
    prices = pd.Series({"A": 100.0, "B": 50.0})
    scenario = StressScenario("down", price_shocks={"A": -0.10, "B": 0.10})
    result = run_stress_scenario(holdings, prices, scenario)

    assert np.isclose(result.base_value, 1250.0)
    assert np.isclose(result.pnl, -75.0)


def test_scenario_set_with_default_factor_scenario():
    holdings = pd.Series({"A": 10.0, "B": 5.0})
    prices = pd.Series({"A": 100.0, "B": 50.0})
    exposures = pd.DataFrame({"equity": [1.0, 0.2], "duration": [0.0, 1.0]}, index=holdings.index)
    scenarios = default_scenarios()[:2]
    result = run_scenario_set(holdings, prices, scenarios, factor_exposures=exposures)

    assert "equity_crash" in result.index
    assert result.loc["equity_crash", "pnl"] < 0

