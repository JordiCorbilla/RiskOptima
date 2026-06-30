###############################################################################
#                         test_optimizer_hardening.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Optimizer hardening tests
###############################################################################

import numpy as np
import pandas as pd
import pytest

from riskoptima.optim import (
    Constraints,
    OptimizationResult,
    ewma_covariance,
    nearest_psd,
    optimize_max_sharpe,
    optimize_min_variance,
    sample_covariance,
)


def _inputs():
    assets = ["A", "B", "C"]
    expected = pd.Series([0.10, 0.07, 0.05], index=assets)
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.03, 0.00], [0.00, 0.00, 0.02]],
        index=assets,
        columns=assets,
    )
    return assets, expected, cov


def test_return_result_diagnostics():
    _, expected, cov = _inputs()
    result = optimize_max_sharpe(expected, cov, return_result=True)

    assert isinstance(result, OptimizationResult)
    assert np.isclose(result.weights.sum(), 1.0)
    assert result.success
    assert result.volatility > 0


def test_weight_and_leverage_constraints():
    _, expected, cov = _inputs()
    constraints = Constraints(weight_bounds=(0.0, 0.6), leverage_limit=1.0)
    weights = optimize_max_sharpe(expected, cov, constraints=constraints)

    assert weights.max() <= 0.600001
    assert np.sum(np.abs(weights)) <= 1.000001


def test_turnover_constraint_enforced():
    _, expected, cov = _inputs()
    previous = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})
    constraints = Constraints(turnover_limit=0.05)
    weights = optimize_min_variance(cov, expected_returns=expected, constraints=constraints, previous_weights=previous)

    assert np.sum(np.abs(weights - previous)) <= 0.050001


def test_factor_sector_asset_class_constraints():
    assets, expected, cov = _inputs()
    exposures = pd.DataFrame({"quality": [1.0, 0.0, 0.0]}, index=assets)
    metadata = pd.DataFrame(
        {"sector": ["Tech", "Utilities", "Utilities"], "asset_class": ["Equity", "Equity", "Bond"]},
        index=assets,
    )
    constraints = Constraints(
        factor_bounds={"quality": (0.0, 0.7)},
        sector_bounds={"Utilities": (0.2, 0.8)},
        asset_class_bounds={"Bond": (0.1, 0.6)},
    )
    weights = optimize_max_sharpe(expected, cov, constraints=constraints, factor_exposures=exposures, metadata=metadata)

    assert float(weights["A"]) <= 0.700001
    assert 0.2 <= float(weights[["B", "C"]].sum()) <= 0.800001
    assert 0.1 <= float(weights["C"]) <= 0.600001


def test_infeasible_constraints_raise():
    _, expected, cov = _inputs()
    constraints = Constraints(weight_bounds=(0.0, 0.2))

    with pytest.raises(ValueError):
        optimize_max_sharpe(expected, cov, constraints=constraints)


def test_covariance_helpers():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(100, 3)), columns=["A", "B", "C"])
    sample = sample_covariance(returns)
    ewma = ewma_covariance(returns)
    psd = nearest_psd(pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=["A", "B"], columns=["A", "B"]))

    assert sample.shape == (3, 3)
    assert ewma.shape == (3, 3)
    assert np.linalg.eigvalsh(psd.values).min() >= -1e-9

