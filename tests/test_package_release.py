###############################################################################
#                          test_package_release.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Package version and cross-module integration tests
###############################################################################

from pathlib import Path
import tomllib

import numpy as np
import pytest

import riskoptima
from riskoptima import RiskOptima
from riskoptima.rates import DiscountCurve, price_fixed_rate_bond


def test_package_versions_have_one_runtime_source():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))

    assert riskoptima.__version__ == "2.7.0"
    assert RiskOptima.VERSION == riskoptima.__version__
    assert metadata["tool"]["poetry"]["version"] == riskoptima.__version__


def test_curve_prices_fixed_rate_bond_cash_flows():
    curve = DiscountCurve.from_zero_rates([0.5, 1.0, 2.0], [0.04, 0.04, 0.04])
    payment_times = np.array([0.5, 1.0, 1.5, 2.0])
    cash_flows = np.array([2.5, 2.5, 2.5, 102.5])
    expected = float(np.dot(cash_flows, np.exp(-0.04 * payment_times)))

    assert price_fixed_rate_bond(curve, 2.0, 0.05, face_value=100, payment_frequency=2) == pytest.approx(expected)

