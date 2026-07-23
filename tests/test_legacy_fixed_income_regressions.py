###############################################################################
#                 test_legacy_fixed_income_regressions.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Correctness regressions for backwards-compatible legacy APIs
###############################################################################

import numpy as np
import pandas as pd
import pytest

from riskoptima import RiskOptima


def test_terminal_stats_uses_reach_mask_and_positive_surplus():
    returns = pd.DataFrame([[-0.50, 0.00, 0.50]], columns=["breach", "middle", "reach"])

    stats = RiskOptima.terminal_stats(returns, floor=0.8, cap=1.2)

    assert stats.loc["p_breach", "Stats"] == pytest.approx(1 / 3)
    assert stats.loc["p_reach", "Stats"] == pytest.approx(1 / 3)
    assert stats.loc["e_short", "Stats"] == pytest.approx(0.3)
    assert stats.loc["e_surplus", "Stats"] == pytest.approx(0.3)


def test_frequency_aware_bond_price_and_duration():
    cash_flows = RiskOptima.bond_cash_flows_v2(n_periods=4, par=1000, coupon_rate=0.06, freq=2)
    periods = np.arange(1, 5)
    expected_discount_factors = (1 + 0.05 / 2) ** (-periods)
    expected_price = float(np.dot(cash_flows, expected_discount_factors))
    expected_duration = float(
        np.dot(periods / 2, cash_flows * expected_discount_factors) / expected_price
    )

    assert RiskOptima.bond_price_v2(cash_flows, yield_rate=0.05, freq=2) == pytest.approx(expected_price)
    assert RiskOptima.macaulay_duration_v2(cash_flows, yield_rate=0.05, freq=2) == pytest.approx(expected_duration)

    _, metrics = RiskOptima.macaulay_duration_v3(cash_flows, yield_rate=0.05, freq=2)
    modified_duration = expected_duration / (1 + 0.05 / 2)
    assert metrics.loc[0, "PVBP (DV01)"] == pytest.approx(modified_duration * expected_price * 0.0001)

