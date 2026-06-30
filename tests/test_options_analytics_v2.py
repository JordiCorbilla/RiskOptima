###############################################################################
#                         test_options_analytics_v2.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Options analytics v2 tests
###############################################################################

from datetime import date

import numpy as np
import pandas as pd

from riskoptima.options import (
    OptionBook,
    OptionContract,
    black_scholes_price,
    build_implied_vol_surface,
    event_straddle_analysis,
    option_scenario_grid,
    value_option_contract,
)


def test_put_call_parity_with_dividend_yield():
    s, k, t, r, q, sigma = 100, 100, 1.0, 0.04, 0.01, 0.2
    call = black_scholes_price(s, k, t, r, sigma, "call", q=q)
    put = black_scholes_price(s, k, t, r, sigma, "put", q=q)

    assert np.isclose(call - put, s * np.exp(-q * t) - k * np.exp(-r * t), atol=1e-10)


def test_option_contract_valuation_and_book_aggregation():
    contracts = [
        OptionContract("SPY", 100, date(2027, 6, 30), "call", quantity=2, implied_volatility=0.2),
        OptionContract("SPY", 95, date(2027, 6, 30), "put", quantity=-1, implied_volatility=0.25),
    ]
    book = OptionBook(contracts)
    detail = book.value({"SPY": 101.0}, valuation_date=date(2026, 6, 30), risk_free_rate=0.04)
    summary = book.aggregate(detail)

    assert len(detail) == 2
    assert "market_value" in summary["total"]
    assert detail["notional_delta"].abs().sum() > 0


def test_option_scenario_grid_signs():
    contract = OptionContract("SPY", 100, date(2027, 6, 30), "call", implied_volatility=0.2)
    grid = option_scenario_grid(contract, 100, date(2026, 6, 30), 0.04, 0.2, spot_shocks=(-0.1, 0.1), volatility_shocks=(0.0,))

    low = grid.loc[grid["spot_shock"] < 0, "pnl"].iloc[0]
    high = grid.loc[grid["spot_shock"] > 0, "pnl"].iloc[0]
    assert low < high


def test_implied_vol_surface_round_trip():
    price = black_scholes_price(100, 100, 1.0, 0.04, 0.22, "call")
    chain = pd.DataFrame(
        [{"option_type": "call", "strike": 100.0, "expiry": date(2027, 6, 30), "market_price": price}]
    )
    surface = build_implied_vol_surface(chain, 100, date(2026, 6, 30), 0.04)

    assert np.isclose(surface["implied_volatility"].iloc[0], 0.22, atol=1e-4)


def test_event_straddle_analysis_has_breakevens():
    result = event_straddle_analysis(100, 100, date(2027, 6, 30), date(2026, 6, 30), 0.04, 0.25)

    assert result["upper_breakeven"] > 100
    assert result["lower_breakeven"] < 100
    assert result["straddle_price"] > 0


def test_invalid_option_contract_raises():
    bad = OptionContract("SPY", 0, date(2027, 6, 30), "call", implied_volatility=0.2)

    try:
        value_option_contract(bad, 100, date(2026, 6, 30), 0.04)
    except ValueError as exc:
        assert "strike" in str(exc)
    else:
        raise AssertionError("expected ValueError")

