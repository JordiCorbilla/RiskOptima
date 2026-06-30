###############################################################################
#                               scenarios.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Portfolio scenario and stress analytics
###############################################################################

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressScenario:
    name: str
    price_shocks: dict[str, float] = field(default_factory=dict)
    factor_shocks: dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass(frozen=True)
class StressResult:
    scenario: StressScenario
    base_value: float
    stressed_value: float
    pnl: float
    pnl_pct: float
    asset_pnl: pd.Series


def apply_price_shock(prices, shocks: dict[str, float]) -> pd.Series:
    series = pd.Series(prices, dtype=float)
    shocked = series.copy()
    for asset, shock in shocks.items():
        if asset not in shocked.index:
            raise ValueError(f"price shock references unknown asset: {asset}")
        shocked.loc[asset] = shocked.loc[asset] * (1.0 + float(shock))
    return shocked


def apply_factor_shock(exposures, factor_shocks: dict[str, float]) -> pd.Series:
    frame = pd.DataFrame(exposures).astype(float)
    missing = [factor for factor in factor_shocks if factor not in frame.columns]
    if missing:
        raise ValueError(f"factor shocks reference unknown factors: {missing}")
    impact = pd.Series(0.0, index=frame.index)
    for factor, shock in factor_shocks.items():
        impact = impact + frame[factor] * float(shock)
    return impact.rename("factor_return_shock")


def run_stress_scenario(
    holdings,
    prices,
    scenario: StressScenario,
    factor_exposures=None,
) -> StressResult:
    quantities = pd.Series(holdings, dtype=float)
    price_series = pd.Series(prices, dtype=float).reindex(quantities.index)
    if price_series.isna().any():
        raise ValueError("prices must align with holdings")
    shocked_prices = apply_price_shock(price_series, scenario.price_shocks)
    if scenario.factor_shocks:
        if factor_exposures is None:
            raise ValueError("factor_exposures are required for factor shocks")
        factor_returns = apply_factor_shock(pd.DataFrame(factor_exposures).reindex(quantities.index), scenario.factor_shocks)
        shocked_prices = shocked_prices * (1.0 + factor_returns)
    base_asset_values = quantities * price_series
    stressed_asset_values = quantities * shocked_prices
    asset_pnl = (stressed_asset_values - base_asset_values).rename("asset_pnl")
    base_value = float(base_asset_values.sum())
    stressed_value = float(stressed_asset_values.sum())
    pnl = stressed_value - base_value
    pnl_pct = pnl / base_value if base_value else np.nan
    return StressResult(
        scenario=scenario,
        base_value=base_value,
        stressed_value=stressed_value,
        pnl=float(pnl),
        pnl_pct=float(pnl_pct),
        asset_pnl=asset_pnl,
    )


def run_scenario_set(holdings, prices, scenarios, factor_exposures=None) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        result = run_stress_scenario(holdings, prices, scenario, factor_exposures=factor_exposures)
        rows.append(
            {
                "scenario": scenario.name,
                "base_value": result.base_value,
                "stressed_value": result.stressed_value,
                "pnl": result.pnl,
                "pnl_pct": result.pnl_pct,
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def default_scenarios() -> list[StressScenario]:
    return [
        StressScenario("equity_crash", factor_shocks={"equity": -0.20}, description="Broad equity market selloff"),
        StressScenario("rates_up", factor_shocks={"duration": -0.08}, description="Rates rise and duration assets fall"),
        StressScenario("credit_spread_widening", factor_shocks={"credit": -0.10}, description="Credit spread widening"),
        StressScenario("volatility_spike", factor_shocks={"volatility": 0.15}, description="Volatility shock"),
        StressScenario("commodity_shock", factor_shocks={"commodity": -0.12}, description="Commodity selloff"),
        StressScenario("usd_shock", factor_shocks={"usd": 0.08}, description="US dollar shock"),
    ]

