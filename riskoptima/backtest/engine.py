###############################################################################
#                                  engine.py                                   
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from riskoptima.core.types import BacktestConfig
from riskoptima.optim.costs import SimpleCostModel
from .portfolio import PortfolioState
from .strategy import Strategy


def _should_rebalance(date: pd.Timestamp, rule: str, last_reb_date: Optional[pd.Timestamp]) -> bool:
    normalized_rule = str(rule).upper()
    if normalized_rule == "D":
        return True
    if normalized_rule == "M":
        return last_reb_date is None or date.to_period("M") != last_reb_date.to_period("M")
    if normalized_rule == "W":
        return last_reb_date is None or date.to_period("W") != last_reb_date.to_period("W")
    if last_reb_date is None:
        return True
    return False


def run_backtest(
    prices: pd.DataFrame,
    strategy: Strategy,
    config: Optional[BacktestConfig] = None,
    cost_model: Optional[SimpleCostModel] = None,
    adv: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute at the current bar using strategy prices strictly before that bar.

    Past observations before config.start can warm up a strategy. Missing
    quotes use the last close for valuation and cannot execute trades.
    ADV observations must also precede execution.
    """
    if config is None:
        config = BacktestConfig()
    if cost_model is None:
        cost_model = SimpleCostModel()

    if not isinstance(prices, pd.DataFrame) or prices.empty:
        raise ValueError("prices must be a non-empty pandas DataFrame")
    if not np.isfinite(config.initial_cash) or config.initial_cash <= 0:
        raise ValueError("initial_cash must be positive and finite")
    if not np.isfinite(config.slippage_bps) or config.slippage_bps < 0:
        raise ValueError("slippage_bps must be non-negative and finite")

    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.hasnans:
        raise ValueError("prices must have a DatetimeIndex without missing timestamps")
    if not prices.index.is_unique or not prices.columns.is_unique:
        raise ValueError("price timestamps and asset labels must be unique")

    quotes = prices.sort_index().apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if config.end is not None:
        quotes = quotes.loc[quotes.index <= config.end]
    if (quotes <= 0).any().any():
        raise ValueError("prices must be strictly positive")
    history = quotes.ffill()
    data = history if config.start is None else history.loc[history.index >= config.start]
    if data.dropna(how="all").empty:
        raise ValueError("prices must contain at least one finite observation as of the requested period")

    assets = data.columns
    if adv is not None:
        if not isinstance(adv.index, pd.DatetimeIndex) or adv.index.hasnans:
            raise ValueError("ADV must have a DatetimeIndex without missing timestamps")
        if not adv.index.is_unique or not adv.columns.is_unique:
            raise ValueError("ADV timestamps and asset labels must be unique")
        adv_data = adv.reindex(columns=assets).sort_index().apply(pd.to_numeric, errors="coerce")
        adv_data = adv_data.where(np.isfinite(adv_data) & adv_data.gt(0)).ffill()
        adv_positions = adv_data.index.searchsorted(data.index, side="left") - 1

    positions = pd.Series(0.0, index=assets)
    state = PortfolioState(positions=positions, cash=config.initial_cash)

    equity_rows = []
    weights_rows = []
    last_reb_date = None

    for offset, (date, row) in enumerate(data.iterrows()):
        price_row = row.dropna()
        if price_row.empty:
            continue

        port_value = state.value(price_row)
        costs = 0.0
        turnover = 0.0

        if _should_rebalance(date, config.rebalance_rule, last_reb_date):
            cutoff = history.index.searchsorted(date)
            snapshot = PortfolioState(positions=state.positions.copy(), cash=state.cash)
            target_weights = strategy.generate_target_weights(date, history.iloc[:cutoff].copy(), state=snapshot)
            if not target_weights.index.is_unique:
                raise ValueError("target weight labels must be unique")
            target_weights = target_weights.reindex(assets).fillna(0.0)
            if not np.isfinite(target_weights).all():
                raise ValueError("target weights must be finite")
            if target_weights.sum() != 0:
                target_weights = target_weights / target_weights.sum()

            target_values = target_weights * port_value
            tradable = quotes.loc[date].notna()
            marked_prices = price_row.reindex(assets)
            if not tradable.all():
                frozen_value = (state.positions[~tradable] * marked_prices[~tradable]).sum()
                available_value = max(port_value - frozen_value, 0.0)
                target_net_value = target_values[tradable].sum()
                if target_net_value > available_value:
                    target_values = target_values * (available_value / target_net_value)
            target_shares = (target_values / marked_prices).where(tradable, state.positions)

            orders = target_shares - state.positions
            order_notional = (orders.abs() * price_row.reindex(assets)).fillna(0.0)
            turnover = float(order_notional.sum() / port_value) if port_value > 0 else 0.0

            for asset, notional in order_notional.items():
                adv_value = None
                if adv is not None and adv_positions[offset] >= 0:
                    prior_value = adv_data.iloc[adv_positions[offset]][asset]
                    if pd.notna(prior_value):
                        adv_value = float(prior_value)
                costs += cost_model.estimate_cost(float(notional), adv=adv_value)

            slippage_cost = order_notional.sum() * (config.slippage_bps * 1e-4)
            costs += float(slippage_cost)

            trade_value = float((orders * price_row.reindex(assets)).fillna(0.0).sum())
            state.cash = state.cash - trade_value - costs
            state.positions = target_shares
            last_reb_date = date

        port_value = state.value(price_row)
        asset_values = state.positions * price_row.reindex(assets).fillna(0.0)
        weights = asset_values / port_value if port_value != 0 else pd.Series(0.0, index=assets)

        equity_rows.append(
            {
                "Date": date,
                "PortfolioValue": port_value,
                "Cash": state.cash,
                "Costs": costs,
                "Turnover": turnover,
            }
        )
        weights_rows.append(weights.rename(date))

    equity_curve = pd.DataFrame(equity_rows).set_index("Date")
    weights_history = pd.DataFrame(weights_rows)
    return equity_curve, weights_history
