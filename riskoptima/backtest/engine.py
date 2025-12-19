from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from riskoptima.core.types import BacktestConfig
from riskoptima.optim.costs import SimpleCostModel
from .portfolio import PortfolioState
from .strategy import Strategy


def _should_rebalance(date: pd.Timestamp, rule: str, last_reb_date: Optional[pd.Timestamp]) -> bool:
    if rule == "D":
        return True
    if rule == "M":
        return date.is_month_end
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
    if config is None:
        config = BacktestConfig()
    if cost_model is None:
        cost_model = SimpleCostModel()

    data = prices.copy().sort_index()
    if config.start is not None:
        data = data[data.index >= config.start]
    if config.end is not None:
        data = data[data.index <= config.end]

    assets = data.columns
    positions = pd.Series(0.0, index=assets)
    state = PortfolioState(positions=positions, cash=config.initial_cash)

    equity_rows = []
    weights_rows = []
    last_reb_date = None

    for date, row in data.iterrows():
        price_row = row.dropna()
        if price_row.empty:
            continue

        port_value = state.value(price_row)
        weights = pd.Series(0.0, index=assets)
        costs = 0.0
        turnover = 0.0

        if _should_rebalance(date, config.rebalance_rule, last_reb_date):
            target_weights = strategy.generate_target_weights(date, data, state=state)
            target_weights = target_weights.reindex(assets).fillna(0.0)
            if target_weights.sum() != 0:
                target_weights = target_weights / target_weights.sum()

            target_values = target_weights * port_value
            target_shares = target_values / price_row.reindex(assets)
            target_shares = target_shares.replace([np.inf, -np.inf], 0.0).fillna(0.0)

            orders = target_shares - state.positions
            order_notional = (orders.abs() * price_row.reindex(assets)).fillna(0.0)
            turnover = float(order_notional.sum() / port_value) if port_value > 0 else 0.0

            for asset, notional in order_notional.items():
                adv_value = None
                if adv is not None and asset in adv.columns:
                    adv_value = adv.loc[date, asset]
                costs += cost_model.estimate_cost(float(notional), adv=adv_value)

            slippage_cost = order_notional.sum() * (config.slippage_bps * 1e-4)
            costs += float(slippage_cost)

            trade_value = float((orders * price_row.reindex(assets)).fillna(0.0).sum())
            state.cash = state.cash - trade_value - costs
            state.positions = target_shares
            last_reb_date = date

            weights = target_weights

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
