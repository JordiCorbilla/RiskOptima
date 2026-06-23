###############################################################################
#                                 __init__.py                                  
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from .engine import run_backtest
from .strategy import Strategy, SMACrossStrategy
from .portfolio import PortfolioState
from .sma import (
    build_sma_signal_frame,
    trades_from_sma_signals,
    run_sma_strategy_with_risk,
    run_strategy_on_portfolio,
    run_and_plot_sma_strategy,
    plot_sma_strategy_cumulative_return,
    plot_sma_strategy_trades,
)

__all__ = [
    "run_backtest",
    "Strategy",
    "SMACrossStrategy",
    "PortfolioState",
    "build_sma_signal_frame",
    "trades_from_sma_signals",
    "run_sma_strategy_with_risk",
    "run_strategy_on_portfolio",
    "run_and_plot_sma_strategy",
    "plot_sma_strategy_cumulative_return",
    "plot_sma_strategy_trades",
]
