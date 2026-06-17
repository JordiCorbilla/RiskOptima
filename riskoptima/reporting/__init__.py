###############################################################################
#                                 __init__.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Reporting module
###############################################################################

from .market_risk import (
    build_market_risk_report,
    plot_drawdown_curve,
    plot_rolling_volatility,
    plot_var_cvar_distribution,
    plot_correlation_heatmap,
)

__all__ = [
    "build_market_risk_report",
    "plot_drawdown_curve",
    "plot_rolling_volatility",
    "plot_var_cvar_distribution",
    "plot_correlation_heatmap",
]
