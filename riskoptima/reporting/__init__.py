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
from .sophistication import (
    available_sophistication_methods,
    build_portfolio_sophistication_report,
    plot_portfolio_sophistication_report,
)
from .markov_regimes import (
    build_markov_regime_report,
    plot_markov_regime_chart,
    plot_markov_regime_probabilities,
)

__all__ = [
    "build_market_risk_report",
    "plot_drawdown_curve",
    "plot_rolling_volatility",
    "plot_var_cvar_distribution",
    "plot_correlation_heatmap",
    "available_sophistication_methods",
    "build_portfolio_sophistication_report",
    "plot_portfolio_sophistication_report",
    "build_markov_regime_report",
    "plot_markov_regime_chart",
    "plot_markov_regime_probabilities",
]
