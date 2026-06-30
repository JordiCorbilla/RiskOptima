###############################################################################
#                                 __init__.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Option pricing engine
###############################################################################

from .black_scholes import black_scholes_price
from .greeks import black_scholes_greeks
from .implied_vol import implied_volatility
from .binomial import binomial_tree_price
from .monte_carlo import monte_carlo_european_option
from .analytics import (
    OptionBook,
    OptionContract,
    OptionValuationResult,
    build_greeks_grid,
    build_implied_vol_surface,
    event_straddle_analysis,
    option_scenario_grid,
    price_option_chain,
    value_option_contract,
)

__all__ = [
    "black_scholes_price",
    "black_scholes_greeks",
    "implied_volatility",
    "binomial_tree_price",
    "monte_carlo_european_option",
    "OptionBook",
    "OptionContract",
    "OptionValuationResult",
    "value_option_contract",
    "price_option_chain",
    "build_greeks_grid",
    "build_implied_vol_surface",
    "option_scenario_grid",
    "event_straddle_analysis",
]
