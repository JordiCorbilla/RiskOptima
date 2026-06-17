###############################################################################
#                     example_option_pricing_engine.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Option pricing engine example
###############################################################################

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.options import (
    binomial_tree_price,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
    monte_carlo_european_option,
)


if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    call = black_scholes_price(S, K, T, r, sigma, option_type="call")
    put = black_scholes_price(S, K, T, r, sigma, option_type="put")
    greeks = pd.Series(black_scholes_greeks(S, K, T, r, sigma, option_type="call"))
    iv = implied_volatility(call, S, K, T, r, option_type="call")
    tree = binomial_tree_price(S, K, T, r, sigma, steps=250, option_type="call")
    mc = monte_carlo_european_option(S, K, T, r, sigma, option_type="call", random_state=42)

    print(f"Call price: {call:.4f}")
    print(f"Put price: {put:.4f}")
    print("Greeks:")
    print(greeks.round(4))
    print(f"Implied volatility: {iv:.4f}")
    print(f"Binomial call: {tree:.4f}")
    print(f"Monte Carlo call: {mc:.4f}")
