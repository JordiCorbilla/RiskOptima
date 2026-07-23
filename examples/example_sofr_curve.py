###############################################################################
#                         example_sofr_curve.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Synthetic USD SOFR-style curve bootstrap and calibration
###############################################################################

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.rates import (
    bootstrap_discount_curve,
    plot_yield_curve,
    price_fixed_rate_bond,
    reprice_curve_instruments,
)


def main():
    # Illustrative decimal quotes; replace this table with a platform market-data response.
    quotes = pd.DataFrame(
        {
            "instrument_type": ["deposit", "deposit", "swap", "swap", "swap", "swap"],
            "maturity": [0.25, 0.50, 1.0, 2.0, 5.0, 10.0],
            "rate": [0.0520, 0.0510, 0.0490, 0.0460, 0.0425, 0.0415],
            "payment_frequency": [np.nan, np.nan, 2, 2, 2, 2],
        }
    )

    curve = bootstrap_discount_curve(
        quotes,
        name="Illustrative USD SOFR Discount Curve",
        valuation_date="2026-07-22",
    )

    print(curve.to_frame([0.25, 0.5, 1.0, 2.0, 5.0, 10.0]).to_string(index=False))
    print("\nCalibration")
    print(reprice_curve_instruments(curve, quotes).to_string(index=False))
    print(f"\n1Y x 1Y forward: {curve.forward_rate(1.0, 2.0):.4%}")
    print(f"5Y par swap rate: {curve.par_swap_rate(5.0):.4%}")
    print(f"5Y 4.50% fixed-rate bond: {price_fixed_rate_bond(curve, 5.0, 0.045):.4f}")

    plot_yield_curve(curve)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

