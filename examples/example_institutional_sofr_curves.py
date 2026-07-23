###############################################################################
#                 example_institutional_sofr_curves.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Dated SOFR, multi-curve, Jacobian, and key-rate DV01 example
###############################################################################

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.rates import (
    DiscountCurve,
    MultiCurveSet,
    bootstrap_projection_curve,
    bootstrap_sofr_curve,
    curve_calibration_jacobian,
    key_rate_dv01,
    plot_yield_curve,
    reprice_projection_instruments,
    reprice_sofr_instruments,
)


def dated_sofr_quotes():
    return pd.DataFrame(
        [
            {"quote_id": "ON", "instrument_type": "deposit", "maturity_date": "2026-07-02", "rate": 0.0430},
            {
                "quote_id": "FOMC-SEP",
                "instrument_type": "fomc",
                "start_date": "2026-07-02",
                "end_date": "2026-09-17",
                "rate": 0.0425,
            },
            {
                "quote_id": "FOMC-OCT",
                "instrument_type": "fomc",
                "start_date": "2026-09-17",
                "end_date": "2026-10-29",
                "rate": 0.0415,
            },
            {
                "quote_id": "SR3-H7",
                "instrument_type": "future",
                "start_date": "2026-10-29",
                "end_date": "2027-01-29",
                "price": 95.95,
                "convexity_adjustment_bps": 0.5,
            },
            {
                "quote_id": "OIS-2Y",
                "instrument_type": "swap",
                "maturity_date": "2028-07-03",
                "rate": 0.0390,
                "payment_frequency": 1,
            },
            {
                "quote_id": "OIS-5Y",
                "instrument_type": "swap",
                "maturity_date": "2031-07-03",
                "rate": 0.0375,
                "payment_frequency": 1,
            },
        ]
    )


def projection_quotes():
    return pd.DataFrame(
        [
            {"quote_id": "DEP-3M", "instrument_type": "deposit", "maturity": 0.25, "rate": 0.0450},
            {"quote_id": "FRA-3X6", "instrument_type": "fra", "start": 0.25, "maturity": 0.50, "rate": 0.0440},
            {"quote_id": "SWAP-1Y", "instrument_type": "swap", "maturity": 1.0, "rate": 0.0430},
            {"quote_id": "SWAP-2Y", "instrument_type": "swap", "maturity": 2.0, "rate": 0.0420},
            {"quote_id": "SWAP-5Y", "instrument_type": "swap", "maturity": 5.0, "rate": 0.0410},
        ]
    )


def main():
    valuation_date = "2026-07-01"
    ois_quotes = dated_sofr_quotes()
    discount_curve = bootstrap_sofr_curve(ois_quotes, valuation_date=valuation_date)
    print("Dated SOFR calibration")
    print(reprice_sofr_instruments(discount_curve, ois_quotes).to_string(index=False))

    forward_quotes = projection_quotes()
    projection_curve = bootstrap_projection_curve(forward_quotes, discount_curve, name="SOFR 3M projection")
    print("\nProjection-curve calibration")
    print(reprice_projection_instruments(projection_curve, discount_curve, forward_quotes).to_string(index=False))

    curves = MultiCurveSet(discount_curve, {"SOFR-3M": projection_curve})
    print(f"\n5Y multi-curve par swap rate: {curves.par_swap_rate(5.0, 'SOFR-3M'):.4%}")

    jacobian = curve_calibration_jacobian(
        ois_quotes,
        curve_builder=bootstrap_sofr_curve,
        builder_kwargs={"valuation_date": valuation_date},
    )
    print("\nZero-rate calibration Jacobian (bp per bp)")
    print(jacobian.round(4).to_string())

    cash_flows = np.array([2.25, 2.25, 2.25, 2.25, 102.25])
    payment_times = np.arange(1.0, 6.0)
    print("\n5Y bond key-rate DV01")
    print(key_rate_dv01(discount_curve, cash_flows, payment_times).round(6).to_string())

    # Compare the discount and projection term structures on the same axes.
    ax = plot_yield_curve(discount_curve, show_forwards=True)
    grid = np.linspace(0.05, 5.0, 200)
    ax.plot(
        grid,
        projection_curve.zero_rate(grid) * 100,
        color="#6b7280",
        linestyle=":",
        linewidth=1.8,
        label="SOFR 3M projection zero rate",
    )
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

