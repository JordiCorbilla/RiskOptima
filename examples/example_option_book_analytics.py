from datetime import date
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.options import (
    OptionBook,
    OptionContract,
    event_straddle_analysis,
    option_scenario_grid,
)


def main():
    valuation_date = date(2026, 6, 30)
    expiry = date(2027, 6, 30)
    contracts = [
        OptionContract("SPY", strike=500, expiry=expiry, option_type="call", quantity=2, implied_volatility=0.22),
        OptionContract("SPY", strike=480, expiry=expiry, option_type="put", quantity=-1, implied_volatility=0.25),
    ]
    book = OptionBook(contracts)
    detail = book.value({"SPY": 505.0}, valuation_date=valuation_date, risk_free_rate=0.04)
    summary = book.aggregate(detail)

    scenario_grid = option_scenario_grid(contracts[0], 505.0, valuation_date, 0.04, 0.22)
    straddle = event_straddle_analysis(505.0, 505.0, expiry, valuation_date, 0.04, 0.24)

    print(detail[["underlying", "option_type", "strike", "price", "market_value", "notional_delta"]])
    print(pd.Series(summary["total"]))
    print(scenario_grid.head())
    print(straddle)


if __name__ == "__main__":
    main()
