"""Run an offline dislocation demo, or use Yahoo Finance with --live."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from riskoptima.dislocation import (
    DislocationConfig, MarketSnapshot, analyze_symbol, generate_markdown_report, scan_universe,
)


class SyntheticProvider:
    """Explicitly synthetic, deterministic daily data; never a live fallback."""

    def get_snapshot(self, symbol: str, period: str = "5y") -> MarketSnapshot:
        rng = np.random.default_rng(7)
        history = pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, 756))},
            index=pd.bdate_range("2023-01-02", periods=756),
        )
        info = {"quoteType": "EQUITY" if symbol == "MSFT" else "MUTUALFUND", "currency": "USD"}
        if symbol == "MSFT":
            info.update(freeCashflow=8_000_000_000, sharesOutstanding=1_000_000_000,
                        revenueGrowth=0.05, forwardPE=20, enterpriseToEbitda=15, priceToBook=4,
                        marketCap=100_000_000_000, profitMargins=0.2, returnOnEquity=0.25)
        return MarketSnapshot(symbol, history, info, name=f"Synthetic {symbol}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use live Yahoo data instead of synthetic inputs")
    args = parser.parse_args()
    provider = None if args.live else SyntheticProvider()
    print("Data source:", "Yahoo Finance (live)" if args.live else "Synthetic (not market data)")
    config = DislocationConfig(use_markov_regime=args.live)
    summary, reports = scan_universe(
        [("0P0000KM24.L", "Global Bond Index Fund"), ("MSFT", "Microsoft")],
        provider=provider, config=config,
    )
    print(summary.to_string(index=False))
    if not reports:
        raise RuntimeError("No instruments could be analyzed; inspect the error rows")
    equity = analyze_symbol("MSFT", provider=provider, config=config)
    print(generate_markdown_report(equity))


if __name__ == "__main__":
    main()
