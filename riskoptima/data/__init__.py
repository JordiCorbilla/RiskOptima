###############################################################################
#                                 __init__.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Packaged deterministic sample data
###############################################################################

from __future__ import annotations

from importlib.resources import files

import pandas as pd


def _read_sample_csv(filename: str, **kwargs) -> pd.DataFrame:
    resource = files(__package__).joinpath(filename)
    with resource.open("rb") as stream:
        return pd.read_csv(stream, **kwargs)


def load_sample_market_returns() -> pd.DataFrame:
    """Load two years of packaged deterministic multi-asset business-day returns."""
    return _read_sample_csv("synthetic_market_returns.csv", index_col=0, parse_dates=True)


def load_sample_credit_portfolio() -> pd.DataFrame:
    """Load the packaged deterministic PD/LGD/EAD obligor sample."""
    return _read_sample_csv("synthetic_credit_portfolio.csv")


__all__ = ["load_sample_market_returns", "load_sample_credit_portfolio"]
