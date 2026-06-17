###############################################################################
#                                 portfolio.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Credit portfolio helpers
###############################################################################

from __future__ import annotations

import pandas as pd

from .metrics import expected_loss


def _resolve_column(df: pd.DataFrame, *names: str) -> str:
    lookup = {col.lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(f"portfolio must contain one of: {', '.join(names)}")


def portfolio_expected_loss(df: pd.DataFrame):
    """
    Computes total expected loss for a portfolio DataFrame.

    Required columns are PD, LGD, and EAD. Column matching is case-insensitive.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    pd_col = _resolve_column(df, "pd", "probability_of_default")
    lgd_col = _resolve_column(df, "lgd", "loss_given_default")
    ead_col = _resolve_column(df, "ead", "exposure_at_default")

    losses = expected_loss(df[pd_col], df[lgd_col], df[ead_col])
    return float(pd.Series(losses, index=df.index).sum())
