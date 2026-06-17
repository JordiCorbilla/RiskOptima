###############################################################################
#                                  metrics.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Credit risk metrics
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_loss(pd_, lgd, ead):
    """
    Computes expected credit loss as PD * LGD * EAD.

    Inputs may be scalars, numpy arrays, or pandas Series.
    """
    return np.asarray(pd_) * np.asarray(lgd) * np.asarray(ead)


def unexpected_loss(pd_, lgd, ead, asset_correlation=0.2):
    """
    Approximates unexpected credit loss with a one-factor correlation adjustment.

    The approximation is:
        EAD * LGD * sqrt(PD * (1 - PD)) * sqrt(1 + rho)
    """
    pd_arr = np.asarray(pd_, dtype=float)
    lgd_arr = np.asarray(lgd, dtype=float)
    ead_arr = np.asarray(ead, dtype=float)
    rho = float(asset_correlation)

    if np.any((pd_arr < 0) | (pd_arr > 1)):
        raise ValueError("PD values must be between 0 and 1")
    if np.any((lgd_arr < 0) | (lgd_arr > 1)):
        raise ValueError("LGD values must be between 0 and 1")
    if rho < -1:
        raise ValueError("asset_correlation must be greater than or equal to -1")

    return ead_arr * lgd_arr * np.sqrt(pd_arr * (1.0 - pd_arr)) * np.sqrt(1.0 + rho)


def credit_var(losses, confidence=0.99):
    """
    Computes Credit VaR from a simulated loss vector.
    """
    loss_arr = pd.Series(losses, dtype=float).dropna().to_numpy()
    if loss_arr.size == 0:
        raise ValueError("losses must contain at least one finite value")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return float(np.quantile(loss_arr, confidence))


def credit_cvar(losses, confidence=0.99):
    """
    Computes Credit CVaR / expected shortfall from a simulated loss vector.
    """
    loss_arr = pd.Series(losses, dtype=float).dropna().to_numpy()
    if loss_arr.size == 0:
        raise ValueError("losses must contain at least one finite value")
    var = credit_var(loss_arr, confidence=confidence)
    tail = loss_arr[loss_arr >= var]
    return float(tail.mean()) if tail.size else var
