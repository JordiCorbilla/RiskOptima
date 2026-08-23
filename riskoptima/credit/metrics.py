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


def _validated_credit_inputs(pd_, lgd, ead):
    pd_arr, lgd_arr, ead_arr = np.broadcast_arrays(
        np.asarray(pd_, dtype=float),
        np.asarray(lgd, dtype=float),
        np.asarray(ead, dtype=float),
    )
    if not (np.isfinite(pd_arr).all() and np.isfinite(lgd_arr).all() and np.isfinite(ead_arr).all()):
        raise ValueError("PD, LGD, and EAD values must be finite")
    if np.any((pd_arr < 0) | (pd_arr > 1)):
        raise ValueError("PD values must be between 0 and 1")
    if np.any((lgd_arr < 0) | (lgd_arr > 1)):
        raise ValueError("LGD values must be between 0 and 1")
    if np.any(ead_arr < 0):
        raise ValueError("EAD values must be non-negative")
    return pd_arr, lgd_arr, ead_arr


def expected_loss(pd_, lgd, ead):
    """
    Computes expected credit loss as PD * LGD * EAD.

    Inputs may be scalars, numpy arrays, or pandas Series.
    """
    pd_arr, lgd_arr, ead_arr = _validated_credit_inputs(pd_, lgd, ead)
    return pd_arr * lgd_arr * ead_arr


def unexpected_loss(pd_, lgd, ead, asset_correlation=0.2):
    """
    Approximates unexpected credit loss with a one-factor correlation adjustment.

    The approximation is:
        EAD * LGD * sqrt(PD * (1 - PD)) * sqrt(1 + rho)
    """
    pd_arr, lgd_arr, ead_arr = _validated_credit_inputs(pd_, lgd, ead)
    rho = float(asset_correlation)

    if not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError("asset_correlation must be between 0 and 1")

    return ead_arr * lgd_arr * np.sqrt(pd_arr * (1.0 - pd_arr)) * np.sqrt(1.0 + rho)


def credit_var(losses, confidence=0.99):
    """
    Computes Credit VaR from a simulated loss vector.
    """
    loss_arr = pd.Series(losses, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if loss_arr.size == 0:
        raise ValueError("losses must contain at least one finite value")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    return float(np.quantile(loss_arr, confidence))


def credit_cvar(losses, confidence=0.99):
    """
    Computes Credit CVaR / expected shortfall from a simulated loss vector.
    """
    loss_arr = pd.Series(losses, dtype=float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if loss_arr.size == 0:
        raise ValueError("losses must contain at least one finite value")
    var = credit_var(loss_arr, confidence=confidence)
    tail = loss_arr[loss_arr >= var]
    return float(tail.mean()) if tail.size else var
