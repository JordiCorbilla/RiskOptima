"""Empirical expected shortfall for equal-weight observations."""

import math

import numpy as np


def _empirical_tail_weights(losses: np.ndarray, confidence: float) -> np.ndarray:
    """Allocate exactly the upper tail mass, sharing boundary mass among ties."""
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if losses.ndim != 1 or not losses.size or not np.isfinite(losses).all():
        raise ValueError("losses must be a non-empty vector of finite values")
    if confidence == 1:
        worst = losses == losses.max()
        return worst / worst.sum()
    mass = (1.0 - confidence) * losses.size
    boundary_index = losses.size - math.ceil(mass)
    cutoff = np.partition(losses, boundary_index)[boundary_index]
    above = losses > cutoff
    tied = losses == cutoff
    boundary_weight = max(mass - above.sum(), 0.0) / tied.sum()
    return (above + tied * boundary_weight) / mass


def _empirical_expected_shortfall(losses: np.ndarray, confidence: float) -> float:
    return float(_empirical_tail_weights(losses, confidence) @ losses)
