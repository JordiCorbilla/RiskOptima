###############################################################################
#                                branding.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Shared chart branding helpers
###############################################################################

from __future__ import annotations

RISKOPTIMA_VERSION = "2.5.1"


def riskoptima_signature() -> str:
    return f"Created by RiskOptima v{RISKOPTIMA_VERSION}"


def add_riskoptima_signature(target, x: float = 0.995, y: float = -0.12, fontsize: int = 10, alpha: float = 0.7):
    """
    Add the RiskOptima version signature to a Matplotlib Axes or Figure.
    """
    signature = riskoptima_signature()
    if hasattr(target, "transAxes"):
        target.text(x, y, signature, fontsize=fontsize, color="gray", alpha=alpha, transform=target.transAxes, ha="right")
        return target
    if hasattr(target, "transFigure"):
        target.text(x, y, signature, fontsize=fontsize, color="gray", alpha=alpha, transform=target.transFigure, ha="right")
        return target
    raise TypeError("target must be a Matplotlib Axes or Figure")
