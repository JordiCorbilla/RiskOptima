###############################################################################
#                             markov_regimes.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Market regime reporting with Markov models
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd

from riskoptima.core import RiskReport
from riskoptima.risk import fit_markov_regime_model, regime_summary


def _to_return_series(data, input_type: str = "auto") -> pd.Series:
    series = pd.Series(data, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        raise ValueError("data must contain at least one finite value")
    input_type = input_type.lower()
    if input_type not in {"auto", "returns", "prices"}:
        raise ValueError("input_type must be 'auto', 'returns', or 'prices'")
    if input_type == "prices" or (input_type == "auto" and (series > 1.5).all()):
        series = series.pct_change().dropna()
    return series.rename("returns")


def build_markov_regime_report(
    data,
    input_type: str = "auto",
    n_regimes: int = 3,
    n_iter: int = 100,
    random_state: int | None = None,
    initial_value: float = 100.0,
) -> RiskReport:
    """
    Build a market-regime report using a univariate Gaussian HMM.

    Pass daily returns directly or price levels with ``input_type='prices'``.
    The report contains returns, cumulative wealth, fitted model parameters,
    state probabilities, most likely regimes, and regime-level summary stats.
    """
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")
    returns = _to_return_series(data, input_type=input_type)
    model = fit_markov_regime_model(
        returns,
        n_regimes=n_regimes,
        n_iter=n_iter,
        random_state=random_state,
    )
    wealth = ((1.0 + returns).cumprod() * initial_value).rename("wealth")
    summary = regime_summary(returns, model.regimes)
    return RiskReport(
        metrics={
            "returns": returns,
            "wealth": wealth,
            "model": model,
            "regimes": model.regimes,
            "regime_probabilities": model.regime_probabilities,
            "transition_matrix": model.transition_matrix,
            "regime_summary": summary,
        }
    )


def plot_markov_regime_chart(
    report: RiskReport,
    title: str = "Market Regimes Identified by Hidden Markov Model",
    ax=None,
    cmap: str = "tab10",
    point_size: float = 12.0,
):
    """
    Plot cumulative portfolio value colored by the most likely Markov regime.
    """
    import matplotlib.pyplot as plt

    wealth = report.metrics["wealth"]
    regimes = report.metrics["regimes"].reindex(wealth.index)
    ax = ax or plt.gca()
    palette = plt.get_cmap(cmap)
    for regime in sorted(regimes.dropna().unique()):
        mask = regimes == regime
        ax.scatter(
            wealth.index[mask],
            wealth.loc[mask],
            s=point_size,
            color=palette(int(regime) % palette.N),
            label=f"Regime {int(regime)}",
            alpha=0.8,
            edgecolors="none",
        )
    ax.plot(wealth.index, wealth, color="#333333", linewidth=0.8, alpha=0.35)
    ax.set_title(title)
    ax.set_ylabel("Cumulative value")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_markov_regime_probabilities(
    report: RiskReport,
    title: str = "Markov Regime Probabilities",
    ax=None,
):
    """Plot smoothed probabilities for each fitted regime."""
    import matplotlib.pyplot as plt

    probabilities = report.metrics["regime_probabilities"]
    ax = ax or plt.gca()
    probabilities.plot(ax=ax, linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Date")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    return ax

