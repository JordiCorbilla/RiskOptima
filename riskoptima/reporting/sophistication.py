###############################################################################
#                            sophistication.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Portfolio sophistication comparison reports
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from riskoptima.core import RiskReport
from riskoptima.optim import optimize_max_sharpe, optimize_min_variance


DEFAULT_METHODS = ("MV", "CVaR", "EVaR", "RLVaR", "WR", "CDaR", "EDaR", "RLDaR", "MDD", "1N")


@dataclass(frozen=True)
class SophisticationMethod:
    label: str
    description: str


METHOD_DESCRIPTIONS = {
    "MV": SophisticationMethod("MV", "minimum variance portfolio"),
    "CVaR": SophisticationMethod("CVaR", "historical expected shortfall proxy"),
    "EVaR": SophisticationMethod("EVaR", "entropic value-at-risk proxy"),
    "RLVaR": SophisticationMethod("RLVaR", "robust tail loss proxy"),
    "WR": SophisticationMethod("WR", "return-to-risk weighted portfolio"),
    "CDaR": SophisticationMethod("CDaR", "conditional drawdown-at-risk proxy"),
    "EDaR": SophisticationMethod("EDaR", "entropic drawdown-at-risk proxy"),
    "RLDaR": SophisticationMethod("RLDaR", "robust large drawdown proxy"),
    "MDD": SophisticationMethod("MDD", "minimum maximum drawdown proxy"),
    "1N": SophisticationMethod("1N", "equal-weight portfolio"),
}

_METHOD_DESCRIPTIONS_BY_KEY = {key.upper(): value for key, value in METHOD_DESCRIPTIONS.items()}


def _clean_returns(returns) -> pd.DataFrame:
    data = pd.DataFrame(returns).copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    data = data.dropna(axis=1, how="all").dropna(how="any")
    if data.empty:
        raise ValueError("returns must contain at least one complete finite row")
    if data.shape[1] < 2:
        raise ValueError("portfolio sophistication reports require at least two assets")
    return data


def _portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.dot(weights)


def _bounds(n_assets: int):
    return [(0.0, 1.0)] * n_assets


def _constraints():
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def _fallback_equal_weight(index: pd.Index) -> pd.Series:
    return pd.Series(np.repeat(1.0 / len(index), len(index)), index=index)


def _solve_long_only(returns: pd.DataFrame, objective, label: str) -> pd.Series:
    n_assets = returns.shape[1]
    initial = np.repeat(1.0 / n_assets, n_assets)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=_bounds(n_assets),
        constraints=_constraints(),
        options={"maxiter": 750, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"{label} optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns, name=label)


def _cvar_loss(values: np.ndarray, confidence: float) -> float:
    losses = -values
    threshold = np.quantile(losses, confidence)
    tail = losses[losses >= threshold]
    return float(tail.mean()) if tail.size else float(threshold)


def _evar_loss(values: np.ndarray, confidence: float) -> float:
    losses = -values
    alpha = max(1.0 - confidence, 1e-8)
    theta_grid = np.geomspace(0.25, 150.0, 60)
    scores = []
    for theta in theta_grid:
        scaled = theta * losses
        scores.append((logsumexp(scaled) - np.log(len(losses)) - np.log(alpha)) / theta)
    return float(np.min(scores))


def _drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1.0 + returns).cumprod()
    return wealth / wealth.cummax() - 1.0


def _drawdown_losses(returns: pd.Series) -> np.ndarray:
    return -_drawdown_series(returns).to_numpy(dtype=float)


def _max_drawdown_loss(returns: pd.Series) -> float:
    drawdowns = _drawdown_losses(returns)
    return float(np.nanmax(drawdowns)) if drawdowns.size else 0.0


def _metric_table(
    portfolio_returns: pd.DataFrame,
    periods_per_year: int,
    risk_free_rate: float,
    var_confidence: float,
) -> pd.DataFrame:
    rows = {}
    rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    for label, series in portfolio_returns.items():
        clean = series.dropna()
        wealth = (1.0 + clean).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        total_return = wealth.iloc[-1] - 1.0
        annual_return = wealth.iloc[-1] ** (periods_per_year / len(clean)) - 1.0
        annual_vol = clean.std(ddof=0) * np.sqrt(periods_per_year)
        downside = clean[clean < rf_per_period].std(ddof=0) * np.sqrt(periods_per_year)
        max_drawdown = abs(float(drawdown.min()))
        max_dd_duration = _max_drawdown_duration(drawdown)
        excess = clean - rf_per_period
        sharpe = excess.mean() * periods_per_year / annual_vol if annual_vol else np.nan
        sortino = excess.mean() * periods_per_year / downside if downside else np.nan
        calmar = annual_return / max_drawdown if max_drawdown else np.nan
        losses = -clean
        var_value = float(np.quantile(losses, var_confidence))
        pos = clean[clean > 0]
        neg = clean[clean < 0]
        omega = pos.sum() / abs(neg.sum()) if not neg.empty and neg.sum() != 0 else np.nan
        upper_tail = clean.quantile(0.95)
        lower_tail = abs(clean.quantile(0.05))
        tail_ratio = upper_tail / lower_tail if lower_tail else np.nan
        common_sense = omega * tail_ratio if np.isfinite(omega) and np.isfinite(tail_ratio) else np.nan
        rows[label] = {
            "Start": clean.index[0],
            "End": clean.index[-1],
            "Period": clean.index[-1] - clean.index[0] if hasattr(clean.index, "__sub__") else len(clean),
            "Total Return [%]": total_return * 100.0,
            "Annualized Return [%]": annual_return * 100.0,
            "Annualized Volatility [%]": annual_vol * 100.0,
            "Max Drawdown [%]": max_drawdown * 100.0,
            "Max Drawdown Duration": max_dd_duration,
            "Sharpe Ratio": sharpe,
            "Calmar Ratio": calmar,
            "Omega Ratio": omega,
            "Sortino Ratio": sortino,
            "Skew": clean.skew(),
            "Kurtosis": clean.kurtosis(),
            "Tail Ratio": tail_ratio,
            "Common Sense Ratio": common_sense,
            "Value at Risk": -var_value,
        }
    return pd.DataFrame(rows)


def _max_drawdown_duration(drawdown: pd.Series):
    underwater = drawdown < 0
    if not underwater.any():
        return pd.Timedelta(0)
    groups = underwater.ne(underwater.shift()).cumsum()
    durations = underwater.groupby(groups).sum()
    max_len = int(durations.max())
    if isinstance(drawdown.index, pd.DatetimeIndex):
        end_positions = []
        for _, group in underwater.groupby(groups):
            if group.iloc[0]:
                end_positions.append(group.index[-1] - group.index[0])
        return max(end_positions) if end_positions else pd.Timedelta(0)
    return max_len


def _method_weights(
    returns: pd.DataFrame,
    label: str,
    confidence: float,
    risk_free_rate: float,
    periods_per_year: int,
) -> pd.Series:
    display_label = label
    label = label.upper()
    expected_returns = returns.mean() * periods_per_year
    covariance = returns.cov() * periods_per_year

    if label == "1N":
        return _fallback_equal_weight(returns.columns).rename(display_label)
    if label == "MV":
        return optimize_min_variance(covariance, expected_returns=expected_returns).rename(display_label)
    if label == "WR":
        try:
            return optimize_max_sharpe(expected_returns, covariance, risk_free_rate=risk_free_rate).rename(display_label)
        except ValueError:
            scores = expected_returns.clip(lower=0.0)
            if np.isclose(scores.sum(), 0.0):
                return _fallback_equal_weight(returns.columns).rename(display_label)
            return (scores / scores.sum()).rename(display_label)

    if label == "CVAR":
        return _solve_long_only(
            returns,
            lambda w: _cvar_loss(_portfolio_returns(returns, w).to_numpy(), confidence),
            display_label,
        )
    if label == "EVAR":
        return _solve_long_only(
            returns,
            lambda w: _evar_loss(_portfolio_returns(returns, w).to_numpy(), confidence),
            display_label,
        )
    if label == "RLVAR":
        return _solve_long_only(
            returns,
            lambda w: (
                _cvar_loss(_portfolio_returns(returns, w).to_numpy(), confidence)
                + 0.25 * _portfolio_returns(returns, w).std(ddof=0)
                - 0.10 * _portfolio_returns(returns, w).mean()
            ),
            display_label,
        )
    if label == "CDAR":
        return _solve_long_only(
            returns,
            lambda w: _cvar_loss(_drawdown_losses(_portfolio_returns(returns, w)), confidence),
            display_label,
        )
    if label == "EDAR":
        return _solve_long_only(
            returns,
            lambda w: _evar_loss(_drawdown_losses(_portfolio_returns(returns, w)), confidence),
            display_label,
        )
    if label == "RLDAR":
        return _solve_long_only(
            returns,
            lambda w: (
                _cvar_loss(_drawdown_losses(_portfolio_returns(returns, w)), confidence)
                + 0.25 * _max_drawdown_loss(_portfolio_returns(returns, w))
            ),
            display_label,
        )
    if label == "MDD":
        return _solve_long_only(
            returns,
            lambda w: _max_drawdown_loss(_portfolio_returns(returns, w)),
            display_label,
        )
    raise ValueError(f"unsupported sophistication method: {label}")


def build_portfolio_sophistication_report(
    returns,
    methods: Sequence[str] = DEFAULT_METHODS,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    initial_value: float = 100.0,
) -> RiskReport:
    """
    Build a risk-measure comparison report for portfolio construction methods.

    The default labels follow the portfolio comparison chart popularised in the
    "Mas sofisticacion = mejor portfolio?" discussion: minimum variance, tail
    risk, drawdown risk, return-to-risk, and equal-weight baselines.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if initial_value <= 0:
        raise ValueError("initial_value must be positive")

    data = _clean_returns(returns)
    ordered_methods = list(dict.fromkeys(methods))
    weights = pd.concat(
        [
            _method_weights(data, method, confidence, risk_free_rate, periods_per_year)
            for method in ordered_methods
        ],
        axis=1,
    )
    portfolio_returns = pd.DataFrame(
        {method: _portfolio_returns(data, weights[method]) for method in ordered_methods},
        index=data.index,
    )
    wealth = (1.0 + portfolio_returns).cumprod() * initial_value
    performance_table = _metric_table(
        portfolio_returns,
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
        var_confidence=confidence,
    )
    descriptions = {
        method: _METHOD_DESCRIPTIONS_BY_KEY.get(method.upper(), SophisticationMethod(method, method)).description
        for method in ordered_methods
    }
    return RiskReport(
        metrics={
            "returns": portfolio_returns,
            "wealth": wealth,
            "weights": weights,
            "performance_table": performance_table,
            "method_descriptions": descriptions,
            "confidence": confidence,
            "initial_value": initial_value,
        }
    )


def plot_portfolio_sophistication_report(
    report: RiskReport,
    title: str = "Portfolio Sophistication Comparison",
    figsize: tuple[float, float] = (15.0, 9.0),
    table_fontsize: int = 8,
):
    """
    Plot cumulative wealth and a performance table for a sophistication report.
    """
    import matplotlib.pyplot as plt

    wealth = report.metrics["wealth"]
    table = report.metrics["performance_table"]
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.55])
    ax_chart = fig.add_subplot(grid[0, 0])
    ax_table = fig.add_subplot(grid[1, 0])

    wealth.plot(ax=ax_chart, linewidth=1.4)
    ax_chart.set_title(title)
    ax_chart.set_ylabel("Portfolio value")
    ax_chart.set_xlabel("")
    ax_chart.grid(True, alpha=0.25)
    ax_chart.legend(loc="upper left", ncol=min(5, len(wealth.columns)), fontsize=8)

    ax_table.axis("off")
    formatted = _format_performance_table(table)
    mpl_table = ax_table.table(
        cellText=formatted.values,
        rowLabels=formatted.index,
        colLabels=formatted.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(table_fontsize)
    mpl_table.scale(1.0, 1.25)
    for (row, _), cell in mpl_table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e6e6e6")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f3f3f3")
    return fig


def _format_performance_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for row in formatted.index:
        if row in {"Start", "End"}:
            formatted.loc[row] = formatted.loc[row].map(_format_index_value)
        elif row in {"Period", "Max Drawdown Duration"}:
            formatted.loc[row] = formatted.loc[row].map(str)
        else:
            formatted.loc[row] = formatted.loc[row].map(lambda x: f"{float(x):.6f}" if pd.notna(x) else "")
    return formatted


def _format_index_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, np.datetime64):
        return str(pd.Timestamp(value).date())
    if isinstance(value, datetime):
        return str(value.date())
    if isinstance(value, date):
        return str(value)
    return str(value)


def available_sophistication_methods() -> pd.DataFrame:
    """Return the built-in method labels and their descriptions."""
    return pd.DataFrame(
        [{"method": item.label, "description": item.description} for item in METHOD_DESCRIPTIONS.values()]
    )
