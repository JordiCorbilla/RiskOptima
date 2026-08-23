###############################################################################
#                            test_chart_branding.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Chart signature tests
###############################################################################

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskoptima import RiskOptima
from riskoptima.branding import add_riskoptima_signature, riskoptima_signature
from riskoptima.reporting import (
    build_markov_regime_report,
    build_portfolio_sophistication_report,
    plot_correlation_heatmap,
    plot_drawdown_curve,
    plot_markov_regime_chart,
    plot_markov_regime_probabilities,
    plot_portfolio_sophistication_report,
    plot_rolling_volatility,
    plot_var_cvar_distribution,
)
from riskoptima.rates import DiscountCurve, plot_yield_curve


ROOT = Path(__file__).resolve().parents[1]


def test_riskoptima_signature_uses_package_version():
    assert riskoptima_signature() == f"Created by RiskOptima v{RiskOptima.VERSION}"


def test_add_riskoptima_signature_to_axes():
    fig, ax = plt.subplots()
    add_riskoptima_signature(ax)
    add_riskoptima_signature(ax)

    assert sum(riskoptima_signature() in text.get_text() for text in ax.texts) == 1
    plt.close(fig)


def test_add_riskoptima_signature_to_figure():
    fig = plt.figure()
    add_riskoptima_signature(fig, y=0.01)
    add_riskoptima_signature(fig, y=0.01)

    assert sum(riskoptima_signature() in text.get_text() for text in fig.texts) == 1
    plt.close(fig)


def test_reporting_charts_include_signature():
    returns = pd.Series([0.01, -0.02, 0.015, -0.004, 0.006, 0.002, -0.003, 0.004, 0.001, -0.002])

    ax = plot_drawdown_curve(returns)
    assert any(riskoptima_signature() in text.get_text() for text in ax.texts)
    plt.close(ax.figure)

    report = build_markov_regime_report(returns, n_regimes=2, n_iter=10, random_state=42)
    ax = plot_markov_regime_chart(report)
    assert any(riskoptima_signature() in text.get_text() for text in ax.texts)
    plt.close(ax.figure)


def test_all_modular_chart_helpers_include_one_signature():
    rng = np.random.default_rng(12)
    index = pd.bdate_range("2024-01-01", periods=120)
    returns = pd.DataFrame(
        rng.normal(0.0003, 0.01, size=(len(index), 3)),
        index=index,
        columns=["A", "B", "C"],
    )
    regime_report = build_markov_regime_report(returns.mean(axis=1), n_regimes=2, random_state=3)
    figures_and_axes = [plt.subplots() for _ in range(4)]
    chart_axes = [
        plot_rolling_volatility(returns.mean(axis=1), ax=figures_and_axes[0][1]),
        plot_var_cvar_distribution(returns.mean(axis=1), ax=figures_and_axes[1][1]),
        plot_correlation_heatmap(returns, ax=figures_and_axes[2][1]),
        plot_markov_regime_probabilities(regime_report, ax=figures_and_axes[3][1]),
        plot_yield_curve(DiscountCurve.from_zero_rates([1.0, 2.0], [0.04, 0.035])),
    ]

    for ax in chart_axes:
        assert sum(riskoptima_signature() in text.get_text() for text in ax.texts) == 1
        plt.close(ax.figure)

    sophistication = build_portfolio_sophistication_report(
        returns,
        methods=("MV", "1N"),
    )
    fig = plot_portfolio_sophistication_report(sophistication)
    assert sum(riskoptima_signature() in text.get_text() for text in fig.texts) == 1
    plt.close(fig)


def test_reporting_helper_can_suppress_signature_in_composed_dashboard():
    returns = pd.Series([0.01, -0.02, 0.015, -0.004, 0.006])
    fig, ax = plt.subplots()

    plot_drawdown_curve(returns, ax=ax, add_signature=False)
    add_riskoptima_signature(fig, y=0.01)

    assert not any(riskoptima_signature() in text.get_text() for text in ax.texts)
    assert sum(riskoptima_signature() in text.get_text() for text in fig.texts) == 1
    plt.close(fig)


def test_readme_chart_assets_are_valid_nonblank_images():
    filenames = {
        "algorithmic_backtesting.png",
        "portfolio_optimization.png",
        "market_risk_dashboard.png",
        "option_pricing_engine.png",
        "credit_risk_model.png",
    }

    for filename in filenames:
        chart = plt.imread(ROOT / "docs" / "assets" / filename)
        assert chart.shape[0] >= 600
        assert chart.shape[1] >= 1_000
        assert np.std(chart[..., :3]) > 0.02


def test_efficient_frontier_helper_can_suppress_embedded_signature():
    expected_returns = pd.Series([0.08, 0.05], index=["A", "B"])
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.02]], index=expected_returns.index, columns=expected_returns.index)

    fig, ax = plt.subplots()
    RiskOptima.plot_ef_ax(10, expected_returns, cov, ax=ax, add_signature=False)
    assert not any(riskoptima_signature() in text.get_text() for text in ax.texts)
    plt.close(fig)

    fig, ax = plt.subplots()
    RiskOptima.plot_ef_ax(10, expected_returns, cov, ax=ax)
    assert sum(riskoptima_signature() in text.get_text() for text in ax.texts) == 1
    plt.close(fig)
