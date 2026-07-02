###############################################################################
#                            test_chart_branding.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Chart signature tests
###############################################################################

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from riskoptima import RiskOptima
from riskoptima.branding import add_riskoptima_signature, riskoptima_signature
from riskoptima.reporting import build_markov_regime_report, plot_drawdown_curve, plot_markov_regime_chart


def test_riskoptima_signature_uses_package_version():
    assert riskoptima_signature() == f"Created by RiskOptima v{RiskOptima.VERSION}"


def test_add_riskoptima_signature_to_axes():
    fig, ax = plt.subplots()
    add_riskoptima_signature(ax)

    assert any(riskoptima_signature() in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_add_riskoptima_signature_to_figure():
    fig = plt.figure()
    add_riskoptima_signature(fig, y=0.01)

    assert any(riskoptima_signature() in text.get_text() for text in fig.texts)
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
