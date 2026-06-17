###############################################################################
#                    streamlit_market_risk_dashboard.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Optional Streamlit market risk dashboard
###############################################################################

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from riskoptima.reporting import (
    build_market_risk_report,
    plot_correlation_heatmap,
    plot_drawdown_curve,
    plot_rolling_volatility,
    plot_var_cvar_distribution,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic_market_returns.csv"


def _load_streamlit():
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is optional. Install it with `pip install streamlit` and run "
            "`streamlit run examples/streamlit_market_risk_dashboard.py`."
        ) from exc
    return st


def main():
    st = _load_streamlit()
    st.set_page_config(page_title="RiskOptima Market Risk Dashboard", layout="wide")
    st.title("RiskOptima Market Risk Dashboard")

    uploaded = st.sidebar.file_uploader("Upload returns CSV", type=["csv"])
    if uploaded is not None:
        returns = pd.read_csv(uploaded, index_col=0, parse_dates=True)
    else:
        returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

    st.sidebar.caption("Weights are normalized automatically.")
    weights = []
    for col in returns.columns:
        weights.append(st.sidebar.number_input(col, min_value=0.0, value=1.0 / len(returns.columns), step=0.05))

    report = build_market_risk_report(returns, weights=weights, benchmark_returns=returns.iloc[:, 0])
    metrics = report.metrics

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    kpi_cols[1].metric("Annualized Volatility", f"{metrics['annualized_volatility']:.2%}")
    kpi_cols[2].metric("Sharpe", f"{metrics['sharpe']:.2f}")
    kpi_cols[3].metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")

    risk_cols = st.columns(4)
    risk_cols[0].metric("VaR 95%", f"{metrics['historical_var'][0.95]:.2%}")
    risk_cols[1].metric("CVaR 95%", f"{metrics['cvar'][0.95]:.2%}")
    risk_cols[2].metric("Beta", f"{metrics['beta']:.2f}")
    risk_cols[3].metric("Tracking Error", f"{metrics['tracking_error']:.2%}")

    portfolio_returns = metrics["portfolio_returns"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_drawdown_curve(portfolio_returns, ax=axes[0, 0])
    plot_rolling_volatility(portfolio_returns, ax=axes[0, 1])
    plot_var_cvar_distribution(portfolio_returns, confidence=0.95, ax=axes[1, 0])
    plot_correlation_heatmap(returns, ax=axes[1, 1])
    fig.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()

