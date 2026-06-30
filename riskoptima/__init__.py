#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
#                                 __init__.py                                  
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

#----------------------------------------------------------------------------
# Created By  : Jordi Corbilla
# Created Date: 2025
# version ='2.5.0'
# ---------------------------------------------------------------------------

from .riskoptima import RiskOptima
from .core import MarketData, Portfolio, BacktestConfig, RiskReport
from .risk import (
    FactorRiskModel,
    StressScenario,
    component_volatility_contribution,
    fit_markov_regime_model,
    run_stress_scenario,
)
from .optim import Constraints, OptimizationResult, optimize_max_sharpe, optimize_min_variance, SimpleCostModel
from .backtest import run_backtest, Strategy, SMACrossStrategy, PortfolioState
from .backtest import build_sma_signal_frame, run_sma_strategy_with_risk
from .credit import expected_loss, credit_var, merton_pd
from .reporting import (
    build_market_risk_report,
    build_markov_regime_report,
    build_portfolio_sophistication_report,
)
from .options import OptionBook, OptionContract, black_scholes_price, black_scholes_greeks, implied_volatility
from .volatility import (
    ewma_volatility,
    historical_volatility,
    realized_volatility,
    rolling_volatility,
)

__all__ = [
    "RiskOptima",
    "MarketData",
    "Portfolio",
    "BacktestConfig",
    "RiskReport",
    "FactorRiskModel",
    "fit_markov_regime_model",
    "component_volatility_contribution",
    "StressScenario",
    "run_stress_scenario",
    "Constraints",
    "OptimizationResult",
    "optimize_max_sharpe",
    "optimize_min_variance",
    "SimpleCostModel",
    "run_backtest",
    "Strategy",
    "SMACrossStrategy",
    "PortfolioState",
    "build_sma_signal_frame",
    "run_sma_strategy_with_risk",
    "expected_loss",
    "credit_var",
    "merton_pd",
    "build_market_risk_report",
    "build_markov_regime_report",
    "build_portfolio_sophistication_report",
    "black_scholes_price",
    "black_scholes_greeks",
    "implied_volatility",
    "OptionBook",
    "OptionContract",
    "historical_volatility",
    "rolling_volatility",
    "realized_volatility",
    "ewma_volatility",
]
