###############################################################################
#                                 __init__.py                                  
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: RiskOptima module
###############################################################################

from .factor_model import FactorRiskModel
from .markov_regime import (
    MarkovRegimeModel,
    classify_markov_regimes,
    fit_markov_regime_model,
    regime_summary,
)
from .attribution import (
    component_cvar,
    component_expected_shortfall,
    component_var,
    component_volatility_contribution,
    factor_risk_contribution,
    marginal_var,
    marginal_volatility_contribution,
    tracking_error_contribution,
)
from .scenarios import (
    StressResult,
    StressScenario,
    apply_factor_shock,
    apply_price_shock,
    default_scenarios,
    run_scenario_set,
    run_stress_scenario,
)

__all__ = [
    "FactorRiskModel",
    "MarkovRegimeModel",
    "classify_markov_regimes",
    "fit_markov_regime_model",
    "regime_summary",
    "marginal_volatility_contribution",
    "component_volatility_contribution",
    "marginal_var",
    "component_var",
    "component_expected_shortfall",
    "component_cvar",
    "factor_risk_contribution",
    "tracking_error_contribution",
    "StressScenario",
    "StressResult",
    "apply_price_shock",
    "apply_factor_shock",
    "run_stress_scenario",
    "run_scenario_set",
    "default_scenarios",
]
