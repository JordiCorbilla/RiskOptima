# Risk Attribution And Scenarios

RiskOptima includes portfolio risk decomposition and deterministic stress testing utilities.

Attribution APIs:

- `marginal_volatility_contribution`
- `component_volatility_contribution`
- `marginal_var`
- `component_var`
- `component_expected_shortfall`
- `component_cvar`
- `factor_risk_contribution`
- `tracking_error_contribution`

Scenario APIs:

- `StressScenario`
- `StressResult`
- `apply_price_shock`
- `apply_factor_shock`
- `run_stress_scenario`
- `run_scenario_set`
- `default_scenarios`

```python
import pandas as pd
from riskoptima.risk import StressScenario, component_volatility_contribution, run_stress_scenario

weights = pd.Series({"A": 0.6, "B": 0.4})
cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.03]], index=weights.index, columns=weights.index)
print(component_volatility_contribution(weights, cov))

holdings = pd.Series({"A": 10.0, "B": 5.0})
prices = pd.Series({"A": 100.0, "B": 50.0})
scenario = StressScenario("equity_down", price_shocks={"A": -0.10, "B": -0.05})
print(run_stress_scenario(holdings, prices, scenario))
```

