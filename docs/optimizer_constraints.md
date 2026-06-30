# Optimizer Constraints

The optimizer keeps backwards compatibility: `optimize_max_sharpe` and `optimize_min_variance` still return a `pd.Series` of weights by default.

Set `return_result=True` to receive an `OptimizationResult` with diagnostics:

```python
from riskoptima.optim import Constraints, optimize_max_sharpe

constraints = Constraints(
    weight_bounds=(0.0, 0.4),
    leverage_limit=1.0,
    turnover_limit=0.2,
    factor_bounds={"quality": (0.0, 0.6)},
)

result = optimize_max_sharpe(
    expected_returns,
    cov,
    constraints=constraints,
    previous_weights=previous_weights,
    factor_exposures=factor_exposures,
    return_result=True,
)
```

Supported constraints:

- Per-asset weight bounds
- Gross leverage limit
- Turnover limit when `previous_weights` is supplied
- Factor exposure bounds
- Sector bounds when metadata includes a `sector` column
- Asset-class bounds when metadata includes an `asset_class` column

Covariance helpers:

- `sample_covariance`
- `ewma_covariance`
- `ledoit_wolf_covariance`
- `nearest_psd`

