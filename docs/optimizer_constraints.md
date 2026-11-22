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



## Legacy Monte Carlo allocation

Set `enforce_allocation_bounds=True` on
`RiskOptima.perform_mean_variance_optimization` to apply
`min_allocation` and `max_allocation` to the final portfolio. The limits
must allow a fully invested, long-only portfolio. The optimizer evaluates
the bounded minimum-variance portfolio alongside feasible random candidates
and selects the highest sampled Sharpe ratio within the annualized
volatility cap. A fixed `random_state` makes the sampling reproducible.

The compatibility default applies the bounds to the draws before
normalization. It emits a FutureWarning when the result exceeds the
requested limits. For the machine-learning portfolio workflow, use the same
opt-in with `min_weight` and `max_weight`. Its `num_portfolios` argument
sets the random sampling budget for each of the two allocations.
