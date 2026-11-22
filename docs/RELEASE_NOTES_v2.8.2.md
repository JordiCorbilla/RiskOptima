# RiskOptima v2.8.2 release notes proposal

This is a draft for the Phase 1 correctness changes.

## Numerical changes

- Merton call pricing now includes the risk-neutral jump compensator and exact compound Poisson terminal sampling. The legacy `num_steps` argument is retained. Use the keyword `random_state` for reproducible independent draws.
- The event backtest exposes pre-execution prices to strategies. Missing quotes are marked with the most recent price and cannot trade until a fresh quote arrives. Execution costs use prior ADV.
- Financial ML now forecasts the next simple return and reports expanding-window out-of-sample scores with fold-local preprocessing. The portfolio workflow annualizes return views and ignores nonpositive validation R² when weighting them.
- Optimizers validate covariance labels, symmetry, PSD and frequency metadata. For a factor model fitted to daily returns, use `model.covariance_matrix(periods_per_year=252)` with annualized optimization inputs. Untagged matrices follow the existing annualized assumption.
- Pass `return_result=True` to `implied_volatility` for convergence, iterations and price residual. The float-only fallback on non-convergence is deprecated and emits a `FutureWarning`. IV surfaces expose the same diagnostics.
- Markov likelihoods and posteriors now use final fitted parameters. Use `filtered_regime_probabilities` or `smoothed_regime_probabilities` to state which estimate is needed. Existing `regime_probabilities` remains a smoothed compatibility alias.
- Historical and credit CVaR and expected-shortfall attribution now use exact empirical tail mass, including fractional mass at ties. Results can change for small samples and discrete losses. The threshold-conditional mean convention is retired from these APIs.

## Verification

Phase 1 adds independent QuantLib and analytical references, property-based regression tests, standalone coverage gates and Ruff. Refer to the [numerical audit](audit/NUMERICAL_AUDIT.md) for formulas and the [architecture audit](audit/ARCHITECTURE_AUDIT.md) for validated integrations. Proposed release versioning is a separate step.
