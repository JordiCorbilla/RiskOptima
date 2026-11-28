# RiskOptima v2.8.4 release notes

## Heston and SABR

- The Heston call pricer and path simulator share a log-spot, full-truncation Euler step. European calls use the analytical solution when variance is deterministic. The stochastic pricer remains a Monte Carlo estimate.
- The SABR simulator applies the diffusion `level * F**beta`, supports signed forwards with `beta=0`, and evolves its diffusion level with a lognormal step. With `0 < beta < 1`, the Euler forward step absorbs at zero.
- Both path simulators return `N` samples including the initial point and reaching `T` when `N > 1`. An optional `random_state` makes runs reproducible. The numerical corrections and Heston draw ordering change earlier seeded path and price values.
- Heston `sigma0` and `theta` continue to denote annualized variance. For 20% initial volatility pass `sigma0=0.2**2`. Take the square root of the returned variance for annualized volatility. Notebook 04 also shows how to convert SABR's diffusion level to relative volatility.

## Numerical stability

- HMM fitting computes forward, backward and transition probabilities in log space. It retains earlier parameters for effectively unvisited states and checks the representable range.
- Merton pricing computes jump compensation and discounted payoffs with more stable arithmetic. Monte Carlo sampling uncertainty remains material for rare-event payoffs.

The [numerical validation ledger](NUMERICAL_VALIDATION_LEDGER.md) records analytical and QuantLib references, tolerances and the remaining approximation limits. The [stochastic volatility decision record](adr/0003-stochastic-volatility-conventions.md) describes the API conventions.
