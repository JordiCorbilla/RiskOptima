# ADR 0005 — Legacy portfolio Monte Carlo conventions

**Status** Accepted for Phase 2

## Context

The legacy probability workflow used positionally ordered allocation
weights with provider-ordered price columns. Its terminal simulator
required positive definite covariance, and multiplying two negative daily
gross factors could restore positive fund value after a loss above 100%.

## Decision

- Order the price, return and covariance axes by the allocation mapping.
  Sort timestamps and reject duplicates. Compute daily simple returns
  from adjacent observed rows without filling missing quotes. Estimate
  stock covariance from complete rows and the benchmark from its own
  available observations.
- Use the labeled daily sample covariance with denominator `n - 1`.
  Preserve normalized uniform weight draws and arithmetic annualization.
  The candidate mean is the daily mean times the trading-day count and
  candidate volatility is the daily volatility times its square root.
  Preserve the legacy return table columns and selection rules. Ties in
  volatility use simulation order, and the first candidate is selected
  when every Sharpe score is undefined.
- Simulate daily correlated Gaussian simple returns with a Cholesky
  factor for positive definite covariance and an eigenfactor for singular
  positive semidefinite covariance. Keep the default NumPy random stream.
  An optional `random_state` provides an isolated generator. Changing
  allocation insertion order changes the assignment of seeded draws.
- Start each fund at 10,000 monetary units and define zero wealth as an
  absorbing state. A nonpositive daily gross factor exhausts the fund.
  Compute terminal return dispersion with the population standard
  deviation. Benchmark Sharpe is undefined when terminal dispersion is
  zero and is returned as `NaN`.

## Scope

The workflow estimates historical moments and applies a constant-weight,
independent daily shock model. Trading costs, funding liabilities and
historical vendor-gap frequencies lie outside the model. Terminal
distributions describe compounded wealth, while candidate scores use
arithmetic annualized moments.

[NumPy Cholesky requirements](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html) · [NumPy covariance sampling conventions](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html)
