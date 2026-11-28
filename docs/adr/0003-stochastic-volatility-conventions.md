# ADR 0003 — Heston and SABR simulation conventions

**Status** Accepted for Phase 2

## Context

The Heston path returned variance under a volatility label and used a
price-level Euler step that could yield negative spots. The SABR path used
`F**(1 + beta)` for its forward diffusion and clipped signed normal forwards
at zero. Both path routines stopped one interval short of the requested
horizon.

## Decision

- Keep the existing positional parameters, defaults and returned array
  shapes. Heston `sigma0` and `theta` keep their annualized variance units.
  For 20% initial volatility pass `sigma0=0.2**2` and plot the square root
  of the returned variance. This supersedes the previous volatility label.
- Heston pricing and simulation share a log-spot, full-truncation Euler
  step. The European call uses the analytical limit when the variance
  diffusion vanishes and retains Monte Carlo for stochastic variance.
- SABR `sigma0` is the initial diffusion level and `alpha` is the level's
  vol-of-vol. These correspond to Hagan's alpha and nu. For positive `F`,
  relative volatility is `level * F**(beta - 1)`. Use a lognormal step for
  the level and freeze it over each forward step. Fractional beta uses
  Euler with an absorbing zero. Normal SABR (`beta=0`) allows signed
  forwards. Lognormal SABR (`beta=1`) uses an exponential forward step.
- Place the `N` path samples at both endpoints when `N > 1`. Preserve
  global NumPy seeding and add an optional keyword-only `random_state`.
  Seeded calls are repeatable under the corrected algorithms. The new
  discretizations and Heston draw ordering change previous sample values.

## Consequences

The stochastic estimates have discretization error. The reference tests
cover representative Heston parameters on both sides of the Feller
condition and a fractional SABR call away from zero. They do not establish
coarse-grid accuracy near the boundary or uniform convergence across all
parameters. Existing Platform payload keys retain their values' units.
Consumers of `heston_vol` and `sabr_vol` must convert them for a relative
volatility display. Notebook 04 demonstrates the conversion.
