# RiskOptima 3.0 development ledger

## Starting point

- The campaign resumed from the clean v2.8.2 merge at `5a6f3c0`. The original
  [phase 1 audit and validation results](audit/VALIDATION_LEDGER.md) are the
  historical baseline.
- The original checkout contained the roadmap and the Phase 1 validation
  ledger. It had no root AGENTS.md or cross-phase development or validation
  ledger in the available history.

## 22 November 2026 — Phase 2 compatibility and conventions

| Roadmap item | Status | Evidence and decision |
| --- | --- | --- |
| 2.1 Legacy and Platform compatibility fixtures | Complete for the audited option, discounting and annualization contracts | Added independent QuantLib and analytical fixtures. Checked the Platform library example and the legacy tuple consumed by its notebook service. Preserved the legacy 260-observation default and modular 252 default, along with the applicable sample or population standard deviation. |
| 2.2 Delegate equivalent legacy computations | Black-Scholes pricing and Greeks completed | The facade uses the same vectorized numerical kernels as the modular scalar API. It retains broadcasting and its Greek reporting units. The unwarned legacy put fallback is deprecated. Option expiry prices now use intrinsic value. |
| Related rate defects | Corrected and verified | Explicit U.S. 30/360 aliases now apply February month-end adjustments. Bare 30/360 retains Bond Basis and historical schedule defaults. Actual day counts use local calendar dates across daylight-saving changes. |
| Regression and downstream verification | Passed | 188 package tests on Windows Python 3.12.10, 68.1388% of lines and 56.6852% of branches. Six RiskOptima Platform backend tests at `4c30d162b4ca01919dde8c171464d21d3356fcc5` passed with a local source override. Both Ruff gates and mypy passed. |

See the [numerical validation ledger](NUMERICAL_VALIDATION_LEDGER.md) for the
inputs, tolerances and measured results.

## 22 November 2026 — Legacy allocation bounds

Added an opt-in final-weight mode to the existing Monte Carlo optimizer
and to the legacy ML portfolio workflow. It uses a bounded minimum-variance
candidate to cover tight volatility caps and allocates the remaining
sampling budget on the feasible fully invested set. The comparison keeps
the existing zero-risk score convention. The default keeps the existing
normalized sampling and global NumPy seed behavior. An out-of-bounds
result now prompts callers to select the explicit mode. Both arms of the
legacy ML workflow now use the supplied simulation budget.

Validation used independent two-asset analytical solutions, a seeded
legacy replay, generated budget and bounds cases, and a synthetic fixture
through both allocations in the legacy ML workflow. All 194 package tests
passed after removing a flaky wall-clock deadline from one existing
backtest property test. Its case count and numerical assertions are
unchanged. Both coverage gates, Ruff and mypy passed. Live notebook 02
(portfolio optimization) and notebook 06 (options and Greeks) executed
against the branch.

## 22 November 2026 — Package import boundary

The root package exposes the legacy facade on demand. Its class identity,
star imports, reflection and pickling are preserved. Standalone option,
curve and structural credit calculations run without importing the facade,
Yahoo Finance, plotting or ML dependencies. Curve plotting, factor fitting
and credit classifier fitting import their own dependencies when requested.
Legacy callers that explicitly import RiskOptima still load its existing
plotting and vendor adapters.

Fresh-process tests blocked adapter imports and ran representative numeric
calls. The classifier, chart, factor and public API tests passed, along with
the full 197-test suite and six Platform backend tests. Coverage was 69.4655%
of executable lines and 57.9376% of branches.

## 22 November 2026 — SMA execution and source tags

The legacy SMA trade-log APIs and facade now offer explicit next-close
execution. The same-close mode remains the compatibility default. Pending
orders fill at a subsequent positive, finite close. Stops and takes are
also observed at the close and become pending exits in next-close mode.
The Yahoo adapter tags the session-date index and extraction settings. The
single-asset signal frame and trade log preserve the tags.

Validated timestamps and closing equity against the event engine and
hand-calculated a stop through a missing quote and a take-profit gap.
Notebook 05 ran live against this branch. The full suite passed 202 tests.
Daily closes still leave intraday ordering and vendor publication or
revision times unverified.

## 22 November 2026 — Out-of-sample Markov filtering

Added a fixed-parameter forward filter to the fitted model. Dated continuations
reject data that overlaps the fit. The log-space filter preserves relative
likelihoods when both raw densities underflow. The continued and restarted
outputs matched independent probability-space recursions within 1e-12.

All 206 tests and six Platform backend tests passed. Line coverage was
69.9351% and branch coverage was 58.7811%. Ruff, mypy for Python 3.11 and
3.12, and local distribution builds passed.

## 23 November 2026 — HMM and Merton tail stability

HMM fitting now shares its log-space forward pass with the fixed-parameter
filter and calculates backward state and transition probabilities in log
space. The fit keeps previous parameters for effectively unvisited states
and rejects arithmetic outside the representable range. The independent
comparison enumerates hidden paths, including underflow and sparse
transition cases.

Merton pricing now keeps the discounted stock move separate from the
rate, scales the jump compensation before exponentiating, and averages
payoffs in log units. Seed handling and callable signatures are unchanged.
The ledger separates arithmetic stability from rare-event Monte Carlo
sampling error. Local Windows verification passed the 179 tests that ran
without QuantLib. The existing QuantLib regressions run in the Linux CI matrix.

## 27 November 2026 — Heston and SABR conventions

Corrected the SABR forward coefficient, signed normal forwards and path
endpoints. Heston now shares a log-spot, full-truncation variance step
between simulation and pricing. The call is exact when variance is
deterministic. The legacy Heston variance units and the Platform payload
shape are preserved. Notebook 04 plots annualized relative volatility.
See [ADR 0003](adr/0003-stochastic-volatility-conventions.md) and the
[numerical validation ledger](NUMERICAL_VALIDATION_LEDGER.md).

## 28 November 2026 — RiskOptima 2.8.4

Published the wheel and source archive built from `cf3d9e4`. The public
SHA-256 digests match the validated artifacts. Installing from either
artifact passed numerical smoke checks, and the source archive rebuilt
the same wheel. All 234 tests passed. Line coverage was 71.0487% and
branch coverage was 59.9617%. Six Platform backend tests and the
stochastic-volatility endpoint passed against the public installation.

Executed all 14 notebooks in fresh kernels against the published wheel
and refreshed their saved outputs and installation pins. The live
dislocation notebook completed 11 of 11 analyses. See the
[release notes](RELEASE_NOTES_v2.8.4.md) and
[PyPI release](https://pypi.org/project/riskoptima/2.8.4/).

## 28 November 2026 — VIX divergence execution

Release coverage identified the legacy VIX workflow as an untested
entry point with 16 uncovered branches in its entry method. Added
caller-level tests and fixed the empty-table failure and the exit scan
that previously stopped on an ineligible entry-day trigger. The
`next_close` mode records pivot, confirmation, entry, exit trigger and
exit fill dates. The default retrospective return calculation emits a
deprecation warning.

Notebook 03 documents both conventions. Both live examples ran with
`next_close` against the development checkout and passed the timestamp
checks. All 249 tests passed. The VIX calculation module covered all
68 measured lines and 26 measured branches. See
[ADR 0004](adr/0004-vix-execution-timing.md) and the
[numerical validation ledger](NUMERICAL_VALIDATION_LEDGER.md).

## Next outstanding item

Add offline caller-level regressions for `run_portfolio_analysis`, the
legacy Monte Carlo portfolio workflow used by notebook 02. Independently
check covariance label alignment and the sampling and payoff conventions.
Historical vendor publication and revision timing, Platform volatility
label migration and coarse-grid accuracy near the stochastic volatility
boundaries remain open.
