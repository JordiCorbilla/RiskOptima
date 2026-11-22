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

## Next outstanding item

Reconcile SMA execution timing in Phase 2.3. Preserve the legacy
same-close trade log for established callers and provide a causal alternative
for signals observed at the close. Independently compare its entries and
exits to the event engine through a deterministic price path.
