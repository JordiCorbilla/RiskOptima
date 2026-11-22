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

## Next outstanding item

Review the allocation bounds of the legacy Monte Carlo optimizer. Specify how
to offer final weight limits without breaking existing callers that use its
normalized sampling bounds. Validate the investment budget, volatility cap,
label order, seeding and the legacy portfolio workflow before changing its
default.
