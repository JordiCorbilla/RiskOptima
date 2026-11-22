# RiskOptima 3.0 roadmap

## Phase 1 — numerical integrity

The patch candidate corrects Merton pricing, event-time access, financial ML validation, covariance and optimization contracts, implied-volatility diagnostics, Markov posteriors, and empirical expected shortfall. The regression suite uses analytical fixtures, QuantLib and Hypothesis. The ledger and coverage floors provide acceptance criteria for future changes. Phase 1 keeps existing entry points and the MIT licence.

| Evidence | Purpose |
| --- | --- |
| [Numerical audit](audit/NUMERICAL_AUDIT.md) | Financial contracts, fixes and remaining validation gaps |
| [Architecture audit](audit/ARCHITECTURE_AUDIT.md) | API inventory, ownership, notebook runs and Platform compatibility |
| [Coverage baseline](audit/COVERAGE_BASELINE.md) | Exact baseline counts, warnings, CI floors and performance |
| [Validation ledger](audit/VALIDATION_LEDGER.md) | Independently reproduced prices, probabilities, weights, and risk measures |
| [v2.8.2 release notes](RELEASE_NOTES_v2.8.2.md) | Upgrading and reading corrected outputs |

## Phase 2 — consolidate with compatibility tests

1. Add representative legacy and Platform caller fixtures. Audit scalar versus array option inputs, rate and Greek reporting units, and the 260 versus 252 trading-day convention.
2. Make legacy quantitative methods delegate to validated modular kernels where contracts agree. Use explicit deprecations and migration examples where callers rely on differences. Replace legacy normalized sampling bounds only after specifying the intended portfolio constraints and testing callers.
3. Separate package imports from vendor adapters and plotting. Introduce timestamp and data provenance at the input boundary. Test the legacy SMA and event backtest together for signal time, next executable quote, and intrabar behavior.
4. Provide a fixed-parameter HMM inference path for an out-of-sample regime evaluation. Stress-test extreme emission ranges and Merton tails. Validate Heston and SABR parameter conventions against independent reference engines.
5. Increase legacy branch coverage based on decision impact. Raise coverage floors only after new coverage is reproducible across supported Python versions.

## Phase 3 — release readiness

- Confirm the compatibility matrix against the supported Python versions and the 3.0 release candidate wheel installed in RiskOptima Platform. Compare Portfolio, options, regime, curve, and risk outputs at the interface boundaries.
- Verify pinned examples, offline data fixtures, notebook execution and published documentation. Record any intentional numerical differences in the ledger and migration guide.
- Review public deprecations and hold a separate release decision after the correctness and downstream gates pass.
