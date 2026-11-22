# ADR 0001 — Numerical contracts at module boundaries

**Status** Accepted for Phase 1

## Context

The same portfolio and option concepts had different implicit units or numerical guarantees across legacy and modular APIs. Silent conversions made a successful call insufficient evidence of a correct result.

## Decision

- Annualized optimizers expect decimal annual return assumptions and covariance in decimal annual return squared. Tag matrices produced by covariance estimators and factor models. Preserve the annualized assumption for untagged inputs. Reject incompatible metadata and materially asymmetric or indefinite matrices. Keep PSD projection explicit.
- Use the empirical quantile integral for Expected Shortfall and allocate threshold ties pro rata. Preserve existing VaR interpolation and names.
- Keep existing scalar IV calls and add a diagnostic result. Warn when returning an unconverged float and provide convergence in option surfaces.
- Route the facade's Merton method through the exact terminal pricing kernel. Keep the legacy simulation-count and step arguments and validate them at the public boundary.

## Consequences

Corrected results change for models affected by the audited defects. Percent-vs-decimal and source frequency for externally supplied untagged covariance still depend on callers. Versioned examples document the conventions and the validation ledger records independent references.
