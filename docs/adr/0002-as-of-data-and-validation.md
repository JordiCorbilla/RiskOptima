# ADR 0002 — Event-time and financial model validation

**Status** Accepted for Phase 1

## Context

Current and future observations were accessible to strategy callbacks, and random ML validation learned from observations later than its test sample. A fitted HMM also exposed retrospective probabilities without an explicit label.

## Decision

- Pass isolated price history strictly earlier than the execution timestamp to each strategy. Allow prior warm-up observations. Execute only against current observations and cost trades using prior ADV.
- Forecast next-observation returns using an expanding validation window, a one-observation gap and preprocessing fitted inside each fold. Report out-of-sample R² and MSE and refit on known labels for the next forecast.
- Recompute the HMM E-step after the final parameter update. Publish explicit filtered and smoothed series with in-sample parameters documented.

## Consequences

Strategies that read the execution row must switch to earlier signals. The engine still trusts caller-supplied timestamps and quote provenance. For a historical filtered HMM signal, estimate parameters with an earlier training cutoff.
