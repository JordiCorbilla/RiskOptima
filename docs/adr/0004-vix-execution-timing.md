# ADR 0004 — Daily-close execution for VIX divergence

**Status** Accepted for Phase 2

## Context

A local minimum needs two neighbouring closes. The legacy workflow used
the lower minimum's date and price as its entry while checking the next
close to confirm the pivot. The exit loop could stop on the entry date
even when same-day exits were disabled. A period with no signals could
also fail during plotting because its result table had no columns.

## Decision

- Keep the positional API and the default retrospective chart convention.
  `run_index_vol_divergence_signals()` emits a `FutureWarning` when
  calculating retrospective returns. Use `execution="next_close"` for
  point-in-time fills on the observed daily series.
- Compare the two pivot closes and their VIX values at the later pivot's
  confirmation. Calculate the rolling mean and sample standard deviation
  from base closes through the pivot date. A missing VIX quote prevents
  that comparison. It does not remove a valid base close from the fill
  calendar.
- In next-close mode, expose `PivotDate` and `ConfirmationDate` and
  retain `SignalDate` and `base_Close` as the fill date and price consumed
  by the legacy exit and return interfaces. Exits fill after
  `ExitSignalDate`. Skip unavailable closes and leave an order unfilled
  if the series ends without a later valid base close.
- With `intraday=False`, start the exit scan after the entry date.
  Preserve named entry, exit and return columns when no rows qualify.

## Scope

The tests check chronology using session dates and observed closes.
Historical vendor publication and revision times remain unverified.
The workflow can report overlapping hypothetical positions and sums
their per-trade returns. These figures do not represent a portfolio
equity curve or include execution costs.
