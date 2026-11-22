# RiskOptima v2.8.3 release notes

## Options and rates

- The legacy and modular Black-Scholes entry points share their numerical kernels. The legacy facade retains broadcastable inputs, daily theta and percentage-point vega. European call prices use intrinsic value at expiry.
- Use lowercase call or put with the legacy Greek interface. The fallback that treated an unrecognized side as a put emits a FutureWarning.
- Explicit U.S. 30/360 aliases apply February month-end rules. The default 30/360 alias continues to use Bond Basis. Actual day counts use calendar dates across daylight-saving transitions.

## Portfolio analysis and backtesting

- Pass `enforce_allocation_bounds=True` to apply allocation limits to final long-only weights. The legacy sampling mode remains the default and warns when the chosen weights exceed the requested interval.
- Pass `execution="next_close"` to fill SMA signals and close-based risk exits at the next available close. The legacy same-close behavior remains available. Missing quotes cannot trigger or fill these orders.
- The Yahoo Finance SMA adapter carries source and session-date labels into single-asset trade logs. Historical publication and revision timestamps remain provider-dependent.

## Markov regimes and imports

- Use `model.filter_probabilities(validation_returns)` to filter new observations under fitted parameters. Dated continuations enforce the training cutoff. The log-space forward filter handles tested emission underflow cases. Existing fitted and smoothed outputs retain their contracts.
- Standalone numerical imports defer the legacy facade and its vendor and chart dependencies until requested.

The [development ledger](DEVELOPMENT_LEDGER.md) records the completed checkpoints and open work. The [numerical validation ledger](NUMERICAL_VALIDATION_LEDGER.md) records independent references and tolerances.
