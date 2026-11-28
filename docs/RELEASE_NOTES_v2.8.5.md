# RiskOptima v2.8.5 release notes

## VIX divergence

- Pass `execution="next_close"` to `RiskOptima.run_index_vol_divergence_signals()` to use daily-close fill timing. An entry fills after the next observed close confirms the pivot. An exit fills after the close that triggers it. The tables expose those dates separately.
- Empty entry, exit and return tables keep their columns. The exit scan skips the entry day when same-day exits are disabled and continues looking for later triggers.
- The legacy retrospective mode remains the default for compatibility and emits a `FutureWarning` when used to calculate trade returns. Notebook 03 uses the next-close mode. Historical vendor publication and revision timestamps remain unverified.

## Portfolio Monte Carlo

- Align historical prices, daily sample covariance and weights by asset label. Accept singular positive semidefinite covariance, including identical histories. Estimate daily stock returns from complete observations without filling missing prices.
- Keep the legacy candidate weight distribution and annualization. Terminal simulations use correlated Gaussian simple returns and treat zero wealth as absorbing. Candidate scores use arithmetic annualization, while terminal statistics use compounded payoffs.
- Pass `random_state` for a reproducible run with an isolated NumPy generator. Positive-payoff seeded fixtures match the previous sampling convention when the provider axes were correctly aligned. Notebook 02 uses an explicit seed for the portfolio probability report.

[Numerical validation ledger](NUMERICAL_VALIDATION_LEDGER.md) · [VIX timing ADR](adr/0004-vix-execution-timing.md) · [Portfolio Monte Carlo ADR](adr/0005-legacy-portfolio-monte-carlo.md)
