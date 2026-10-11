# Dislocation Research

Available in RiskOptima 2.8.0 under `riskoptima.dislocation`. This is a research
screen, not a validated investment recommendation or market-bottom predictor.

## Inputs and Execution

- `analyze_snapshot(MarketSnapshot(...), config=...)`: network-free analysis.
- `analyze_symbol(symbol, provider=..., config=...)`: fetch and analyze a ticker.
- `scan_universe([(symbol, name), ...], provider=..., config=...)`: ranked pandas
  summary and a list of successful `DislocationAnalysis` objects.
- `generate_markdown_report(analysis)`: readable, versioned quantitative report.

`MarketSnapshot.history` requires a daily `DatetimeIndex` and a `Close` or
`Adj Close` column. Single-instrument Yahoo MultiIndex columns are accepted.
Duplicate timestamps retain the last record; non-finite/non-positive prices
are removed. Use consistently adjusted prices and verify provider currency,
share class, reporting period and units. Annualization assumes 252 sessions.

The default `YahooFinanceProvider` fetches adjusted closes and ticker metadata.
An ISIN is not necessarily a Yahoo symbol. For example, Yahoo ISIN search for
IE00B50W2R13 resolved to `0P0000KM24.L`, the GBP-hedged accumulation share class.
Identifiers and upstream availability can change; errors are explicit rather
than silently substituting another instrument or synthetic data.

## Assumptions and Score Definitions

Equities combine quality (25%), valuation (30%), historical dislocation (25%),
regime (10%) and upside to model value (10%). Missing components are omitted
and available weights renormalized. Non-positive forward P/E, EV/EBITDA or
price/book receive zero component scores and diagnostics, never a cheapness
reward. Funds/non-equities combine dislocation (70%), regime (20%) and
12-month trend health (10%). No independent NAV is supplied or compared.

Regimes are fitted with the existing Gaussian HMM. Failures are documented
before using a momentum/volatility fallback. Historical HMM regimes are
descriptive, not an out-of-sample trading signal.

The equity DCF discounts projected positive `freeCashflow` plus a perpetuity
terminal value, divided by positive `sharesOutstanding`. Growth is clipped
to the configured bounds; missing growth defaults to 4% within those bounds.
Discount rate must be positive and exceed terminal growth. The calculation
uses an **unadjusted FCF proxy**, without explicit net borrowing or normalized
FCFE reconciliation. Do not describe it as a conservative or certified FCFE
valuation. Missing/non-positive cash flow or shares yields unavailable values.

For model value V and observed price P:

- `margin_of_safety = (V - P) / V`: discount relative to model value.
- `upside_to_model_value = V / P - 1`: hypothetical upside relative to price.
- `implied_annualized_return = (V / P) ** (1 / horizon_years) - 1`:
  hypothetical convergence to today's model value, without distributions.
  It is not a forecast, implied market yield or total-return IRR.
- `expected_annualized_return` retains the same convergence value for backwards
  compatibility. New consumers should use `implied_annualized_return`.
- `confidence_score`: input-coverage heuristic, not a probability of accuracy.
  Reports call this **Data coverage**. High completeness does not establish
  valuation quality, freshness or predictive skill.

The definitions of margin of safety and annualized convergence should be
displayed beside these metrics in downstream applications. In 2.8.0 the
margin-of-safety denominator is corrected from price to model value; the old
upside quantity remains separately accessible.

## Platform Contract

Use `analysis.as_dict()` rather than serializing the dataclass directly. It
contains ISO-compatible date strings, nullable model values, both legacy and
new return fields, diagnostics, metrics and `riskoptima_version`. Missing
model fields are `None` (JSON `null`). Serialize scanner error rows with
`summary.to_json(orient="records")` to convert pandas NaNs to JSON nulls.

Preserve `ERROR` rows and diagnostics. An all-error scan must be shown as
failure, not an empty successful dashboard. Keep provider access server-side;
accept ticker/config input, validate with `DislocationConfig`, and render
Markdown through the platform's safe Markdown component. Narrative callbacks
are optional application adapters; the package requires no LLM dependency.

Display the source, price date and model assumptions. Separate equity
valuation details from non-equity price/regime analysis; hide unavailable
DCF metrics instead of displaying zero. Label coverage honestly and retain
`RiskOptima vX` report provenance.

## Examples

`python examples/example_dislocation_model.py` runs entirely synthetic inputs.
`python examples/example_dislocation_model.py --live` requests Yahoo data.
Notebook `14-dislocation_model_demo.ipynb` runs eleven funds and an active MSFT
equity example. Set `RISKOPTIMA_DEMO_LIVE=0` to execute its synthetic mode. It
never replaces a failed live request with synthetic data automatically.
