# Interest-Rate Curves

`riskoptima.rates` provides the term-structure layer used to turn market quotes into valuation inputs. It is suitable for notebooks and for JSON-backed RiskOptima platform services.

## Conventions

- Tenors are year fractions from the valuation date.
- Input rates are decimals, so `0.0425` means 4.25%.
- The default `log_linear_discount` interpolation is linear in log discount factors. This produces piecewise-constant instantaneous forward rates and is a conservative valuation default.
- `linear_zero` and `cubic_zero` are available when a smooth zero-rate curve is preferred. A visually smooth curve is not automatically an arbitrage-free curve.
- Deposits use simple compounding: `DF(T) = 1 / (1 + rT)`.
- Swaps use a single-curve OIS-style identity: the fixed leg equals `1 - DF(T)`.

## Bootstrap And Repricing

```python
import pandas as pd
from riskoptima.rates import bootstrap_discount_curve, price_fixed_rate_bond, reprice_curve_instruments

quotes = pd.DataFrame({
    "instrument_type": ["deposit", "swap", "swap"],
    "maturity": [0.5, 2.0, 5.0],
    "rate": [0.0500, 0.0460, 0.0425],
    "payment_frequency": [None, 2, 2],
})

curve = bootstrap_discount_curve(quotes, valuation_date="2026-07-22")
calibration = reprice_curve_instruments(curve, quotes)

print(curve.discount_factor(3.0))
print(curve.zero_rate(3.0))
print(curve.forward_rate(1.0, 2.0))
print(curve.par_swap_rate(5.0))
print(price_fixed_rate_bond(curve, 5.0, 0.045))
print(calibration[["maturity", "error_bps"]])
```

`curve.to_dict()` returns a JSON-compatible payload containing the curve name, valuation date, interpolation convention, tenors, and discount factors. `DiscountCurve.from_dict(...)` restores the same curve in a service or dashboard process.

## Dated SOFR Calibration

`bootstrap_sofr_curve` accepts mixed instrument rows with explicit dates:

- `deposit`: `maturity_date`, `rate`, and optional `effective_date`
- `fomc`: `start_date`, `end_date`, and `rate`
- `future`: `start_date`, `end_date`, and either `rate` or a price such as `95.95`; `convexity_adjustment_bps` is optional
- `swap`: `maturity_date`, `rate`, and optional `effective_date`, `payment_frequency`, and `fixed_day_count`

Curve time defaults to `ACT/365F`. Money-market, FOMC, futures, and OIS accruals default to `ACT/360`. `BusinessCalendar`, `spot_date`, and `generate_schedule` make weekend, holiday, spot-lag, and payment-date assumptions explicit.

```python
from riskoptima.rates import bootstrap_sofr_curve, reprice_sofr_instruments

curve = bootstrap_sofr_curve(quotes, valuation_date="2026-07-01")
calibration = reprice_sofr_instruments(curve, quotes)
assert calibration["error_bps"].abs().max() < 1e-6
```

## Multi-Curve Valuation

`bootstrap_projection_curve` calibrates a forward curve from deposits, FRAs, and swaps while discounting cash flows with an existing OIS curve. `MultiCurveSet` packages one discount curve and named projection curves in a JSON-compatible platform payload.

```python
from riskoptima.rates import MultiCurveSet, bootstrap_projection_curve

projection = bootstrap_projection_curve(forward_quotes, curve, name="SOFR 3M")
curves = MultiCurveSet(curve, {"SOFR-3M": projection})
par_rate = curves.par_swap_rate(5.0, "SOFR-3M")
payload = curves.to_dict()
```

## Calibration And Curve Risk

`curve_calibration_jacobian` reports zero-rate sensitivity to each market quote in bp-per-bp terms. It supports both tenor bootstraps and dated builders, including futures prices. `key_rate_dv01` and `key_rate_dv01_dates` report the PV loss from a one-basis-point bump at each curve knot.

```python
from riskoptima.rates import curve_calibration_jacobian, key_rate_dv01

jacobian = curve_calibration_jacobian(quotes)
bucketed_dv01 = key_rate_dv01(curve, cash_flows, payment_times)
```

Run `examples/example_institutional_sofr_curves.py` for an end-to-end example without an external data dependency.

## Model Scope

The dated OIS bootstrap uses single-curve par-swap identities and explicit simple-compounded forward intervals. Futures convexity adjustments are inputs, not estimated by an interest-rate model. The calendar supports supplied holidays but does not ship an exchange holiday database. Cross-currency curves, collateral optionality, and stochastic convexity calibration remain outside this deterministic curve layer.
