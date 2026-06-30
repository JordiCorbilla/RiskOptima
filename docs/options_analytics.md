# Options Analytics

RiskOptima's options layer now supports both single-option pricing and book-level analytics.

Core APIs:

- `OptionContract`
- `OptionBook`
- `value_option_contract`
- `price_option_chain`
- `build_greeks_grid`
- `build_implied_vol_surface`
- `option_scenario_grid`
- `event_straddle_analysis`

Greeks follow the existing options engine conventions:

- `theta` is annualized.
- `vega` is per 1.00 volatility point.
- Contract market value uses `price * quantity * multiplier`.
- Notional delta uses `delta * spot * quantity * multiplier`.

```python
from datetime import date
from riskoptima.options import OptionBook, OptionContract

book = OptionBook([
    OptionContract("SPY", strike=500, expiry=date(2027, 6, 30), option_type="call", quantity=2, implied_volatility=0.22),
    OptionContract("SPY", strike=480, expiry=date(2027, 6, 30), option_type="put", quantity=-1, implied_volatility=0.25),
])

detail = book.value({"SPY": 505.0}, valuation_date=date(2026, 6, 30), risk_free_rate=0.04)
summary = book.aggregate(detail)
```

