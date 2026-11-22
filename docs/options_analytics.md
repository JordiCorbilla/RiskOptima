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

The modular Black-Scholes functions take scalar inputs, annual decimal
volatility, and continuously compounded decimal rates. Dated option
contracts use actual days divided by 365.

| Interface | Return | Theta | Vega | Rho |
| --- | --- | --- | --- | --- |
| Modular Greeks and option books | Named fields | Price change per year of calendar-time passage | Per unit decimal volatility | Per unit continuous decimal rate |
| Legacy facade and Greeks simulator | (delta, gamma, theta, vega) | Price change per calendar day | Per one percentage-point rise in volatility | Omitted |

For example, convert modular theta to the legacy scale by dividing by 365,
and convert modular vega by dividing by 100. Existing legacy broadcastable
NumPy inputs still work. Pass an explicit lowercase call or put to the
facade. Its previous fallback to a put for unrecognized option types is
deprecated. Greeks require a positive term and volatility. Prices support
expiry and the zero-volatility limit.

Contract market value equals price times quantity times multiplier. Notional
delta equals delta times spot times quantity times multiplier.

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

