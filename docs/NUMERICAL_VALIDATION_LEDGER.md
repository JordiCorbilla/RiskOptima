# RiskOptima 3.0 numerical validation ledger

The [Phase 1 validation ledger](audit/VALIDATION_LEDGER.md) records the
regressions completed before v2.8.2. This ledger records subsequent work.

## 22 November 2026 — Option, rate and frequency compatibility

Independent references used QuantLib 1.44 and NumPy. The regression fixtures
are in [test_phase2_compatibility.py](../tests/test_phase2_compatibility.py).
Values below are rounded for display. The comparisons use full precision.

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Platform library European call | QuantLib BlackCalculator | S 100, K 102, 45/365 years, r 4%, sigma 22% | 2.412230244278232 | 2.412230244278227 | 1e-12 absolute | Pass |
| Platform modular and legacy Greeks | QuantLib BlackCalculator plus unit conversion | ATM at 100, 30/365 years, r 4%, sigma 20%, no dividends | Annual theta -15.90306108943074, unit vega 11.39509542199982. Daily theta -0.043570030382, percentage-point vega 0.113950954220 | Annual theta -15.90306108943081, unit vega 11.39509542199982. Legacy -0.043570030382 and 0.113950954220 | 1e-12 absolute on all fields | Pass |
| Dividend-paying option with negative interest rate | QuantLib BlackCalculator | S 102.5, K 108, T 1.3, r -1.5%, q 2.3%, sigma 28%, both sides | Put 19.276544616764824 and rho -97.88245425411466 | Put 19.276544616764838 and rho -97.88245425411465. Both sides and all Greeks matched the independent reference | 1e-12 absolute | Pass |
| Legacy array call price | Exact boundary formulas plus QuantLib | S columns 100 and 110 against three strikes and a mix of zero term, zero volatility and regular inputs | At-the-money expiry 0. Regular prices 6.97631264 and 12.47266801 | Same full 2 x 3 matrix. v2.8.2 had returned NaN for scalar at-the-money expiry | 1e-12 absolute | Pass |
| U.S. 30/360 and Bond Basis | QuantLib Thirty360.USA and QuantLib Thirty360.BondBasis | 28 February to 31 March 2023, plus leap-day and end-of-month cases | 30/360 for U.S. and 33/360 for Bond Basis | 0.08333333333333333 and 0.09166666666666666. The v2.8.2 U.S. alias had returned 33/360 | Exact | Pass |
| Actual accrual across DST | QuantLib Actual360 and Actual365Fixed | 29 to 30 March 2026 at midnight in Europe/London | 1/360 and 1/365 | 0.002777777777777778 and 0.00273972602739726. v2.8.2 treated the elapsed 23 hours as zero days | Exact | Pass |
| Nominal and continuous discounting | QuantLib InterestRate | 6% nominal semiannual or 6% continuous, 2.5 years | 0.86260878438416 and 0.86070797642506 | 0.86260878438416 and 0.86070797642506 | 1e-12 absolute | Pass |
| Dated option on a leap calendar | QuantLib Actual365Fixed and BlackCalculator | 28 February to 30 March 2024, short 3 calls, multiplier 100 | 31/365 years and -300 times independent call value | Exact maturity, matching sign and contract scale | 1e-10 absolute price units | Pass |
| Legacy and modular annualization | NumPy variance and independent compounding | Returns [0.005, -0.009, 0.013, 0.003, -0.008], effective annual risk-free rate 2% | 252-period population volatility, 260-period population and sample volatilities, legacy 260-period Sharpe | 0.131825338991, 0.133901456303, 0.149706379290, and 1.310814356962 | pytest default numeric comparison | Pass |

The scalar Platform pricing workload had a local median of 66.10 microseconds
per call for the installed v2.8.2 wheel and 37.18 microseconds for the
consolidated kernel. Measured on Windows Python 3.12.10 with one OpenBLAS
thread, five batches of 5,000 calls. This timing has no CI threshold.

Primary references

- [QuantLib BlackCalculator](https://github.com/lballabio/QuantLib/blob/v1.44/ql/pricingengines/blackcalculator.hpp)
- [QuantLib Thirty360 implementation](https://github.com/lballabio/QuantLib/blob/v1.44/ql/time/daycounters/thirty360.cpp)
