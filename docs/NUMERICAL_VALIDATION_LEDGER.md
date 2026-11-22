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


## 22 November 2026 — Legacy allocation bounds

The fixture builds prices from orthogonal shocks with diagonal annual
variances of 0.04 and 0.09 and the legacy 260-observation convention.
See [test_legacy_allocation_bounds.py](../tests/test_legacy_allocation_bounds.py).

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Opt-in maximum sampled Sharpe | Analytic monotone Sharpe in the second weight | Annual return 0 for A and 0.12 for B, shuffled Series labels, bounds [0.35, 0.65], cap 0.30, 5,000 draws, seed 73 | Optimum [0.35, 0.65] within the constraints | [0.350087393909, 0.649912606091], volatility 0.207164724157, exact seed replay | 0.001 weight error from sampled optimum, 1e-12 on budget | Pass |
| Tight volatility cap | Normalize inverse variances | Bounds [0.20, 0.80], cap 0.166410058867569 plus 1e-10, one random draw | [0.692307692308, 0.307692307692] and volatility 0.166410058867569 | [0.692307692308, 0.307692307692] with the same measured volatility | 1e-7 on weights, explicit cap | Pass |
| Legacy default | Independent normalization, quadratic variance and argmax of the seeded draws | Seed 9, 40 draws, raw bounds [0.01, 0.35], annual means [0.02, 0.13] | [0.256532754652, 0.743467245348] and warning about the breached bound | Same weights and expected warning | Exact seed replay | Pass |
| Budget and feasible bounds | Long-only investment budget and component inequalities | 25 generated cases with up to 30 assets, plus a two-asset cap of 0.50 per asset | Weights sum to one and satisfy interval, boundary [0.5, 0.5] | All inequalities and boundary satisfied | 1e-12 on sums, exact interval | Pass |
| Infeasible portfolio | Sum of upper bounds and analytical minimum variance | Two assets with default cap 0.35 each, or a volatility cap below the independent minimum | Reject infeasible bounds and volatility caps | Expected errors | Exact failure mode | Pass |


## 22 November 2026 — SMA execution timing

See [test_sma_execution_timing.py](../tests/test_sma_execution_timing.py).
The event and trade-log comparison uses a single symbol, daily evaluation
and zero transaction costs.

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Next-close crossover | Manually compute SMA(1) and SMA(2) crossings, then execute on the next close | [102, 100, 99, 103, 105, 100, 97, 98, 97] from 5 January 2026 | Entry 9 January at 105, exit 13 January at 97, return -0.076190476190, equity 923.809523809524 from 1,000 | Exact dates and prices, same return and event-engine equity | 1e-12 return, 1e-10 equity | Pass |
| Legacy same-close mode | Execute on the observed crossover rows | Same closes | Entry 8 January at 103, exit 12 January at 100, return -0.029126213592 | Exact dates and prices, same return | 1e-12 return | Pass |
| Stop through a missing close | Queue an exit at the 96 close after entry at 105. Wait through a missing quote. | Stop 5%, next available exit price 90 | Fill 90, return -0.142857142857, no closed trade if data ends without a quote | Same trigger reason, price and return | 1e-12 return | Pass |
| Take profit through a gap | Queue an exit at the 112 close after entry at 101 | Take 10%, next available exit price 126 | Fill 126, return 0.247524752475 | Same trigger reason, price and return | 1e-12 return | Pass |

## 22 November 2026 — Out-of-sample Markov filtering

The tests compare fixed-parameter inference to an independent forward
recursion in probability space. [Regression tests](../tests/test_markov_out_of_sample.py) · [Rabiner's forward procedure](https://www.ee.columbia.edu/~dpwe/e6820/papers/Rabiner89-hmm.pdf)

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Continued filter | Gaussian likelihood and propagated filtered training posterior | Two states, ten training rows, two EM updates, seed 4. Validation [-0.01, missing, 0.025, 0.03]. | First posterior [0.677186320072, 0.322813679928], last [0.002721461254, 0.997278538746] | Same posteriors, maximum absolute difference 1.67e-16 | 1e-12 | Pass |
| Restarted filter | Direct recurrence from fitted initial probabilities | Same data and fitted parameters | First posterior [0.999857294826, 0.000142705174], last [0.002799438009, 0.997200561991] | Same posteriors, maximum absolute difference 2.78e-17 | 1e-12 | Pass |
| Underflow | Gaussian likelihood ratio and two-state transition | Means [0, 0], variances [1, 4], transition rows [0.9, 0.1] and [0.2, 0.8], prior [0.6, 0.4], observations [100, 0] | [0, 1], then [1/3, 2/3] | Same posteriors when both raw densities at 100 round to zero | 1e-12 | Pass |
| Chronology | Strict date comparison and exact prefix comparison | Overlapping and out-of-order dates, and appended observations | Reject invalid dates in continuation, preserve already filtered rows and fitted state | Expected errors and equal rows | Exact | Pass |
