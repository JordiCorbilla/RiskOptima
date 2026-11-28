# RiskOptima 3.0 numerical validation ledger

The [Phase 1 validation ledger](audit/VALIDATION_LEDGER.md) records the
regressions completed before v2.8.2. This ledger records subsequent work.

## 28 November 2026 — Legacy portfolio Monte Carlo

The scalar reference computes sample covariance from centered returns,
uses a two-asset analytical Cholesky factor and compounds each sampled
daily portfolio return. It preserves the legacy NumPy seed and draw
order. Displayed amounts are rounded. [Regression tests](../tests/test_portfolio_monte_carlo.py) · [Model conventions](adr/0005-legacy-portfolio-monte-carlo.md)

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Daily covariance and candidate ranking | Sample covariance and scalar quadratic form using the ordered asset returns | Allocation Z 75%, A 25%, four daily return rows, seed 41, four candidates, three periods, risk-free rate 1% | Daily variances 0.000466666667 and 0.000691666667, covariance -0.000366666667, optimum at candidate index 3 | All covariance and ranking values matched | 1e-12 absolute | Pass |
| Portfolio and benchmark payoffs | Scalar correlated draws and cumulative product | Same fixture, two simulations, provider columns reversed | Current portfolio terminal values 10233.910789687508 and 10567.244173811576. First benchmark value 10120.964090516570 | Same values. v2.8.4 gave 10488.718755093618 for the first current portfolio path with reversed provider columns. | 2e-10 absolute payoff, 1e-12 return statistics | Pass |
| Provider permutations | Same labeled daily histories and seeded draw streams | Thirty Hypothesis examples with varied seeds and allocations, provider columns and timestamps in reverse order | Identical labeled covariance, candidate rows and terminal values | Exact matches | Exact | Pass |
| Singular covariance | Identical asset returns with exact two-period Gaussian gross-return moments, where ruin probability is negligible | Duplicated Z history, 10,000 paths, two periods, seed 41 | Rank one, terminal return mean 0.020100 and standard deviation 0.030859538414 | Rank one, mean 0.019610744394 and standard deviation 0.030644583510. v2.8.4 raised a Cholesky error. | Five standard errors on mean, 5% relative on standard deviation | Pass |
| Wealth after ruin | Absorbing zero payoff | Historical returns [-0.5, 0.75, 0] and two -3 standard-normal shocks | Daily gross factor -0.804125275484 exhausts the fund, terminal value zero | Zero for all portfolios and benchmark. v2.8.4 returned positive terminal value 6466.174586727890. | Exact | Pass |
| Missing and deterministic histories | Complete-case adjacent returns and constant gross factors | Missing quote on the third row in one asset. Separate flat-price fixture with five paths. | Preserve only the first and fourth asset returns in the missing-quote case. All flat paths end at 10000 with zero dispersion. | Matched. Flat benchmark Sharpe is undefined and returned as NaN. | Exact dates and flat metrics, 1e-12 return values | Pass |

The Windows timing fixture used 10 assets, 100 historical returns, 2,000
candidate allocations, 100 paths and 60 simulation steps, with seed 41.
The median of three calls was 6.20 seconds before the change and 0.044
seconds after it. The optimum index remained 1768, and the first current
portfolio and benchmark terminal values matched within 1e-10.

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

## 23 November 2026 — HMM fitting and Merton tail stability

The references use enumerated hidden paths, Decimal arithmetic and
truncated lognormal moments. The high-count Merton case uses a Poisson
mixture with omitted second-moment tilted probability below 1e-50.
[Regression tests](../tests/test_tail_stability.py) · [Rabiner's forward-backward procedure](https://www.cs.ubc.ca/~murphyk/Software/HMM/rabiner.pdf) · [Merton's jump-diffusion pricing paper](https://www.cmat.edu.uy/~mordecki/hk2010/merton76.pdf)

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| HMM E-step underflow | Enumerated two-state paths and transition counts | Means [0, 0], variances [1, 4], observations [0, 100, 0, -110], initial [0.6, 0.4], transition rows [0.9, 0.1] and [0.2, 0.8] | At 100, filtered [0, 1]. Full log likelihood -2770.15498588794. Three expected transitions. | Matched. v2.8.3 gave [0.725, 0.275] and log likelihood -1383.77019328225. | 1e-10 on arrays and likelihood, 1e-12 row sums | Pass |
| HMM fit across separated states | Enumerate paths under final parameters and compare EM likelihoods | Eleven observations spanning ~0 to 108, two states, seed 18 | Correct filtered and smoothed rows and non-decreasing likelihood from one to 25 iterations | All checks passed, including stochastic transition rows | 1e-12 probabilities and 1e-10 likelihood | Pass |
| Merton rate cancellation | Discounted lognormal paired shocks | S = K = 100, T = 1, sigma = 20%, shocks [-1, 1], zero jumps | 99.98680805421445 at r = 1e16 and zero at r = -750 | Matched. v2.8.3 rounded the positive-rate result to 100. | 1e-12 absolute | Pass |
| Large jump compensation at tiny intensity | Decimal weighted jump drift conditional on the sampled zero-count paths | S = 1, K = 1e-250, T = 1, lambda = 1e-310, mean log jump = 720, eight paths, seed 3 | About 1.980039659e-214 for the sampled paths | About 1.980039659e-214. v2.8.3 returned zero. | 2e-10 relative | Pass |
| Merton payoff aggregation | Decimal formula for one large payoff out of ten paths | S = K = 1, T = 1, sigma = 1, shocks [710.5, -1, ...] | (exp(710) - 1) / 10, approximately 2.234e307 | Finite and within tolerance. Single-path price raises OverflowError. | 2e-13 relative | Pass |
| Merton deep OTM at high counts | Analytical Poisson mixture and second payoff moment | S = 100, K = 350, T = 2, r = -3.5%, sigma = 16%, lambda = 18, log jump mean = 0.04, SD = 0.14, 500,000 paths, seed 320 | 6.640950396609, standard error 0.093808223614 | 6.688766659574, about 0.51 standard errors above reference | Five standard errors (0.469041118071) | Pass |

The tiny-intensity test checks the estimator's arithmetic conditional on
what it sampled. A finite sample can miss the rare jumps that dominate an
option's expectation. A small or zero estimate in that case provides no
validation of the population price.

## 27 November 2026 — Heston and SABR conventions

The Heston Monte Carlo standard errors come from six independent batches
of 20,000 terminal payoffs, using seeds 105 through 110. The SABR
standard error comes from 6,000 path payoffs, using seed 774. The
statistical tolerances are five standard errors. These comparisons assess
agreement at the stated grids and parameters. They do not bound
discretization error for arbitrary settings.

[Regression tests](../tests/test_stochastic_volatility.py) · [Heston model](https://finance.martinsewell.com/stylized-facts/volatility/Heston1993.pdf) · [Hagan et al., equation 2.15](https://lesniewski.us/papers/published/ManagingSmileRisk.pdf) · [QuantLib Heston engine](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/analytichestonengine.cpp) · [QuantLib SABR engine](https://github.com/lballabio/QuantLib/blob/master/ql/pricingengines/vanilla/fdsabrvanillaengine.cpp)

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Heston deterministic variance call | Integrated CIR variance and analytical lognormal call | S = 100, K = 105, T = 1.4, r = -0.02, kappa = 1.5, theta = 0.04, eta = 0, v0 = 0.07 | 7.669693726565 | 7.669693726565 | 1e-12 absolute | Pass |
| Heston call satisfying Feller | QuantLib 1.44 `AnalyticHestonEngine`, order 144 | S = 100, K = 105, T = 1, r = -0.02, kappa = 1.5, theta = 0.04, eta = 0.3, rho = -0.65, v0 = 0.07, 512 steps | 5.775029923523 | 5.761112854398 | 0.178478824010 absolute | Pass |
| Heston call violating Feller | Same engine | Same inputs with eta = 0.65 | 4.540518563362 | 4.565201338013 | 0.120224402290 absolute | Pass |
| Heston path endpoint | Zero-variance risk-neutral spot, `S0 * exp(r*T)` | S0 = 100, r = 0.05, T = 1, N = 2, v0 = theta = 0 | 105.127109637602 | 105.127109637602. The previous endpoint was 102.5. | 1e-12 absolute | Pass |
| SABR fractional forward and level | Hagan diffusion coefficient and exact correlated lognormal level | F0 = 100, level = 0.2, T = 1, N = 2, beta = 0.5, nu = 0.3, rho = -0.4, seed 42 | End forward 100.60943416 and level 0.13849067 | Matched both values at full precision | 1e-12 absolute | Pass |
| SABR normal and lognormal limits | Exact Brownian and geometric Brownian paths | nu = 0, level = 0.4, beta = 0 and 1, T = 1.5, N = 8, seed 42 | Signed forward path for beta = 0, exact positive forward path for beta = 1 | Both paths matched | 1e-12 absolute | Pass |
| SABR fractional-beta call | QuantLib 1.44 `FdSabrVanillaEngine`, grid 100 × 400 × 50 | F0 = 100, K = 102, T = 183/365, level = 2, beta = 0.5, nu = 0.4, rho = -0.35, no discount, 64 simulation steps | 4.713201221780 | 4.720527853330 | 0.489517803275 absolute | Pass |
| SABR scale invariance | Same Brownian path after rescaling forward by c and level by c^(1 − beta) | Hypothesis over c in [0.01, 10], beta in [0, 1], nine points and 40 seeds or parameter combinations | Entire forward and level paths scale by their unit conversion factors | All generated cases matched | 1e-12 absolute and 1e-13 relative | Pass |

The Windows Heston timing fixture used 10,000 paths, 100 steps and the
first stochastic call's parameters. One run took 13.27 seconds before the
change and 0.39 seconds after vectorizing the path updates. The seeded
sample estimate changed from 5.708306 to 5.859125 because draw ordering and
the discretization changed.

## 28 November 2026 — VIX divergence execution

The fixture uses business-day closes with the consecutive lows at 96 and
95 and their VIX quotes at 21 and 30. The entry observation at 107 confirms
the second pivot. The first later base close is 112. An unavailable base
quote follows, then the 114 close exceeds the three-close mean of 111.
The order exits at the next 108 close. [Regression tests](../tests/test_vix_timing.py) · [Execution conventions](adr/0004-vix-execution-timing.md)

| Model or invariant | Reference | Input | Expected | Actual | Tolerance | Result |
| --- | --- | --- | --- | --- | --- | --- |
| Causal VIX entry | Strict three-close pivot and next-close execution | 9 January 2025 pivot at 95, 10 January confirmation at 107, VIX missing on 13 January | 13 January fill at 112, pivot VIX 30 | Same dates, prices and pivot VIX | Exact | Pass |
| Causal VIX exit | Mean of the last three observed closes and next-close execution | 14 January missing base close, 15 January close at 114 above mean 111 | Trigger 15 January, exit 16 January at 108, trade return -0.035714285714 | Same trigger, fill and return | Exact dates and prices, 1e-12 return | Pass |
| Lower-band exit | Skip a neutral close and wait through an unavailable quote | Eligible closes 96, 91, missing and 88 with lower band 95 | Trigger at 91, fill at 88 | Same trigger and fill | Exact | Pass |
| Legacy later exit | Start observing strictly after entry date | Entry at 100 above its mean, next close at 105 above its mean | Later exit at 105 when entry-day exits are disabled | Exit at 105. v2.8.4 returned no exit. | Exact | Pass |
| Empty and incomplete histories | No qualified pivot or no valid later fill | Monotone prices, truncated histories and missing VIX at a pivot | Empty result tables with stable columns or an unfilled exit | Matched | Exact | Pass |
| Prefix stability | Append observations after both fills | Thirty Hypothesis cases with varying future closes | Existing entry and exit records unchanged | Matched | Exact | Pass |
