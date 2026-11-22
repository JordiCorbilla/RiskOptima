# RiskOptima Phase 1 architecture audit

## Public compatibility and package boundaries

The v2.8.1 audit covered the legacy `RiskOptima` facade, documented module exports, examples, notebooks, and RiskOptima Platform's repository integration. The facade retains all 144 baseline public attributes. The root and each module retain their baseline `__all__` entries. `ImpliedVolatilityResult` is additive at root and in options. `merton_jump_diffusion_call` is additive in options. The MIT licence and production dependency declarations are unchanged.

| Area | Ownership after Phase 1 | Remaining duplication and migration risk |
| --- | --- | --- |
| Options and volatility | `options` owns the corrected Merton terminal pricer and IV solver. The facade and volatility module share these implementations. | The legacy Black-Scholes call, spot Greeks, Heston and plot-facing simulations still differ from modular APIs in boundary semantics or units. Preserve scalar, array and legacy report conventions while inventorying callers before delegation. Vendor-quoted IV on the screener is a separate input source from solver IV. |
| Risk and expected shortfall | The empirical tail routines in `risk/_tail.py` serve credit, market, legacy, contribution, and portfolio optimization code. | Historical VaR interpolation and parametric risk-free conventions still differ by interface. Platform's fallback report independently implements threshold-mean CVaR, so it can differ if Platform cannot use the installed RiskOptima report. |
| Portfolio optimization | Modular optimizers share covariance validation and return annualized volatility. The factor model can annualize its per-observation output explicitly. | The legacy frontier and Monte Carlo optimizers still duplicate objectives and convention handling. The legacy sampler normalizes candidate weights after drawing from `min_allocation` to `max_allocation`, so those arguments are sampling bounds with no strict post-normalization guarantee. Replace or deprecate that behavior in a separate compatibility-reviewed change. |
| Backtesting and market data | The event engine owns the strategy cutoff, marks, execution, and prior-ADV policy. The forecast computation accepts supplied close data and estimators, and the legacy wrapper downloads prices. | The separate SMA stop-loss/take-profit notebook workflow owns its own signal, fill and intrabar assumptions. Legacy Yahoo helpers and live notebook adapters use current provider datasets. Timestamp provenance and repeatable historical snapshots are open. |
| Returns and rates | Curve construction exposes schedules and day-count conventions. The new ML consumer uses one internally consistent legacy annual basis. | Legacy fixed-income conventions differ from the dated curve APIs. Legacy trading days default to 260 and modular sample analytics commonly default to 252. Preserve defaults during Phase 1, require explicit frequency where the bases join, then consolidate with caller migration tests. |
| Imports and reporting | Changed numeric kernels operate on supplied data and do not fetch quotes or render figures. Plotting remains in existing plot entry points. | Importing `riskoptima` still eagerly loads the legacy facade and plotting/vendor dependencies. Modular reporting includes both computation and plotting wrappers. Import isolation needs a measured migration and Platform verification. |

## Downstream and notebook evidence

- RiskOptima Platform backend at `4c30d162b4ca01919dde8c171464d21d3356fcc5` passed all 6 backend tests with both the Phase 1 source checkout and an installed copy of the candidate wheel. This exercises API endpoints, options, optimizers, risk, backtesting, and dislocation through the tested backend paths. Platform's upstream dependency file still pins 2.7.1. The compatibility runs used the candidate via a local import override. [Platform repository](https://github.com/JordiCorbilla/RiskOptima-Platform)
- All 14 tracked notebooks executed from top to bottom in fresh Python 3.12 kernels on 22 November 2026. The live Yahoo demos ran in their default mode, and notebook 14 reported successful analysis of 11 of 11 funds. Each notebook used an isolated output directory and the candidate source checkout.
- Notebook 02 initially received zero BKH training closes from Yahoo for January 2022 through November 2023. It now uses AEP in the same utility allocation and the legacy optimizer identifies missing histories directly. A subsequent live run exposed a pandas scalar-vs-Series assumption in the benchmark statistics table. The table now normalizes its single return series before calculating metrics.
- Notebook 07 annualizes factor covariance explicitly before optimization. Saved outputs for notebooks 02 and 07 were refreshed. Live verification covered data available at run time. It did not establish provider availability for future reruns.

## Baseline export inventory

The names below come from importing the baseline package and reading each declared `__all__`. The last entry lists the public attributes declared on the facade class. This inventory records the supported named surface and is not a guarantee about incidental third-party imports.

### riskoptima (47)

`BacktestConfig`, `BusinessCalendar`, `Constraints`, `DiscountCurve`, `FactorRiskModel`, `MarketData`, `MultiCurveSet`, `OptimizationResult`, `OptionBook`, `OptionContract`, `Portfolio`, `PortfolioState`, `RiskOptima`, `RiskReport`, `SMACrossStrategy`, `SimpleCostModel`, `Strategy`, `StressScenario`, `__version__`, `black_scholes_greeks`, `black_scholes_price`, `bootstrap_discount_curve`, `bootstrap_sofr_curve`, `build_market_risk_report`, `build_markov_regime_report`, `build_portfolio_sophistication_report`, `build_sma_signal_frame`, `component_volatility_contribution`, `credit_var`, `ewma_volatility`, `expected_loss`, `fit_markov_regime_model`, `historical_volatility`, `implied_volatility`, `key_rate_dv01`, `load_sample_credit_portfolio`, `load_sample_market_returns`, `merton_pd`, `optimize_max_sharpe`, `optimize_min_variance`, `price_fixed_rate_bond`, `realized_volatility`, `reprice_curve_instruments`, `rolling_volatility`, `run_backtest`, `run_sma_strategy_with_risk`, `run_stress_scenario`

### riskoptima.backtest (11)

`PortfolioState`, `SMACrossStrategy`, `Strategy`, `build_sma_signal_frame`, `plot_sma_strategy_cumulative_return`, `plot_sma_strategy_trades`, `run_and_plot_sma_strategy`, `run_backtest`, `run_sma_strategy_with_risk`, `run_strategy_on_portfolio`, `trades_from_sma_signals`

### riskoptima.core (4)

`BacktestConfig`, `MarketData`, `Portfolio`, `RiskReport`

### riskoptima.credit (12)

`credit_cvar`, `credit_var`, `expected_loss`, `logistic_pd_model`, `merton_distance_to_default`, `merton_pd`, `portfolio_expected_loss`, `predict_pd`, `simulate_credit_losses`, `simulate_rating_migration`, `unexpected_loss`, `validate_transition_matrix`

### riskoptima.data (2)

`load_sample_credit_portfolio`, `load_sample_market_returns`

### riskoptima.dislocation (10)

`DislocationAnalysis`, `DislocationConfig`, `MarketDataProvider`, `MarketSnapshot`, `RESEARCH_DISCLAIMER`, `YahooFinanceProvider`, `analyze_snapshot`, `analyze_symbol`, `generate_markdown_report`, `scan_universe`

### riskoptima.optim (9)

`Constraints`, `OptimizationResult`, `SimpleCostModel`, `ewma_covariance`, `ledoit_wolf_covariance`, `nearest_psd`, `optimize_max_sharpe`, `optimize_min_variance`, `sample_covariance`

### riskoptima.options (14)

`OptionBook`, `OptionContract`, `OptionValuationResult`, `binomial_tree_price`, `black_scholes_greeks`, `black_scholes_price`, `build_greeks_grid`, `build_implied_vol_surface`, `event_straddle_analysis`, `implied_volatility`, `monte_carlo_european_option`, `option_scenario_grid`, `price_option_chain`, `value_option_contract`

### riskoptima.rates (20)

`BusinessCalendar`, `DiscountCurve`, `MultiCurveSet`, `bootstrap_discount_curve`, `bootstrap_projection_curve`, `bootstrap_sofr_curve`, `curve_calibration_jacobian`, `generate_schedule`, `key_rate_dv01`, `key_rate_dv01_dates`, `multi_curve_par_swap_rate`, `normalize_day_count`, `plot_yield_curve`, `price_fixed_rate_bond`, `price_fixed_rate_bond_dates`, `reprice_curve_instruments`, `reprice_projection_instruments`, `reprice_sofr_instruments`, `spot_date`, `year_fraction`

### riskoptima.reporting (11)

`available_sophistication_methods`, `build_market_risk_report`, `build_markov_regime_report`, `build_portfolio_sophistication_report`, `plot_correlation_heatmap`, `plot_drawdown_curve`, `plot_markov_regime_chart`, `plot_markov_regime_probabilities`, `plot_portfolio_sophistication_report`, `plot_rolling_volatility`, `plot_var_cvar_distribution`

### riskoptima.risk (20)

`FactorRiskModel`, `MarkovRegimeModel`, `StressResult`, `StressScenario`, `apply_factor_shock`, `apply_price_shock`, `classify_markov_regimes`, `component_cvar`, `component_expected_shortfall`, `component_var`, `component_volatility_contribution`, `default_scenarios`, `factor_risk_contribution`, `fit_markov_regime_model`, `marginal_var`, `marginal_volatility_contribution`, `regime_summary`, `run_scenario_set`, `run_stress_scenario`, `tracking_error_contribution`

### riskoptima.volatility (9)

`ewma_volatility`, `garman_klass_volatility`, `historical_volatility`, `implied_volatility`, `parkinson_volatility`, `realized_volatility`, `rogers_satchell_volatility`, `rolling_volatility`, `yang_zhang_volatility`

### RiskOptima public attributes (144)

`TRADING_DAYS`, `VERSION`, `add_features`, `add_portfolio_terms_explanation`, `add_ratio_explanation`, `add_stats_text_box`, `add_table_to_plot`, `ann_to_inst`, `annualize_returns`, `annualize_volatility`, `backtest_ws`, `black_litterman_adjust_returns`, `black_scholes`, `black_scholes_greeks`, `bond_cash_flows`, `bond_cash_flows_v2`, `bond_price`, `bond_price_v2`, `bond_total_return`, `bt_mix`, `build_correlation_matrix`, `build_factor_risk_model`, `calculate_performance_metrics`, `calculate_portfolio_allocation`, `calculate_statistics`, `calculate_total_returns`, `cc_covariance`, `cir`, `compound`, `compute_market_returns`, `consolidate_stats_to_dataframe`, `convexity`, `correlation_with_market`, `create_heatmap`, `create_lagged_features`, `create_portfolio_area_chart`, `cvar_historic`, `discount`, `discount_v2`, `dollar_duration`, `download_data_yfinance`, `drawdown`, `drawdown_allocator`, `efficient_frontier`, `equal_risk_contributions`, `evaluate_model`, `exit_strategy`, `fetch_historical_stock_prices`, `ff_analysis`, `fixedmix_allocator`, `floor_allocator`, `funding_ratio`, `gbm`, `generate_predictions_tickers`, `generate_stock_predictions`, `get_fff_returns`, `get_ffme_returns`, `get_hfi_returns`, `get_ind_file`, `get_ind_market_caps`, `get_ind_nfirms`, `get_ind_returns`, `get_ind_size`, `get_market_statistics`, `get_previous_working_day`, `get_previous_year_date`, `get_total_market_index_returns`, `get_trading_days`, `glidepath_allocator`, `global_minimum_volatility`, `greeks_simulator`, `heston`, `implied_volatility_screener`, `information_ratio`, `inst_to_ann`, `is_normal`, `kurtosis`, `macaulay_duration`, `macaulay_duration_v2`, `macaulay_duration_v3`, `match_durations`, `max_sharpe_ratio`, `merton_jump_diffusion`, `min_volatility`, `minimize_volatility`, `modified_duration`, `optimal_weights`, `optimize_max_sharpe_with_factors`, `optimize_min_variance_with_factors`, `perform_mean_variance_optimization`, `plot_ef`, `plot_ef2`, `plot_ef_ax`, `plot_efficient_frontier_monte_carlo`, `plot_exit_strategy`, `plot_performance_metrics`, `plot_performance_table`, `plot_probability_table`, `plot_sma_strategy_cumulative_return`, `plot_sma_strategy_trades`, `plot_weights_table`, `portfolio_performance`, `portfolio_return`, `portfolio_tracking_error`, `portfolio_volatility`, `predict_with_model`, `pv`, `pvpb`, `regress`, `risk_contribution`, `run_and_plot_sma_strategy`, `run_backtest_daily`, `run_base_vix_strategy`, `run_cppi`, `run_index_vol_divergence_signals`, `run_monte_carlo_simulation`, `run_portfolio_analysis`, `run_portfolio_optimization_mv_ml`, `run_portfolio_probability_analysis`, `run_sma_strategy_with_risk`, `run_strategy_on_portfolio`, `sample_covariance`, `semideviation`, `setup_chart_aesthetics`, `sharpe_ratio`, `shrinkage_covariance`, `simulate_heston`, `simulate_hull_white`, `simulate_sabr`, `skewness`, `sortino_ratio`, `straddle_backtester`, `style_analysis`, `summary_stats`, `target_risk_contributions`, `terminal_stats`, `terminal_values`, `tracking_error`, `var_gaussian`, `var_historic`, `weight_cw`, `weight_erc`, `weight_ew`, `weight_gmv`
