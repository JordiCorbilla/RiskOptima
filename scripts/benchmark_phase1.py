"""Comparable seeded calls, exercised on baseline and candidate."""
from statistics import median
from timeit import repeat

import numpy as np
import pandas as pd

import riskoptima
from riskoptima import RiskOptima, SMACrossStrategy, run_backtest
from riskoptima.core import BacktestConfig
from riskoptima.risk import fit_markov_regime_model

rng = np.random.default_rng(5)
returns = pd.DataFrame(rng.normal(.0002, .01, (1000, 5)), index=pd.bdate_range("2019-01-01", periods=1000))
prices = 100 * (1 + returns).cumprod()

def merton():
    np.random.seed(240)
    return RiskOptima.merton_jump_diffusion(100, 105, 1.25, .04, .2, 4., -.14, .28, 10_000, 100)

benchmarks = {
    "Merton call (10,000 paths, legacy step argument 100)": (merton, 10),
    "SMA backtest (1,000 bars, five assets, daily rebalance)": (
        lambda: run_backtest(prices, SMACrossStrategy(), BacktestConfig(initial_cash=1_000_000)), 1
    ),
    "Markov HMM (600 rows, three regimes, 25 iterations)": (
        lambda: fit_markov_regime_model(returns.iloc[:600, 0], n_regimes=3, n_iter=25, tol=0, random_state=5), 2
    ),
}
print(riskoptima.__file__)
for label, (call, repeats_per_sample) in benchmarks.items():
    call()
    batch = repeat(call, number=repeats_per_sample, repeat=5)
    print(f"{label}: {1e3 * median(batch) / repeats_per_sample:.3f} ms (median)")
