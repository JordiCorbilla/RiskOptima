# RiskOptima v2.8.1 test and performance baseline

Baseline commit `756293c2948daae570c11fae2e969383f5b07749`, measured from a detached checkout on 22 November 2026. Scope is pytest's configured `tests/` directory and all package modules under `riskoptima`. The repository-root `test.py` and `test_data.py` are outside the configured suite.

## Test and coverage results

| Measure | Baseline v2.8.1 | Phase 1 candidate |
| --- | ---: | ---: |
| Tests passed | 151 | 174 |
| Failures, errors, skips | 0 | 0 |
| Executable lines covered | 3,162 / 4,826 (65.5201%) | 3,423 / 5,054 (67.7285%) |
| Possible branches covered | 696 / 1,296 (53.7037%) | 810 / 1,436 (56.4067%) |
| Combined coverage displayed by coverage.py | 63.0186% | 65.2234% |

The legacy facade alone had 555 of 1,788 executable lines covered (31.0403%) and 33 of 314 branches covered (10.5096%). Combined coverage divides covered lines plus branches by the total opportunities and differs from each individual rate. CI enforces the measured baseline floors rounded down to two decimals, 65.52% for lines and 53.70% for branches. It runs Ruff's correctness selectors across package, tests, examples and scripts, plus the configured lint set on new audit tests, scripts and numerical helpers. Existing stylistic and bugbear diagnostics in legacy sources remain repair work for later phases.

Baseline warnings

1. Pytest selected `pytest.ini` and warned that it was ignoring pytest configuration in `pyproject.toml`. The candidate consolidates settings into `pytest.ini`.
2. Seaborn's heatmap test triggered Matplotlib's pending deprecation of `Colormap.set_bad` in `seaborn/matrix.py`. The upstream warning remains present in the candidate suite.
3. Pandas warned about `DataFrame.set_index(inplace=True)` in the legacy efficient-frontier table. The candidate assigns the returned frame.

The desktop used Windows x64, an Intel Core 7 240H, CPython 3.12.10, NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, pytest 9.2.0, pytest-cov 7.1.0, coverage.py 7.17.4 and the headless Agg backend for test runs. The initial cold instrumented run took 106.70 seconds inside pytest and 109.17 seconds wall time. A repeat with one OpenBLAS and OMP thread took 8.23 seconds inside pytest and 9.20 seconds wall time. The 174-test candidate took 13.89 seconds inside pytest and 15.05 seconds wall time on the same one-thread settings. These times include different test counts and transient host load.

## Focused performance measurements

Median milliseconds per call over five timed batches after one warm-up, with one OpenBLAS and OMP thread. These are local measurements, without a timing gate.

| Workload | Baseline (ms) | Candidate (ms) |
| --- | ---: | ---: |
| Legacy Merton call with 10,000 paths and step argument 100 | 54.143 | 0.812 |
| Daily SMA backtest with 1,000 bars and five assets | 1,927.445 | 1,869.959 |
| Gaussian HMM with 600 observations, three regimes and 25 EM updates | 153.220 | 155.578 |

The Merton comparison uses exact terminal sampling in the candidate and a time-stepped simulation in the baseline. The backtest now copies the causal prefix for each callback. Its cost can grow quadratically with the number of bars, so the measured 1,000-bar result does not establish scalability. The HMM candidate performs the final E-step under the returned parameters.

Reproduce the tests and gate from the selected checkout.

```shell
python -m pytest -q --cov=riskoptima --cov-branch --cov-report=term-missing --cov-report=json:coverage.json --durations=15 -W always
python scripts/check_coverage.py coverage.json
```

Run `scripts/benchmark_phase1.py` with `PYTHONPATH` pointing to each checkout. The benchmark prints the imported package path so the baseline can be verified even when the candidate is installed editable.
