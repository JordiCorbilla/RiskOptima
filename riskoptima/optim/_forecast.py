"""Chronological model evaluation and next-observation return features."""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def evaluate_regressor(model, features, target, *, gap=0):
    """Return expanding-window R² and MSE, then refit on all supplied rows."""
    if isinstance(features, (pd.DataFrame, pd.Series)):
        if not features.index.is_unique or not features.index.is_monotonic_increasing:
            raise ValueError("feature rows must have a unique chronological index")
        if isinstance(target, (pd.DataFrame, pd.Series)) and not target.index.equals(features.index):
            raise ValueError("feature and target rows must have the same chronological index")
    test_size = max(2, len(features) // 6)
    folds = min(5, (len(features) - gap - 1) // test_size)
    if folds < 2:
        raise ValueError("insufficient observations for expanding-window validation")
    scores = cross_validate(
        model, features, target,
        cv=TimeSeriesSplit(n_splits=folds, test_size=test_size, gap=gap),
        scoring={"r2": "r2", "mse": "neg_mean_squared_error"}, error_score="raise",
    )
    model.fit(features, target)
    return float(scores["test_r2"].mean()), float(-scores["test_mse"].mean())


def forecast_next_return(prices, estimator):
    """Use end-of-bar features and targets known one observation later."""
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError("prices must have a single close column")
        prices = prices.iloc[:, 0]
    close = pd.Series(prices, dtype=float).sort_index()
    if close.empty or not close.index.is_unique or close.index.hasnans:
        raise ValueError("prices must have unique, non-missing observation dates")
    if (close.dropna() <= 0).any() or np.isinf(close).any():
        raise ValueError("observed close prices must be positive and finite")
    last_observed = close.last_valid_index()
    if last_observed is None:
        raise ValueError("no observed close prices")
    close = close.loc[:last_observed]

    features = pd.DataFrame(index=close.index)
    for window in (5, 10):
        features[f"mean_{window}"] = close.rolling(window).mean().div(close) - 1.0
    for lag in range(1, 6):
        features[f"return_{lag}"] = close.div(close.shift(lag)) - 1.0
    target = close.shift(-1).div(close) - 1.0
    # Require a complete lookback for the starting warm-up. Internal gaps
    # remain missing and are imputed separately inside each training fold.
    features = features.iloc[9:]
    target = target.reindex(features.index).dropna()
    pipeline = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), estimator)
    score, _ = evaluate_regressor(pipeline, features.loc[target.index], target, gap=1)
    return float(pipeline.predict(features.iloc[[-1]])[0]), score
