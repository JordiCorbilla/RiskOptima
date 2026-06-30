###############################################################################
#                                  ohlc.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: OHLC volatility estimators
###############################################################################

from __future__ import annotations

import numpy as np
import pandas as pd


def _ohlc_frame(data) -> pd.DataFrame:
    frame = pd.DataFrame(data).copy()
    lower_map = {str(col).lower(): col for col in frame.columns}
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in lower_map]
    if missing:
        raise ValueError(f"OHLC data is missing columns: {missing}")
    ohlc = frame[[lower_map[col] for col in required]].astype(float)
    ohlc.columns = required
    ohlc = ohlc.replace([np.inf, -np.inf], np.nan).dropna()
    if ohlc.empty:
        raise ValueError("OHLC data must contain at least one complete finite row")
    if (ohlc <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (ohlc["high"] < ohlc[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("high must be at least open, close, and low")
    if (ohlc["low"] > ohlc[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("low must be at most open, close, and high")
    return ohlc


def parkinson_volatility(ohlc, periods_per_year: int = 252) -> float:
    """Annualized Parkinson high-low volatility estimator."""
    data = _ohlc_frame(ohlc)
    hl = np.log(data["high"] / data["low"])
    variance = np.square(hl).mean() / (4.0 * np.log(2.0))
    return float(np.sqrt(variance * periods_per_year))


def garman_klass_volatility(ohlc, periods_per_year: int = 252) -> float:
    """Annualized Garman-Klass OHLC volatility estimator."""
    data = _ohlc_frame(ohlc)
    hl = np.log(data["high"] / data["low"])
    co = np.log(data["close"] / data["open"])
    variance = (0.5 * np.square(hl) - (2.0 * np.log(2.0) - 1.0) * np.square(co)).mean()
    return float(np.sqrt(max(variance, 0.0) * periods_per_year))


def rogers_satchell_volatility(ohlc, periods_per_year: int = 252) -> float:
    """Annualized Rogers-Satchell OHLC volatility estimator."""
    data = _ohlc_frame(ohlc)
    ho = np.log(data["high"] / data["open"])
    hc = np.log(data["high"] / data["close"])
    lo = np.log(data["low"] / data["open"])
    lc = np.log(data["low"] / data["close"])
    variance = (ho * hc + lo * lc).mean()
    return float(np.sqrt(max(variance, 0.0) * periods_per_year))


def yang_zhang_volatility(ohlc, periods_per_year: int = 252) -> float:
    """Annualized Yang-Zhang volatility estimator."""
    data = _ohlc_frame(ohlc)
    if len(data) < 2:
        raise ValueError("Yang-Zhang volatility requires at least two OHLC rows")
    open_close = np.log(data["close"] / data["open"])
    overnight = np.log(data["open"] / data["close"].shift(1)).dropna()
    rs_terms = (
        np.log(data["high"] / data["open"]) * np.log(data["high"] / data["close"])
        + np.log(data["low"] / data["open"]) * np.log(data["low"] / data["close"])
    )
    rs = rs_terms.iloc[1:]
    open_close = open_close.iloc[1:]
    n = len(open_close)
    k = 0.34 / (1.34 + (n + 1.0) / max(n - 1.0, 1.0))
    variance = overnight.var(ddof=1) + k * open_close.var(ddof=1) + (1.0 - k) * rs.mean()
    return float(np.sqrt(max(float(variance), 0.0) * periods_per_year))

