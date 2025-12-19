from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


class Strategy:
    def generate_target_weights(self, date, prices: pd.DataFrame, state=None) -> pd.Series:
        raise NotImplementedError


@dataclass
class SMACrossStrategy(Strategy):
    short_window: int = 20
    long_window: int = 50

    def generate_target_weights(self, date, prices: pd.DataFrame, state=None) -> pd.Series:
        hist = prices.loc[:date]
        if hist.shape[0] < self.long_window:
            return pd.Series(0.0, index=prices.columns)

        short_ma = hist.rolling(self.short_window).mean().iloc[-1]
        long_ma = hist.rolling(self.long_window).mean().iloc[-1]
        signal = (short_ma > long_ma).astype(float)
        if signal.sum() == 0:
            return pd.Series(0.0, index=prices.columns)
        return signal / signal.sum()
