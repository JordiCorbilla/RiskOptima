"""SMA decision times, executable quotes and source metadata."""

import numpy as np
import pandas as pd
import pytest

from riskoptima import RiskOptima
from riskoptima.backtest import SMACrossStrategy, build_sma_signal_frame, run_backtest, trades_from_sma_signals
from riskoptima.backtest.sma import download_close_prices
from riskoptima.core import BacktestConfig
from riskoptima.optim import SimpleCostModel


def test_next_close_sma_matches_event_backtest_and_preserves_legacy_trade_dates():
    dates = pd.bdate_range("2026-01-05", periods=9)
    prices = pd.DataFrame({"Close": [102., 100., 99., 103., 105., 100., 97., 98., 97.]}, index=dates)
    signal_frame = build_sma_signal_frame(prices.iloc[::-1], short_window=1, long_window=2)
    assert signal_frame.index.equals(dates)
    long_cross, short_cross = dates[3], dates[5]
    assert signal_frame.loc[long_cross, "Signal"] == 1 and signal_frame.loc[short_cross, "Signal"] == -1

    old_trade = trades_from_sma_signals(signal_frame, "A").iloc[0]
    assert (old_trade["Entry Date"], old_trade["Exit Date"]) == (long_cross, short_cross)
    assert old_trade["Return"] == pytest.approx(100 / 103 - 1, abs=1e-12, rel=0)

    trades = trades_from_sma_signals(signal_frame, "A", execution="next_close")
    assert trades.attrs["execution"] == "next_close"
    assert trades.attrs["signal_observation"] == "close"
    assert len(trades) == 1
    trade = trades.iloc[0]
    assert (trade["Entry Date"], trade["Exit Date"]) == (dates[4], dates[6])
    assert (trade["Entry Price"], trade["Exit Price"], trade["Exit Reason"]) == (105., 97., "Sell Signal")
    assert trade["Return"] == pytest.approx(97 / 105 - 1, abs=1e-12, rel=0)

    equity, weights = run_backtest(
        prices.rename(columns={"Close": "A"}), SMACrossStrategy(short_window=1, long_window=2),
        BacktestConfig(initial_cash=1000), SimpleCostModel(spread_bps=0),
    )
    assert weights.loc[long_cross, "A"] == 0
    assert weights.loc[dates[4], "A"] == pytest.approx(1.0)
    assert weights.loc[dates[6], "A"] == 0
    assert equity.loc[dates[6], "PortfolioValue"] == pytest.approx(1000 * 97 / 105, abs=1e-10, rel=0)


@pytest.mark.parametrize(("closes", "rule", "reason", "expected"), [
    ([100., 105., 96., np.nan, 90.], {"stop_loss": .05}, "Stop Loss", 90 / 105 - 1),
    ([100., 101., 112., np.nan, 126.], {"take_profit": .10}, "Take Profit", 126 / 101 - 1),
])
def test_next_close_stop_and_take_profit_trigger_at_observation_and_fill_after_gap(closes, rule, reason, expected):
    dates = pd.bdate_range("2026-01-05", periods=5)
    bars = pd.DataFrame({"Close": closes, "Signal": [1, 0, 0, 0, 0]}, index=dates)
    assert trades_from_sma_signals(bars.iloc[:-1], "A", execution="next_close", **rule).empty

    trade = trades_from_sma_signals(bars, "A", execution="next_close", **rule).iloc[0]
    assert (trade["Entry Date"], trade["Exit Date"]) == (dates[1], dates[4])
    assert trade["Exit Reason"] == reason
    assert trade["Return"] == pytest.approx(expected, abs=1e-12, rel=0)


def test_sma_propagates_supplied_vendor_tags_and_validates_execution_inputs(monkeypatch):
    import yfinance

    dates = pd.bdate_range("2026-01-05", periods=5)
    raw = pd.DataFrame({"Close": [102., 100., 99., 103., 105.]}, index=dates)
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append((ticker, kwargs["auto_adjust"]))
        return raw

    monkeypatch.setattr(yfinance, "download", fake_download)
    bars = download_close_prices("A", "2026-01-05", "2026-01-13")
    assert calls == [("A", False)]
    signals = build_sma_signal_frame(bars, short_window=1, long_window=2)
    trade_log = trades_from_sma_signals(signals, "A", execution="next_close")
    assert trade_log.attrs == {
        "source": "Yahoo Finance", "symbol": "A", "price_field": "Close",
        "auto_adjust": False, "index_role": "session_date", "signal_observation": "close",
        "execution": "next_close",
    }
    with pytest.raises(ValueError, match="execution"):
        trades_from_sma_signals(signals, "A", execution="intrabar")
    with pytest.raises(ValueError, match="unique"):
        trades_from_sma_signals(signals.iloc[[0, 0]], "A", execution="next_close")
    with pytest.raises(ValueError, match="finite"):
        trades_from_sma_signals(signals, "A", stop_loss=np.inf, execution="next_close")


def test_legacy_facade_forwards_explicit_sma_execution(monkeypatch):
    from riskoptima.backtest import sma

    def fake_sma(*args, execution, **kwargs):
        return execution

    monkeypatch.setattr(sma, "run_sma_strategy_with_risk", fake_sma)
    monkeypatch.setattr(sma, "run_strategy_on_portfolio", fake_sma)
    monkeypatch.setattr(sma, "run_and_plot_sma_strategy", fake_sma)
    for public in (
        RiskOptima.run_sma_strategy_with_risk, RiskOptima.run_strategy_on_portfolio, RiskOptima.run_and_plot_sma_strategy,
    ):
        assert public("A", "2026-01-05", "2026-01-13", execution="next_close") == "next_close"
