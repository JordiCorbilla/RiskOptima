"""Caller-level regressions for the index and volatility divergence workflow."""

import importlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from riskoptima import RiskOptima
from riskoptima.backtest._vix import vix_close_frame, vix_entries, vix_exits

DATES = pd.date_range("2025-01-06", periods=10, freq="B")
BASE = [100, 96, 104, 95, 107, 112, np.nan, 114, 108, 109]
VIX = [20, 21, 19, 30, 17, np.nan, 25, 18, 24, 19]


@pytest.fixture
def sources(monkeypatch, tmp_path):
    module = importlib.import_module("riskoptima.riskoptima")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module.plt, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.plt, "show", lambda: None)

    def configure(base, vix, dates=DATES, *, reverse=False):
        def download(symbol, *, start, end, progress, auto_adjust):
            assert start == "2025-01-06" and end == "2025-01-20"
            assert not progress and not auto_adjust
            assert symbol in {"BASE", "^VIX"}
            values = base if symbol == "BASE" else vix
            frame = pd.DataFrame({("Close", symbol): pd.array(values, dtype="Float64")}, index=dates)
            return frame.iloc[::-1] if reverse else frame

        monkeypatch.setattr(module.yf, "download", download)

    yield configure
    module.plt.close("all")


def _workflow(**kwargs):
    return RiskOptima.run_index_vol_divergence_signals("2025-01-06", "2025-01-20", "BASE", "^VIX", 3, **kwargs)


def test_next_close_workflow_prices_the_first_available_base_quotes(sources):
    sources(BASE, VIX, reverse=True)
    entries, exits, returns = _workflow(execution="next_close")
    assert len(entries) == len(exits) == len(returns) == 1
    entry, exit_trade, result = entries.iloc[0], exits.iloc[0], returns.iloc[0]
    # The lower low at 95 is confirmed when 107 arrives. The next base
    # close is 112, even though the VIX quote is unavailable that day.
    assert (entry.PivotDate, entry.ConfirmationDate, entry.SignalDate) == (DATES[3], DATES[4], DATES[5])
    assert (entry.base_Close, entry.VIX_Close) == (112, 30)
    # The missing base quote cannot be used. The 114 close is above the
    # three-close mean of 111, so the exit fills at the subsequent 108.
    assert (exit_trade.EntryDate, exit_trade.ExitSignalDate, exit_trade.ExitDate) == (DATES[5], DATES[7], DATES[8])
    assert (exit_trade.EntryPrice, exit_trade.ExitPrice, exit_trade.Reason) == (112, 108, "Above moving average")
    assert result.PnL == pytest.approx(-4 / 112, abs=1e-12, rel=0)
    assert result.TotalReturn == result.PnL


@pytest.mark.parametrize("limit, entry_count", [(4, 0), (5, 0), (8, 1)])
def test_workflow_leaves_orders_unfilled_at_the_end(sources, limit, entry_count):
    sources(BASE[:limit], VIX[:limit], DATES[:limit])
    entries, exits, returns = _workflow(execution="next_close")
    assert len(entries) == entry_count
    assert exits.empty and returns.empty
    assert "ExitSignalDate" in exits and "PnL" in returns


def test_missing_pivot_vix_quote_leaves_no_entry(sources):
    missing_vix = VIX.copy()
    missing_vix[3] = np.nan
    sources(BASE, missing_vix)
    entries, exits, returns = _workflow(execution="next_close")
    assert entries.empty and exits.empty and returns.empty


@pytest.mark.parametrize("second_low, second_vix, window", [(97, 30, 3), (95, 21, 3), (95, 30, 5)])
def test_entry_requires_a_lower_low_higher_vix_and_available_bands(second_low, second_vix, window):
    base = pd.Series(BASE, index=DATES, name="Close")
    vix = pd.Series(VIX, index=DATES, name="Close")
    base.iloc[3] = second_low
    vix.iloc[3] = second_vix
    data = vix_close_frame(base, vix, window, "next_close")
    assert vix_entries(data, "next_close").empty


def test_legacy_empty_results_keep_columns_and_plot(sources):
    sources(list(range(100, 110)), [20] * 10)
    with pytest.warns(FutureWarning, match="retrospective VIX default"):
        entries, exits, returns = _workflow()
    assert entries.empty and exits.empty and returns.empty
    assert list(entries) == ["SignalDate", "base_Close", "VIX_Close", "Comment"]
    assert list(exits) == ["EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "Reason"]
    assert list(returns) == ["EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "PnL", "TotalReturn"]


def test_legacy_entries_retain_retrospective_pivot_price(sources):
    sources(BASE, VIX)
    with pytest.warns(FutureWarning, match="retrospective VIX default"):
        entries, exits, returns = _workflow()
    assert len(entries) == len(exits) == len(returns) == 1
    assert (entries.iloc[0].SignalDate, entries.iloc[0].base_Close) == (DATES[3], 95)
    assert "ConfirmationDate" not in entries and "ExitSignalDate" not in exits


def test_exit_checks_skip_entry_day_without_discarding_later_exits():
    data = pd.DataFrame({"base_Close": [100, 105, 106], "MA30": [99, 102, 103], "Lower_Band": [95, 96, 97]},
                        index=DATES[:3])
    entries = pd.DataFrame({"SignalDate": DATES[:1], "base_Close": [100]})
    same_day = RiskOptima.exit_strategy(data, entries, intraday=True).iloc[0]
    later = RiskOptima.exit_strategy(data, entries, intraday=False).iloc[0]
    assert (same_day.ExitDate, same_day.ExitPrice) == (DATES[0], 100)
    assert (later.ExitDate, later.ExitPrice) == (DATES[1], 105)


def test_causal_lower_band_trigger_fills_through_a_missing_base_quote():
    data = pd.DataFrame(
        {"base_Close": [100, 96, 91, np.nan, 88], "MA30": [102, 99, 98, 97, 95], "Lower_Band": [95] * 5},
        index=DATES[:5],
    )
    entries = pd.DataFrame({"SignalDate": DATES[:1], "base_Close": [100]})
    result = RiskOptima.exit_strategy(data, entries, intraday=False, execution="next_close").iloc[0]
    assert (result.ExitSignalDate, result.ExitDate, result.ExitPrice) == (DATES[2], DATES[4], 88)
    assert result.Reason == "Below Lower Band"


def test_invalid_execution_is_rejected_before_downloading(monkeypatch):
    module = importlib.import_module("riskoptima.riskoptima")

    def unexpected_download(*args, **kwargs):
        pytest.fail("invalid execution reached the data provider")

    monkeypatch.setattr(module.yf, "download", unexpected_download)
    with pytest.raises(ValueError, match="execution"):
        _workflow(execution="open")
    with pytest.raises(ValueError, match="execution"):
        RiskOptima.exit_strategy(pd.DataFrame(), pd.DataFrame(), execution="open")


def test_causal_workflow_rejects_duplicate_sessions(sources):
    dates = DATES.to_list()
    dates[-1] = dates[-2]
    sources(BASE, VIX, dates)
    with pytest.raises(ValueError, match="unique, non-missing"):
        _workflow(execution="next_close")


@given(future=st.lists(st.floats(min_value=40, max_value=180), min_size=1, max_size=7))
@settings(max_examples=30, deadline=None)
def test_future_closes_do_not_revise_completed_causal_fills(future):
    prefix_base = pd.Series(BASE, index=DATES, name="Close")
    prefix_vix = pd.Series(VIX, index=DATES, name="Close")
    later = pd.bdate_range(DATES[-1] + pd.offsets.BDay(), periods=len(future))
    base = pd.concat([prefix_base, pd.Series(future, index=later, name="Close")])
    vix = pd.concat([prefix_vix, pd.Series(25., index=later, name="Close")])
    prefix = vix_close_frame(prefix_base, prefix_vix, 3, "next_close")
    extended = vix_close_frame(base, vix, 3, "next_close")
    prefix_entries = vix_entries(prefix, "next_close")
    extended_entries = vix_entries(extended, "next_close")
    pd.testing.assert_frame_equal(prefix_entries, extended_entries.loc[extended_entries.SignalDate <= DATES[-1]])
    prefix_exits = vix_exits(prefix, prefix_entries, False, "next_close")
    extended_exits = vix_exits(extended, extended_entries, False, "next_close")
    pd.testing.assert_frame_equal(prefix_exits, extended_exits.loc[extended_exits.ExitDate <= DATES[-1]])
