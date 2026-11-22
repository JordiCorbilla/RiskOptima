###############################################################################
#                                    sma.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: SMA crossover helpers
###############################################################################

from __future__ import annotations

from datetime import datetime
import os
from typing import Iterable

import numpy as np
import pandas as pd

from riskoptima.branding import add_riskoptima_signature


TRADE_COLUMNS = [
    "Ticker",
    "Entry Date",
    "Exit Date",
    "Entry Price",
    "Exit Price",
    "Return",
    "Exit Reason",
]

PORTFOLIO_TRADE_COLUMNS = TRADE_COLUMNS + ["Weight", "Weighted Return"]


def _close_frame(prices) -> pd.DataFrame:
    data = pd.DataFrame(prices).copy()
    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            close_data = data.xs("Close", axis=1, level=0)
        elif "Close" in data.columns.get_level_values(-1):
            close_data = data.xs("Close", axis=1, level=-1)
        else:
            raise ValueError("prices must contain a Close column or a single price column")
        data = close_data.iloc[:, [0]].copy()
        data.columns = ["Close"]
    elif "Close" in data.columns:
        data = data[["Close"]].copy()
    elif data.shape[1] == 1:
        data.columns = ["Close"]
    else:
        raise ValueError("prices must contain a Close column or a single price column")
    data = _ordered_closes(data)
    if isinstance(prices, (pd.Series, pd.DataFrame)):
        data.attrs.update(prices.attrs)
    return data


def _ordered_closes(data: pd.DataFrame) -> pd.DataFrame:
    if not data.index.is_unique or data.index.hasnans:
        raise ValueError("price rows must have unique, non-missing index values")
    data = data.copy()
    close = pd.to_numeric(data["Close"], errors="coerce")
    data["Close"] = close.where(np.isfinite(close))
    data = data.dropna(subset=["Close"]).sort_index()
    if data["Close"].le(0).any():
        raise ValueError("close prices must be positive")
    return data


def download_close_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads close prices with yfinance.
    """
    import yfinance as yf

    prices = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)[["Close"]].copy()
    prices.attrs.update(source="Yahoo Finance", symbol=ticker, price_field="Close", auto_adjust=False,
                        index_role="session_date")
    return prices


def build_sma_signal_frame(prices, short_window: int = 20, long_window: int = 50) -> pd.DataFrame:
    """
    Builds close, SMA, and crossover signal columns.

    Signal is +1 on bullish short-over-long crosses, -1 on bearish crosses, and 0 otherwise.
    """
    if short_window <= 0 or long_window <= 0:
        raise ValueError("short_window and long_window must be positive")
    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    df = _close_frame(prices)
    df[f"SMA{short_window}"] = df["Close"].rolling(short_window).mean()
    df[f"SMA{long_window}"] = df["Close"].rolling(long_window).mean()

    short_ma = df[f"SMA{short_window}"]
    long_ma = df[f"SMA{long_window}"]
    bullish = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
    bearish = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

    df["Signal"] = 0
    df.loc[bullish, "Signal"] = 1
    df.loc[bearish, "Signal"] = -1
    df.attrs["signal_observation"] = "close"
    return df


def _sma_order(entry, price, signal, stop_loss, take_profit):
    if entry is None:
        return "Buy Signal" if signal == 1 else None
    if signal == -1:
        return "Sell Signal"
    pnl = (price - entry[1]) / entry[1]
    if stop_loss is not None and pnl <= -stop_loss:
        return "Stop Loss"
    if take_profit is not None and pnl >= take_profit:
        return "Take Profit"
    return None


def _fill_sma_order(trades, ticker, order, entry, date, price):
    if order is None:
        return entry
    if entry is None:
        return date, price
    entry_date, entry_price = entry
    trades.append(
        {
            "Ticker": ticker,
            "Entry Date": entry_date,
            "Exit Date": date,
            "Entry Price": entry_price,
            "Exit Price": price,
            "Return": (price - entry_price) / entry_price,
            "Exit Reason": order,
        }
    )
    return None


def trades_from_sma_signals(
    signal_frame: pd.DataFrame,
    ticker: str,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    *,
    execution: str = "same_close",
) -> pd.DataFrame:
    """Convert crossover signals and close-based risk triggers to a trade log.

    The compatibility default fills at the signal close. Use next_close
    for signals observed at the close. It fills at the next available
    close, including exits triggered by stops or take profits. Missing
    closes cannot trigger or fill an order. Returns contain closed trades.
    """
    missing = {"Close", "Signal"} - set(signal_frame.columns)
    if missing:
        raise ValueError(f"signal_frame is missing required columns: {sorted(missing)}")
    if execution not in {"same_close", "next_close"}:
        raise ValueError("execution must be 'same_close' or 'next_close'")
    for parameter, threshold in (("stop_loss", stop_loss), ("take_profit", take_profit)):
        if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
            raise ValueError(f"{parameter} must be non-negative and finite")

    bars = _ordered_closes(signal_frame.loc[:, ["Close", "Signal"]])
    if not bars["Signal"].isin([-1, 0, 1]).all():
        raise ValueError("Signal must contain only -1, 0, or 1")
    trades: list[dict] = []
    entry = None
    pending = None
    for date, close, signal in bars.itertuples():
        price = float(close)
        if execution == "next_close":
            entry = _fill_sma_order(trades, ticker, pending, entry, date, price)
        order = _sma_order(entry, price, signal, stop_loss, take_profit)
        if execution == "same_close":
            entry = _fill_sma_order(trades, ticker, order, entry, date, price)
        else:
            pending = order

    result = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    result.attrs.update(signal_frame.attrs)
    result.attrs["execution"] = execution
    return result


def run_sma_strategy_with_risk(
    ticker: str,
    start: str,
    end: str,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    short_window: int = 20,
    long_window: int = 50,
    prices: pd.DataFrame | None = None,
    *,
    execution: str = "same_close",
) -> pd.DataFrame:
    """
    Runs an SMA crossover strategy for one ticker and returns a trade log.
    """
    price_data = prices if prices is not None else download_close_prices(ticker, start, end)
    signals = build_sma_signal_frame(price_data, short_window=short_window, long_window=long_window)
    return trades_from_sma_signals(
        signals, ticker=ticker, stop_loss=stop_loss, take_profit=take_profit, execution=execution,
    )


def run_strategy_on_portfolio(
    asset_table: pd.DataFrame,
    start: str,
    end: str,
    stop_loss: float | None = None,
    take_profit: float | None = None,
    short_window: int = 20,
    long_window: int = 50,
    *,
    execution: str = "same_close",
) -> pd.DataFrame:
    """
    Runs the SMA strategy across an asset table with Asset and Weight columns.
    """
    if not {"Asset", "Weight"}.issubset(asset_table.columns):
        raise ValueError("asset_table must contain Asset and Weight columns")

    results = []
    for _, row in asset_table.iterrows():
        trades_df = run_sma_strategy_with_risk(
            row["Asset"],
            start,
            end,
            stop_loss=stop_loss,
            take_profit=take_profit,
            short_window=short_window,
            long_window=long_window,
            execution=execution,
        )
        if trades_df.empty:
            continue
        trades_df["Weight"] = row["Weight"]
        trades_df["Weighted Return"] = trades_df["Return"] * row["Weight"]
        results.append(trades_df)

    if not results:
        return pd.DataFrame(columns=PORTFOLIO_TRADE_COLUMNS)

    all_trades = pd.concat(results, ignore_index=True)
    return all_trades.sort_values(by="Entry Date")


def plot_sma_strategy_cumulative_return(trade_log: pd.DataFrame, title="Portfolio Return"):
    """
    Plots cumulative weighted returns from an SMA trade log.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(20, 12))

    if not trade_log.empty:
        trade_log = trade_log.sort_values("Exit Date").copy()
        trade_log["Cumulative Return"] = (1 + trade_log["Weighted Return"]).cumprod()
        plt.plot(trade_log["Exit Date"], trade_log["Cumulative Return"], marker="o")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.3)
    add_riskoptima_signature(ax, y=-0.20, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_sma_strategy_trades(signal_frame: pd.DataFrame, ticker: str):
    """
    Plots close prices, SMA bands, and buy/sell markers.
    """
    import matplotlib.pyplot as plt

    df = signal_frame.copy()
    sma_cols = [col for col in df.columns if col.startswith("SMA")]

    fig, ax = plt.subplots(figsize=(20, 12))
    plt.plot(df.index, df["Close"], label="Close Price", alpha=0.5)
    for col in sma_cols:
        plt.plot(df.index, df[col], label=col, alpha=0.8)

    plt.scatter(df.index[df["Signal"] == 1], df["Close"][df["Signal"] == 1], marker="^", color="green", s=100, label="Buy Signal")
    plt.scatter(df.index[df["Signal"] == -1], df["Close"][df["Signal"] == -1], marker="v", color="red", s=100, label="Sell Signal")
    plt.title(f"{ticker} - SMA Strategy with Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    add_riskoptima_signature(ax, y=-0.12, fontsize=12)
    plt.tight_layout()

    plots_folder = "plots"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    plot_path = os.path.join(plots_folder, f"riskoptima_sma_strategy_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()


def normalize_asset_table(tickers) -> pd.DataFrame:
    """
    Normalizes a ticker string, ticker list, or asset table into Asset/Weight rows.
    """
    if isinstance(tickers, str):
        return pd.DataFrame([{"Asset": tickers, "Weight": 1.0}])
    if isinstance(tickers, Iterable) and not isinstance(tickers, pd.DataFrame):
        tickers = list(tickers)
        return pd.DataFrame([{"Asset": ticker, "Weight": 1.0 / len(tickers)} for ticker in tickers])
    if isinstance(tickers, pd.DataFrame):
        return tickers.copy()
    raise ValueError("Tickers must be a string, list, or DataFrame.")


def run_and_plot_sma_strategy(
    tickers,
    start_date,
    end_date,
    stop_loss=None,
    take_profit=None,
    short_window: int = 20,
    long_window: int = 50,
    *,
    execution: str = "same_close",
):
    """
    Runs and plots SMA signals plus cumulative weighted trade returns.
    """
    asset_table = normalize_asset_table(tickers)
    portfolio_trades = run_strategy_on_portfolio(
        asset_table,
        start=start_date,
        end=end_date,
        stop_loss=stop_loss,
        take_profit=take_profit,
        short_window=short_window,
        long_window=long_window,
        execution=execution,
    )

    for ticker in asset_table["Asset"]:
        prices = download_close_prices(ticker, start_date, end_date)
        signals = build_sma_signal_frame(prices, short_window=short_window, long_window=long_window)
        plot_sma_strategy_trades(signals, ticker)

    plot_sma_strategy_cumulative_return(portfolio_trades, title="SMA Strategy - Cumulative Return")
    return portfolio_trades
