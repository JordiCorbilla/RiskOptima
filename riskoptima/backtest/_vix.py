"""Daily close rules for the legacy VIX divergence workflow."""

from itertools import pairwise

import numpy as np
import pandas as pd

from .sma import _close_frame, _ordered_closes

ENTRY_COLUMNS = ["SignalDate", "base_Close", "VIX_Close", "Comment"]
EXIT_COLUMNS = ["EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "Reason"]
RETURN_COLUMNS = ["EntryDate", "ExitDate", "EntryPrice", "ExitPrice", "PnL", "TotalReturn"]


def validate_vix_execution(execution):
    if execution not in {"same_close", "next_close"}:
        raise ValueError("execution must be 'same_close' or 'next_close'")


def vix_close_frame(base, vix, ma_window, execution):
    if execution == "next_close":
        # Keep base trading sessions when the VIX close is missing.
        base = _close_frame(base).rename(columns={"Close": "base_Close"})
        vix = _close_frame(vix).rename(columns={"Close": "VIX_Close"})
        data = base.join(vix, how="left")
    else:
        base = base[["Close"]].set_axis(["base_Close"], axis="columns")
        vix = vix[["Close"]].set_axis(["VIX_Close"], axis="columns")
        data = base.join(vix, how="inner")

    rolling = data["base_Close"].rolling(ma_window)
    data["MA30"] = rolling.mean()
    data["STD30"] = rolling.std()
    data["Upper_Band"] = data["MA30"] + 2 * data["STD30"]
    data["Lower_Band"] = data["MA30"] - 2 * data["STD30"]
    return data


def vix_entries(data, execution):
    causal = execution == "next_close"
    columns = ENTRY_COLUMNS + (["PivotDate", "ConfirmationDate"] if causal else [])
    close = data["base_Close"].to_numpy(dtype=float, na_value=np.nan)
    vix = data["VIX_Close"].to_numpy(dtype=float, na_value=np.nan)
    lower = data["Lower_Band"].to_numpy()
    upper = data["Upper_Band"].to_numpy()
    pivots = np.flatnonzero((close[1:-1] < close[:-2]) & (close[1:-1] < close[2:])) + 1
    entries = []
    for previous, pivot in pairwise(pivots):
        if not (close[pivot] < close[previous] and vix[pivot] > vix[previous]
                and lower[pivot] <= close[pivot] <= upper[pivot]):
            continue
        fill = pivot + 2 if causal else pivot
        if fill >= len(data):
            break
        entry = {
            "SignalDate": data.index[fill],
            "base_Close": close[fill],
            "VIX_Close": vix[pivot],
            "Comment": "Second lower low + higher VIX + within ±2σ",
        }
        if causal:
            entry.update(PivotDate=data.index[pivot], ConfirmationDate=data.index[pivot + 1])
        entries.append(entry)
    return pd.DataFrame.from_records(entries, columns=columns)


def vix_exits(data, signals, intraday, execution):
    causal = execution == "next_close"
    columns = EXIT_COLUMNS + (["ExitSignalDate"] if causal else [])
    if causal:
        data = _ordered_closes(data.rename(columns={"base_Close": "Close"})).rename(
            columns={"Close": "base_Close"},
        )
    exits = []
    for signal in signals.itertuples(index=False):
        entry_date = signal.SignalDate
        eligible = data.loc[entry_date:] if intraday else data.loc[data.index > entry_date]
        for position, (observed_date, row) in enumerate(eligible.iterrows()):
            if row["base_Close"] > row["MA30"]:
                reason = "Above moving average" if causal else "Above 30-day MA"
            elif row["base_Close"] < row["Lower_Band"]:
                reason = "Below Lower Band"
            else:
                continue
            fill = position + 1 if causal else position
            if fill >= len(eligible):
                break
            exit_record = {
                "EntryDate": entry_date,
                "ExitDate": eligible.index[fill],
                "EntryPrice": signal.base_Close,
                "ExitPrice": eligible["base_Close"].iloc[fill],
                "Reason": reason,
            }
            if causal:
                exit_record["ExitSignalDate"] = observed_date
            exits.append(exit_record)
            break
    return pd.DataFrame.from_records(exits, columns=columns)
