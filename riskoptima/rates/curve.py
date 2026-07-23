###############################################################################
#                                  curve.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Discount curves, zero/forward rates, and curve bootstrapping
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from riskoptima.branding import add_riskoptima_signature
from riskoptima.rates.conventions import normalize_day_count, year_fraction


_INTERPOLATION_METHODS = {"log_linear_discount", "linear_zero", "cubic_zero"}


def _as_float_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _restore_shape(values: np.ndarray, source):
    return float(values.item()) if np.ndim(source) == 0 else values


def _payment_schedule(maturity: float, payment_frequency: int) -> tuple[np.ndarray, np.ndarray]:
    if maturity <= 0:
        raise ValueError("maturity must be positive")
    if payment_frequency <= 0:
        raise ValueError("payment_frequency must be positive")

    interval = 1.0 / payment_frequency
    regular_dates = np.arange(interval, maturity, interval)
    payment_times = np.append(regular_dates, maturity)
    accruals = np.diff(np.insert(payment_times, 0, 0.0))
    return payment_times, accruals


@dataclass(frozen=True, init=False)
class DiscountCurve:
    """Continuously compounded zero curve represented by discount-factor knots.

    Tenors are year fractions from the valuation date. The default interpolation
    is linear in log discount factors, equivalent to piecewise-constant forwards.
    """

    tenors: np.ndarray
    discount_factors: np.ndarray
    interpolation: str = "log_linear_discount"
    name: str = "Discount curve"
    valuation_date: pd.Timestamp | None = None
    day_count: str = "ACT/365F"

    def __init__(
        self,
        tenors: Iterable[float],
        discount_factors: Iterable[float],
        interpolation: str = "log_linear_discount",
        name: str = "Discount curve",
        valuation_date: pd.Timestamp | str | None = None,
        day_count: str = "ACT/365F",
    ):
        object.__setattr__(self, "tenors", _as_float_array(tenors, "tenors"))
        object.__setattr__(
            self,
            "discount_factors",
            _as_float_array(discount_factors, "discount_factors"),
        )
        object.__setattr__(self, "interpolation", interpolation)
        object.__setattr__(self, "name", name)
        normalized_date = None if valuation_date is None else pd.Timestamp(valuation_date).normalize()
        object.__setattr__(self, "valuation_date", normalized_date)
        object.__setattr__(self, "day_count", normalize_day_count(day_count))
        self.__post_init__()

    def __post_init__(self):
        tenors = _as_float_array(self.tenors, "tenors")
        discount_factors = _as_float_array(self.discount_factors, "discount_factors")
        if tenors.size != discount_factors.size:
            raise ValueError("tenors and discount_factors must have the same length")
        if np.any(tenors < 0) or np.any(np.diff(tenors) <= 0):
            raise ValueError("tenors must be non-negative and strictly increasing")
        if np.any(discount_factors <= 0):
            raise ValueError("discount_factors must be strictly positive")
        if self.interpolation not in _INTERPOLATION_METHODS:
            allowed = ", ".join(sorted(_INTERPOLATION_METHODS))
            raise ValueError(f"interpolation must be one of: {allowed}")

        if tenors[0] > 0:
            tenors = np.insert(tenors, 0, 0.0)
            discount_factors = np.insert(discount_factors, 0, 1.0)
        elif not np.isclose(discount_factors[0], 1.0, atol=1e-12):
            raise ValueError("the discount factor at tenor 0 must equal 1")
        if tenors.size < 2:
            raise ValueError("at least one positive tenor is required")

        object.__setattr__(self, "tenors", tenors.copy())
        object.__setattr__(self, "discount_factors", discount_factors.copy())

    @classmethod
    def from_zero_rates(
        cls,
        tenors,
        zero_rates,
        *,
        compounding: str = "continuous",
        frequency: int = 1,
        interpolation: str = "log_linear_discount",
        name: str = "Discount curve",
        valuation_date=None,
        day_count: str = "ACT/365F",
    ) -> "DiscountCurve":
        """Build a curve from annualized zero rates."""
        tenor_array = _as_float_array(tenors, "tenors")
        rate_array = _as_float_array(zero_rates, "zero_rates")
        if tenor_array.size != rate_array.size:
            raise ValueError("tenors and zero_rates must have the same length")
        if np.any(tenor_array <= 0):
            raise ValueError("zero-rate tenors must be positive")

        compounding = compounding.lower()
        if compounding == "continuous":
            discount_factors = np.exp(-rate_array * tenor_array)
        elif compounding == "simple":
            denominator = 1.0 + rate_array * tenor_array
            if np.any(denominator <= 0):
                raise ValueError("simple-compounded rates imply non-positive discount factors")
            discount_factors = 1.0 / denominator
        elif compounding == "periodic":
            if frequency <= 0:
                raise ValueError("frequency must be positive")
            base = 1.0 + rate_array / frequency
            if np.any(base <= 0):
                raise ValueError("periodically compounded rates imply non-positive discount factors")
            discount_factors = base ** (-frequency * tenor_array)
        else:
            raise ValueError("compounding must be 'continuous', 'simple', or 'periodic'")

        return cls(
            tenor_array,
            discount_factors,
            interpolation=interpolation,
            name=name,
            valuation_date=valuation_date,
            day_count=day_count,
        )

    @classmethod
    def from_dict(cls, payload: Mapping) -> "DiscountCurve":
        """Restore a curve from a JSON-compatible platform payload."""
        return cls(
            payload["tenors"],
            payload["discount_factors"],
            interpolation=payload.get("interpolation", "log_linear_discount"),
            name=payload.get("name", "Discount curve"),
            valuation_date=payload.get("valuation_date"),
            day_count=payload.get("day_count", "ACT/365F"),
        )

    def time_from_date(self, dates):
        """Convert one or more dates into curve year-fraction tenors."""
        if self.valuation_date is None:
            raise ValueError("valuation_date is required for date-based curve operations")
        collection_types = (list, tuple, np.ndarray, pd.Series, pd.Index)
        if not isinstance(dates, collection_types):
            return year_fraction(self.valuation_date, dates, self.day_count)
        return np.asarray(
            [year_fraction(self.valuation_date, date, self.day_count) for date in dates],
            dtype=float,
        )

    def discount_factor_date(self, dates):
        """Return discount factors for one or more calendar dates."""
        return self.discount_factor(self.time_from_date(dates))

    def zero_rate_date(self, dates, *, compounding: str = "continuous", frequency: int = 1):
        """Return zero rates for one or more calendar dates."""
        return self.zero_rate(self.time_from_date(dates), compounding=compounding, frequency=frequency)

    def _continuous_zero_knots(self) -> np.ndarray:
        rates = np.empty_like(self.tenors)
        rates[1:] = -np.log(self.discount_factors[1:]) / self.tenors[1:]
        rates[0] = rates[1]
        return rates

    def _log_discount(self, tenors) -> np.ndarray:
        requested = np.asarray(tenors, dtype=float)
        if np.any(~np.isfinite(requested)) or np.any(requested < 0):
            raise ValueError("tenors must be finite and non-negative")

        if self.interpolation == "log_linear_discount":
            knots = np.log(self.discount_factors)
            values = np.interp(requested, self.tenors, knots)
            beyond = requested > self.tenors[-1]
            if np.any(beyond):
                slope = (knots[-1] - knots[-2]) / (self.tenors[-1] - self.tenors[-2])
                values = np.asarray(values)
                values[beyond] = knots[-1] + slope * (requested[beyond] - self.tenors[-1])
            return values

        zero_knots = self._continuous_zero_knots()
        if self.interpolation == "linear_zero":
            zero_rates = np.interp(requested, self.tenors, zero_knots)
            beyond = requested > self.tenors[-1]
            if np.any(beyond):
                zero_rates = np.asarray(zero_rates)
                zero_rates[beyond] = zero_knots[-1]
        else:
            zero_rates = CubicSpline(self.tenors, zero_knots, bc_type="natural", extrapolate=True)(requested)
        return -np.asarray(zero_rates) * requested

    def discount_factor(self, tenors):
        """Return discount factors at one or more year-fraction tenors."""
        values = np.exp(self._log_discount(tenors))
        return _restore_shape(values, tenors)

    def zero_rate(self, tenors, *, compounding: str = "continuous", frequency: int = 1):
        """Return annualized zero rates under the requested compounding convention."""
        requested = np.asarray(tenors, dtype=float)
        discount_factors = np.asarray(self.discount_factor(requested))
        safe_tenors = np.where(requested == 0, 1.0, requested)
        continuous = -np.log(discount_factors) / safe_tenors
        if np.any(requested == 0):
            continuous = np.asarray(continuous)
            continuous[requested == 0] = self._continuous_zero_knots()[0]

        compounding = compounding.lower()
        if compounding == "continuous":
            result = continuous
        elif compounding == "simple":
            result = np.expm1(continuous * safe_tenors) / safe_tenors
            if np.any(requested == 0):
                result = np.asarray(result)
                result[requested == 0] = continuous[requested == 0]
        elif compounding == "periodic":
            if frequency <= 0:
                raise ValueError("frequency must be positive")
            result = frequency * np.expm1(continuous / frequency)
        else:
            raise ValueError("compounding must be 'continuous', 'simple', or 'periodic'")
        return _restore_shape(np.asarray(result), tenors)

    def forward_rate(self, start, end, *, compounding: str = "continuous"):
        """Return the annualized forward rate between two year-fraction tenors."""
        start_array, end_array = np.broadcast_arrays(np.asarray(start, dtype=float), np.asarray(end, dtype=float))
        if np.any(start_array < 0) or np.any(end_array <= start_array):
            raise ValueError("forward-rate tenors must satisfy 0 <= start < end")
        accrual = end_array - start_array
        start_df = np.asarray(self.discount_factor(start_array))
        end_df = np.asarray(self.discount_factor(end_array))

        compounding = compounding.lower()
        if compounding == "continuous":
            result = np.log(start_df / end_df) / accrual
        elif compounding == "simple":
            result = (start_df / end_df - 1.0) / accrual
        else:
            raise ValueError("compounding must be 'continuous' or 'simple'")
        return _restore_shape(np.asarray(result), start if np.ndim(start) >= np.ndim(end) else end)

    def par_swap_rate(self, maturity: float, *, payment_frequency: int = 2) -> float:
        """Return the single-curve par fixed rate for an OIS-style swap."""
        payment_times, accruals = _payment_schedule(float(maturity), int(payment_frequency))
        annuity = float(np.dot(accruals, self.discount_factor(payment_times)))
        if annuity <= 0:
            raise ValueError("swap annuity must be positive")
        return (1.0 - float(self.discount_factor(maturity))) / annuity

    def present_value(self, cash_flows, payment_times) -> float:
        """Present value deterministic cash flows using this curve."""
        flows = _as_float_array(cash_flows, "cash_flows")
        times = _as_float_array(payment_times, "payment_times")
        if flows.size != times.size:
            raise ValueError("cash_flows and payment_times must have the same length")
        return float(np.dot(flows, self.discount_factor(times)))

    def to_frame(self, tenors=None) -> pd.DataFrame:
        """Return curve values in a dashboard- and notebook-friendly table."""
        requested = self.tenors if tenors is None else _as_float_array(tenors, "tenors")
        return pd.DataFrame(
            {
                "tenor": requested,
                "discount_factor": self.discount_factor(requested),
                "zero_rate": self.zero_rate(requested),
            }
        )

    def to_dict(self) -> dict:
        """Return a JSON-compatible representation for platform APIs."""
        return {
            "name": self.name,
            "valuation_date": self.valuation_date.isoformat() if self.valuation_date is not None else None,
            "day_count": self.day_count,
            "interpolation": self.interpolation,
            "tenors": self.tenors.tolist(),
            "discount_factors": self.discount_factors.tolist(),
        }


def bootstrap_discount_curve(
    instruments: pd.DataFrame,
    *,
    interpolation: str = "log_linear_discount",
    default_payment_frequency: int = 2,
    name: str = "SOFR discount curve",
    valuation_date=None,
) -> DiscountCurve:
    """Bootstrap a single discount curve from deposit and par-swap quotes.

    Required columns are ``instrument_type``, ``maturity`` (year fraction), and
    ``rate`` (decimal). Swap rows may supply ``payment_frequency``.
    """
    if not isinstance(instruments, pd.DataFrame):
        raise TypeError("instruments must be a pandas DataFrame")
    required = {"instrument_type", "maturity", "rate"}
    missing = required.difference(instruments.columns)
    if missing:
        raise ValueError(f"instruments is missing columns: {', '.join(sorted(missing))}")
    if instruments.empty:
        raise ValueError("instruments must not be empty")

    quotes = instruments.copy()
    quotes["instrument_type"] = quotes["instrument_type"].astype(str).str.lower().str.strip()
    quotes["maturity"] = pd.to_numeric(quotes["maturity"], errors="raise")
    quotes["rate"] = pd.to_numeric(quotes["rate"], errors="raise")
    if quotes[["maturity", "rate"]].isna().any().any() or not np.isfinite(quotes[["maturity", "rate"]]).all().all():
        raise ValueError("maturity and rate must contain finite values")
    if (quotes["maturity"] <= 0).any():
        raise ValueError("instrument maturities must be positive")
    if quotes["maturity"].duplicated().any():
        raise ValueError("instrument maturities must be unique")
    if not quotes["instrument_type"].isin({"deposit", "swap"}).all():
        raise ValueError("instrument_type must be 'deposit' or 'swap'")
    quotes = quotes.sort_values("maturity").reset_index(drop=True)

    tenors = [0.0]
    discount_factors = [1.0]
    swap_seen = False
    for row in quotes.itertuples(index=False):
        instrument_type = row.instrument_type
        maturity = float(row.maturity)
        rate = float(row.rate)

        if instrument_type == "deposit":
            if swap_seen:
                raise ValueError("deposit maturities must precede swap maturities")
            denominator = 1.0 + rate * maturity
            if denominator <= 0:
                raise ValueError("deposit quote implies a non-positive discount factor")
            terminal_df = 1.0 / denominator
        else:
            swap_seen = True
            frequency_value = getattr(row, "payment_frequency", default_payment_frequency)
            if pd.isna(frequency_value):
                frequency_value = default_payment_frequency
            frequency = int(frequency_value)
            payment_times, accruals = _payment_schedule(maturity, frequency)

            def pricing_error(terminal_df: float) -> float:
                trial_curve = DiscountCurve(
                    [*tenors, maturity],
                    [*discount_factors, terminal_df],
                    interpolation=interpolation,
                )
                fixed_leg = rate * float(np.dot(accruals, trial_curve.discount_factor(payment_times)))
                floating_leg = 1.0 - terminal_df
                return fixed_leg - floating_leg

            try:
                terminal_df = brentq(pricing_error, 1e-10, 5.0, xtol=1e-13, rtol=1e-13)
            except ValueError as exc:
                raise ValueError(f"could not bootstrap the {maturity:g}Y swap quote") from exc

        tenors.append(maturity)
        discount_factors.append(float(terminal_df))

    return DiscountCurve(
        tenors,
        discount_factors,
        interpolation=interpolation,
        name=name,
        valuation_date=valuation_date,
    )


def reprice_curve_instruments(curve: DiscountCurve, instruments: pd.DataFrame) -> pd.DataFrame:
    """Reprice curve instruments and return calibration errors in basis points."""
    required = {"instrument_type", "maturity", "rate"}
    missing = required.difference(instruments.columns)
    if missing:
        raise ValueError(f"instruments is missing columns: {', '.join(sorted(missing))}")

    rows = []
    for row in instruments.itertuples(index=False):
        instrument_type = str(row.instrument_type).lower().strip()
        maturity = float(row.maturity)
        market_rate = float(row.rate)
        if instrument_type == "deposit":
            model_rate = (1.0 / float(curve.discount_factor(maturity)) - 1.0) / maturity
        elif instrument_type == "swap":
            frequency_value = getattr(row, "payment_frequency", 2)
            frequency = 2 if pd.isna(frequency_value) else int(frequency_value)
            model_rate = curve.par_swap_rate(maturity, payment_frequency=frequency)
        else:
            raise ValueError("instrument_type must be 'deposit' or 'swap'")
        rows.append(
            {
                "instrument_type": instrument_type,
                "maturity": maturity,
                "market_rate": market_rate,
                "model_rate": model_rate,
                "error_bps": (model_rate - market_rate) * 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def price_fixed_rate_bond(
    curve: DiscountCurve,
    maturity: float,
    coupon_rate: float,
    *,
    face_value: float = 100.0,
    payment_frequency: int = 2,
) -> float:
    """Price a deterministic fixed-rate bond from the discount curve."""
    if face_value <= 0:
        raise ValueError("face_value must be positive")
    payment_times, accruals = _payment_schedule(float(maturity), int(payment_frequency))
    cash_flows = face_value * float(coupon_rate) * accruals
    cash_flows[-1] += face_value
    return curve.present_value(cash_flows, payment_times)


def plot_yield_curve(
    curve: DiscountCurve,
    *,
    tenors=None,
    ax=None,
    show_forwards: bool = True,
):
    """Plot zero and short forward rates with calibrated curve knots."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))
    if tenors is None:
        tenors = np.linspace(max(0.01, curve.tenors[1] / 10), curve.tenors[-1], 240)
    else:
        tenors = _as_float_array(tenors, "tenors")

    zero_color = "#1f77b4"
    forward_color = "#d97706"
    zero_rates = np.asarray(curve.zero_rate(tenors)) * 100.0
    ax.plot(tenors, zero_rates, color=zero_color, linewidth=2.2, label="Zero rate")
    if show_forwards:
        horizon = min(0.25, max(curve.tenors[-1] / 40.0, 0.01))
        forward_rates = np.asarray(curve.forward_rate(tenors, tenors + horizon)) * 100.0
        ax.plot(
            tenors,
            forward_rates,
            color=forward_color,
            linewidth=1.6,
            linestyle="--",
            label=f"{horizon:g}Y forward",
        )

    knot_rates = np.asarray(curve.zero_rate(curve.tenors[1:])) * 100.0
    ax.scatter(
        curve.tenors[1:],
        knot_rates,
        s=42,
        facecolor="white",
        edgecolor=zero_color,
        linewidth=1.5,
        zorder=3,
        label="Calibrated knots",
    )
    ax.set_title(curve.name, loc="left", pad=28)
    valuation_label = (
        f"Valuation date: {curve.valuation_date.date()}"
        if curve.valuation_date is not None
        else "Year-fraction tenor"
    )
    interpolation_label = curve.interpolation.replace("_", " ")
    ax.text(
        0.0,
        1.01,
        f"{valuation_label} | {curve.day_count} curve time | Continuous compounding | {interpolation_label}",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Rate (%)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    add_riskoptima_signature(ax, y=-0.18)
    return ax

