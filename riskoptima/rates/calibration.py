###############################################################################
#                              calibration.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Dated SOFR, multi-curve, Jacobian, and key-rate analytics
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .conventions import BusinessCalendar, generate_schedule, spot_date, year_fraction
from .curve import DiscountCurve, _payment_schedule, bootstrap_discount_curve


_DATED_INSTRUMENT_TYPES = {"deposit", "fomc", "future", "swap"}
_PROJECTION_INSTRUMENT_TYPES = {"deposit", "fra", "swap"}


def _finite_number(value) -> bool:
    try:
        return value is not None and bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _record_value(record: Mapping, key: str, default):
    value = record.get(key)
    return default if value is None or bool(pd.isna(value)) else value


def _record_rate(record: Mapping) -> float:
    instrument_type = str(record["instrument_type"]).lower().strip()
    if _finite_number(record.get("rate")):
        return float(record["rate"])
    if instrument_type == "future" and _finite_number(record.get("price")):
        implied_rate = (100.0 - float(record["price"])) / 100.0
        adjustment_bps = float(_record_value(record, "convexity_adjustment_bps", 0.0))
        return implied_rate - adjustment_bps / 10_000.0
    raise ValueError(f"{instrument_type} instruments require a finite rate or futures price")


def _dated_records(
    instruments: pd.DataFrame,
    valuation_date,
    calendar: BusinessCalendar,
    business_day_convention: str,
    spot_lag: int,
) -> list[dict]:
    if not isinstance(instruments, pd.DataFrame):
        raise TypeError("instruments must be a pandas DataFrame")
    if instruments.empty or "instrument_type" not in instruments.columns:
        raise ValueError("instruments must contain instrument_type rows")

    valuation = pd.Timestamp(valuation_date).normalize()
    default_spot = spot_date(valuation, spot_lag, calendar)
    prepared = []
    for raw in instruments.to_dict("records"):
        record = dict(raw)
        instrument_type = str(record.get("instrument_type", "")).lower().strip()
        if instrument_type not in _DATED_INSTRUMENT_TYPES:
            raise ValueError(f"instrument_type must be one of: {', '.join(sorted(_DATED_INSTRUMENT_TYPES))}")

        if instrument_type in {"deposit", "swap"}:
            if pd.isna(record.get("maturity_date")):
                raise ValueError(f"{instrument_type} instruments require maturity_date")
            start = record.get("effective_date")
            if pd.isna(start):
                start = valuation if instrument_type == "deposit" else default_spot
            end = record["maturity_date"]
        else:
            if pd.isna(record.get("start_date")) or pd.isna(record.get("end_date")):
                raise ValueError(f"{instrument_type} instruments require start_date and end_date")
            start = record["start_date"]
            end = record["end_date"]

        start_date = calendar.adjust(start, business_day_convention)
        end_date = calendar.adjust(end, business_day_convention)
        if end_date <= start_date:
            raise ValueError("instrument end dates must be after start dates")
        record.update(
            {
                "instrument_type": instrument_type,
                "_start_date": start_date,
                "_end_date": end_date,
                "_market_rate": _record_rate(record),
            }
        )
        prepared.append(record)

    prepared.sort(key=lambda item: item["_end_date"])
    end_dates = [item["_end_date"] for item in prepared]
    if len(end_dates) != len(set(end_dates)):
        raise ValueError("dated curve instruments must have unique pillar dates")
    return prepared


def _curve_at_nodes(
    tenors,
    discount_factors,
    *,
    interpolation: str,
    name: str,
    valuation_date,
    day_count: str,
) -> DiscountCurve:
    return DiscountCurve(
        tenors,
        discount_factors,
        interpolation=interpolation,
        name=name,
        valuation_date=valuation_date,
        day_count=day_count,
    )


def bootstrap_sofr_curve(
    instruments: pd.DataFrame,
    *,
    valuation_date,
    calendar: BusinessCalendar | None = None,
    interpolation: str = "log_linear_discount",
    curve_day_count: str = "ACT/365F",
    default_instrument_day_count: str = "ACT/360",
    default_swap_frequency: int = 1,
    business_day_convention: str = "modified_following",
    spot_lag: int = 2,
    name: str = "USD SOFR discount curve",
) -> DiscountCurve:
    """Bootstrap a dated SOFR curve from deposits, FOMC intervals, futures, and OIS swaps."""
    valuation = pd.Timestamp(valuation_date).normalize()
    calendar = calendar or BusinessCalendar()
    records = _dated_records(instruments, valuation, calendar, business_day_convention, spot_lag)

    tenors = [0.0]
    discount_factors = [1.0]
    for record in records:
        instrument_type = record["instrument_type"]
        start_date = record["_start_date"]
        end_date = record["_end_date"]
        market_rate = float(record["_market_rate"])
        start_t = year_fraction(valuation, start_date, curve_day_count)
        end_t = year_fraction(valuation, end_date, curve_day_count)
        if end_t <= tenors[-1] + 1e-12:
            raise ValueError("instrument pillar dates must be strictly increasing")

        if start_t <= 1e-12:
            start_df = 1.0
        else:
            if len(tenors) == 1 or start_t > tenors[-1] + 1e-12:
                raise ValueError(f"the {instrument_type} start date is beyond the calibrated curve")
            current_curve = _curve_at_nodes(
                tenors,
                discount_factors,
                interpolation=interpolation,
                name=name,
                valuation_date=valuation,
                day_count=curve_day_count,
            )
            start_df = float(current_curve.discount_factor(start_t))

        if instrument_type in {"deposit", "fomc", "future"}:
            instrument_day_count = str(_record_value(record, "day_count", default_instrument_day_count))
            accrual = year_fraction(start_date, end_date, instrument_day_count)
            denominator = 1.0 + market_rate * accrual
            if denominator <= 0:
                raise ValueError(f"the {instrument_type} quote implies a non-positive discount factor")
            terminal_df = start_df / denominator
        else:
            frequency = int(_record_value(record, "payment_frequency", default_swap_frequency))
            fixed_day_count = str(_record_value(record, "fixed_day_count", default_instrument_day_count))
            schedule = generate_schedule(
                start_date,
                end_date,
                payment_frequency=frequency,
                day_count=fixed_day_count,
                calendar=calendar,
                business_day_convention=business_day_convention,
            )
            payment_times = np.asarray(
                [year_fraction(valuation, date, curve_day_count) for date in schedule["payment_date"]],
                dtype=float,
            )
            accruals = schedule["accrual_factor"].to_numpy(dtype=float)

            def pricing_error(candidate_df: float) -> float:
                trial_curve = _curve_at_nodes(
                    [*tenors, end_t],
                    [*discount_factors, candidate_df],
                    interpolation=interpolation,
                    name=name,
                    valuation_date=valuation,
                    day_count=curve_day_count,
                )
                fixed_leg = market_rate * float(np.dot(accruals, trial_curve.discount_factor(payment_times)))
                floating_leg = start_df - candidate_df
                return fixed_leg - floating_leg

            try:
                terminal_df = brentq(pricing_error, 1e-10, 5.0, xtol=1e-13, rtol=1e-13)
            except ValueError as exc:
                raise ValueError(f"could not calibrate the OIS swap ending {end_date.date()}") from exc

        tenors.append(end_t)
        discount_factors.append(float(terminal_df))

    return _curve_at_nodes(
        tenors,
        discount_factors,
        interpolation=interpolation,
        name=name,
        valuation_date=valuation,
        day_count=curve_day_count,
    )


def reprice_sofr_instruments(
    curve: DiscountCurve,
    instruments: pd.DataFrame,
    *,
    calendar: BusinessCalendar | None = None,
    default_instrument_day_count: str = "ACT/360",
    default_swap_frequency: int = 1,
    business_day_convention: str = "modified_following",
    spot_lag: int = 2,
) -> pd.DataFrame:
    """Reprice dated SOFR calibration instruments in rate-equivalent basis points."""
    if curve.valuation_date is None:
        raise ValueError("curve.valuation_date is required")
    calendar = calendar or BusinessCalendar()
    records = _dated_records(
        instruments,
        curve.valuation_date,
        calendar,
        business_day_convention,
        spot_lag,
    )

    rows = []
    for record in records:
        instrument_type = record["instrument_type"]
        start_date = record["_start_date"]
        end_date = record["_end_date"]
        market_rate = float(record["_market_rate"])
        start_df = float(curve.discount_factor_date(start_date))
        end_df = float(curve.discount_factor_date(end_date))

        if instrument_type in {"deposit", "fomc", "future"}:
            instrument_day_count = str(_record_value(record, "day_count", default_instrument_day_count))
            accrual = year_fraction(start_date, end_date, instrument_day_count)
            model_rate = (start_df / end_df - 1.0) / accrual
        else:
            frequency = int(_record_value(record, "payment_frequency", default_swap_frequency))
            fixed_day_count = str(_record_value(record, "fixed_day_count", default_instrument_day_count))
            schedule = generate_schedule(
                start_date,
                end_date,
                payment_frequency=frequency,
                day_count=fixed_day_count,
                calendar=calendar,
                business_day_convention=business_day_convention,
            )
            payment_dfs = np.asarray(curve.discount_factor_date(schedule["payment_date"]), dtype=float)
            annuity = float(np.dot(schedule["accrual_factor"].to_numpy(dtype=float), payment_dfs))
            model_rate = (start_df - end_df) / annuity

        rows.append(
            {
                "instrument_type": instrument_type,
                "start_date": start_date,
                "end_date": end_date,
                "market_rate": market_rate,
                "model_rate": model_rate,
                "error_bps": (model_rate - market_rate) * 10_000.0,
            }
        )
    return pd.DataFrame(rows)


def multi_curve_par_swap_rate(
    discount_curve: DiscountCurve,
    projection_curve: DiscountCurve,
    maturity: float,
    *,
    fixed_frequency: int = 2,
    floating_frequency: int = 4,
) -> float:
    """Return a par swap rate using OIS discounting and a separate projection curve."""
    fixed_times, fixed_accruals = _payment_schedule(float(maturity), int(fixed_frequency))
    float_times, float_accruals = _payment_schedule(float(maturity), int(floating_frequency))
    float_starts = np.insert(float_times[:-1], 0, 0.0)
    forward_rates = np.asarray(
        projection_curve.forward_rate(float_starts, float_times, compounding="simple"),
        dtype=float,
    )
    floating_pv = float(
        np.dot(float_accruals * forward_rates, discount_curve.discount_factor(float_times))
    )
    fixed_annuity = float(np.dot(fixed_accruals, discount_curve.discount_factor(fixed_times)))
    if fixed_annuity <= 0:
        raise ValueError("fixed-leg annuity must be positive")
    return floating_pv / fixed_annuity


def bootstrap_projection_curve(
    instruments: pd.DataFrame,
    discount_curve: DiscountCurve,
    *,
    interpolation: str = "log_linear_discount",
    default_fixed_frequency: int = 2,
    default_floating_frequency: int = 4,
    name: str = "SOFR projection curve",
) -> DiscountCurve:
    """Bootstrap a projection curve from deposits, FRAs, and swaps under OIS discounting."""
    required = {"instrument_type", "maturity", "rate"}
    if not isinstance(instruments, pd.DataFrame):
        raise TypeError("instruments must be a pandas DataFrame")
    missing = required.difference(instruments.columns)
    if instruments.empty or missing:
        raise ValueError(f"projection instruments are missing columns: {', '.join(sorted(missing))}")

    quotes = instruments.copy()
    quotes["instrument_type"] = quotes["instrument_type"].astype(str).str.lower().str.strip()
    quotes["maturity"] = pd.to_numeric(quotes["maturity"], errors="raise")
    quotes["rate"] = pd.to_numeric(quotes["rate"], errors="raise")
    if not np.isfinite(quotes[["maturity", "rate"]].to_numpy(dtype=float)).all():
        raise ValueError("projection maturities and rates must be finite")
    if not quotes["instrument_type"].isin(_PROJECTION_INSTRUMENT_TYPES).all():
        raise ValueError(f"instrument_type must be one of: {', '.join(sorted(_PROJECTION_INSTRUMENT_TYPES))}")
    if (quotes["maturity"] <= 0).any() or quotes["maturity"].duplicated().any():
        raise ValueError("projection maturities must be positive and unique")
    quotes = quotes.sort_values("maturity").reset_index(drop=True)

    tenors = [0.0]
    projection_dfs = [1.0]
    for record in quotes.to_dict("records"):
        instrument_type = record["instrument_type"]
        maturity = float(record["maturity"])
        rate = float(record["rate"])
        if not np.isfinite(rate):
            raise ValueError("projection rates must be finite")

        if instrument_type == "deposit":
            denominator = 1.0 + rate * maturity
            if denominator <= 0:
                raise ValueError("deposit quote implies a non-positive projection discount factor")
            terminal_df = 1.0 / denominator
        elif instrument_type == "fra":
            start = float(record.get("start", tenors[-1]))
            if start < 0 or start >= maturity or start > tenors[-1] + 1e-12:
                raise ValueError("FRA start must be calibrated and satisfy 0 <= start < maturity")
            if start <= 1e-12:
                start_df = 1.0
            else:
                current_curve = DiscountCurve(
                    tenors,
                    projection_dfs,
                    interpolation=interpolation,
                    name=name,
                    valuation_date=discount_curve.valuation_date,
                    day_count=discount_curve.day_count,
                )
                start_df = float(current_curve.discount_factor(start))
            terminal_df = start_df / (1.0 + rate * (maturity - start))
        else:
            fixed_value = record.get("fixed_frequency", default_fixed_frequency)
            floating_value = record.get("floating_frequency", default_floating_frequency)
            fixed_frequency = default_fixed_frequency if pd.isna(fixed_value) else int(fixed_value)
            floating_frequency = default_floating_frequency if pd.isna(floating_value) else int(floating_value)

            def pricing_error(candidate_df: float) -> float:
                trial_curve = DiscountCurve(
                    [*tenors, maturity],
                    [*projection_dfs, candidate_df],
                    interpolation=interpolation,
                    name=name,
                    valuation_date=discount_curve.valuation_date,
                    day_count=discount_curve.day_count,
                )
                model_rate = multi_curve_par_swap_rate(
                    discount_curve,
                    trial_curve,
                    maturity,
                    fixed_frequency=fixed_frequency,
                    floating_frequency=floating_frequency,
                )
                return model_rate - rate

            try:
                terminal_df = brentq(pricing_error, 1e-10, 5.0, xtol=1e-13, rtol=1e-13)
            except ValueError as exc:
                raise ValueError(f"could not calibrate the {maturity:g}Y projection swap") from exc

        tenors.append(maturity)
        projection_dfs.append(float(terminal_df))

    return DiscountCurve(
        tenors,
        projection_dfs,
        interpolation=interpolation,
        name=name,
        valuation_date=discount_curve.valuation_date,
        day_count=discount_curve.day_count,
    )


def reprice_projection_instruments(
    projection_curve: DiscountCurve,
    discount_curve: DiscountCurve,
    instruments: pd.DataFrame,
    *,
    default_fixed_frequency: int = 2,
    default_floating_frequency: int = 4,
) -> pd.DataFrame:
    """Return projection-curve calibration errors in basis points."""
    rows = []
    for record in instruments.sort_values("maturity").to_dict("records"):
        instrument_type = str(record["instrument_type"]).lower().strip()
        maturity = float(record["maturity"])
        market_rate = float(record["rate"])
        if instrument_type == "deposit":
            model_rate = (1.0 / float(projection_curve.discount_factor(maturity)) - 1.0) / maturity
        elif instrument_type == "fra":
            start = float(record.get("start", 0.0))
            model_rate = float(projection_curve.forward_rate(start, maturity, compounding="simple"))
        elif instrument_type == "swap":
            fixed_value = record.get("fixed_frequency", default_fixed_frequency)
            floating_value = record.get("floating_frequency", default_floating_frequency)
            fixed_frequency = default_fixed_frequency if pd.isna(fixed_value) else int(fixed_value)
            floating_frequency = default_floating_frequency if pd.isna(floating_value) else int(floating_value)
            model_rate = multi_curve_par_swap_rate(
                discount_curve,
                projection_curve,
                maturity,
                fixed_frequency=fixed_frequency,
                floating_frequency=floating_frequency,
            )
        else:
            raise ValueError(f"instrument_type must be one of: {', '.join(sorted(_PROJECTION_INSTRUMENT_TYPES))}")
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


@dataclass(frozen=True)
class MultiCurveSet:
    """OIS discount curve plus named forward-projection curves."""

    discount_curve: DiscountCurve
    projection_curves: Mapping[str, DiscountCurve]

    def __post_init__(self):
        if not self.projection_curves:
            raise ValueError("projection_curves must not be empty")
        projection_curves = dict(self.projection_curves)
        for name, curve in projection_curves.items():
            if not isinstance(name, str) or not isinstance(curve, DiscountCurve):
                raise TypeError("projection_curves must map names to DiscountCurve objects")
            if (
                self.discount_curve.valuation_date is not None
                and curve.valuation_date is not None
                and curve.valuation_date != self.discount_curve.valuation_date
            ):
                raise ValueError("discount and projection curves must share a valuation date")
        object.__setattr__(self, "projection_curves", projection_curves)

    def projection_curve(self, name: str) -> DiscountCurve:
        try:
            return self.projection_curves[name]
        except KeyError as exc:
            raise KeyError(f"unknown projection curve: {name}") from exc

    def par_swap_rate(
        self,
        maturity: float,
        projection_curve: str,
        *,
        fixed_frequency: int = 2,
        floating_frequency: int = 4,
    ) -> float:
        return multi_curve_par_swap_rate(
            self.discount_curve,
            self.projection_curve(projection_curve),
            maturity,
            fixed_frequency=fixed_frequency,
            floating_frequency=floating_frequency,
        )

    def to_dict(self) -> dict:
        return {
            "discount_curve": self.discount_curve.to_dict(),
            "projection_curves": {name: curve.to_dict() for name, curve in self.projection_curves.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "MultiCurveSet":
        return cls(
            discount_curve=DiscountCurve.from_dict(payload["discount_curve"]),
            projection_curves={
                name: DiscountCurve.from_dict(curve_payload)
                for name, curve_payload in payload["projection_curves"].items()
            },
        )


def curve_calibration_jacobian(
    instruments: pd.DataFrame,
    *,
    curve_builder: Callable[..., DiscountCurve] = bootstrap_discount_curve,
    builder_kwargs: Mapping | None = None,
    output_tenors=None,
    bump_bps: float = 1.0,
) -> pd.DataFrame:
    """Finite-difference zero-rate Jacobian with quote-rate rows expressed bp-for-bp."""
    if bump_bps <= 0:
        raise ValueError("bump_bps must be positive")
    kwargs = dict(builder_kwargs or {})
    base_curve = curve_builder(instruments.copy(), **kwargs)
    tenors = base_curve.tenors[1:] if output_tenors is None else np.asarray(output_tenors, dtype=float)
    bump = bump_bps / 10_000.0

    sensitivities = {}
    for position, (_, row) in enumerate(instruments.iterrows()):
        up = instruments.copy()
        down = instruments.copy()
        instrument_type = str(row.get("instrument_type", "quote")).lower().strip()
        if "rate" in instruments.columns and _finite_number(row.get("rate")):
            up.iloc[position, up.columns.get_loc("rate")] = float(row["rate"]) + bump
            down.iloc[position, down.columns.get_loc("rate")] = float(row["rate"]) - bump
        elif instrument_type == "future" and "price" in instruments.columns and _finite_number(row.get("price")):
            price_bump = bump * 100.0
            up.iloc[position, up.columns.get_loc("price")] = float(row["price"]) - price_bump
            down.iloc[position, down.columns.get_loc("price")] = float(row["price"]) + price_bump
        else:
            raise ValueError("each Jacobian quote requires rate or a futures price")

        up_curve = curve_builder(up, **kwargs)
        down_curve = curve_builder(down, **kwargs)
        label = row.get("quote_id")
        if pd.isna(label):
            pillar = row.get("maturity", row.get("maturity_date", row.get("end_date", position)))
            label = f"{instrument_type}_{pillar}"
        label = str(label)
        if label in sensitivities:
            raise ValueError(f"Jacobian quote labels must be unique: {label}")
        sensitivities[label] = (
            np.asarray(up_curve.zero_rate(tenors)) - np.asarray(down_curve.zero_rate(tenors))
        ) / (2.0 * bump)

    jacobian = pd.DataFrame(sensitivities, index=np.asarray(tenors, dtype=float))
    jacobian.index.name = "output_tenor"
    return jacobian


def key_rate_dv01(
    curve: DiscountCurve,
    cash_flows,
    payment_times,
    *,
    bump_bps: float = 1.0,
) -> pd.Series:
    """Return PV loss for a one-basis-point bump at each zero-curve knot."""
    if bump_bps <= 0:
        raise ValueError("bump_bps must be positive")
    flows = np.asarray(cash_flows, dtype=float)
    times = np.asarray(payment_times, dtype=float)
    if flows.ndim != 1 or times.ndim != 1 or flows.size != times.size or flows.size == 0:
        raise ValueError("cash_flows and payment_times must be equal-length one-dimensional arrays")
    if np.any(~np.isfinite(flows)) or np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError("cash flows and payment times must be finite, with non-negative times")

    base_pv = curve.present_value(flows, times)
    knot_tenors = curve.tenors[1:]
    zero_rates = np.asarray(curve.zero_rate(knot_tenors), dtype=float)
    bump = bump_bps / 10_000.0
    values = []
    for position in range(knot_tenors.size):
        bumped_rates = zero_rates.copy()
        bumped_rates[position] += bump
        bumped_curve = DiscountCurve.from_zero_rates(
            knot_tenors,
            bumped_rates,
            interpolation=curve.interpolation,
            name=curve.name,
            valuation_date=curve.valuation_date,
            day_count=curve.day_count,
        )
        values.append(base_pv - bumped_curve.present_value(flows, times))
    return pd.Series(values, index=knot_tenors, name="dv01").rename_axis("key_tenor")


def key_rate_dv01_dates(
    curve: DiscountCurve,
    cash_flows,
    payment_dates,
    *,
    bump_bps: float = 1.0,
) -> pd.Series:
    """Date-based wrapper around :func:`key_rate_dv01`."""
    return key_rate_dv01(
        curve,
        cash_flows,
        curve.time_from_date(payment_dates),
        bump_bps=bump_bps,
    )


def price_fixed_rate_bond_dates(
    curve: DiscountCurve,
    effective_date,
    maturity_date,
    coupon_rate: float,
    *,
    face_value: float = 100.0,
    payment_frequency: int = 2,
    fixed_day_count: str = "30/360",
    calendar: BusinessCalendar | None = None,
    business_day_convention: str = "modified_following",
) -> float:
    """Price a fixed-rate bond from a dated schedule and discount curve."""
    if face_value <= 0:
        raise ValueError("face_value must be positive")
    schedule = generate_schedule(
        effective_date,
        maturity_date,
        payment_frequency=payment_frequency,
        day_count=fixed_day_count,
        calendar=calendar,
        business_day_convention=business_day_convention,
    )
    cash_flows = face_value * float(coupon_rate) * schedule["accrual_factor"].to_numpy(dtype=float)
    cash_flows[-1] += face_value
    return curve.present_value(cash_flows, curve.time_from_date(schedule["payment_date"]))
