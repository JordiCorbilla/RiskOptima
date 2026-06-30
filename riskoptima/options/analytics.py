###############################################################################
#                               analytics.py
###############################################################################
# Product: RiskOptima
# Author: Jordi Corbilla
# Description: Professional options analytics layer
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal, Optional

import numpy as np
import pandas as pd

from .black_scholes import black_scholes_price
from .greeks import black_scholes_greeks
from .implied_vol import implied_volatility as solve_implied_volatility

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionContract:
    underlying: str
    strike: float
    expiry: date | pd.Timestamp
    option_type: OptionType
    symbol: Optional[str] = None
    quantity: float = 1.0
    premium: Optional[float] = None
    multiplier: float = 100.0
    implied_volatility: Optional[float] = None


@dataclass(frozen=True)
class OptionValuationResult:
    contract: OptionContract
    spot: float
    valuation_date: pd.Timestamp
    time_to_expiry: float
    price: float
    intrinsic_value: float
    extrinsic_value: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: float
    notional_delta: float
    market_value: float


def _validate_option_type(option_type: str) -> str:
    value = option_type.lower()
    if value not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return value


def _time_to_expiry(expiry, valuation_date, day_count: float = 365.0) -> float:
    expiry_ts = pd.Timestamp(expiry)
    valuation_ts = pd.Timestamp(valuation_date)
    days = (expiry_ts - valuation_ts).days
    if days < 0:
        raise ValueError("expiry must be on or after valuation_date")
    return max(days / day_count, 0.0)


def _validate_contract(contract: OptionContract):
    _validate_option_type(contract.option_type)
    if contract.strike <= 0:
        raise ValueError("strike must be positive")
    if contract.multiplier <= 0:
        raise ValueError("multiplier must be positive")
    if contract.implied_volatility is not None and contract.implied_volatility <= 0:
        raise ValueError("implied_volatility must be positive")
    if contract.premium is not None and contract.premium < 0:
        raise ValueError("premium must be non-negative")


def value_option_contract(
    contract: OptionContract,
    spot: float,
    valuation_date,
    risk_free_rate: float,
    volatility: Optional[float] = None,
    dividend_yield: float = 0.0,
) -> OptionValuationResult:
    """Value one option contract with Black-Scholes and full Greeks."""
    _validate_contract(contract)
    if spot <= 0:
        raise ValueError("spot must be positive")
    sigma = volatility if volatility is not None else contract.implied_volatility
    if sigma is None or sigma <= 0:
        raise ValueError("volatility or contract.implied_volatility must be positive")
    option_type = _validate_option_type(contract.option_type)
    tte = _time_to_expiry(contract.expiry, valuation_date)
    price = black_scholes_price(spot, contract.strike, tte, risk_free_rate, sigma, option_type=option_type, q=dividend_yield)
    if option_type == "call":
        intrinsic = max(spot - contract.strike, 0.0)
    else:
        intrinsic = max(contract.strike - spot, 0.0)
    greeks = (
        {"delta": 1.0 if option_type == "call" and spot > contract.strike else -1.0 if option_type == "put" and spot < contract.strike else 0.0,
         "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
        if tte <= 0
        else black_scholes_greeks(spot, contract.strike, tte, risk_free_rate, sigma, option_type=option_type, q=dividend_yield)
    )
    market_value = price * contract.quantity * contract.multiplier
    notional_delta = greeks["delta"] * spot * contract.quantity * contract.multiplier
    return OptionValuationResult(
        contract=contract,
        spot=float(spot),
        valuation_date=pd.Timestamp(valuation_date),
        time_to_expiry=float(tte),
        price=float(price),
        intrinsic_value=float(intrinsic),
        extrinsic_value=float(max(price - intrinsic, 0.0)),
        delta=float(greeks["delta"]),
        gamma=float(greeks["gamma"]),
        theta=float(greeks["theta"]),
        vega=float(greeks["vega"]),
        rho=float(greeks["rho"]),
        implied_volatility=float(sigma),
        notional_delta=float(notional_delta),
        market_value=float(market_value),
    )


class OptionBook:
    """Collection of option contracts with valuation and Greek aggregation."""

    def __init__(self, contracts: list[OptionContract]):
        if not contracts:
            raise ValueError("contracts must not be empty")
        self.contracts = list(contracts)
        for contract in self.contracts:
            _validate_contract(contract)

    def value(
        self,
        spots: dict[str, float] | pd.Series,
        valuation_date,
        risk_free_rate: float,
        volatilities: Optional[dict[str, float] | pd.Series] = None,
        dividend_yields: Optional[dict[str, float] | pd.Series] = None,
    ) -> pd.DataFrame:
        rows = []
        spot_series = pd.Series(spots, dtype=float)
        vol_series = pd.Series(volatilities, dtype=float) if volatilities is not None else pd.Series(dtype=float)
        div_series = pd.Series(dividend_yields, dtype=float) if dividend_yields is not None else pd.Series(dtype=float)
        for contract in self.contracts:
            if contract.underlying not in spot_series.index:
                raise ValueError(f"missing spot for underlying: {contract.underlying}")
            sigma = vol_series.get(contract.underlying, contract.implied_volatility)
            result = value_option_contract(
                contract,
                spot=float(spot_series[contract.underlying]),
                valuation_date=valuation_date,
                risk_free_rate=risk_free_rate,
                volatility=sigma,
                dividend_yield=float(div_series.get(contract.underlying, 0.0)),
            )
            rows.append(_valuation_row(result))
        return pd.DataFrame(rows)

    def aggregate(self, valuation_detail: pd.DataFrame, by: Optional[list[str]] = None) -> dict[str, pd.DataFrame | dict]:
        by = by or ["underlying", "expiry", "option_type"]
        greek_cols = ["market_value", "notional_delta", "delta_exposure", "gamma_exposure", "theta_exposure", "vega_exposure", "rho_exposure"]
        total = valuation_detail[greek_cols].sum().to_dict()
        grouped = valuation_detail.groupby(by, dropna=False)[greek_cols].sum()
        return {"total": total, "by_group": grouped}


def _valuation_row(result: OptionValuationResult) -> dict:
    c = result.contract
    return {
        "underlying": c.underlying,
        "symbol": c.symbol,
        "option_type": c.option_type,
        "strike": c.strike,
        "expiry": pd.Timestamp(c.expiry),
        "quantity": c.quantity,
        "multiplier": c.multiplier,
        "spot": result.spot,
        "time_to_expiry": result.time_to_expiry,
        "price": result.price,
        "intrinsic_value": result.intrinsic_value,
        "extrinsic_value": result.extrinsic_value,
        "delta": result.delta,
        "gamma": result.gamma,
        "theta": result.theta,
        "vega": result.vega,
        "rho": result.rho,
        "implied_volatility": result.implied_volatility,
        "market_value": result.market_value,
        "notional_delta": result.notional_delta,
        "delta_exposure": result.delta * c.quantity * c.multiplier,
        "gamma_exposure": result.gamma * c.quantity * c.multiplier,
        "theta_exposure": result.theta * c.quantity * c.multiplier,
        "vega_exposure": result.vega * c.quantity * c.multiplier,
        "rho_exposure": result.rho * c.quantity * c.multiplier,
    }


def price_option_chain(chain: pd.DataFrame, spot: float, valuation_date, risk_free_rate: float, dividend_yield: float = 0.0) -> pd.DataFrame:
    """Price a DataFrame with columns option_type, strike, expiry, and volatility."""
    rows = []
    for row in chain.to_dict("records"):
        contract = OptionContract(
            underlying=row.get("underlying", "UNDERLYING"),
            symbol=row.get("symbol"),
            option_type=row["option_type"],
            strike=float(row["strike"]),
            expiry=row["expiry"],
            quantity=float(row.get("quantity", 1.0)),
            multiplier=float(row.get("multiplier", 100.0)),
            implied_volatility=float(row["volatility"]),
        )
        rows.append(_valuation_row(value_option_contract(contract, spot, valuation_date, risk_free_rate, dividend_yield=dividend_yield)))
    return pd.DataFrame(rows)


def build_greeks_grid(spot: float, strikes, expiries, volatility: float, risk_free_rate: float, valuation_date, option_type: str = "call", dividend_yield: float = 0.0) -> pd.DataFrame:
    rows = []
    for strike in strikes:
        for expiry in expiries:
            contract = OptionContract("UNDERLYING", float(strike), expiry, _validate_option_type(option_type), multiplier=1.0, implied_volatility=volatility)
            rows.append(_valuation_row(value_option_contract(contract, spot, valuation_date, risk_free_rate, dividend_yield=dividend_yield)))
    return pd.DataFrame(rows)


def build_implied_vol_surface(chain: pd.DataFrame, spot: float, valuation_date, risk_free_rate: float, dividend_yield: float = 0.0) -> pd.DataFrame:
    rows = []
    for row in chain.to_dict("records"):
        tte = _time_to_expiry(row["expiry"], valuation_date)
        iv = solve_implied_volatility(
            float(row["market_price"]),
            spot,
            float(row["strike"]),
            tte,
            risk_free_rate,
            option_type=row["option_type"],
            q=dividend_yield,
        )
        rows.append({**row, "time_to_expiry": tte, "implied_volatility": iv})
    return pd.DataFrame(rows)


def option_scenario_grid(
    contract: OptionContract,
    spot: float,
    valuation_date,
    risk_free_rate: float,
    volatility: float,
    spot_shocks=(-0.1, 0.0, 0.1),
    volatility_shocks=(-0.05, 0.0, 0.05),
) -> pd.DataFrame:
    base = value_option_contract(contract, spot, valuation_date, risk_free_rate, volatility)
    rows = []
    for spot_shock in spot_shocks:
        for vol_shock in volatility_shocks:
            shocked_spot = spot * (1.0 + spot_shock)
            shocked_vol = max(volatility + vol_shock, 1e-6)
            shocked = value_option_contract(contract, shocked_spot, valuation_date, risk_free_rate, shocked_vol)
            rows.append({
                "spot_shock": spot_shock,
                "volatility_shock": vol_shock,
                "spot": shocked_spot,
                "volatility": shocked_vol,
                "price": shocked.price,
                "pnl": (shocked.price - base.price) * contract.quantity * contract.multiplier,
            })
    return pd.DataFrame(rows)


def event_straddle_analysis(spot: float, strike: float, expiry, valuation_date, risk_free_rate: float, volatility: float, quantity: float = 1.0, multiplier: float = 100.0) -> dict:
    call = OptionContract("UNDERLYING", strike, expiry, "call", quantity=quantity, multiplier=multiplier, implied_volatility=volatility)
    put = OptionContract("UNDERLYING", strike, expiry, "put", quantity=quantity, multiplier=multiplier, implied_volatility=volatility)
    call_value = value_option_contract(call, spot, valuation_date, risk_free_rate, volatility)
    put_value = value_option_contract(put, spot, valuation_date, risk_free_rate, volatility)
    premium = call_value.price + put_value.price
    return {
        "call_price": call_value.price,
        "put_price": put_value.price,
        "straddle_price": premium,
        "upper_breakeven": strike + premium,
        "lower_breakeven": strike - premium,
        "total_market_value": (call_value.market_value + put_value.market_value),
    }

