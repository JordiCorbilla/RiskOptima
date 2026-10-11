###############################################################################
#                              engine.py
###############################################################################
# Product: RiskOptima
# Description: Quality, valuation, and market-dislocation analysis
###############################################################################

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd


Number = int | float | np.number
NarrativeGenerator = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class DislocationConfig:
    """Configuration for the dislocation model."""

    history_period: str = "5y"
    min_observations: int = 126
    trend_window: int = 200
    zscore_window: int = 126
    volatility_window: int = 63
    valuation_horizon_years: int = 5
    discount_rate: float = 0.09
    terminal_growth: float = 0.025
    max_growth: float = 0.12
    min_growth: float = -0.05
    use_markov_regime: bool = True
    random_state: int = 7


@dataclass
class MarketSnapshot:
    """Market and metadata snapshot supplied by a data provider."""

    symbol: str
    history: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)
    name: str | None = None


class MarketDataProvider(Protocol):
    """Protocol for injectable market-data providers."""

    def get_snapshot(self, symbol: str, period: str = "5y") -> MarketSnapshot:
        ...


@dataclass
class DislocationAnalysis:
    """Complete model output for one instrument."""

    symbol: str
    name: str
    instrument_type: str
    valuation_mode: str
    currency: str | None
    price: float
    as_of: pd.Timestamp
    opportunity_score: float
    dislocation_score: float
    regime_score: float
    quality_score: float | None
    valuation_score: float | None
    confidence_score: float
    classification: str
    metrics: dict[str, float | str | None]
    intrinsic_value: float | None = None
    margin_of_safety: float | None = None
    expected_5y_irr: float | None = None
    narrative: str = ""
    diagnostics: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["as_of"] = str(self.as_of)
        result["generated_at"] = self.generated_at.isoformat()
        return result


class YahooFinanceProvider:
    """yfinance-backed market-data provider.

    The import is intentionally lazy so the analytical core remains testable
    without a network connection or a yfinance session.
    """

    def get_snapshot(self, symbol: str, period: str = "5y") -> MarketSnapshot:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period, auto_adjust=True)
        info: dict[str, Any]
        try:
            info = dict(ticker.info or {})
        except Exception:
            info = {}
        return MarketSnapshot(
            symbol=symbol,
            history=history,
            info=info,
            name=info.get("longName") or info.get("shortName"),
        )


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _clip(value: Number, low: float = 0.0, high: float = 100.0) -> float:
    return float(np.clip(float(value), low, high))


def _linear_score(value: float | None, bad: float, good: float) -> float | None:
    if value is None:
        return None
    if good == bad:
        return 50.0
    return _clip(100.0 * (value - bad) / (good - bad))


def _weighted_available(items: Sequence[tuple[float | None, float]]) -> float | None:
    available = [(v, w) for v, w in items if v is not None and np.isfinite(v)]
    if not available:
        return None
    total = sum(w for _, w in available)
    if total <= 0:
        return None
    return float(sum(float(v) * w for v, w in available) / total)


def _normalize_history(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if history is None or history.empty:
        raise ValueError(f"No price history returned for {symbol}")
    frame = history.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(-1)
    close_column = next((c for c in ("Close", "Adj Close", "close", "adjclose") if c in frame.columns), None)
    if close_column is None:
        raise ValueError(f"Price history for {symbol} does not contain a Close column")
    frame = frame.rename(columns={close_column: "Close"})
    frame["Close"] = pd.to_numeric(frame["Close"], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    frame = frame[frame["Close"] > 0].sort_index()
    if frame.empty:
        raise ValueError(f"No finite positive closes returned for {symbol}")
    return frame


def _instrument_type(info: Mapping[str, Any]) -> str:
    quote_type = str(info.get("quoteType") or info.get("typeDisp") or "").upper().replace(" ", "")
    if "MUTUAL" in quote_type:
        return "MUTUAL_FUND"
    if quote_type == "ETF" or "ETF" in quote_type:
        return "ETF"
    if quote_type in {"EQUITY", "STOCK"}:
        return "EQUITY"
    if any(info.get(key) is not None for key in ("marketCap", "sharesOutstanding", "freeCashflow", "trailingPE")):
        return "EQUITY"
    return quote_type or "FUND_OR_OTHER"


def _market_metrics(prices: pd.Series, config: DislocationConfig) -> dict[str, float]:
    prices = prices.astype(float).dropna()
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    price = float(prices.iloc[-1])

    one_year = prices.tail(min(252, len(prices)))
    peak_1y = float(one_year.max())
    drawdown_1y = price / peak_1y - 1.0

    all_peak = float(prices.max())
    drawdown_period = price / all_peak - 1.0

    trend_window = min(config.trend_window, len(prices))
    sma = float(prices.tail(trend_window).mean())
    sma_deviation = price / sma - 1.0 if sma else 0.0

    zwindow = prices.tail(min(config.zscore_window, len(prices)))
    zstd = float(zwindow.std(ddof=0))
    price_zscore = (price - float(zwindow.mean())) / zstd if zstd > 1e-12 else 0.0

    ret_3m = price / float(prices.iloc[-min(63, len(prices))]) - 1.0 if len(prices) > 1 else 0.0
    ret_12m = price / float(prices.iloc[-min(252, len(prices))]) - 1.0 if len(prices) > 1 else 0.0

    vol_window = min(config.volatility_window, len(returns))
    recent_vol = float(returns.tail(vol_window).std(ddof=1) * np.sqrt(252)) if vol_window >= 2 else 0.0

    if len(returns) >= max(20, vol_window):
        rolling_vol = returns.rolling(vol_window, min_periods=max(10, vol_window // 2)).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()
        if not rolling_vol.empty:
            vol_percentile = float((rolling_vol <= recent_vol).mean())
        else:
            vol_percentile = 0.5
    else:
        vol_percentile = 0.5

    downside = returns[returns < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(252)) if len(downside) >= 2 else 0.0

    return {
        "price": price,
        "drawdown_1y": drawdown_1y,
        "drawdown_period": drawdown_period,
        "sma_deviation": sma_deviation,
        "price_zscore": price_zscore,
        "return_3m": ret_3m,
        "return_12m": ret_12m,
        "annualized_volatility": recent_vol,
        "volatility_percentile": vol_percentile,
        "downside_volatility": downside_vol,
    }


def _dislocation_score(metrics: Mapping[str, float]) -> float:
    drawdown = _linear_score(-metrics["drawdown_1y"], bad=0.0, good=0.35) or 0.0
    period_drawdown = _linear_score(-metrics["drawdown_period"], bad=0.0, good=0.45) or 0.0
    zscore = _linear_score(-metrics["price_zscore"], bad=-1.0, good=2.0) or 0.0
    trend_gap = _linear_score(-metrics["sma_deviation"], bad=-0.10, good=0.25) or 0.0
    vol = _linear_score(metrics["volatility_percentile"], bad=0.20, good=0.95) or 0.0
    return _clip(0.30 * drawdown + 0.15 * period_drawdown + 0.25 * zscore + 0.20 * trend_gap + 0.10 * vol)


def _regime_score(prices: pd.Series, config: DislocationConfig, diagnostics: list[str]) -> tuple[float, str]:
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if len(returns) < 30:
        return 50.0, "insufficient_history"

    if config.use_markov_regime:
        try:
            from riskoptima.risk.markov_regime import fit_markov_regime_model

            tail = returns.tail(min(756, len(returns)))
            model = fit_markov_regime_model(tail, n_regimes=3, random_state=config.random_state)
            current = int(model.regimes.iloc[-1])
            probs = model.regime_probabilities.iloc[-1].to_numpy(dtype=float)
            score = float(100.0 * (probs[0] + 0.50 * probs[1]))
            label = ("weak", "neutral", "strong")[current]
            return _clip(score), label
        except Exception as exc:
            diagnostics.append(f"Markov regime fallback used: {exc}")

    vol = float(returns.tail(min(63, len(returns))).std(ddof=1) * np.sqrt(252))
    momentum = float(prices.iloc[-1] / prices.iloc[-min(63, len(prices))] - 1.0)
    score = _clip(50.0 + 120.0 * vol - 100.0 * momentum)
    label = "weak" if momentum < -0.08 else "strong" if momentum > 0.08 else "neutral"
    return score, label


def _quality_score(info: Mapping[str, Any]) -> tuple[float | None, dict[str, float | None]]:
    roe = _finite(info.get("returnOnEquity"))
    roa = _finite(info.get("returnOnAssets"))
    margin = _finite(info.get("profitMargins"))
    operating_margin = _finite(info.get("operatingMargins"))
    debt_to_equity = _finite(info.get("debtToEquity"))
    revenue_growth = _finite(info.get("revenueGrowth"))
    earnings_growth = _finite(info.get("earningsGrowth"))

    leverage = None if debt_to_equity is None else debt_to_equity / 100.0

    components = {
        "roe_score": _linear_score(roe, bad=0.00, good=0.30),
        "roa_score": _linear_score(roa, bad=0.00, good=0.15),
        "profit_margin_score": _linear_score(margin, bad=0.00, good=0.25),
        "operating_margin_score": _linear_score(operating_margin, bad=0.00, good=0.30),
        "leverage_score": None if leverage is None else _linear_score(-leverage, bad=-2.0, good=0.0),
        "revenue_growth_score": _linear_score(revenue_growth, bad=-0.05, good=0.20),
        "earnings_growth_score": _linear_score(earnings_growth, bad=-0.10, good=0.25),
    }
    score = _weighted_available(
        [
            (components["roe_score"], 0.20),
            (components["roa_score"], 0.10),
            (components["profit_margin_score"], 0.20),
            (components["operating_margin_score"], 0.15),
            (components["leverage_score"], 0.15),
            (components["revenue_growth_score"], 0.10),
            (components["earnings_growth_score"], 0.10),
        ]
    )
    return score, components


def _valuation_score(info: Mapping[str, Any]) -> tuple[float | None, dict[str, float | None]]:
    market_cap = _finite(info.get("marketCap"))
    fcf = _finite(info.get("freeCashflow"))
    trailing_pe = _finite(info.get("trailingPE"))
    forward_pe = _finite(info.get("forwardPE"))
    ev_to_ebitda = _finite(info.get("enterpriseToEbitda"))
    price_to_book = _finite(info.get("priceToBook"))

    fcf_yield = fcf / market_cap if fcf is not None and market_cap and market_cap > 0 else None
    earnings_yield = 1.0 / trailing_pe if trailing_pe is not None and trailing_pe > 0 else None

    scores = {
        "fcf_yield": fcf_yield,
        "earnings_yield": earnings_yield,
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "ev_to_ebitda": ev_to_ebitda,
        "price_to_book": price_to_book,
        "fcf_yield_score": _linear_score(fcf_yield, bad=0.015, good=0.09),
        "earnings_yield_score": _linear_score(earnings_yield, bad=0.025, good=0.10),
        "forward_pe_score": None if forward_pe is None else _linear_score(-forward_pe, bad=-40.0, good=-8.0),
        "ev_to_ebitda_score": None if ev_to_ebitda is None else _linear_score(-ev_to_ebitda, bad=-30.0, good=-6.0),
        "price_to_book_score": None if price_to_book is None else _linear_score(-price_to_book, bad=-10.0, good=-1.0),
    }
    score = _weighted_available(
        [
            (scores["fcf_yield_score"], 0.35),
            (scores["earnings_yield_score"], 0.20),
            (scores["forward_pe_score"], 0.20),
            (scores["ev_to_ebitda_score"], 0.15),
            (scores["price_to_book_score"], 0.10),
        ]
    )
    return score, scores


def _select_growth(info: Mapping[str, Any], config: DislocationConfig) -> float:
    candidates = [_finite(info.get("revenueGrowth")), _finite(info.get("earningsGrowth"))]
    valid = [x for x in candidates if x is not None]
    if not valid:
        return 0.04
    return float(np.clip(np.median(valid), config.min_growth, config.max_growth))


def _equity_dcf(
    info: Mapping[str, Any],
    price: float,
    config: DislocationConfig,
) -> tuple[float | None, float | None, float | None, dict[str, float | None]]:
    fcf = _finite(info.get("freeCashflow"))
    shares = _finite(info.get("sharesOutstanding"))
    if fcf is None or fcf <= 0 or shares is None or shares <= 0:
        return None, None, None, {"dcf_growth": None, "dcf_discount_rate": None, "dcf_terminal_growth": None}

    discount = config.discount_rate
    terminal = min(config.terminal_growth, discount - 0.01)
    growth = _select_growth(info, config)
    projected = fcf
    pv = 0.0
    for year in range(1, config.valuation_horizon_years + 1):
        projected *= 1.0 + growth
        pv += projected / ((1.0 + discount) ** year)
    terminal_value = projected * (1.0 + terminal) / (discount - terminal)
    pv += terminal_value / ((1.0 + discount) ** config.valuation_horizon_years)
    intrinsic = pv / shares
    if not np.isfinite(intrinsic) or intrinsic <= 0:
        return None, None, None, {"dcf_growth": growth, "dcf_discount_rate": discount, "dcf_terminal_growth": terminal}

    margin = intrinsic / price - 1.0
    expected_irr = (intrinsic / price) ** (1.0 / config.valuation_horizon_years) - 1.0 if price > 0 else None
    return float(intrinsic), float(margin), float(expected_irr), {
        "dcf_growth": growth,
        "dcf_discount_rate": discount,
        "dcf_terminal_growth": terminal,
    }


def _confidence(
    n_obs: int,
    info: Mapping[str, Any],
    is_equity: bool,
    intrinsic_value: float | None,
) -> float:
    history_score = _clip(100.0 * n_obs / (252.0 * 3.0))
    metadata_keys = ("quoteType", "currency", "longName", "shortName")
    metadata_score = 100.0 * sum(info.get(k) is not None for k in metadata_keys) / len(metadata_keys)
    if not is_equity:
        return _clip(0.80 * history_score + 0.20 * metadata_score)

    fundamental_keys = (
        "marketCap",
        "freeCashflow",
        "sharesOutstanding",
        "returnOnEquity",
        "profitMargins",
        "forwardPE",
        "enterpriseToEbitda",
    )
    fundamental_score = 100.0 * sum(info.get(k) is not None for k in fundamental_keys) / len(fundamental_keys)
    dcf_score = 100.0 if intrinsic_value is not None else 0.0
    return _clip(0.35 * history_score + 0.15 * metadata_score + 0.40 * fundamental_score + 0.10 * dcf_score)


def _classify(score: float, confidence: float) -> str:
    if confidence < 30:
        return "LOW_CONFIDENCE"
    if score >= 85:
        return "EXCEPTIONAL_DISLOCATION"
    if score >= 72:
        return "ATTRACTIVE"
    if score >= 60:
        return "WATCH"
    if score >= 42:
        return "NEUTRAL"
    return "EXTENDED"


def _default_narrative(analysis: DislocationAnalysis) -> str:
    parts = [
        f"{analysis.name} ({analysis.symbol}) scores {analysis.opportunity_score:.1f}/100 and is classified as "
        f"{analysis.classification.replace('_', ' ').lower()}."
    ]
    if analysis.valuation_mode == "fundamental":
        if analysis.quality_score is not None and analysis.valuation_score is not None:
            parts.append(
                f"Fundamental quality is {analysis.quality_score:.1f}/100 while the valuation score is "
                f"{analysis.valuation_score:.1f}/100."
            )
        if analysis.intrinsic_value is not None and analysis.margin_of_safety is not None:
            direction = "above" if analysis.margin_of_safety >= 0 else "below"
            parts.append(
                f"The model's conservative FCFE valuation is {analysis.intrinsic_value:.2f}, "
                f"{abs(analysis.margin_of_safety):.1%} {direction} the observed price."
            )
    else:
        parts.append(
            "This instrument is treated as a fund/pooled vehicle, so RiskOptima does not fabricate a "
            "company-style intrinsic value; the signal is based on price, drawdown, volatility and regime dislocation."
        )
    parts.append(
        f"The market-dislocation component is {analysis.dislocation_score:.1f}/100 and the regime-opportunity "
        f"component is {analysis.regime_score:.1f}/100. Model confidence is {analysis.confidence_score:.1f}/100."
    )
    if analysis.confidence_score < 55:
        parts.append("The result should be treated cautiously because data coverage is incomplete.")
    return " ".join(parts)


def analyze_snapshot(
    snapshot: MarketSnapshot,
    *,
    config: DislocationConfig | None = None,
    narrative_generator: NarrativeGenerator | None = None,
) -> DislocationAnalysis:
    """Analyze an already-fetched market snapshot without network calls."""

    config = config or DislocationConfig()
    history = _normalize_history(snapshot.history, snapshot.symbol)
    if len(history) < config.min_observations:
        raise ValueError(
            f"{snapshot.symbol} has {len(history)} observations; at least {config.min_observations} are required"
        )

    info = dict(snapshot.info or {})
    diagnostics: list[str] = []
    instrument_type = _instrument_type(info)
    is_equity = instrument_type == "EQUITY"
    metrics = _market_metrics(history["Close"], config)
    price = metrics["price"]

    dislocation = _dislocation_score(metrics)
    regime, regime_label = _regime_score(history["Close"], config, diagnostics)
    quality: float | None = None
    valuation: float | None = None
    intrinsic: float | None = None
    margin: float | None = None
    expected_irr: float | None = None

    metric_payload: dict[str, float | str | None] = dict(metrics)
    metric_payload["regime"] = regime_label

    if is_equity:
        quality, quality_components = _quality_score(info)
        valuation, valuation_components = _valuation_score(info)
        intrinsic, margin, expected_irr, dcf_metrics = _equity_dcf(info, price, config)
        metric_payload.update(quality_components)
        metric_payload.update(valuation_components)
        metric_payload.update(dcf_metrics)

        intrinsic_score = None if margin is None else _linear_score(margin, bad=-0.25, good=0.50)
        opportunity = _weighted_available(
            [
                (valuation, 0.30),
                (quality, 0.25),
                (dislocation, 0.25),
                (regime, 0.10),
                (intrinsic_score, 0.10),
            ]
        )
        valuation_mode = "fundamental"
    else:
        trend_health = _linear_score(metrics["return_12m"], bad=-0.30, good=0.20)
        opportunity = _weighted_available(
            [
                (dislocation, 0.70),
                (regime, 0.20),
                (trend_health, 0.10),
            ]
        )
        valuation_mode = "market_dislocation"
        diagnostics.append("Corporate intrinsic valuation skipped for non-equity instrument.")

    opportunity = float(opportunity if opportunity is not None else 50.0)
    confidence = _confidence(len(history), info, is_equity, intrinsic)
    classification = _classify(opportunity, confidence)
    currency = str(info.get("currency")) if info.get("currency") else None
    name = snapshot.name or info.get("longName") or info.get("shortName") or snapshot.symbol

    analysis = DislocationAnalysis(
        symbol=snapshot.symbol,
        name=str(name),
        instrument_type=instrument_type,
        valuation_mode=valuation_mode,
        currency=currency,
        price=price,
        as_of=pd.Timestamp(history.index[-1]),
        opportunity_score=_clip(opportunity),
        dislocation_score=dislocation,
        regime_score=regime,
        quality_score=quality,
        valuation_score=valuation,
        confidence_score=confidence,
        classification=classification,
        metrics=metric_payload,
        intrinsic_value=intrinsic,
        margin_of_safety=margin,
        expected_5y_irr=expected_irr,
        diagnostics=diagnostics,
    )
    analysis.narrative = (
        str(narrative_generator(analysis.as_dict()))
        if narrative_generator is not None
        else _default_narrative(analysis)
    )
    return analysis


def analyze_symbol(
    symbol: str,
    *,
    name: str | None = None,
    provider: MarketDataProvider | None = None,
    config: DislocationConfig | None = None,
    narrative_generator: NarrativeGenerator | None = None,
) -> DislocationAnalysis:
    """Fetch and analyze one ticker symbol."""

    config = config or DislocationConfig()
    provider = provider or YahooFinanceProvider()
    snapshot = provider.get_snapshot(symbol, period=config.history_period)
    if name:
        snapshot.name = name
    return analyze_snapshot(snapshot, config=config, narrative_generator=narrative_generator)


def scan_universe(
    instruments: Iterable[tuple[str, str] | Mapping[str, str]],
    *,
    provider: MarketDataProvider | None = None,
    config: DislocationConfig | None = None,
    narrative_generator: NarrativeGenerator | None = None,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, list[DislocationAnalysis]]:
    """Analyze a universe and return a sortable summary plus full reports."""

    config = config or DislocationConfig()
    provider = provider or YahooFinanceProvider()
    reports: list[DislocationAnalysis] = []
    rows: list[dict[str, Any]] = []

    for item in instruments:
        if isinstance(item, Mapping):
            symbol = str(item["symbol"])
            name = str(item.get("name") or symbol)
        else:
            symbol, name = str(item[0]), str(item[1])
        try:
            report = analyze_symbol(
                symbol,
                name=name,
                provider=provider,
                config=config,
                narrative_generator=narrative_generator,
            )
            reports.append(report)
            rows.append(
                {
                    "symbol": report.symbol,
                    "name": report.name,
                    "instrument_type": report.instrument_type,
                    "price": report.price,
                    "currency": report.currency,
                    "opportunity_score": report.opportunity_score,
                    "dislocation_score": report.dislocation_score,
                    "quality_score": report.quality_score,
                    "valuation_score": report.valuation_score,
                    "confidence_score": report.confidence_score,
                    "classification": report.classification,
                    "valuation_mode": report.valuation_mode,
                    "error": None,
                }
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "instrument_type": None,
                    "price": np.nan,
                    "currency": None,
                    "opportunity_score": np.nan,
                    "dislocation_score": np.nan,
                    "quality_score": np.nan,
                    "valuation_score": np.nan,
                    "confidence_score": 0.0,
                    "classification": "ERROR",
                    "valuation_mode": None,
                    "error": str(exc),
                }
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["opportunity_score", "confidence_score"],
            ascending=[False, False],
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
    return summary, reports
