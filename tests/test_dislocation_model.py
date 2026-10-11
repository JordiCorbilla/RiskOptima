from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from riskoptima import __version__

from riskoptima.dislocation import (
    DislocationConfig,
    MarketSnapshot,
    analyze_snapshot,
    generate_markdown_report,
    scan_universe,
)


def _history(start: float = 100.0, end: float = 82.0, n: int = 400) -> pd.DataFrame:
    trend = np.linspace(start, end, n)
    cycle = 2.0 * np.sin(np.linspace(0.0, 8.0 * np.pi, n))
    index = pd.bdate_range("2025-01-01", periods=n)
    return pd.DataFrame({"Close": trend + cycle}, index=index)


def test_fund_uses_market_dislocation_not_corporate_dcf():
    snapshot = MarketSnapshot(
        symbol="FUND.L",
        name="Example Fund",
        history=_history(),
        info={"quoteType": "MUTUALFUND", "currency": "GBP"},
    )
    report = analyze_snapshot(snapshot, config=DislocationConfig(use_markov_regime=False))

    assert report.instrument_type == "MUTUAL_FUND"
    assert report.valuation_mode == "market_dislocation"
    assert report.intrinsic_value is None
    assert report.quality_score is None
    assert 0 <= report.opportunity_score <= 100
    assert "does not fabricate" in report.narrative


def test_equity_uses_fundamentals_and_produces_intrinsic_value():
    snapshot = MarketSnapshot(
        symbol="GOOD",
        name="Good Business",
        history=_history(start=80.0, end=100.0),
        info={
            "quoteType": "EQUITY",
            "currency": "USD",
            "marketCap": 100_000_000_000,
            "freeCashflow": 7_500_000_000,
            "sharesOutstanding": 1_000_000_000,
            "returnOnEquity": 0.28,
            "returnOnAssets": 0.12,
            "profitMargins": 0.22,
            "operatingMargins": 0.27,
            "debtToEquity": 35.0,
            "revenueGrowth": 0.08,
            "earningsGrowth": 0.10,
            "trailingPE": 18.0,
            "forwardPE": 16.0,
            "enterpriseToEbitda": 12.0,
            "priceToBook": 4.0,
            "longName": "Good Business",
        },
    )
    report = analyze_snapshot(
        snapshot,
        config=DislocationConfig(use_markov_regime=False, valuation_horizon_years=10),
    )

    assert report.instrument_type == "EQUITY"
    assert report.valuation_mode == "fundamental"
    assert report.quality_score is not None
    assert report.valuation_score is not None
    assert report.intrinsic_value is not None
    assert report.margin_of_safety is not None
    assert report.expected_annualized_return is not None
    assert report.valuation_horizon_years == 10
    markdown = generate_markdown_report(report)
    assert "Implied 10Y annualized return to model value" in markdown
    assert 0 <= report.confidence_score <= 100


def test_report_is_generated_and_llm_adapter_is_optional():
    snapshot = MarketSnapshot(
        symbol="FUND.L",
        name="Example Fund",
        history=_history(),
        info={"quoteType": "MUTUALFUND", "currency": "GBP"},
    )
    analysis = analyze_snapshot(snapshot, config=DislocationConfig(use_markov_regime=False))
    markdown = generate_markdown_report(analysis)
    generated = generate_markdown_report(
        analysis,
        narrative_generator=lambda context: f"Generated narrative for {context['symbol']}.",
    )

    assert "# RiskOptima Dislocation Report" in markdown
    assert "Generated interpretation" in markdown
    assert "Generated narrative for FUND.L." in generated


class _Provider:
    def get_snapshot(self, symbol: str, period: str = "5y") -> MarketSnapshot:
        if symbol == "BAD":
            raise RuntimeError("unavailable")
        return MarketSnapshot(
            symbol=symbol,
            name=symbol,
            history=_history(),
            info={"quoteType": "MUTUALFUND", "currency": "GBP"},
        )


def test_scanner_is_resilient_to_one_bad_ticker():
    summary, reports = scan_universe(
        [("OK", "Available"), ("BAD", "Unavailable")],
        provider=_Provider(),
        config=DislocationConfig(use_markov_regime=False),
    )

    assert len(summary) == 2
    assert len(reports) == 1
    bad = summary.loc[summary["symbol"] == "BAD"].iloc[0]
    assert bad["classification"] == "ERROR"
    assert "unavailable" in bad["error"]


def test_custom_narrative_generator_is_used_at_analysis_time():
    snapshot = MarketSnapshot(
        symbol="X",
        history=_history(),
        info={"quoteType": "MUTUALFUND"},
    )
    report = analyze_snapshot(
        snapshot,
        config=DislocationConfig(use_markov_regime=False),
        narrative_generator=lambda context: f"score={context['opportunity_score']:.1f}",
    )
    assert report.narrative.startswith("score=")


@pytest.mark.parametrize("field,value", [
    ("valuation_horizon_years", 0), ("valuation_horizon_years", 1.5),
    ("valuation_horizon_years", True), ("min_observations", 1),
    ("trend_window", 0), ("zscore_window", -1), ("volatility_window", 1),
    ("discount_rate", 0), ("discount_rate", float("nan")),
    ("discount_rate", "0.09"), ("terminal_growth", 0.09),
    ("terminal_growth", -1), ("max_growth", float("inf")),
    ("min_growth", -1), ("min_growth", 0.2),
    ("history_period", ""), ("random_state", -1), ("use_markov_regime", "False"),
])
def test_invalid_configuration_fails_early(field, value):
    with pytest.raises(ValueError):
        DislocationConfig(**{field: value})


@pytest.mark.parametrize("value", [-10.0, 0.0])
def test_nonpositive_multiples_are_not_rewarded(value):
    analysis = analyze_snapshot(
        MarketSnapshot("DISTRESSED", _history(), {
            "quoteType": "EQUITY", "forwardPE": value,
            "enterpriseToEbitda": value, "priceToBook": value,
        }), config=DislocationConfig(use_markov_regime=False),
    )
    assert analysis.valuation_score == 0
    assert len([d for d in analysis.diagnostics if "non-positive" in d]) == 3


def test_dcf_known_value_margin_and_compatibility_payload():
    # Zero growth makes the DCF a perpetuity: annual FCF / discount rate.
    analysis = analyze_snapshot(
        MarketSnapshot("PERPETUITY", _history(100, 100), {
            "quoteType": "EQUITY", "freeCashflow": 20, "sharesOutstanding": 1,
            "revenueGrowth": 0,
        }), config=DislocationConfig(use_markov_regime=False, discount_rate=0.1, terminal_growth=0),
    )
    assert analysis.intrinsic_value == pytest.approx(200)
    assert analysis.margin_of_safety == pytest.approx(0.5)
    assert analysis.upside_to_model_value == pytest.approx(1.0)
    assert analysis.implied_annualized_return == pytest.approx(2**(1/5) - 1)
    assert analysis.expected_annualized_return == analysis.implied_annualized_return
    payload = analysis.as_dict()
    assert payload["riskoptima_version"] == __version__
    assert payload["upside_to_model_value"] == pytest.approx(1)
    json.dumps(payload, allow_nan=False)
    markdown = generate_markdown_report(analysis)
    assert "| Margin of safety | 50.0% |" in markdown
    assert "| Upside to model value | 100.0% |" in markdown
    assert "not reconciled FCFE" in markdown
    assert "not an expected-return forecast" in markdown
    assert markdown.count(f"RiskOptima v{__version__}") == 1


def test_missing_growth_is_clipped_to_configured_bounds():
    analysis = analyze_snapshot(
        MarketSnapshot("BOUNDS", _history(), {
            "quoteType": "EQUITY", "freeCashflow": 20, "sharesOutstanding": 1,
        }), config=DislocationConfig(use_markov_regime=False, min_growth=0, max_growth=0.01),
    )
    assert analysis.metrics["dcf_growth"] == 0.01


def test_nonfinite_fundamentals_do_not_inflate_coverage():
    base = {"quoteType": "EQUITY", "currency": "USD"}
    clean = analyze_snapshot(MarketSnapshot("X", _history(), base), config=DislocationConfig(use_markov_regime=False))
    missing = dict(base, marketCap=np.nan, freeCashflow=np.inf, sharesOutstanding=np.nan,
                   returnOnEquity=np.nan, profitMargins=np.nan, forwardPE=np.nan, enterpriseToEbitda=np.inf)
    invalid = analyze_snapshot(MarketSnapshot("X", _history(), missing), config=DislocationConfig(use_markov_regime=False))
    assert invalid.confidence_score == clean.confidence_score
    json.dumps(invalid.as_dict(), allow_nan=False)


def test_reports_do_not_claim_nav_measurement_or_fixed_trend_window():
    report = analyze_snapshot(MarketSnapshot("FUND", _history(), {"quoteType": "MUTUALFUND"}),
                              config=DislocationConfig(use_markov_regime=False, trend_window=50))
    markdown = generate_markdown_report(report)
    assert "50D trend deviation" in markdown
    assert "no NAV premium or discount is measured" in markdown
    assert "market/NAV-price" not in markdown
    assert "Data coverage" in markdown


def test_history_normalization_and_small_volatility_window():
    history = _history()
    history.columns = pd.MultiIndex.from_tuples([("Close", "X")])
    history = pd.concat([history, history.tail(1)])
    result = analyze_snapshot(MarketSnapshot("X", history, {"quoteType": "ETF"}),
                              config=DislocationConfig(use_markov_regime=False, volatility_window=2))
    assert result.as_of == history.index[-1]
    assert np.isfinite(result.opportunity_score)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        analyze_snapshot(MarketSnapshot("X", history.reset_index(drop=True)))


@pytest.mark.parametrize("cash_flow,shares", [(0, 1), (-20, 1), (20, 0), (np.nan, 1)])
def test_unavailable_dcf_remains_null(cash_flow, shares):
    analysis = analyze_snapshot(MarketSnapshot("X", _history(), {
        "quoteType": "EQUITY", "freeCashflow": cash_flow, "sharesOutstanding": shares,
    }), config=DislocationConfig(use_markov_regime=False))
    assert analysis.intrinsic_value is None
    assert analysis.margin_of_safety is None
    assert analysis.upside_to_model_value is None
    assert analysis.implied_annualized_return is None
    assert any("DCF unavailable" in d for d in analysis.diagnostics)
    json.dumps(analysis.as_dict(), allow_nan=False)
