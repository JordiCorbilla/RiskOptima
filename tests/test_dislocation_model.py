from __future__ import annotations

import numpy as np
import pandas as pd

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
