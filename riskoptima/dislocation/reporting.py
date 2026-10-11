###############################################################################
#                            reporting.py
###############################################################################
# Product: RiskOptima
# Description: Generative Markdown reporting for dislocation analyses
###############################################################################

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from riskoptima._version import __version__

from .engine import DislocationAnalysis, RESEARCH_DISCLAIMER


NarrativeGenerator = Callable[[Mapping[str, Any]], str]


def _fmt_number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def generate_markdown_report(
    analysis: DislocationAnalysis,
    *,
    narrative_generator: NarrativeGenerator | None = None,
) -> str:
    """Generate a self-contained investment/dislocation note.

    ``narrative_generator`` is an optional adapter for an LLM or other text
    generation service. The core package deliberately has no LLM dependency;
    without an adapter it generates a deterministic narrative from the model
    outputs, making notebooks and CI fully reproducible.
    """

    narrative = analysis.narrative
    if narrative_generator is not None:
        narrative = str(narrative_generator(analysis.as_dict()))

    m = analysis.metrics
    coverage_note = (
        "price history/basic identifiers, not complete fund valuation or a probability of accuracy"
        if analysis.instrument_type in {"MUTUAL_FUND", "ETF"}
        else "not a probability of accuracy"
    )
    lines = [
        f"# RiskOptima Dislocation Report — {analysis.name}",
        "",
        f"**Ticker:** `{analysis.symbol}`  ",
        f"**As of:** {analysis.as_of.date()}  ",
        f"**Instrument type:** {analysis.instrument_type}  ",
        f"**Valuation mode:** {analysis.valuation_mode}  ",
        f"**Price:** {_fmt_number(analysis.price)} {analysis.currency or ''}".rstrip() + "  ",
        "",
        "## Executive signal",
        "",
        f"- **Opportunity score:** {analysis.opportunity_score:.1f}/100",
        f"- **Classification:** {analysis.classification}",
        f"- **Data coverage:** {analysis.confidence_score:.1f}/100 ({coverage_note})",
        f"- **Dislocation score:** {analysis.dislocation_score:.1f}/100",
        f"- **Regime opportunity:** {analysis.regime_score:.1f}/100",
        "",
        "## Generated interpretation",
        "",
        narrative,
        "",
        "## Market dislocation",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| 1Y drawdown | {_fmt_pct(_as_float(m.get('drawdown_1y')))} |",
        f"| Period drawdown | {_fmt_pct(_as_float(m.get('drawdown_period')))} |",
        f"| {int(float(m.get('trend_window') or 200))}D trend deviation | {_fmt_pct(_as_float(m.get('sma_deviation')))} |",
        f"| Price z-score | {_fmt_number(_as_float(m.get('price_zscore')))} |",
        f"| 3M return | {_fmt_pct(_as_float(m.get('return_3m')))} |",
        f"| 12M return | {_fmt_pct(_as_float(m.get('return_12m')))} |",
        f"| Annualized volatility | {_fmt_pct(_as_float(m.get('annualized_volatility')))} |",
        f"| Volatility percentile | {_fmt_pct(_as_float(m.get('volatility_percentile')))} |",
        f"| Regime | {m.get('regime', 'n/a')} |",
        "",
    ]

    if analysis.valuation_mode == "fundamental":
        lines.extend(
            [
                "## Fundamental valuation",
                "",
                "| Metric | Value |",
                "|---|---:|",
                f"| Quality score | {_fmt_number(analysis.quality_score, 1)}/100 |",
                f"| Valuation score | {_fmt_number(analysis.valuation_score, 1)}/100 |",
                f"| FCF yield | {_fmt_pct(_as_float(m.get('fcf_yield')))} |",
                f"| Earnings yield | {_fmt_pct(_as_float(m.get('earnings_yield')))} |",
                f"| Forward P/E | {_fmt_number(_as_float(m.get('forward_pe')))} |",
                f"| EV / EBITDA | {_fmt_number(_as_float(m.get('ev_to_ebitda')))} |",
                f"| Intrinsic value | {_fmt_number(analysis.intrinsic_value)} |",
                f"| Margin of safety | {_fmt_pct(analysis.margin_of_safety)} |",
                f"| Upside to model value | {_fmt_pct(analysis.upside_to_model_value)} |",
                f"| Implied {analysis.valuation_horizon_years}Y annualized return to model value | "
                f"{_fmt_pct(analysis.expected_annualized_return)} |",
                f"| DCF growth assumption | {_fmt_pct(_as_float(m.get('dcf_growth')))} |",
                f"| DCF discount rate | {_fmt_pct(_as_float(m.get('dcf_discount_rate')))} |",
                f"| Terminal growth | {_fmt_pct(_as_float(m.get('dcf_terminal_growth')))} |",
                "",
                "> The intrinsic-value estimate is a scenario model, not a precise fair-price claim. "
                "It discounts an unadjusted FCF proxy, not reconciled FCFE; net borrowing is not modeled. "
                "Margin of safety is (model value - price) / model value. Upside uses price as the denominator. "
                "The annualized figure assumes convergence to today's model value over the stated horizon, "
                "excludes distributions, and is not an expected-return forecast. "
                "Sensitivity to cash flow, growth and discount rates is material.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Valuation treatment",
                "",
                "This instrument is not an operating-company equity. A fund may hold equities, but its units "
                "are not corporate shares. RiskOptima deliberately does **not** apply a corporate DCF to the fund itself. "
                "The opportunity score measures historical price and regime dislocation, not company intrinsic value. "
                "Fund metadata alone does not establish holdings-level fair value; no NAV premium or discount is measured.",
                "",
            ]
        )

    if analysis.instrument_type in {"MUTUAL_FUND", "ETF"}:
        lines.extend(["## Fund metadata (provider-reported)", "", "| Metric | Value |", "|---|---:|"])
        for label, key, percentage in (
            ("Total assets (provider units; currency not independently verified)", "fund_total_assets", False),
            ("Distribution yield", "fund_distribution_yield", True),
            ("Annual-report expense ratio", "fund_expense_ratio", True),
            ("Reported fund P/E (methodology unverified)", "fund_reported_pe", False),
            ("NAV reference (date/share class unverified)", "fund_nav_price", False),
        ):
            value = _as_float(m.get(key))
            lines.append(f"| {label} | {_fmt_pct(value) if percentage else _fmt_number(value)} |")
        for label, key in (("Category", "fund_category"), ("Fund family", "fund_family")):
            text_value = str(m.get(key) or "n/a").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {label} | {text_value} |")
        lines.extend(["", "These optional fields are descriptive and do not change the dislocation score. "
                      "Missing values are shown as n/a. Metadata may differ in date or units from the price history.", ""])

    if analysis.diagnostics:
        lines.extend(["## Diagnostics", ""])
        lines.extend([f"- {item}" for item in analysis.diagnostics])
        lines.append("")

    lines.extend(
        [
            "## Model note",
            "",
            "The score is designed to detect potentially attractive dislocations, not to forecast an exact market bottom. "
            "Interpret scores alongside input coverage and, for corporate equities, fundamental quality; "
            "neither establishes predictive accuracy.",
            "",
            "## Research notice",
            "",
            RESEARCH_DISCLAIMER,
            "",
            f"Generated by RiskOptima v{__version__}",
            "",
        ]
    )
    return "\n".join(lines)
