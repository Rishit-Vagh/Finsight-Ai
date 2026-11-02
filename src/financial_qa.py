"""Local question answering over the structured financial workbook data."""

from __future__ import annotations

import pandas as pd


METRICS = {
    "revenue": {
        "label": "Revenue", "column": "revenue", "unit": "INR crore",
        "terms": ("revenue", "sales", "turnover"),
    },
    "operating_profit": {
        "label": "Operating profit / income", "column": "operating_profit", "unit": "INR crore",
        "terms": ("operating profit", "operating income"),
    },
    "ebitda": {"label": "EBITDA", "column": "ebitda", "unit": "INR crore", "terms": ("ebitda",)},
    "net_profit": {
        "label": "Net profit / PAT", "column": "net_profit", "unit": "INR crore",
        "terms": ("net profit", "pat", "net income", "profit after tax"),
    },
    "total_debt": {
        "label": "Total debt", "column": "total_debt", "unit": "INR crore",
        "terms": ("total debt", "debt", "borrowings"),
    },
    "total_equity": {"label": "Total equity", "column": "total_equity", "unit": "INR crore", "terms": ("equity",)},
    "current_assets": {"label": "Current assets", "column": "current_assets", "unit": "INR crore", "terms": ("current assets",)},
    "current_liabilities": {"label": "Current liabilities", "column": "current_liabilities", "unit": "INR crore", "terms": ("current liabilities",)},
    "operating_cash_flow": {
        "label": "Cash from operations", "column": "operating_cash_flow", "unit": "INR crore",
        "terms": ("operating cash flow", "cash flow from operations", "cash from operations", "cash flow"),
    },
    "revenue_growth_pct": {
        "label": "Revenue growth", "column": "revenue_growth_pct", "unit": "%",
        "terms": ("revenue growth", "sales growth", "growth"),
    },
    "net_profit_margin_pct": {
        "label": "Net profit margin", "column": "net_profit_margin_pct", "unit": "%",
        "terms": ("net profit margin", "profit margin", "net margin"),
    },
    "operating_margin_pct": {
        "label": "Operating margin", "column": "operating_margin_pct", "unit": "%",
        "terms": ("operating margin",),
    },
    "debt_to_equity": {
        "label": "Debt-to-equity", "column": "debt_to_equity", "unit": "×",
        "terms": ("debt to equity", "debt-to-equity", "leverage ratio"),
    },
    "current_ratio": {
        "label": "Current ratio", "column": "current_ratio", "unit": "×",
        "terms": ("current ratio", "liquidity ratio"),
    },
    "operating_cash_flow_margin_pct": {
        "label": "Operating cash flow margin", "column": "operating_cash_flow_margin_pct", "unit": "%",
        "terms": ("cash flow margin", "operating cash flow margin"),
    },
}


def _citation(row: pd.Series) -> str:
    return f"[{row['source_file']}, {row['period_label']}, {row['reporting_basis']}]"


def _format_value(value: object, unit: str) -> str:
    if pd.isna(value):
        return "not reported"
    if unit == "INR crore":
        return f"₹{float(value):,.2f} crore"
    if unit == "%":
        return f"{float(value):.2f}%"
    return f"{float(value):.2f}×"


def answer_company_question(question: str, company_rows: pd.DataFrame) -> tuple[str, list[dict[str, object]]]:
    """Answer common metric and trend questions using local calculations only."""
    if company_rows.empty:
        return "There are no financial observations for this company in the supplied workbooks.", []
    query = question.lower().strip()
    ordered = company_rows.sort_values("period_end").copy()
    # Keep every specifically mentioned metric so combined questions such as
    # "revenue and net profit trends" receive both answers. Longer phrases win
    # over their embedded generic terms (e.g. margin over net profit).
    candidates = sorted(METRICS.items(), key=lambda pair: max(map(len, pair[1]["terms"])), reverse=True)
    matched_specs: list[tuple[str, dict[str, object], str]] = []
    for key, spec in candidates:
        present = [term for term in spec["terms"] if term in query]
        if present:
            matched_specs.append((key, spec, max(present, key=len)))
    matched_specs = [
        item for item in matched_specs
        if not any(item[2] != other[2] and item[2] in other[2] for other in matched_specs)
    ]

    causal_question = (
        query.startswith("why ")
        or any(word in query for word in ("cause", "reason", "because", "due to"))
    )
    if any(word in query for word in ("risk", "risks", "management commentary")):
        return (
            "The supplied workbooks contain financial figures, not risk disclosures or management commentary. "
            "Use the **Annual report PDF Q&A** tab to ask that question against a cited report.",
            [],
        )

    summary_question = any(word in query for word in ("performance", "overview", "summari", "financial health"))
    if not matched_specs and not summary_question:
        supported = ", ".join(spec["label"] for spec in METRICS.values())
        return f"I couldn’t map that question to a supported field. Try asking about one of these: {supported}.", []

    if summary_question and not matched_specs:
        latest = ordered.iloc[-1]
        earliest = ordered.iloc[0]
        metrics = ["revenue", "operating_profit", "net_profit", "operating_cash_flow"]
        lines = [f"**{latest['company']} — reported performance snapshot**"]
        evidence = []
        for column in metrics:
            value_latest = latest.get(column)
            value_earliest = earliest.get(column)
            if pd.isna(value_latest):
                continue
            label = next(spec["label"] for spec in METRICS.values() if spec["column"] == column)
            lines.append(f"- Latest {label}: {_format_value(value_latest, 'INR crore')} {_citation(latest)}")
            if not pd.isna(value_earliest) and earliest.name != latest.name:
                lines.append(f"  Earliest available: {_format_value(value_earliest, 'INR crore')} {_citation(earliest)}")
            evidence.extend([earliest.to_dict(), latest.to_dict()])
        lines.append("This is a figure summary; the workbook does not explain the causes behind changes.")
        return "\n".join(lines), evidence

    trend_question = any(word in query for word in (
        "trend", "over time", "history", "historical", "last ", "year", "quarter",
        "growth", "increase", "increased", "increasing", "decrease", "decreased",
        "decreasing", "rise", "rose", "fall", "fell", "change", "changed", "higher", "lower",
    ))
    if len(matched_specs) > 1:
        lines = [f"**{ordered.iloc[-1]['company']} — requested financial figures**"]
        evidence: list[dict[str, object]] = []
        for _key, spec, _term in matched_specs:
            column, label, unit = str(spec["column"]), str(spec["label"]), str(spec["unit"])
            available = ordered.dropna(subset=[column])
            if available.empty:
                lines.append(f"- {label}: blank or unavailable in this workbook.")
                continue
            if trend_question:
                lines.append(f"\n**{label}**")
                previous = None
                for _, row in available.iterrows():
                    line = f"- {row['period_label']}: {_format_value(row[column], unit)} {_citation(row)}"
                    if previous is not None:
                        days = (row["period_end"] - previous["period_end"]).days
                        comparable = row["period_type"] == previous["period_type"] and 300 <= days <= 400
                        if comparable and not str(row["period_type"]).startswith("Non-standard"):
                            change = float(row[column]) - float(previous[column])
                            change_pct = change / abs(float(previous[column])) * 100 if float(previous[column]) else None
                            direction = "up" if change > 0 else "down" if change < 0 else "unchanged"
                            change_text = _format_value(abs(change), unit)
                            pct_text = f" ({abs(change_pct):.2f}% {direction})" if change_pct is not None else f" ({direction})"
                            line += f" · {change_text} {direction} vs prior period{pct_text}"
                    lines.append(line)
                    evidence.append(row.to_dict())
                    previous = row
            else:
                row = available.iloc[-1]
                lines.append(f"- Latest {label}: {_format_value(row[column], unit)} for {row['period_label']} {_citation(row)}")
                evidence.append(row.to_dict())
        if causal_question:
            lines.append(
                "\n**What this dataset can establish:** it shows the reported balances and their period-to-period changes. "
                "It does not state why debt changed. Check the company’s annual report debt notes and management discussion for a reported explanation."
            )
        return "\n".join(lines), evidence

    assert matched_specs
    _matched_key, matched, _matched_term = matched_specs[0]
    column = str(matched["column"])
    label = str(matched["label"])
    unit = str(matched["unit"])
    available = ordered.dropna(subset=[column])
    if available.empty:
        return f"{label} is blank or unavailable in this company’s workbook.", []

    if trend_question:
        lines = [f"**{label} for {available.iloc[0]['company']} by reported period**"]
        evidence: list[dict[str, object]] = []
        previous = None
        for _, row in available.iterrows():
            line = f"- {row['period_label']}: {_format_value(row[column], unit)} {_citation(row)}"
            if previous is not None:
                days = (row["period_end"] - previous["period_end"]).days
                comparable = row["period_type"] == previous["period_type"] and 300 <= days <= 400
                if comparable and not str(row["period_type"]).startswith("Non-standard"):
                    change = float(row[column]) - float(previous[column])
                    change_pct = change / abs(float(previous[column])) * 100 if float(previous[column]) else None
                    direction = "up" if change > 0 else "down" if change < 0 else "unchanged"
                    change_text = _format_value(abs(change), unit)
                    pct_text = f" ({abs(change_pct):.2f}% {direction})" if change_pct is not None else f" ({direction})"
                    line += f" · {change_text} {direction} vs prior period{pct_text}"
            lines.append(line)
            evidence.append(row.to_dict())
            previous = row
        if len(available) >= 2 and column in {"revenue", "net_profit", "operating_profit", "ebitda", "operating_cash_flow"}:
            first, last = available.iloc[0], available.iloc[-1]
            if not pd.isna(first[column]) and float(first[column]) != 0:
                change = (float(last[column]) - float(first[column])) / abs(float(first[column])) * 100
                direction = "increased" if change > 0 else "decreased" if change < 0 else "was unchanged"
                lines.append(f"\nAcross the listed periods, {label.lower()} {direction} by {abs(change):.2f}% from {first['period_label']} to {last['period_label']}. This is a calculated change between the first and last available observations.")
        if any("15M" in str(label).upper() for label in available["period_label"].tolist()):
            lines.append("\nOne listed period is 15 months; do not treat it as a standard 12-month year when comparing growth.")
        if causal_question:
            lines.append(
                "\n**What this dataset can establish:** it shows the reported balances and their period-to-period changes. "
                "It does not state why the figure changed. Check the company’s annual report notes and management discussion for a reported explanation."
            )
        return "\n".join(lines), evidence

    latest = available.iloc[-1]
    return (
        f"**{label} for {latest['company']}**\n\n"
        f"The latest available value is **{_format_value(latest[column], unit)}** for {latest['period_label']} "
        f"({latest['period_type']}). {_citation(latest)}",
        [latest.to_dict()],
    )

