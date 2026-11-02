"""Load and normalize the Bloomberg Excel workbooks supplied for FinSight AI."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re

import pandas as pd
from openpyxl import load_workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "AI Finance System"

FIELD_MAP = {
    "SALES_REV_TURN": ("revenue", "Revenue"),
    "IS_OPER_INC": ("operating_profit", "Operating income"),
    "EBITDA": ("ebitda", "EBITDA"),
    "NET_INCOME": ("net_profit", "Net income / profit"),
    "SHORT_AND_LONG_TERM_DEBT": ("total_debt", "Short and long-term debt"),
    "BS_CUR_ASSET_REPORT": ("current_assets", "Current assets"),
    "BS_CUR_LIAB": ("current_liabilities", "Current liabilities"),
    "CF_CASH_FROM_OPER": ("operating_cash_flow", "Cash from operations"),
    "TOTAL_EQUITY": ("total_equity", "Total equity"),
}

SECTORS = {
    "ABB": "Industrials", "ADANIENSOL": "Utilities", "ADANIGREEN": "Utilities",
    "ADANIPORTS": "Infrastructure", "ADANIPOWER": "Utilities", "AMBUJACEM": "Cement",
    "APOLLOHOSP": "Healthcare", "ASIANPAINT": "Consumer", "AXISBANK": "Banking",
    "BAJAJ-AUTO": "Automotive", "BAJAJFINSV": "Financial services",
    "BAJAJHLDNG": "Financial services", "BANKBARODA": "Banking", "BEL": "Industrials",
    "BHARTIARTL": "Telecommunications", "BOSCHLTD": "Automotive", "BPCL": "Energy",
    "BRITANNIA": "Consumer staples", "BSE": "Financial services", "CANBK": "Banking",
    "CGPOWER": "Industrials", "CHOLAFIN": "Financial services", "CIPLA": "Pharmaceuticals",
    "COALINDIA": "Energy", "CUMMINSIND": "Industrials", "DMART": "Retail",
    "INFY": "Information technology", "Reliance": "Diversified",
    "TCS": "Information technology",
}


def _as_period_end(value: object) -> pd.Timestamp:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(value)
    parsed = pd.to_datetime(str(value), errors="coerce")
    return parsed if not pd.isna(parsed) else pd.NaT


def _as_number(value: object) -> float:
    try:
        return float(value) if value is not None and value != "" else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def load_financials(data_dir: Path | str = RAW_DATA_DIR) -> pd.DataFrame:
    """Return one observation per company, period, and metric in INR crore."""
    files = sorted(Path(data_dir).glob("*_Bloomberg.xlsx"))
    records: list[dict[str, object]] = []
    for path in files:
        workbook = load_workbook(path, data_only=True, read_only=True)
        sheet = workbook[workbook.sheetnames[0]]
        rows = list(sheet.iter_rows(values_only=True))
        workbook.close()
        if len(rows) < 6:
            continue

        title = str(rows[1][0] or path.stem)
        match = re.match(r"(.+?)\s*\((.*?)\)", title)
        company = match.group(1).strip() if match else path.stem.replace("_Bloomberg", "")
        bloomberg_id = match.group(2).strip() if match else ""
        ticker = path.stem.removesuffix("_Bloomberg")
        metadata = str(rows[2][0] or "")
        basis_lower = metadata.lower()
        if "not shown" in basis_lower or "no acct selector" in basis_lower:
            reporting_basis = "Not stated in workbook"
        elif "con mixed" in basis_lower:
            reporting_basis = "Con Mixed (consolidated)"
        elif "consolidated" in basis_lower:
            reporting_basis = "Consolidated"
        elif "standalone" in basis_lower:
            reporting_basis = "Standalone"
        else:
            reporting_basis = "Unspecified"

        unit_text = str(rows[3][0] or "")
        unit_lower = unit_text.lower()
        if "million" in unit_lower:
            scale_to_crore = 0.1
            source_unit = "INR million"
        elif "thousand" in unit_lower:
            scale_to_crore = 0.0001
            source_unit = "INR thousand"
        else:
            scale_to_crore = 1.0
            source_unit = "INR crore" if "crore" in unit_lower else "INR (unit unspecified)"

        headers = rows[3]
        dates = rows[4]
        value_rows = rows[5:]
        fields: dict[str, tuple[str, str, list[object]]] = {}
        for row in value_rows:
            if len(row) < 2 or not row[1]:
                continue
            code = str(row[1]).strip()
            if code in FIELD_MAP:
                key, label = FIELD_MAP[code]
                fields[key] = (code, label, list(row))

        for col in range(2, len(headers)):
            period_label = str(headers[col] or "").strip()
            if not period_label:
                continue
            period_end = _as_period_end(dates[col] if col < len(dates) else None)
            period_type = "Calendar year" if period_label.upper().startswith("CY") else "Fiscal year"
            if "15m" in period_label.lower():
                period_type = "Non-standard period (15 months)"
            row_record: dict[str, object] = {
                "company": company,
                "ticker": ticker,
                "bloomberg_identifier": bloomberg_id,
                "sector": SECTORS.get(ticker, "Other / unclassified"),
                "period_label": period_label,
                "period_end": period_end,
                "period_type": period_type,
                "reporting_basis": reporting_basis,
                "source_unit": source_unit,
                "source_file": path.name,
                "source": "Bloomberg workbook supplied by user",
                "source_note": metadata,
            }
            for key, (code, label, row) in fields.items():
                raw_value = row[col] if col < len(row) else None
                row_record[key] = _as_number(raw_value) * scale_to_crore
                row_record[f"{key}_field"] = code
                row_record[f"{key}_label"] = label
            records.append(row_record)

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    frame["period_end"] = pd.to_datetime(frame["period_end"], errors="coerce")
    for metric in FIELD_MAP.values():
        frame[metric[0]] = pd.to_numeric(frame[metric[0]], errors="coerce")
    frame = frame.sort_values(["company", "period_end", "period_label"]).reset_index(drop=True)
    frame["revenue_growth_pct"] = float("nan")
    for _, indices in frame.groupby("ticker").groups.items():
        group = frame.loc[indices].sort_values("period_end")
        previous_revenue = group["revenue"].shift(1)
        previous_type = group["period_type"].shift(1)
        day_gap = group["period_end"].diff().dt.days
        growth = (group["revenue"] - previous_revenue) / previous_revenue * 100
        # Do not compare a calendar-year series with fiscal-year data or use a
        # 15-month transition period as if it were a normal 12-month year.
        valid_interval = (group["period_type"] == previous_type) & day_gap.between(300, 400)
        growth = growth.where(valid_interval)
        frame.loc[group.index, "revenue_growth_pct"] = growth
    frame.loc[frame["period_type"].str.startswith("Non-standard"), "revenue_growth_pct"] = float("nan")
    frame["net_profit_margin_pct"] = frame["net_profit"] / frame["revenue"].replace(0, float("nan")) * 100
    frame["operating_margin_pct"] = frame["operating_profit"] / frame["revenue"].replace(0, float("nan")) * 100
    frame["debt_to_equity"] = frame["total_debt"] / frame["total_equity"].replace(0, float("nan"))
    frame["current_ratio"] = frame["current_assets"] / frame["current_liabilities"].replace(0, float("nan"))
    frame["operating_cash_flow_margin_pct"] = frame["operating_cash_flow"] / frame["revenue"].replace(0, float("nan")) * 100
    return frame


def available_workbooks(data_dir: Path | str = RAW_DATA_DIR) -> list[Path]:
    return sorted(Path(data_dir).glob("*_Bloomberg.xlsx"))
