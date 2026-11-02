from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.data_loader import FIELD_MAP, RAW_DATA_DIR, available_workbooks, load_financials
from src.financial_qa import answer_company_question
from src.report_assistant import answer_with_evidence, build_retrieval_index, extract_pdf_passages, retrieve_passages


load_dotenv()
st.set_page_config(page_title="FinSight AI", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#e8edf4; --muted:#9ba9b8; --panel:#101923; --line:#263444; --mint:#76e2b3; }
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 80% -10%, #18372f 0%, #0b1118 42%, #080d13 100%); color:var(--ink); }
    [data-testid="stSidebar"] { background:#0b121a; border-right:1px solid #1e2a36; }
    [data-testid="stSidebar"] * { color:#d9e3ec; }
    .block-container { padding-top:2rem; max-width:1440px; }
    h1,h2,h3 { letter-spacing:-.04em; }
    .eyebrow { color:var(--mint); font:500 .72rem 'DM Mono',monospace; letter-spacing:.16em; text-transform:uppercase; }
    .hero-title { font-size:2.6rem; font-weight:800; line-height:1.1; margin:.25rem 0 .35rem; }
    .hero-sub { color:#a7b4c1; font-size:.98rem; margin-bottom:1.5rem; }
    .metric-card { background:linear-gradient(145deg,rgba(24,38,49,.95),rgba(14,23,32,.92)); border:1px solid #263847; border-radius:16px; padding:17px 19px; min-height:116px; }
    .metric-label { color:#9fb0be; font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; }
    .metric-value { color:#f0f6f8; font-size:1.55rem; font-weight:800; margin-top:8px; letter-spacing:-.04em; }
    .metric-note { color:#7e91a1; font: .74rem 'DM Mono',monospace; margin-top:4px; }
    .section-kicker { color:var(--mint); font:500 .7rem 'DM Mono',monospace; letter-spacing:.15em; text-transform:uppercase; margin:1rem 0 .25rem; }
    .small-note { color:#91a0ae; font-size:.82rem; }
    div[data-testid="stMetric"] { background:#101923; border:1px solid #263847; padding:14px 16px; border-radius:14px; }
    div[data-testid="stMetricLabel"] { color:#9fb0be; }
    div[data-testid="stMetricValue"] { color:#eef7f5; }
    .stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #263444; }
    .stTabs [data-baseweb="tab"] { color:#97a6b4; padding:10px 16px; }
    .stTabs [aria-selected="true"] { color:#8fe7bf !important; }
    .stButton button, .stDownloadButton button { border-radius:10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Reading and normalizing financial workbooks…")
def get_financials() -> pd.DataFrame:
    return load_financials()


def money_crore(value: object) -> str:
    if pd.isna(value):
        return "—"
    amount = float(value)
    sign = "−" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 100000:
        return f"{sign}₹{amount / 100000:,.2f} L Cr"
    if amount >= 1000:
        return f"{sign}₹{amount / 1000:,.2f}k Cr"
    return f"{sign}₹{amount:,.1f} Cr"


def show_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def percent(value: object, digits: int = 1) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}%"


files = available_workbooks()
if not files:
    st.markdown('<div class="eyebrow">Financial intelligence workspace</div><div class="hero-title">FinSight AI</div>', unsafe_allow_html=True)
    st.error(f"No Bloomberg Excel workbooks found. Put files matching `*_Bloomberg.xlsx` in `{RAW_DATA_DIR}`.")
    st.stop()

data = get_financials()
companies = sorted(data["company"].dropna().unique().tolist())

with st.sidebar:
    st.markdown('<div class="eyebrow">FinSight AI</div>', unsafe_allow_html=True)
    st.markdown("### Research desk")
    st.caption("Company fundamentals · annual reports · evidence-led answers")
    selected_company = st.selectbox("Company", companies, index=companies.index("Reliance Industries Ltd") if "Reliance Industries Ltd" in companies else 0)
    current_company_rows = data[data["company"] == selected_company].sort_values("period_end")
    period_labels = current_company_rows["period_label"].astype(str).drop_duplicates().tolist()
    st.divider()
    st.caption(f"{len(companies)} companies in the supplied workbook set")
    st.caption("Values normalized to INR crore for display")

st.markdown('<div class="eyebrow">Company research · fundamentals and filings</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Financial clarity, with evidence.</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Explore reported performance, inspect the source workbook, and ask questions about an annual report.</div>', unsafe_allow_html=True)

dashboard_tab, compare_tab, report_tab = st.tabs(["Company overview", "Compare companies", "Ask an annual report"])

with dashboard_tab:
    latest = current_company_rows.iloc[-1]
    previous = current_company_rows.iloc[-2] if len(current_company_rows) > 1 else None
    st.markdown(f"<div class='section-kicker'>{latest['sector']} · {latest['period_label']} · {latest['reporting_basis']}</div>", unsafe_allow_html=True)
    cards = st.columns(4)
    for column, label, field in zip(cards, ["Revenue", "Operating profit", "Net profit", "Operating cash flow"], ["revenue", "operating_profit", "net_profit", "operating_cash_flow"]):
        with column:
            show_card(label, money_crore(latest[field]), f"INR crore · {latest['period_label']}")

    metric_columns = st.columns(4)
    for column, label, field, is_pct in zip(
        metric_columns,
        ["Revenue growth", "Net profit margin", "Debt / equity", "Current ratio"],
        ["revenue_growth_pct", "net_profit_margin_pct", "debt_to_equity", "current_ratio"],
        [True, True, False, False],
    ):
        with column:
            shown = percent(latest[field]) if is_pct else ("—" if pd.isna(latest[field]) else f"{float(latest[field]):.2f}×")
            show_card(label, shown, "Calculated from reported figures")

    st.markdown("<div class='section-kicker'>Reported trends</div>", unsafe_allow_html=True)
    metric_choice = st.selectbox("Trend metric", list(FIELD_MAP.values()), format_func=lambda item: item[1], key="trend_metric")
    metric_key = metric_choice[0]
    history = current_company_rows.dropna(subset=[metric_key]).copy()
    if history.empty:
        st.info("No values are available for this metric.")
    else:
        fig = px.line(history, x="period_label", y=metric_key, markers=True, labels={"period_label": "Reported period", metric_key: "INR crore"})
        fig.update_traces(line_color="#76e2b3", line_width=3, marker_size=8, hovertemplate="%{x}<br>₹%{y:,.1f} crore<extra></extra>")
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=8, r=8, t=12, b=8), font=dict(color="#b7c4cf"),
            xaxis=dict(title=None, showgrid=False), yaxis=dict(title="INR crore", gridcolor="#243340", zerolinecolor="#334554"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("View source values and data notes"):
        display = current_company_rows[["period_label", "period_end", "period_type", "reporting_basis", "source_unit", *[v[0] for v in FIELD_MAP.values()], "source_file", "source_note"]].copy()
        display["period_end"] = display["period_end"].dt.strftime("%Y-%m-%d")
        display = display.rename(columns={key: label for key, label in FIELD_MAP.values()})
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("The workbook basis and period labels are preserved. Bloomberg values are normalized from INR millions to INR crore only when the workbook declares millions.")

with compare_tab:
    st.markdown("<div class='section-kicker'>Peer review</div>", unsafe_allow_html=True)
    selected_companies = st.multiselect("Select companies", companies, default=companies[: min(6, len(companies))])
    compare_metric = st.selectbox("Metric", list(FIELD_MAP.values()), format_func=lambda item: item[1], key="compare_metric")
    compare_key, compare_label = compare_metric
    if selected_companies:
        latest_rows = data[data["company"].isin(selected_companies)].dropna(subset=[compare_key]).sort_values("period_end").groupby("company", as_index=False).tail(1).copy()
        latest_rows["display_period"] = latest_rows["period_label"] + " · " + latest_rows["reporting_basis"]
        latest_rows = latest_rows.sort_values(compare_key, ascending=False)
        fig = px.bar(latest_rows, x="company", y=compare_key, color="sector", hover_data=["ticker", "display_period"], labels={compare_key: "INR crore", "company": "Company", "sector": "Sector"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=8, r=8, t=12, b=8), font=dict(color="#b7c4cf"), xaxis=dict(title=None, tickangle=-35, showgrid=False), yaxis=dict(title="INR crore", gridcolor="#243340"), legend_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.warning("Latest observations may cover different fiscal or calendar periods. Companies also span different sectors; compare like with like and inspect the period and reporting-basis columns.")
        st.dataframe(latest_rows[["company", "ticker", "sector", "period_label", "period_end", "reporting_basis", compare_key, "source_file"]], use_container_width=True, hide_index=True)
    else:
        st.info("Choose at least one company to compare.")

with report_tab:
    st.markdown("<div class='section-kicker'>Ask about available company data</div>", unsafe_allow_html=True)
    st.write("Choose a company from the supplied workbooks and ask about its figures, ratios, or historical trend. Answers are calculated locally and cite the workbook and reported period.")
    with st.form("company_financial_question_form"):
        company_for_question = st.selectbox("Company in dataset", companies, key="company_qa_company")
        financial_question = st.text_input(
            "Your question",
            placeholder="How did revenue and net profit change over the reported years?",
            key="company_qa_question",
        )
        ask_company = st.form_submit_button("Ask about this company", type="primary")
    if ask_company and financial_question.strip():
        company_rows = data[data["company"] == company_for_question]
        answer, evidence = answer_company_question(financial_question, company_rows)
        st.markdown(answer)
        if evidence:
            with st.expander("Figures used in this answer"):
                evidence_frame = pd.DataFrame(evidence)
                columns = ["period_label", "period_end", "reporting_basis", "source_file"]
                columns += [column for column in ["revenue", "operating_profit", "ebitda", "net_profit", "total_debt", "total_equity", "operating_cash_flow"] if column in evidence_frame]
                evidence_frame = evidence_frame[[column for column in columns if column in evidence_frame]].drop_duplicates()
                st.dataframe(evidence_frame, use_container_width=True, hide_index=True)
        st.caption("Company-data Q&A uses deterministic Python calculations on the local workbook data. It does not send Bloomberg figures to an external model.")

    st.divider()
    st.markdown("<div class='section-kicker'>Ask an annual report PDF</div>", unsafe_allow_html=True)
    st.write("Upload a company annual report PDF to ask about risks, management commentary, or explanations found in the report. Answers show PDF page citations.")
    uploaded = st.file_uploader("Annual report PDF", type=["pdf"], accept_multiple_files=False)
    if uploaded is not None:
        # Bump this suffix when extraction/indexing logic changes so the current
        # PDF is reprocessed with the improved text normalization and retrieval.
        signature = f"{uploaded.name}:{uploaded.size}:identity-v2"
        if (
            st.session_state.get("report_signature") != signature
            or "report_search_index" not in st.session_state
        ):
            with st.spinner("Extracting report text and indexing pages…"):
                st.session_state["report_passages"] = extract_pdf_passages(uploaded.getvalue(), uploaded.name)
                st.session_state["report_search_index"] = build_retrieval_index(st.session_state["report_passages"])
                st.session_state["report_signature"] = signature
                st.session_state["report_chat"] = []
        passages = st.session_state.get("report_passages", [])
        st.caption(f"{len(passages):,} page-aware passages indexed from **{uploaded.name}**.")
        if not passages:
            st.warning("No selectable text was extracted. This PDF may be scanned; OCR support is not included yet.")
        for entry in st.session_state.get("report_chat", []):
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                if entry.get("sources"):
                    with st.expander("Retrieved source passages"):
                        for source in entry["sources"]:
                            st.markdown(f"**{source['id']} · {source['file']} · PDF page {source['page']}**  ")
                            st.write(source["text"])
        question = st.chat_input("Ask about performance, risks, cash flow, or management commentary…", disabled=not passages)
        if question:
            st.session_state["report_chat"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Finding supporting passages…"):
                    retrieved = retrieve_passages(
                        passages, question, limit=3,
                        index=st.session_state.get("report_search_index"),
                    )
                with st.spinner("Writing a cited answer…"):
                    answer, mode = answer_with_evidence(question, retrieved)
                st.markdown(answer)
                st.caption(f"Mode: {mode}")
                source_rows = [{"id": f"S{i}", "file": passage.source, "page": passage.page, "text": passage.text}
                               for i, (passage, _score) in enumerate(retrieved, start=1)]
                if source_rows:
                    with st.expander("Retrieved source passages"):
                        for source in source_rows:
                            st.markdown(f"**{source['id']} · {source['file']} · PDF page {source['page']}**  ")
                            st.write(source["text"])
            st.session_state["report_chat"].append({"role": "assistant", "content": answer, "sources": source_rows})
    else:
        st.info("The ZIP contains financial workbooks, not annual-report PDFs. Upload a report here to try document Q&A.")
    st.caption("When a Gemini or OpenAI key is configured, only the retrieved PDF passages and your question are sent to that model. Without a key, FinSight shows source excerpts without generating an explanation.")

st.divider()
with st.expander("Dataset coverage"):
    basis_counts = data[["ticker", "reporting_basis"]].drop_duplicates()["reporting_basis"].value_counts().rename_axis("Workbook basis").reset_index(name="Companies")
    st.write(f"**{len(companies)} companies · {len(data):,} company-period rows · {data['period_end'].min().date()} to {data['period_end'].max().date()}**")
    st.dataframe(basis_counts, use_container_width=True, hide_index=True)
    st.caption("The supplied ZIP has 29 company workbooks with annual data. It does not contain the full Nifty 100 or quarterly observations. ABB is calendar-year and its workbook does not state reporting basis; BSE contains a non-standard 15-month period. The app preserves those distinctions.")
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button("Download normalized financial data (CSV)", csv, file_name="finsight_financials_normalized.csv", mime="text/csv")

