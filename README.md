# FinSight AI

FinSight AI is a local financial research dashboard built around the Bloomberg Excel workbooks supplied for this project. It combines deterministic Python calculations with page-aware retrieval over annual-report PDFs.

## What is included

- Company overview with revenue, operating profit, net profit, cash flow, and calculated ratios.
- Historical trend charts in INR crore.
- Latest-period company comparison with period and reporting-basis labels.
- Local company-data Q&A for any company in the supplied workbooks, with workbook and period citations.
- Annual-report PDF upload, page-preserving text extraction, TF-IDF retrieval, source excerpts, and optional LLM explanations.
- Normalized CSV download and a dataset coverage panel.

## Run locally

Python 3.10 or newer is recommended.

```powershell
cd "C:\path\to\finsight-ai"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Or double-click `run.bat` after installing the requirements.

The workbooks are expected in `data/raw/AI Finance System/` and should match `*_Bloomberg.xlsx`. They are kept local and excluded from Git by `.gitignore`.

### Optional LLM answers

For Gemini, create an API key in [Google AI Studio](https://aistudio.google.com/app/apikey), copy `.env.example` to `.env`, then set `LLM_PROVIDER=gemini`, `GEMINI_API_KEY=your_key`, and optionally `GEMINI_MODEL=gemini-3.8-flash`. Install dependencies with `python -m pip install -r requirements.txt` and restart Streamlit. Eligible Gemini API usage has a free tier, subject to model and project rate limits; check AI Studio for your account's current eligibility and limits. Google may use free-tier inputs and outputs to improve its products, so use public report material only if that is acceptable to you.

OpenAI remains an option: set `LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and optionally `OPENAI_MODEL`. The report assistant retrieves text locally. When a key is set, the question and retrieved passages are sent to the selected provider for an explanation. Without a key, the app displays retrieved report excerpts with page citations and does not invent an answer.

## Dataset notes

The supplied archive contains **29 Excel workbooks**, not all 100 Nifty 100 companies. It contains annual observations, not quarterly data. Most workbooks label periods FY2022–FY2026; ABB uses calendar years, and BSE includes a 15-month transition period. The app preserves the source period labels and reporting-basis notes rather than forcing these into a misleading uniform timeline.

All the workbooks describe values as INR millions (except per-share fields); reported financial statement values are converted to INR crore for charts. The source values and workbook metadata remain available in the data table. Twenty-five workbooks explicitly state “Consolidated”; three have a “Con Mixed (consolidated)” note; ABB does not state the reporting basis. Treat the latter labels carefully in comparisons.

Ratios are computed in Python from the loaded values. Missing data stays missing; it is not treated as zero. Revenue growth is suppressed when periods are not comparable, including the BSE 15-month period. Banks and other financial firms have sector-specific meanings for revenue, debt, EBITDA, and liquidity ratios, so cross-sector comparisons require care.

Company-data Q&A is local and deterministic: it answers supported metric, ratio, and trend questions from the selected company's workbook and cites the source file and period. Questions about causes, risks, or management commentary are directed to annual-report PDF Q&A. Bloomberg workbook values are not sent to the configured LLM.

## Project structure

```text
finsight-ai/
├── app.py
├── data/
│   └── raw/AI Finance System/   # local workbooks, ignored by Git
├── src/
│   ├── data_loader.py
│   └── report_assistant.py
├── .env.example
├── requirements.txt
└── README.md
```

## Data use

Keep this prototype private unless the data provider’s license permits redistribution or public deployment. The provided workbook files are excluded from Git. Public annual-report PDFs can be uploaded for Q&A; configure a model API only if you are comfortable sending retrieved excerpts and the question to that provider.
