"""Page-aware retrieval and optional LLM answers for uploaded annual reports."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import time
from typing import Iterable

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_BILLING_ERROR_CODES = {
    "insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
}


@dataclass(frozen=True)
class Passage:
    text: str
    page: int
    source: str


def extract_pdf_passages(pdf_bytes: bytes, filename: str, chunk_chars: int = 1300, overlap: int = 180) -> list[Passage]:
    """Extract PDF text and make overlapping chunks that retain printed PDF pages."""
    reader = PdfReader(__import__("io").BytesIO(pdf_bytes))
    passages: list[Passage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        # Some annual reports space out abbreviations (for example, "T C S").
        text = re.sub(
            r"\b(?:[A-Z]\s+){1,}[A-Z]\b",
            lambda match: re.sub(r"\s+", "", match.group(0)),
            text,
        )
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk = text[start:end].strip()
            if len(chunk) >= 80:
                passages.append(Passage(chunk, page_number, filename))
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return passages


def build_retrieval_index(passages: Iterable[Passage]):
    """Fit the local TF-IDF index once per uploaded report, not once per question."""
    docs = list(passages)
    if not docs:
        return docs, None, None
    corpus = [p.text for p in docs]
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=40000)
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return docs, None, None
    return docs, vectorizer, matrix


def retrieve_passages(
    passages: Iterable[Passage], question: str, limit: int = 5, index=None
) -> list[tuple[Passage, float]]:
    if not question.strip():
        return []
    docs, vectorizer, matrix = index if index is not None else build_retrieval_index(passages)
    if not docs or vectorizer is None or matrix is None:
        return []
    # Company-name questions often have almost no word overlap with the report's
    # formal identity section. Prefer the opening pages and reporting-entity cues.
    if _is_company_identity_question(question):
        identity_results = []
        for passage in docs:
            if passage.page > 15:
                continue
            text = passage.text.lower()
            score = 0.0
            if "reporting entity" in text:
                score += 12.0
            if re.search(r"\babout\s+(the\s+)?(company|report|[a-z0-9&.-]+)\b", text):
                score += 4.0
            if re.search(r"\b(limited|ltd\.?|inc\.?|corporation|corp\.?)\b", text):
                score += 3.0
            if re.search(r"\b(tata consultancy services|tcs limited|tcs)\b", text):
                score += 8.0
            if "parent company" in text or "collectively referred to" in text:
                score += 5.0
            score += max(0.0, 2.0 - passage.page / 10)
            if score > 0:
                identity_results.append((passage, score))
        if identity_results:
            identity_results.sort(key=lambda item: item[1], reverse=True)
            return identity_results[:limit]
    try:
        query_vector = vectorizer.transform([question])
        scores = cosine_similarity(query_vector, matrix).ravel()
    except ValueError:
        return []
    selected = scores.argsort()[::-1][:limit]
    return [(docs[i], float(scores[i])) for i in selected if scores[i] > 0]


def _is_company_identity_question(question: str) -> bool:
    text = question.lower()
    asks_name = re.search(r"\b(name|called|identify|identified|mentioned|reporting entity)\b", text)
    about_company = re.search(r"\b(company|entity|organization|organisation|business|issuer|report)\b", text)
    return bool(asks_name and about_company)


def _api_error_info(exc: Exception) -> dict[str, object]:
    body = getattr(exc, "body", None)
    error_body = body.get("error", body) if isinstance(body, dict) else {}
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    code = error_body.get("code") if isinstance(error_body, dict) else None
    error_type = error_body.get("type") if isinstance(error_body, dict) else None
    message = error_body.get("message") if isinstance(error_body, dict) else None
    code = code or getattr(exc, "code", None)
    error_type = error_type or getattr(exc, "type", None)
    message = message or str(exc)
    retry_after = headers.get("retry-after") if headers else None
    try:
        retry_after = float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        retry_after = None
    return {
        "code": str(code or ""),
        "type": str(error_type or ""),
        "message": str(message or ""),
        "status": getattr(exc, "status_code", None),
        "retry_after": retry_after,
    }


def _friendly_api_error(info: dict[str, object], api_key: str) -> str:
    code = str(info["code"]).lower()
    message = str(info["message"])
    lower_message = message.lower()
    if api_key:
        message = message.replace(api_key, "[hidden]")
    if code in _BILLING_ERROR_CODES or any(term in lower_message for term in ("exceeded your current quota", "billing hard limit", "insufficient quota", "credit balance exhausted")):
        return (
            "OpenAI rejected the request because the API organization has no available credits or has reached a spend/usage limit. "
            "Check API Platform → Billing for credits and Settings → Organization/Project → Limits for caps. "
            "Waiting or retrying will not fix a billing or quota limit."
        )

    if info["status"] == 429 or code in {"rate_limit_exceeded", "slow_down"}:
        retry_after = info["retry_after"]
        suffix = f" The server asked you to wait {retry_after:g} seconds before trying again." if isinstance(retry_after, (int, float)) else " Wait briefly, then try again."
        detail = f" OpenAI detail: {message}" if message else ""
        return (
            "OpenAI temporarily throttled the request. The app already limits the prompt to a few retrieved passages. "
            "Check the model's RPM/TPM limits in the API Platform and avoid repeated rapid submissions."
            + suffix + detail
        )

    detail = f" OpenAI detail: {message}" if message else ""
    return f"The OpenAI request was rejected (HTTP {info['status'] or 'unknown'}, code: {info['code'] or 'not provided'}).{detail}"


def answer_with_evidence(question: str, results: list[tuple[Passage, float]]) -> tuple[str, str]:
    """Return an evidence-only fallback or an answer grounded in retrieved passages."""
    if not results:
        return "I couldn't find relevant text in this PDF to support an answer. Try a more specific question or confirm that the PDF contains selectable text.", "retrieval"

    # Keep context bounded to reduce token use and fit entry-level API limits.
    results = results[:3]
    context_blocks = []
    for index, (passage, _score) in enumerate(results, start=1):
        excerpt = passage.text[:1600]
        context_blocks.append(f"[S{index}] {passage.source}, PDF page {passage.page}:\n{excerpt}")
    context = "\n\n".join(context_blocks)
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider not in {"gemini", "openai"}:
        provider = "gemini" if os.getenv("GEMINI_API_KEY", "").strip() else "openai"
    api_key = os.getenv("GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY", "").strip()
    if not api_key:
        key_name = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
        excerpts = "\n\n".join(
            f"**[S{i}] {p.source}, PDF page {p.page}**\n> {p.text[:650]}{'…' if len(p.text) > 650 else ''}"
            for i, (p, _score) in enumerate(results[:3], start=1)
        )
        return (
            f"No {provider.title()} API key is configured, so here are the most relevant report excerpts instead of a generated explanation. "
            f"Add {key_name} to your .env file to enable grounded natural-language answers.\n\n" + excerpts,
            "retrieval only",
        )

    if provider == "gemini":
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            prompt = (
                "You are FinSight AI, an educational financial-report assistant. Answer only from the supplied evidence. "
                "Do not invent figures, causes, risks, or advice. Distinguish reported facts from calculations or interpretation. "
                "Cite every report-based claim with source markers such as [S1]. If evidence is insufficient, say so. "
                "For questions asking who the report is about or the company's name, check passages labelled Reporting Entity, About the Company, or About the Report first. "
                "Give the exact formal entity name stated there. Do not claim the name is absent just because other passages use the generic phrase 'the Company'. "
                "Never give personalized investment recommendations.\n\n"
                f"Question: {question}\n\nEvidence:\n{context}"
            )
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
                contents=prompt,
            )
            answer = response.text or "The model returned no answer."
            return answer, "Gemini + retrieved evidence"
        except Exception as exc:  # retrieval remains available when Gemini fails
            message = str(exc).replace(api_key, "[hidden]")
            lower_message = message.lower()
            if any(word in lower_message for word in ("resource_exhausted", "quota", "429", "rate limit")):
                message = "Gemini rejected the request because a quota or rate limit was reached. Check AI Studio's API usage and limits, then try later or use an eligible model/project."
            elif any(word in lower_message for word in ("api key not valid", "invalid api key", "permission_denied", "403")):
                message = "Gemini rejected the key. Create a Gemini API key in Google AI Studio, replace GEMINI_API_KEY in .env, and restart FinSight."
            else:
                message = f"The Gemini request failed: {message}"
            return f"{message}\n\nRetrieved excerpts are still available below.\n\n" + "\n\n".join(
                f"**[S{i}] {p.source}, PDF page {p.page}**\n> {p.text[:650]}{'…' if len(p.text) > 650 else ''}"
                for i, (p, _score) in enumerate(results[:3], start=1)
            ), "retrieval fallback"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, max_retries=0)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are FinSight AI, an educational financial-report assistant. Answer only from the supplied evidence. "
                    "Do not invent figures, causes, risks, or advice. Distinguish reported facts from calculations or interpretation. "
                    "Cite every report-based claim with the supplied source marker such as [S1]. If evidence is insufficient, say so. "
                    "For questions asking the company's name, prioritize the report's Reporting Entity or About section and state the exact formal name. "
                    "Do not say the name is absent only because other passages call it 'the Company'. "
                    "Never give personalized investment recommendations."
                ),
            },
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
        ]

        def create_completion():
            return client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                temperature=0.1,
                max_tokens=650,
                messages=messages,
            )

        try:
            response = create_completion()
        except Exception as first_error:
            first_info = _api_error_info(first_error)
            first_code = str(first_info["code"]).lower()
            first_message = str(first_info["message"]).lower()
            billing_issue = first_code in _BILLING_ERROR_CODES or any(
                term in first_message for term in ("exceeded your current quota", "billing hard limit", "insufficient quota", "credit balance exhausted")
            )
            retry_after = first_info["retry_after"]
            temporary_throttle = (first_info["status"] == 429 or first_code in {"rate_limit_exceeded", "slow_down"}) and not billing_issue
            delay = retry_after if isinstance(retry_after, (int, float)) else 2.0
            # One bounded retry for a short transient throttle; never retry quota/billing errors.
            if not temporary_throttle or delay > 5:
                return _friendly_api_error(first_info, api_key), "OpenAI error"
            time.sleep(max(1.0, delay))
            try:
                response = create_completion()
            except Exception as retry_error:
                retry_info = _api_error_info(retry_error)
                return _friendly_api_error(retry_info, api_key), "OpenAI error"
        answer = response.choices[0].message.content or "The model returned no answer."
        return answer, "LLM + retrieved evidence"
    except Exception as exc:  # keep document retrieval usable if a key/model call fails
        message = _friendly_api_error(_api_error_info(exc), api_key)
        return f"{message}\n\nRetrieved excerpts are still available below.\n\n" + "\n\n".join(
            f"**[S{i}] {p.source}, PDF page {p.page}**\n> {p.text[:650]}{'…' if len(p.text) > 650 else ''}"
            for i, (p, _score) in enumerate(results[:3], start=1)
        ), "retrieval fallback"

