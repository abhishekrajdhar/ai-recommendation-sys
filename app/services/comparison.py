from __future__ import annotations

import re

from app.models.schemas import Assessment
from app.services.retrieval import HybridRetriever


def is_comparison_query(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ("compare", "difference between", "vs", "versus", "which is better"))


def extract_comparison_names(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = (
        r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:and|with|vs|versus)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+(?:vs|versus)\s+(.+?)(?:\?|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidates.extend([match.group(1), match.group(2)])
            break
    if not candidates:
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
        candidates.extend(quoted)
    cleaned = []
    for candidate in candidates:
        value = re.sub(r"\b(shl|assessment|test|the)\b", "", candidate, flags=re.IGNORECASE)
        value = re.sub(r"\s+", " ", value).strip(" ?.,")
        if value:
            cleaned.append(value)
    return cleaned[:3]


def _value(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item) or "Not specified in catalog"
    return str(value).strip() or "Not specified in catalog"


def build_comparison(assessments: list[Assessment]) -> str:
    headers = " | ".join(["Feature", *[assessment.name for assessment in assessments]])
    separator = " | ".join(["---", *["---" for _ in assessments]])
    rows = [f"| {headers} |", f"| {separator} |"]
    feature_getters = [
        ("Type", lambda item: item.test_type),
        ("Assessment focus", lambda item: item.assessment_type),
        ("Use case", lambda item: item.description),
        ("Duration", lambda item: item.duration),
        ("Remote testing", lambda item: item.remote_testing),
        ("Adaptive support", lambda item: item.adaptive_support),
        ("Skills measured", lambda item: item.skills_measured),
        ("Job levels", lambda item: item.job_levels),
    ]
    for label, getter in feature_getters:
        cells = [_value(getter(assessment)) for assessment in assessments]
        rows.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(rows)


def compare_from_catalog(text: str, retriever: HybridRetriever) -> tuple[str, list[Assessment]]:
    names = extract_comparison_names(text)
    assessments = retriever.find_by_names(names)
    if len(assessments) < 2:
        return (
            "I can compare SHL assessments when I can identify at least two catalog items. "
            "Which two assessment names should I compare?",
            [],
        )
    selected = assessments[:2]
    reply = "Here is a catalog-grounded comparison:\n\n" + build_comparison(selected)
    return reply, selected

