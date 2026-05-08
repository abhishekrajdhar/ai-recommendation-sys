from __future__ import annotations

from app.models.schemas import Assessment, ConversationState


def _contains_any(text: str, values: list[str]) -> bool:
    lowered = text.lower()
    return any(value.lower() in lowered for value in values)


def rerank(
    query: str,
    candidates: list[tuple[Assessment, float]],
    state: ConversationState | None = None,
    limit: int = 10,
) -> list[tuple[Assessment, float]]:
    reranked: list[tuple[Assessment, float]] = []
    for assessment, score in candidates:
        text = assessment.searchable_text()
        bonus = 0.0
        if state:
            if state.technical_skills and _contains_any(text, state.technical_skills):
                bonus += 0.08
            if state.needs_personality and "personality" in text.lower():
                bonus += 0.08
            if state.needs_cognitive and any(term in text.lower() for term in ("cognitive", "ability", "reasoning", "aptitude")):
                bonus += 0.08
            if state.needs_technical and any(term in text.lower() for term in ("technical", "coding", "programming", "skills")):
                bonus += 0.06
            if state.remote_requirement and "remote" in text.lower():
                bonus += 0.03
            if state.excluded_assessment_types and _contains_any(text, state.excluded_assessment_types):
                bonus -= 0.5
        reranked.append((assessment, score + bonus))
    return sorted(reranked, key=lambda item: item[1], reverse=True)[:limit]

