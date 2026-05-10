from __future__ import annotations

from app.models.schemas import (
    Assessment,
    ConversationState,
)
import re

# =========================================================
# HELPERS
# =========================================================

def _contains_any(
    text: str,
    values: list[str],
) -> bool:

    lowered = text.lower()

    return any(
        value.lower() in lowered
        for value in values
    )


# =========================================================
# ROLE BOOSTING
# =========================================================

ROLE_BOOSTS = {
    "developer": [
        "developer",
        "software",
        "coding",
        "programming",
        "engineering",
        "technical",
        "backend",
        "frontend",
        "api",
        "microservices",
    ],

    "engineer": [
        "engineering",
        "developer",
        "backend",
        "software",
        "technical",
        "cloud",
    ],

    "manager": [
        "leadership",
        "stakeholder",
        "manager",
        "executive",
        "behavioral",
        "personality",
        "people management",
        "director",
    ],

    "graduate": [
        "graduate",
        "aptitude",
        "cognitive",
        "reasoning",
        "numerical",
        "verbal",
    ],
}


# =========================================================
# MAIN RERANKER
# =========================================================

def rerank(
    query: str,
    candidates: list[tuple[Assessment, float]],
    state: ConversationState | None = None,
    limit: int = 5,
) -> list[tuple[Assessment, float]]:

    enhanced: list[
        tuple[Assessment, float]
    ] = []

    for assessment, base_score in candidates:

        text = assessment.searchable_text()

        text_lower = text.lower()

        bonus = 0.0

        # =================================================
        # TECHNICAL SKILL BOOST
        # =================================================

        if (
            state
            and state.technical_skills
        ):

            matched_skills = sum(
                1
                for skill in state.technical_skills
                if skill.lower() in text_lower
            )

            bonus += matched_skills * 0.60

        # =================================================
        # PERSONALITY BOOST
        # =================================================

        if (
            state
            and state.needs_personality
        ):

            if any(
                term in text_lower
                for term in [
                    "personality",
                    "behavioral",
                    "behavioural",
                    "opq",
                    "leadership",
                    "motivation",
                    "work style",
                ]
            ):
                bonus += 0.40

        if (
            state
            and state.stakeholder_interaction
        ):

            if any(
                term in text_lower
                for term in [
                    "personality",
                    "behavioral",
                    "leadership",
                    "opq",
                ]
            ):
                bonus += 0.40


        # =================================================
        # COGNITIVE BOOST
        # =================================================

        if (
            state
            and state.needs_cognitive
        ):

            if any(
                term in text_lower
                for term in [
                    "cognitive",
                    "ability",
                    "reasoning",
                    "aptitude",
                    "gsa",
                    "numerical",
                    "verbal",
                    "inductive",
                ]
            ):
                bonus += 0.20

        # =================================================
        # TECHNICAL ASSESSMENT BOOST
        # =================================================

        if (
            state
            and state.needs_technical
        ):

            if any(
                term in text_lower
                for term in [
                    "technical",
                    "coding",
                    "programming",
                    "java",
                    "python",
                    "sql",
                    ".net",
                    "developer",
                    "engineering",
                    "software",
                    "backend",
                    "frontend",
                ]
            ):
                bonus += 0.25

        # =================================================
        # REMOTE BOOST
        # =================================================

        if (
            state
            and state.remote_requirement is True
        ):

            if any(
                term in text_lower
                for term in [
                    "remote",
                    "online",
                    "virtual",
                    "web-based",
                ]
            ):
                bonus += 0.10

        # =================================================
        # ROLE BOOSTING
        # =================================================

        if (
            state
            and state.role
        ):

            role_lower = (
                state.role.lower()
            )

            for (
                role_key,
                role_terms,
            ) in ROLE_BOOSTS.items():

                if role_key in role_lower:

                    matches = sum(
                        1
                        for term in role_terms
                        if term in text_lower
                    )

                    bonus += matches * 0.08

        # =================================================
        # STAKEHOLDER BOOST
        # =================================================

        if (
            state
            and state.stakeholder_interaction
        ):

            if any(
                term in text_lower
                for term in [
                    "stakeholder",
                    "communication",
                    "leadership",
                    "presentation",
                    "people management",
                    "behavioral",
                ]
            ):
                bonus += 0.18

        # =================================================
        # SENIORITY BOOST
        # =================================================

        if (
            state
            and state.seniority
        ):

            seniority = (
                state.seniority.lower()
            )

            if (
                seniority == "senior"
                and any(
                    term in text_lower
                    for term in [
                        "leadership",
                        "executive",
                        "director",
                        "manager",
                    ]
                )
            ):
                bonus += 0.15

            elif (
                seniority == "junior"
                and any(
                    term in text_lower
                    for term in [
                        "graduate",
                        "entry level",
                        "junior",
                    ]
                )
            ):
                bonus += 0.15

        # =================================================
        # EXCLUSION PENALTY
        # =================================================

        if (
            state
            and state.excluded_assessment_types
        ):

            if _contains_any(
                text,
                state.excluded_assessment_types,
            ):
                bonus -= 1.0

        # =================================================
        # TECH DOMAIN PENALTY
        # =================================================

        if (
            state
            and state.needs_technical
        ):

            if any(
                bad in text_lower
                for bad in [
                    "cashier",
                    "bookkeeping",
                    "accounting",
                    "auditing",
                    "retail",
                    "clerk",
                    "branch manager",
                    "administrative professional",
                ]
            ):
                bonus -= 0.80

        # =================================================
        # ROLE MISMATCH PENALTY (technical-only items vs non-technical roles)
        # If the role contains no shared tokens with the assessment text and the
        # assessment is clearly a technical product (framework/language-specific),
        # apply a strong penalty so non-technical roles (e.g., Account Manager)
        # don't receive .NET/Java programming tests.
        if state and state.role:
            role_tokens = [t for t in re.split(r"\W+", state.role.lower()) if t]
            overlap = sum(1 for t in role_tokens if t in text_lower)

            tech_only_terms = [
                ".net",
                "wcf",
                "mvc",
                "mvvm",
                "xaml",
                "wpf",
                "ado.net",
                "c#",
                "asp.net",
                "framework",
            ]

            if overlap == 0 and any(term in text_lower for term in tech_only_terms):
                bonus -= 0.9

        # =================================================
        # FINAL SCORE
        # =================================================

        final_score = (
            base_score + bonus
        )

        enhanced.append(
            (
                assessment,
                final_score,
            )
        )

    # =====================================================
    # SORT
    # =====================================================

    enhanced_sorted = sorted(
        enhanced,
        key=lambda item: item[1],
        reverse=True,
    )

    if not enhanced_sorted:
        return []

    # =====================================================
    # THRESHOLDING
    # =====================================================

    max_score = enhanced_sorted[0][1]

    # Relaxed threshold: allow more candidates to pass through for reranking.
    # Previously: max(0.45, max_score * 0.55)
    # New: max(0.30, max_score * 0.45)
    threshold = max(
        0.30,
        max_score * 0.45,
    )

    filtered = [
        item
        for item in enhanced_sorted
        if item[1] >= threshold
    ]

    # =====================================================
    # DIVERSITY SELECTION
    # =====================================================

    unique: list[
        tuple[Assessment, float]
    ] = []

    seen_types: set[str] = set()

    for assessment, score in filtered:

        tp = (
            assessment.test_type
            or assessment.category
            or assessment.assessment_type
            or "other"
        ).lower()

        if tp not in seen_types:

            unique.append(
                (
                    assessment,
                    score,
                )
            )

            seen_types.add(tp)

        if len(unique) >= limit:
            break

    # =====================================================
    # FILL REMAINING
    # =====================================================

    if len(unique) < min(
        limit,
        len(filtered),
    ):

        for (
            assessment,
            score,
        ) in filtered:

            if (
                assessment,
                score,
            ) not in unique:

                unique.append(
                    (
                        assessment,
                        score,
                    )
                )

            if len(unique) >= limit:
                break

    return unique[:limit]