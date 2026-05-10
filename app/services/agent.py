from __future__ import annotations

from app.models.schemas import ChatResponse, Message, Recommendation
from app.services.comparison import compare_from_catalog, is_comparison_query
from app.services.guardrails import should_refuse
from app.services.llm_client import OptionalLLMClient
from app.services.prompts import REFUSAL_MESSAGE
from app.services.retrieval import HybridRetriever, get_retriever, TECH_DOMAIN_KEYWORDS
from app.services.state_extractor import extract_state, missing_context_questions
import logging
import hashlib
import re

logger = logging.getLogger(__name__)

MAX_TURNS = 8


def _latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _turn_count(messages: list[Message]) -> int:
    return sum(1 for message in messages if message.role == "user")


def _recommendation_models(assessments) -> list[Recommendation]:
    return [
        Recommendation(name=assessment.name, url=assessment.url, test_type=assessment.test_type)
        for assessment in assessments[:10]
    ]


def _filter_by_requested_types(assessments, state):
    requested: list[str] = []
    if state.needs_technical:
        requested.append("technical")
    if state.needs_personality:
        requested.extend(["personality", "behavioral", "behavioural"])
    if state.needs_cognitive:
        requested.extend(["cognitive", "ability", "reasoning", "aptitude"])
    if not requested:
        return assessments
    filtered = [
        assessment
        for assessment in assessments
        if any(term in assessment.searchable_text().lower() for term in requested)
    ]
    return filtered or assessments


def _ensure_requested_types_included(assessments: list, state, retriever: HybridRetriever) -> list:

    included = assessments[:]

    def _has_personality(items):
        return any("personality" in (a.test_type or "").lower() for a in items)

    def _has_cognitive(items):
        return any("cognitive" in (a.test_type or "").lower() for a in items)

    # If personality explicitly requested but not present, try to find one
    if getattr(state, "needs_personality", False) and not _has_personality(included):

        for a in retriever.assessments:
            if a in included:
                continue
            if any(term in a.searchable_text().lower() for term in ["personality", "behavioral", "opq"]):
                included.append(a)
                break

    if getattr(state, "needs_cognitive", False) and not _has_cognitive(included):

        for a in retriever.assessments:
            if a in included:
                continue
            if any(term in a.searchable_text().lower() for term in ["cognitive", "ability", "gsa", "aptitude"]):
                included.append(a)
                break

    return included


class SHLAgent:
    def __init__(self, retriever: HybridRetriever | None = None) -> None:
        self.retriever = retriever or get_retriever()
        self.llm_client = OptionalLLMClient()

    def chat(self, messages: list[Message]) -> ChatResponse:
        latest = _latest_user_text(messages)
        if not latest:
            return ChatResponse(reply="Please send a user message.", recommendations=[], end_of_conversation=False)

        if should_refuse(latest):
            return ChatResponse(reply=REFUSAL_MESSAGE, recommendations=[], end_of_conversation=True)

        if is_comparison_query(latest):
            reply, compared = compare_from_catalog(latest, self.retriever)
            return ChatResponse(
                reply=reply,
                recommendations=_recommendation_models(compared),
                end_of_conversation=_turn_count(messages) >= MAX_TURNS,
            )

        state = extract_state(messages)
        logger.info("extracted state=%s", state.model_dump())
        logger.debug("DEBUG_STATE %s", state.model_dump())
        # Clarification-first routing: if we have a role but lack seniority
        # and the user hasn't expressed an assessment-type preference (personality/cognitive/technical),
        # prefer to ask a clarification question before performing retrieval. This prevents
        # over-aggressive filtering and weak retrieval from returning a small or irrelevant set.
        def _clarification_first(s):
            # Must have a role to clarify about seniority/assessment intent
            if not s.role:
                return False
            # If seniority is already provided, no clarification needed
            if s.seniority:
                return False
            # If any assessment-type preference is explicitly set, proceed to retrieval
            if s.needs_personality or s.needs_cognitive or s.needs_technical:
                return False
            # If the role or query is clearly technical, proceed to retrieval (don't ask seniority)
            role_lower = (s.role or "").lower()
            if any(k in role_lower for k in TECH_DOMAIN_KEYWORDS):
                return False
            # Otherwise prefer clarification (ask seniority / assessment focus)
            return True

        if _clarification_first(state):
            questions = missing_context_questions(state)
            # Prepend a focused seniority prompt when role is present
            focused = "Could you tell me the seniority level you are hiring for (Junior, Mid, Senior)?"
            # missing_context_questions already asks about assessment focus; include it as well
            reply = focused + " " + " ".join(questions)
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=_turn_count(messages) >= MAX_TURNS,
            )

        # If we don't have the minimum context at all (e.g., "I need an assessment"),
        # ask the generic missing-context questions and do not attempt retrieval.
        if not state.has_minimum_context():
            questions = missing_context_questions(state)
            reply = "To recommend the right SHL assessments, I need a bit more context. " + " ".join(questions)
            return ChatResponse(
                reply=reply,
                recommendations=[],
                end_of_conversation=_turn_count(messages) >= MAX_TURNS,
            )

        query = state.retrieval_query() or latest
        assessments = self.retriever.search(query, state=state, initial_k=20, limit=10)
        logger.info("retriever returned %d assessments", len(assessments))
        logger.info("retriever.names=%s", [a.name for a in assessments])
        logger.debug("DEBUG_CANDIDATES %s", [a.name for a in assessments])
        assessments = _filter_by_requested_types(assessments, state)[:10]
        assessments = _ensure_requested_types_included(assessments, state, self.retriever)[:10]
        logger.info("after type filter names=%s", [a.name for a in assessments])
        logger.debug("DEBUG_AFTER_FILTER %s", [a.name for a in assessments])
        if not assessments:
            return ChatResponse(
                reply=(
                    "I could not find a matching Individual Test Solution in the indexed SHL catalog. "
                    "Try a different role, skill, or assessment type from the catalog."
                ),
                recommendations=[],
                end_of_conversation=True,
            )

        # Group assessments into primary (exact skill/name matches) and secondary (similar)
        query_text = state.retrieval_query() or latest
        q_tokens = set(t for t in re.split(r"\W+", query_text.lower()) if t)

        primary_assessments = []
        secondary_assessments = []

        for a in assessments:
            name_lower = a.name.lower()
            skills = [s.lower() for s in (getattr(a, "skills_measured", []) or [])]

            # Primary if any query token exactly matches a skill token or appears in the assessment name
            is_primary = any(tok in skills for tok in q_tokens) or any(tok in name_lower for tok in q_tokens)

            if is_primary:
                primary_assessments.append(a)
            else:
                secondary_assessments.append(a)

        # If nothing matched as primary, promote the top result as primary
        if not primary_assessments and assessments:
            primary_assessments = [assessments[0]]
            secondary_assessments = [a for a in assessments[1:]]

        # Build Recommendation models with primary first
        recommendations = _recommendation_models(primary_assessments + secondary_assessments)

        # Compose deterministic reply when LLM is disabled; otherwise prefer LLM wording
        llm_reply = self.llm_client.recommendation_reply(state, assessments)

        if llm_reply:
            reply = llm_reply
        else:
            primary_names = ", ".join(a.name for a in primary_assessments[:5])
            secondary_names = ", ".join(a.name for a in secondary_assessments[:5])

            if secondary_names:
                templates = [
                    "Primary recommendation(s): {primary}. You can also explore similar assessments: {secondary}. To refine, tell me seniority or whether you prefer personality/cognitive/technical tests.",
                    "I recommend: {primary}. Other similar options you might consider: {secondary}. Specify seniority or assessment focus to refine.",
                    "Top pick(s): {primary}. You may also explore: {secondary}. Mention seniority or type (personality/cognitive/technical) for a tighter match.",
                ]
                seed = (primary_names + secondary_names + (state.role or "")).encode("utf-8")
                idx = int(hashlib.md5(seed).hexdigest(), 16) % len(templates)
                reply = templates[idx].format(primary=primary_names, secondary=secondary_names)
            else:
                templates = [
                    "Based on the role and requirements, I recommend these SHL Individual Test Solutions: {primary}. You can refine by seniority or assessment focus (personality/cognitive/technical).",
                    "Recommended: {primary}. To refine further, add seniority, remote-testing needs, or indicate whether personality/cognitive/technical assessments should be prioritized.",
                ]
                seed = (primary_names + (state.role or "")).encode("utf-8")
                idx = int(hashlib.md5(seed).hexdigest(), 16) % len(templates)
                reply = templates[idx].format(primary=primary_names)

        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=_turn_count(messages) >= MAX_TURNS,
        )
