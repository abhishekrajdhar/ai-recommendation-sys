from __future__ import annotations

from app.models.schemas import ChatResponse, Message, Recommendation
from app.services.comparison import compare_from_catalog, is_comparison_query
from app.services.guardrails import should_refuse
from app.services.llm_client import OptionalLLMClient
from app.services.prompts import REFUSAL_MESSAGE
from app.services.retrieval import HybridRetriever, get_retriever
from app.services.state_extractor import extract_state, missing_context_questions

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
        assessments = _filter_by_requested_types(assessments, state)[:10]
        if not assessments:
            return ChatResponse(
                reply=(
                    "I could not find a matching Individual Test Solution in the indexed SHL catalog. "
                    "Try a different role, skill, or assessment type from the catalog."
                ),
                recommendations=[],
                end_of_conversation=True,
            )

        recommendations = _recommendation_models(assessments)
        llm_reply = self.llm_client.recommendation_reply(state, assessments)
        names = ", ".join(item.name for item in recommendations[:5])
        reply = llm_reply or (
            f"Based on the role and requirements, I recommend these SHL Individual Test Solutions: {names}. "
            "These are grounded in the indexed catalog; you can refine by adding or excluding personality, "
            "cognitive, technical, remote, or seniority requirements."
        )
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=_turn_count(messages) >= MAX_TURNS,
        )
