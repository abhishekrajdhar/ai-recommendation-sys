from __future__ import annotations

import json
import logging
import os

from app.models.schemas import Assessment, ConversationState
from app.services.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class OptionalLLMClient:
    """Small configurable LLM wrapper used only for response wording.

    The deterministic agent owns routing, refusal, retrieval, and recommendation
    validation. This client never decides which catalog items are returned.
    """

    def __init__(self) -> None:
        self.enabled = os.getenv("SHL_USE_LLM", "false").lower() == "true"
        self.provider = os.getenv("LLM_PROVIDER", "openai").lower()

    def recommendation_reply(self, state: ConversationState, assessments: list[Assessment]) -> str | None:
        if not self.enabled:
            return None
        payload = {
            "state": state.model_dump(),
            "catalog_items": [
                {
                    "name": item.name,
                    "url": item.url,
                    "test_type": item.test_type,
                    "description": item.description,
                }
                for item in assessments[:10]
            ],
        }
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Write a concise recommendation reply using only this JSON context. "
            "Do not add assessments or metadata not present here.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        try:
            if self.provider == "gemini":
                return self._gemini(prompt)
            return self._openai(prompt)
        except Exception as exc:
            logger.warning("LLM reply generation failed; using deterministic reply: %s", exc)
            return None

    def _openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        response = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=220,
        )
        return response.output_text.strip()

    def _gemini(self, prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
        response = model.generate_content(prompt)
        return (response.text or "").strip()
