from __future__ import annotations

from fastapi import APIRouter

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import SHLAgent

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return SHLAgent().chat(request.messages)

