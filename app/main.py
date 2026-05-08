from __future__ import annotations

from fastapi import FastAPI

from app.routes.chat import router as chat_router

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(chat_router)

