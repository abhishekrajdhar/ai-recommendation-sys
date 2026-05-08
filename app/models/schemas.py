from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

    @field_validator("recommendations")
    @classmethod
    def max_ten_recommendations(cls, value: list[Recommendation]) -> list[Recommendation]:
        if len(value) > 10:
            raise ValueError("recommendations must contain at most 10 items")
        return value


class Assessment(BaseModel):
    name: str
    url: str
    description: str = ""
    assessment_type: str = ""
    duration: str = ""
    remote_testing: str = ""
    adaptive_support: str = ""
    job_levels: list[str] = Field(default_factory=list)
    skills_measured: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    category: str = ""

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        HttpUrl(value)
        return value

    @property
    def test_type(self) -> str:
        return self.category or self.assessment_type or "Individual Test Solution"

    def searchable_text(self) -> str:
        fields: list[str] = [
            self.name,
            self.description,
            self.assessment_type,
            self.duration,
            self.remote_testing,
            self.adaptive_support,
            self.category,
            " ".join(self.job_levels),
            " ".join(self.skills_measured),
            " ".join(self.languages),
        ]
        return " ".join(part for part in fields if part).strip()


class ConversationState(BaseModel):
    role: str | None = None
    seniority: str | None = None
    experience_years: int | None = None
    technical_skills: list[str] = Field(default_factory=list)
    needs_personality: bool | None = None
    needs_cognitive: bool | None = None
    needs_technical: bool | None = None
    hiring_scale: str | None = None
    stakeholder_interaction: bool | None = None
    remote_requirement: bool | None = None
    excluded_assessment_types: list[str] = Field(default_factory=list)

    def has_minimum_context(self) -> bool:
        has_role = bool(self.role)
        has_dimension = bool(
            self.seniority
            or self.technical_skills
            or self.needs_personality
            or self.needs_cognitive
            or self.needs_technical
        )
        return has_role and has_dimension

    def retrieval_query(self) -> str:
        parts: list[str] = []
        if self.role:
            parts.append(self.role)
        if self.seniority:
            parts.append(self.seniority)
        parts.extend(self.technical_skills)
        if self.needs_personality:
            parts.append("personality behavioral workplace preferences")
        if self.needs_cognitive:
            parts.append("cognitive ability aptitude reasoning")
        if self.needs_technical:
            parts.append("technical skills coding")
        if self.remote_requirement:
            parts.append("remote testing")
        if self.hiring_scale:
            parts.append(self.hiring_scale)
        return " ".join(parts).strip()

