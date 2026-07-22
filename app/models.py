from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def reject_blank_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("الرسالة مطلوبة")
        return value


class Source(BaseModel):
    title: str
    article: str | None = None
    file: str
    source_url: str | None = None


class ChatResponse(BaseModel):
    id: str
    content: str
    declined: bool
    sources: list[Source] = []


class TokenChunk(BaseModel):
    type: Literal["token"] = "token"
    value: str


class DoneChunk(BaseModel):
    type: Literal["done"] = "done"
    id: str
    declined: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    chroma: Literal["up", "down", "not_checked"]
    collection_count: int | None = None

