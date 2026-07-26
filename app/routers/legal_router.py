"""API Router for the Legal Chatbot domain.

Provides buffered and SSE streaming endpoints for answering legal questions
regarding Egyptian Rental Law No. 4 of 1996 with Legal Disclaimer enforcement.
"""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.auth import CurrentUser
from app.llm import LlmProviderError
from app.models import ChatRequest, ChatResponse, DoneChunk, TokenChunk
from app.rag import LegalRagService, RagAnswer, get_rag_service

router = APIRouter(prefix="/legal-chat", tags=["legal-chat"])


@router.post("", response_model=ChatResponse)
async def legal_chat(
    payload: ChatRequest,
    _user: CurrentUser,
    rag: Annotated[LegalRagService, Depends(get_rag_service)],
) -> ChatResponse:
    """Return a buffered JSON legal answer with cited Egyptian law sources."""
    result = await _answer_or_502(rag, payload.message)
    return ChatResponse(
        id=result.id,
        content=result.content,
        declined=result.declined,
        sources=result.sources,
    )


@router.post("/stream")
async def legal_chat_stream(
    payload: ChatRequest,
    _user: CurrentUser,
    rag: Annotated[LegalRagService, Depends(get_rag_service)],
) -> StreamingResponse:
    """Stream token-by-token legal answer frames over Server-Sent Events (SSE)."""
    result = await _answer_or_502(rag, payload.message)

    async def _sse_generator():
        for token in result.content.splitlines(keepends=True):
            words = token.split(" ")
            for index, word in enumerate(words):
                suffix = " " if index < len(words) - 1 else ""
                chunk = TokenChunk(value=f"{word}{suffix}")
                yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        done = DoneChunk(id=result.id, declined=result.declined)
        yield f"data: {json.dumps(done.model_dump(), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _answer_or_502(rag: LegalRagService, message: str) -> RagAnswer:
    """Execute RAG pipeline and catch LLM/retrieval provider errors into 502/503 HTTP exceptions."""
    try:
        return await rag.answer(message)
    except LlmProviderError as exc:
        raise HTTPException(status_code=502, detail="تعذر الحصول على إجابة قانونية الآن") from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="خدمة البحث القانوني غير متاحة الآن") from exc
