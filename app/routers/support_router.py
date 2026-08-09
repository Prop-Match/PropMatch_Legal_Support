"""API Router for the Customer Support AI Assistant domain.

Provides SSE streaming endpoints for answering platform usage questions.
"""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import CurrentUser
from app.models import ChatRequest, DoneChunk, TokenChunk
from app.services.support_rag import SupportRagService, get_support_rag_service

router = APIRouter(prefix="/support", tags=["support-chat"])


@router.post("/ai-chat/stream")
@router.post("/stream")
async def support_chat_stream(
    payload: ChatRequest,
    _user: CurrentUser,
    rag: Annotated[SupportRagService, Depends(get_support_rag_service)],
) -> StreamingResponse:
    """Stream a read-only support answer over Server-Sent Events (SSE).

    Ticket creation is deliberately excluded. Users escalate explicitly through
    the authenticated NestJS customer-support endpoint.
    """

    async def _sse_generator():
        result = await rag.answer(payload.message, payload.history, payload.userContext)
        for token in result.content.splitlines(keepends=True):
            words = token.split(" ")
            for index, word in enumerate(words):
                suffix = " " if index < len(words) - 1 else ""
                chunk = TokenChunk(value=f"{word}{suffix}")
                yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)
        suggested_guides = []
        content_lower = result.content.lower()
        if any(w in content_lower for w in ["kyc", "توثيق", "الهوية"]):
            suggested_guides.append("KYC_GUIDE")
        if any(w in content_lower for w in ["عقار", "property", "إضافة عقار", "اضافة عقار"]):
            suggested_guides.append("PROPERTY_GUIDE")
        if any(w in content_lower for w in ["طلب سكن", "tenant request", "طلبات السكن"]):
            suggested_guides.append("REQUEST_GUIDE")
        done = DoneChunk(
            id=result.id,
            escalated=False,
            suggestedGuide=suggested_guides,
        )
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
