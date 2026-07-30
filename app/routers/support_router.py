"""API Router for the Customer Support AI Assistant domain.

Provides SSE streaming endpoints for answering platform usage questions,
evaluating multi-factor human escalation rules, and recommending ticket creation.
"""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import CurrentUser
from app.models import ChatRequest, DoneChunk, TokenChunk
from app.services.escalation import evaluate_escalation
from app.services.support_rag import SupportRagService, get_support_rag_service

router = APIRouter(prefix="/support", tags=["support-chat"])


@router.post("/ai-chat/stream")
@router.post("/stream")
async def support_chat_stream(
    payload: ChatRequest,
    _user: CurrentUser,
    rag: Annotated[SupportRagService, Depends(get_support_rag_service)],
) -> StreamingResponse:
    """Stream token-by-token support answer frames over Server-Sent Events (SSE).

    Evaluates multi-factor escalation rules first. If human escalation is required,
    yields an initial `escalate` chunk before streaming answer tokens.
    """
    escalation = evaluate_escalation(payload.message, payload.history)

    async def _sse_generator():
        # 1. Yield escalation recommendation chunk if required
        if escalation.get("shouldEscalate"):
            escalate_chunk = {
                "type": "escalate",
                "shouldEscalate": True,
                "reason": escalation.get("reason"),
                "priority": escalation.get("priority"),
            }
            yield f"data: {json.dumps(escalate_chunk, ensure_ascii=False)}\n\n"

        # 2. Stream tokens
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
        if any(
            w in content_lower for w in ["عقار", "property", "إضافة عقار", "اضافة عقار"]
        ):
            suggested_guides.append("PROPERTY_GUIDE")
        if any(
            w in content_lower for w in ["طلب سكن", "tenant request", "طلبات السكن"]
        ):
            suggested_guides.append("REQUEST_GUIDE")
        done = DoneChunk(
            id=result.id,
            escalated=escalation.get("shouldEscalate", False),
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
