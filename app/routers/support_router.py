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
from app.services.support_agent import (
    SupportEscalationAgent,
)
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

    The model chooses whether to answer or request a human handoff. NestJS
    validates and persists the ticket using the already-authenticated user.
    """
    decision = await SupportEscalationAgent().decide(payload.message, payload.history)

    async def _sse_generator():
        # Stream only answer content before the terminal escalation intent.
        # The NestJS gateway owns ticket persistence; this service never calls
        # the database or an obsolete internal ticket endpoint directly.
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
            escalated=decision.should_escalate,
            escalationReason=decision.reason if decision.should_escalate else None,
            priority=decision.priority if decision.should_escalate else None,
            suggestedGuide=suggested_guides,
        )
        yield f"data: {json.dumps(done.model_dump(exclude_none=True), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
