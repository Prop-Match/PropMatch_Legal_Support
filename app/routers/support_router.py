"""API Router for the Customer Support AI Assistant domain.

Provides SSE streaming endpoints for answering platform usage questions,
evaluating multi-factor human escalation rules, and recommending ticket creation.
"""

import asyncio
import json
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.auth import CurrentUser
from app.config import get_settings
from app.models import ChatRequest, DoneChunk, TokenChunk
from app.services.support_agent import (
    AgentDecision,
    SupportEscalationAgent,
    SupportTicketTool,
    new_agent_run_id,
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

    The model chooses whether to answer or call the private
    create-support-ticket tool. NestJS validates and persists any ticket.
    """
    decision = await SupportEscalationAgent().decide(payload.message, payload.history)
    ticket_id: str | None = None
    if decision.should_escalate:
        try:
            run_id = (
                str(payload.clientRequestId)
                if payload.clientRequestId
                else new_agent_run_id()
            )
            ticket = await SupportTicketTool(get_settings()).create(
                run_id=run_id,
                user_id=_user["sub"],
                message=payload.message,
                reason=decision.reason,
                priority=decision.priority,
            )
            ticket_id = ticket.ticket_id
        except (httpx.HTTPError, RuntimeError):
            # Never claim that an escalation completed when the tool failed.
            decision = AgentDecision(action="RESPOND")

    async def _sse_generator():
        # 1. Yield the completed model-selected tool call, if any.
        if ticket_id:
            escalate_chunk = {
                "type": "escalate",
                "shouldEscalate": True,
                "reason": decision.reason,
                "priority": decision.priority,
                "ticketId": ticket_id,
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
        if any(w in content_lower for w in ["عقار", "property", "إضافة عقار", "اضافة عقار"]):
            suggested_guides.append("PROPERTY_GUIDE")
        if any(w in content_lower for w in ["طلب سكن", "tenant request", "طلبات السكن"]):
            suggested_guides.append("REQUEST_GUIDE")
        done = DoneChunk(
            id=result.id,
            escalated=ticket_id is not None,
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
