import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.auth import CurrentUser
from app.config import get_settings
from app.llm import LlmProviderError
from app.models import ChatRequest, ChatResponse, DoneChunk, HealthResponse, TokenChunk
from app.rag import LegalRagService, RagAnswer, get_rag_service
from app.vector_store import get_vector_store

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Grounded RAG API over the supplied Egyptian real-estate law corpus.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    messages = [str(error.get("msg", "بيانات غير صالحة")) for error in exc.errors()]
    return JSONResponse(status_code=400, content={"statusCode": 400, "message": messages})


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"statusCode": exc.status_code, "message": exc.detail},
        headers=exc.headers,
    )


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
async def live() -> HealthResponse:
    return HealthResponse(status="ok", chroma="not_checked")


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def ready() -> HealthResponse:
    store = get_vector_store()
    try:
        await asyncio.to_thread(store.heartbeat)
        count = await asyncio.to_thread(store.count)
        status_value = "ok" if count > 0 else "degraded"
        return HealthResponse(status=status_value, chroma="up", collection_count=count)
    except Exception:
        logger.exception("Chroma readiness check failed")
        return HealthResponse(status="degraded", chroma="down")


@app.post("/legal-chat", response_model=ChatResponse, tags=["legal-chat"])
async def legal_chat(
    payload: ChatRequest,
    _user: CurrentUser,
    rag: Annotated[LegalRagService, Depends(get_rag_service)],
) -> ChatResponse:
    result = await _answer_or_502(rag, payload.message)
    return ChatResponse(
        id=result.id,
        content=result.content,
        declined=result.declined,
        sources=result.sources,
    )


@app.post("/legal-chat/stream", tags=["legal-chat"])
async def legal_chat_stream(
    payload: ChatRequest,
    _user: CurrentUser,
    rag: Annotated[LegalRagService, Depends(get_rag_service)],
) -> StreamingResponse:
    # Resolve gates/retrieval/provider errors before creating the response, so
    # clients receive an ordinary JSON 4xx/5xx rather than a broken 200 stream.
    result = await _answer_or_502(rag, payload.message)
    return StreamingResponse(
        _sse_answer(result),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _answer_or_502(rag: LegalRagService, message: str) -> RagAnswer:
    try:
        return await rag.answer(message)
    except LlmProviderError as exc:
        logger.exception("LLM provider failure")
        raise HTTPException(status_code=502, detail="تعذر الحصول على إجابة قانونية الآن") from exc
    except Exception as exc:
        logger.exception("Legal RAG failure")
        raise HTTPException(status_code=503, detail="خدمة البحث القانوني غير متاحة الآن") from exc


async def _sse_answer(result: RagAnswer) -> AsyncIterator[str]:
    # The ITI provider returns one completed response; split it on word
    # boundaries to retain the frontend's progressive SSE experience.
    for token in result.content.splitlines(keepends=True):
        words = token.split(" ")
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            chunk = TokenChunk(value=f"{word}{suffix}")
            yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)
    done = DoneChunk(id=result.id, declined=result.declined)
    yield f"data: {json.dumps(done.model_dump(), ensure_ascii=False)}\n\n"
