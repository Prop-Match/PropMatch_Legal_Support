"""FastAPI application entrypoint for the PropMatch Unified AI Service.

Exposes REST and SSE streaming endpoints for both the Legal RAG Assistant
and the Customer Support AI Assistant, along with health check endpoints.
"""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models import HealthResponse
from app.routers.legal_router import router as legal_router
from app.routers.support_router import router as support_router
from app.vector_store import get_vector_store

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="PropMatch Unified AI Service for Legal RAG & Customer Support Assistant",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# Register domain routers for Legal Chat and Support Chat
app.include_router(legal_router)
app.include_router(support_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Format Pydantic request validation errors into unified JSON error objects."""
    messages = [str(error.get("msg", "بيانات غير صالحة")) for error in exc.errors()]
    return JSONResponse(status_code=400, content={"statusCode": 400, "message": messages})


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    """Format HTTP exceptions into unified JSON response shape."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"statusCode": exc.status_code, "message": exc.detail},
        headers=exc.headers,
    )


@app.get("/health/live", response_model=HealthResponse, tags=["health"])
async def live() -> HealthResponse:
    """Liveness probe indicating the web application process is running."""
    return HealthResponse(status="ok", chroma="not_checked")


@app.get("/health/ready", response_model=HealthResponse, tags=["health"])
async def ready() -> HealthResponse:
    """Readiness probe checking connection and collection count in Dockerized ChromaDB."""
    store = get_vector_store()
    try:
        store.heartbeat()
        count = store.count()
        status_value = "ok" if count > 0 else "degraded"
        return HealthResponse(status=status_value, chroma="up", collection_count=count)
    except Exception:
        logger.exception("Chroma readiness check failed")
        return HealthResponse(status="degraded", chroma="down")
