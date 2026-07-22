from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal

import httpx
from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings


class EmbeddingProviderError(RuntimeError):
    pass


class ItiEmbeddings(Embeddings):
    """LangChain embeddings adapter for the ITI `/student/embed` API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_sync(texts, "search_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed_sync([text], "search_query")[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_async(texts, "search_document")

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._embed_async([text], "search_query"))[0]

    def _embed_sync(
        self, texts: Sequence[str], input_type: Literal["search_document", "search_query"]
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.settings.embedding_timeout_seconds) as client:
            for batch in _batches(texts, self.settings.embedding_batch_size):
                response = client.post(
                    self.settings.embedding_api_url,
                    headers=self._headers(),
                    json=self._payload(batch, input_type),
                )
                vectors.extend(_handle_response(response, len(batch)))
        return vectors

    async def _embed_async(
        self, texts: Sequence[str], input_type: Literal["search_document", "search_query"]
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.settings.embedding_timeout_seconds) as client:
            for batch in _batches(texts, self.settings.embedding_batch_size):
                response = await client.post(
                    self.settings.embedding_api_url,
                    headers=self._headers(),
                    json=self._payload(batch, input_type),
                )
                vectors.extend(_handle_response(response, len(batch)))
        return vectors

    def _headers(self) -> dict[str, str]:
        api_key = self.settings.resolved_embedding_api_key
        if not api_key:
            raise EmbeddingProviderError("SBG_API_KEY or EMBEDDING_API_KEY is not configured")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _payload(self, texts: Sequence[str], input_type: str) -> dict[str, Any]:
        return {
            "model_id": self.settings.embedding_model_id,
            "texts": list(texts),
            "input_type": input_type,
        }


def _batches(texts: Sequence[str], size: int) -> list[list[str]]:
    return [list(texts[start : start + size]) for start in range(0, len(texts), size)]


def _handle_response(response: httpx.Response, expected_count: int) -> list[list[float]]:
    if not response.is_success:
        raise EmbeddingProviderError(_provider_error_message(response))
    try:
        vectors = extract_embeddings(response.json())
    except (ValueError, TypeError, KeyError) as exc:
        raise EmbeddingProviderError(f"Embedding provider request failed: {exc}") from exc
    if len(vectors) != expected_count:
        raise EmbeddingProviderError(
            f"Embedding provider returned {len(vectors)} vectors for {expected_count} texts"
        )
    dimensions = {len(vector) for vector in vectors}
    if not vectors or dimensions == {0} or len(dimensions) != 1:
        raise EmbeddingProviderError("Embedding provider returned invalid vector dimensions")
    return vectors


def _provider_error_message(response: httpx.Response) -> str:
    code = "UNKNOWN_PROVIDER_ERROR"
    message = response.reason_phrase or "Embedding provider rejected the request"
    details: Any = None
    try:
        payload = response.json()
        error = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(error, dict):
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
            details = error.get("details")
    except ValueError:
        pass
    suffix = f" Details: {details}" if details else ""
    return f"Embedding API HTTP {response.status_code} [{code}]: {message}.{suffix}"


def extract_embeddings(data: Any) -> list[list[float]]:
    """Parse common batch shapes while preserving input order."""
    if not isinstance(data, dict):
        raise TypeError("Embedding response must be a JSON object")
    raw = data.get("embeddings") or data.get("vectors")
    if raw is None and isinstance(data.get("data"), list):
        raw = [item.get("embedding") for item in data["data"] if isinstance(item, dict)]
    if isinstance(raw, dict):
        raw = raw.get("embeddings") or raw.get("vectors")
    if not isinstance(raw, list):
        raise KeyError("No supported embeddings field found")
    vectors: list[list[float]] = []
    for vector in raw:
        is_numeric_vector = isinstance(vector, list) and all(
            isinstance(value, int | float) for value in vector
        )
        if not is_numeric_vector:
            raise TypeError("Each embedding must be a numeric list")
        vectors.append([float(value) for value in vector])
    return vectors


@lru_cache
def get_embeddings() -> ItiEmbeddings:
    return ItiEmbeddings(get_settings())
