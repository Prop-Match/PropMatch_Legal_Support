"""ChromaDB adapter used to ingest and retrieve RAG passages for legal and support domains."""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.chunking import RagChunk
from app.config import Settings, get_settings
from app.embeddings import get_embeddings


@dataclass(frozen=True)
class RetrievedPassage:
    """Retrieved text, source metadata, and its cosine distance from the query."""

    document: str
    metadata: dict[str, Any]
    distance: float


class LegalVectorStore:
    """LangChain Chroma adapter backed by the external Chroma Docker service."""

    def __init__(self, settings: Settings, embeddings: Embeddings, collection_name: str | None = None) -> None:
        self.settings = settings
        self.embeddings = embeddings
        target_collection = collection_name or settings.chroma_legal_collection
        self.store = Chroma(
            collection_name=target_collection,
            embedding_function=embeddings,
            host=settings.chroma_host,
            port=settings.chroma_port,
            ssl=settings.chroma_ssl,
            collection_metadata={"hnsw:space": "cosine", "corpus": "propmatch-rag"},
        )

    def heartbeat(self) -> int:
        """Check whether the external ChromaDB server responds."""
        return self.store._client.heartbeat()

    def count(self) -> int:
        """Return the number of chunks currently indexed in the collection."""
        return self.store._collection.count()

    def ingest(self, chunks: Sequence[RagChunk], batch_size: int = 64) -> int:
        """Upsert deterministic RAG chunks in bounded batches."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [
                Document(page_content=chunk.document, metadata=chunk.metadata, id=chunk.id)
                for chunk in batch
            ]
            self.store.add_documents(documents=documents, ids=[chunk.id for chunk in batch])
        return len(chunks)

    async def query(self, message: str, top_k: int) -> list[RetrievedPassage]:
        """Embed a question and retrieve the nearest indexed passages."""
        if self.count() == 0:
            return []
        results = await self.store.asimilarity_search_with_score(
            message, k=min(top_k, self.count())
        )
        return [
            RetrievedPassage(
                document=document.page_content,
                metadata=document.metadata,
                distance=float(distance),
            )
            for document, distance in results
        ]

    async def asimilarity_search_with_score(self, message: str, k: int):
        """Proxy method for similarity search with distance score."""
        if self.count() == 0:
            return []
        return await self.store.asimilarity_search_with_score(
            message, k=min(k, self.count())
        )


def get_vector_store(collection_name: str | None = None) -> LegalVectorStore:
    """Return the vector store collection adapter for legal or support domains."""
    settings = get_settings()
    target_collection = collection_name or settings.chroma_legal_collection
    return LegalVectorStore(settings, get_embeddings(), collection_name=target_collection)
