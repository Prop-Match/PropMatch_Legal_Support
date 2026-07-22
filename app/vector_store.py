from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.chunking import LawChunk
from app.config import Settings, get_settings
from app.embeddings import ItiEmbeddings, get_embeddings


@dataclass(frozen=True)
class RetrievedPassage:
    document: str
    metadata: dict[str, Any]
    distance: float


class LegalVectorStore:
    """LangChain Chroma adapter backed by the external Chroma Docker service."""

    def __init__(self, settings: Settings, embeddings: ItiEmbeddings) -> None:
        self.settings = settings
        self.embeddings = embeddings
        self.store = Chroma(
            collection_name=settings.chroma_collection,
            embedding_function=embeddings,
            host=settings.chroma_host,
            port=settings.chroma_port,
            ssl=settings.chroma_ssl,
            collection_metadata={"hnsw:space": "cosine", "corpus": "egypt-real-estate-laws"},
        )

    def heartbeat(self) -> int:
        return self.store._client.heartbeat()

    def count(self) -> int:
        return self.store._collection.count()

    def ingest(self, chunks: Sequence[LawChunk], batch_size: int = 64) -> int:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            documents = [
                Document(page_content=chunk.document, metadata=chunk.metadata, id=chunk.id)
                for chunk in batch
            ]
            # LangChain's Chroma integration performs an upsert for supplied IDs,
            # making corpus ingestion safe to repeat.
            self.store.add_documents(documents=documents, ids=[chunk.id for chunk in batch])
        return len(chunks)

    async def query(self, message: str, top_k: int) -> list[RetrievedPassage]:
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


@lru_cache
def get_vector_store() -> LegalVectorStore:
    return LegalVectorStore(get_settings(), get_embeddings())
