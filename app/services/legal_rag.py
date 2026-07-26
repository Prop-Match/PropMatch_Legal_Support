"""Retrieval-Augmented Generation (RAG) service for the Legal Chatbot domain."""

import uuid
from dataclasses import dataclass

from app.config import get_settings
from app.llm import get_llm_client
from app.vector_store import get_vector_store

LEGAL_DISCLAIMER = "\n\n⚠️ *تنبيه: هذه إجابة استرشادية مبنية على قانون الإيجارات المصري ولا تعتبر استشارة قانونية رسمية.*"


@dataclass
class LegalAnswer:
    """Legal answer data with cited law sources and legal disclaimer."""

    id: str
    content: str
    sources: list[dict]
    declined: bool


class LegalRagService:
    """Retrieves Egyptian real-estate law passages from ChromaDB and generates grounded legal answers."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = get_vector_store(
            collection_name=self.settings.chroma_legal_collection
        )
        self.llm = get_llm_client()

    async def answer(self, query: str) -> LegalAnswer:
        """Search egypt_real_estate_laws_v1 collection and generate grounded legal answer."""
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"

        results = await self.vector_store.asimilarity_search_with_score(
            query, k=self.settings.retrieval_top_k
        )
        relevant_docs = [
            doc for doc, score in results if score <= self.settings.relevance_max_distance
        ]

        if not relevant_docs:
            return LegalAnswer(
                id=msg_id,
                content="أقدر أساعدك فقط في أسئلة الإيجار والقانون العقاري في مصر.",
                sources=[],
                declined=True,
            )

        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
        system_prompt = f"أنت المستشار القانوني العقاري لمنصة PropMatch. أجب فقط بناءً على نصوص القوانين المصرية التالية:\n{context_str}"

        response_text = await self.llm.generate(system_prompt=system_prompt, prompt=query)
        final_text = f"{response_text}{LEGAL_DISCLAIMER}"

        return LegalAnswer(
            id=msg_id,
            content=final_text,
            sources=[doc.metadata for doc in relevant_docs],
            declined=False,
        )


def get_legal_rag_service() -> LegalRagService:
    """Return an instance of LegalRagService for FastAPI dependency injection."""
    return LegalRagService()
