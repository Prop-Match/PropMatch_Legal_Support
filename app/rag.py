"""Grounded legal retrieval and answer orchestration.

This module is the stable legal pipeline used by the legal router: retrieve
candidate passages, reject unrelated material, build constrained context, call
the ITI LLM, enforce the disclaimer, and expose source metadata.
"""

import re
import uuid
from dataclasses import dataclass
from functools import lru_cache

from app.config import Settings, get_settings
from app.llm import ItiLlmClient
from app.models import Source
from app.vector_store import LegalVectorStore, RetrievedPassage, get_vector_store

DECLINE_MESSAGE = "أقدر أساعدك فقط في أسئلة الإيجار والقانون العقاري في مصر."
DISCLAIMER = "هذه معلومات قانونية عامة وليست استشارة قانونية ملزمة؛ راجع محاميًا في حالتك الخاصة."
DOMAIN_TERMS = (
    "إيجار",
    "ايجار",
    "عقد",
    "مالك",
    "مؤجر",
    "مستأجر",
    "إخلاء",
    "اخلاء",
    "شقة",
    "عقار",
    "ملكية",
    "تمليك",
    "بناء",
    "مبنى",
    "ترخيص",
    "تصالح",
    "مخالفة",
    "شهر عقاري",
    "تسجيل",
    "ضريبة عقارية",
    "قانون مدني",
    "قانون",
    "صيانة",
    "تأمين",
    "فسخ",
    "إنذار",
    "اخطار",
    "إخطار",
    "أجرة",
    "اجرة",
    "محل",
    "أرض",
)
REAL_ESTATE_PASSAGE_TERMS = (
    "إيجار",
    "ايجار",
    "المؤجر",
    "المستأجر",
    "العين المؤجرة",
    "الأجرة",
    "عقار",
    "الأراضي",
    "المنازل",
    "المباني",
)
OUT_OF_SCOPE_PASSAGE_TERMS = (
    "عقد العمل",
    "رب العمل",
    "العامل",
    "ملحقات الأجر",
    "الأجر المحدد",
    "الشركة",
    "الشرآة",
    "الشركاء",
    "الشرآاء",
)


@dataclass(frozen=True)
class RagAnswer:
    """Internal legal result shared by buffered and streamed endpoints."""

    id: str
    content: str
    declined: bool
    sources: list[Source]


class LegalRagService:
    """Coordinate legal relevance checks, Chroma retrieval, and LLM generation."""

    def __init__(self, settings: Settings, store: LegalVectorStore, llm: ItiLlmClient) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm

    async def answer(self, message: str) -> RagAnswer:
        """Generate a grounded answer or a safe off-topic/insufficient-context reply."""

        candidate_count = min(self.settings.retrieval_top_k * 3, 30)
        candidates = await self.store.query(message, candidate_count)
        passages = [passage for passage in candidates if self._is_real_estate_passage(passage)][
            : self.settings.retrieval_top_k
        ]
        on_topic = self._has_domain_term(message) or self._has_relevant_passage(passages)
        answer_id = f"msg_{uuid.uuid4().hex}"
        if not on_topic:
            return RagAnswer(answer_id, DECLINE_MESSAGE, True, [])
        if not passages:
            content = f"لا تتوفر مقتطفات قانونية كافية للإجابة عن هذا السؤال.\n\n{DISCLAIMER}"
            return RagAnswer(answer_id, content, False, [])

        context = self._format_context(passages)
        content = await self.llm.answer(message, context)
        if not _contains_disclaimer(content):
            content = f"{content.rstrip()}\n\n{DISCLAIMER}"
        return RagAnswer(answer_id, content, False, _unique_sources(passages))

    @staticmethod
    def _has_domain_term(message: str) -> bool:
        """Recognize explicit Arabic real-estate terminology in the question."""

        normalized = message.casefold()
        return any(term.casefold() in normalized for term in DOMAIN_TERMS)

    def _has_relevant_passage(self, passages: list[RetrievedPassage]) -> bool:
        """Use cosine distance as a second signal when keywords are inconclusive."""

        return bool(passages and passages[0].distance <= self.settings.relevance_max_distance)

    @staticmethod
    def _is_real_estate_passage(passage: RetrievedPassage) -> bool:
        """Remove obvious cross-domain legal matches before prompting the LLM."""

        normalized = passage.document.casefold()
        has_real_estate_term = any(
            term.casefold() in normalized for term in REAL_ESTATE_PASSAGE_TERMS
        )
        has_out_of_scope_term = any(
            term.casefold() in normalized for term in OUT_OF_SCOPE_PASSAGE_TERMS
        )
        return has_real_estate_term or not has_out_of_scope_term

    @staticmethod
    def _format_context(passages: list[RetrievedPassage]) -> str:
        """Render retrieved text and provenance into the legal prompt context."""

        blocks: list[str] = []
        for index, passage in enumerate(passages, start=1):
            metadata = passage.metadata
            blocks.append(
                f"[المصدر {index}]\n"
                f"القانون: {metadata.get('title', 'غير محدد')}\n"
                f"المادة/القسم: {metadata.get('article', 'غير محدد')}\n"
                f"طريقة الاستخراج: {metadata.get('extraction_method', 'غير محددة')}\n"
                f"النص:\n{passage.document}"
            )
        return "\n\n".join(blocks)


def _contains_disclaimer(text: str) -> bool:
    """Detect whether the provider already included the required disclaimer."""

    return bool(re.search(r"(ليست|ليس).{0,30}(استشارة|فتوى).{0,30}(قانونية|ملزمة)", text))


def _unique_sources(passages: list[RetrievedPassage]) -> list[Source]:
    """Deduplicate public source records while preserving retrieval order."""

    sources: list[Source] = []
    seen: set[tuple[str, str, str]] = set()
    for passage in passages:
        metadata = passage.metadata
        key = (
            str(metadata.get("file", "")),
            str(metadata.get("article", "")),
            str(metadata.get("title", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            Source(
                title=key[2] or "غير محدد",
                article=key[1] or None,
                file=key[0],
                source_url=str(metadata.get("source_url", "")) or None,
            )
        )
    return sources


@lru_cache
def get_rag_service() -> LegalRagService:
    """Build and cache the legal RAG dependency used by HTTP routes."""

    settings = get_settings()
    return LegalRagService(settings, get_vector_store(), ItiLlmClient(settings))
