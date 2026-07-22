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


@dataclass(frozen=True)
class RagAnswer:
    id: str
    content: str
    declined: bool
    sources: list[Source]


class LegalRagService:
    def __init__(self, settings: Settings, store: LegalVectorStore, llm: ItiLlmClient) -> None:
        self.settings = settings
        self.store = store
        self.llm = llm

    async def answer(self, message: str) -> RagAnswer:
        passages = await self.store.query(message, self.settings.retrieval_top_k)
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
        normalized = message.casefold()
        return any(term.casefold() in normalized for term in DOMAIN_TERMS)

    def _has_relevant_passage(self, passages: list[RetrievedPassage]) -> bool:
        return bool(passages and passages[0].distance <= self.settings.relevance_max_distance)

    @staticmethod
    def _format_context(passages: list[RetrievedPassage]) -> str:
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
    return bool(re.search(r"(ليست|ليس).{0,30}(استشارة|فتوى).{0,30}(قانونية|ملزمة)", text))


def _unique_sources(passages: list[RetrievedPassage]) -> list[Source]:
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
    settings = get_settings()
    return LegalRagService(settings, get_vector_store(), ItiLlmClient(settings))
