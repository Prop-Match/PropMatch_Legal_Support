"""Retrieval-Augmented Generation (RAG) service for the Customer Support AI domain."""

from app.models import UserContext
import uuid
from dataclasses import dataclass
from app.config import get_settings
from app.llm import get_llm_client
from app.vector_store import get_vector_store


@dataclass
class SupportAnswer:
    """Indexed support answer data with metadata and status."""

    id: str
    content: str
    sources: list[dict]
    declined: bool


class SupportRagService:
    """Retrieves platform FAQ passages from ChromaDB and generates grounded support answers."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = get_vector_store(
            collection_name=self.settings.chroma_support_collection
        )
        self.llm = get_llm_client()

    async def answer(
        self,
        question: str,
        history: list[dict] | None = None,
        user_context: UserContext | None = None,
    ) -> SupportAnswer:
        """Search support_kb_v1 collection and generate a grounded support answer."""
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"

        # The model-directed SupportEscalationAgent owns action selection in
        # support_router before this RAG answer runs. Keep this legacy branch
        # unreachable until it is removed with the deprecated rule evaluator.
        if False:  # pragma: no cover
            return SupportAnswer(
                id=msg_id,
                content="جاري تحويل طلبك إلى فريق خدمة العملاء والدعم الفني لمساعدتك مباشرة. يمكنك النقر على زر 'تحدث مع موظف' أعلاه لفتح تذكرة مباشرة مع الإدارة.",
                sources=[],
                declined=False,
            )

        # Search support_kb_v1 collection
        results = await self.vector_store.asimilarity_search_with_score(
            question, k=self.settings.retrieval_top_k
        )

        relevant_docs = [
            doc
            for doc, score in results
            if score <= self.settings.relevance_max_distance
        ]

        if not relevant_docs:
            return SupportAnswer(
                id=msg_id,
                content="عذراً، أستطيع مساعدتك فقط في الأسئلة المتعلقة بمنصة PropMatch وكيفية استخدامها. إذا كنت ترغب في التواصل مع مسؤول الدعم، يسعدنا تحويلك لموظف خدمة العملاء.",
                sources=[],
                declined=True,
            )

        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
        user_context_str = ""
        if user_context:
            user_context_str = (
                f"معلومات المستخدم الحالي:\n"
                f"- الاسم: {user_context.fullName}\n"
                f"- نوع الحساب: {user_context.role}\n"
                f"- حالة التوثيق: {user_context.kycStatus}\n"
            )
            if user_context.kycRejectionReason:
                user_context_str += (
                    f"- سبب رفض التوثيق السابق: {user_context.kycRejectionReason}\n"
                )
        system_prompt = (
            "أنت مساعد الدعم الفني الذكي لمنصة PropMatch.\n"
            f"{user_context_str}\n"
            "تعليمات هامة حول الصور والأدلة البصرية:\n"
            "- عندما تحتوي الإرشادات على دليل بصري أو صورة توضيحية (مثل: ![دليل توثيق الهوية](/images/guides/kyc_guide.png))، يجب عليك عرض الصورة باستخدام صيغة الماركداون المخصصة للصور فقط: `![العنوان](/images/guides/اسم_الصورة.png)`.\n"
            "- لا تقم أبداً بكتابة مسار الصورة أو اسم الملف أو رابطها كأكواد برمجية (مثال: `/images/guides/...`) أو كنص عادي أو كرابط تشعبي عادي في الرد. نريد فقط إظهار الصورة للمستخدم دون إظهار رابط المسار الفني الخاص بها ككود أو نص.\n"
            "- لا تذكر مسارات المجلدات أو الملفات البرمجية للمستخدم.\n\n"
            f"أجب بناءً على الإرشادات التالية فقط:\n"
            f"{context_str}"
        )


        response_text = await self.llm.generate(
            system_prompt=system_prompt, prompt=question
        )

        return SupportAnswer(
            id=msg_id,
            content=response_text,
            sources=[doc.metadata for doc in relevant_docs],
            declined=False,
        )


def get_support_rag_service() -> SupportRagService:
    """Return an instance of SupportRagService for FastAPI dependency injection."""
    return SupportRagService()
