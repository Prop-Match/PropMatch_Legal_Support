import pytest

from app.config import Settings
from app.rag import DECLINE_MESSAGE, DISCLAIMER, LegalRagService
from app.vector_store import RetrievedPassage


class FakeStore:
    def __init__(self, passages):
        self.passages = passages
        self.requested_top_k = None

    async def query(self, _message, top_k):
        self.requested_top_k = top_k
        return self.passages


class FakeLlm:
    def __init__(self, answer="إجابة موثقة"):
        self.result = answer
        self.calls = 0
        self.context = ""

    async def answer(self, _question, context):
        self.calls += 1
        self.context = context
        return self.result


@pytest.mark.asyncio
async def test_off_topic_question_is_declined_without_llm_call():
    llm = FakeLlm()
    store = FakeStore([RetrievedPassage("نص", {}, 1.2)])
    service = LegalRagService(Settings(auth_required=False), store, llm)

    result = await service.answer("ما هو أفضل مطعم؟")

    assert result.declined is True
    assert result.content == DECLINE_MESSAGE
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_legal_question_is_grounded_and_gets_disclaimer():
    llm = FakeLlm()
    passage = RetrievedPassage(
        "المادة الأولى: نص قانوني",
        {
            "title": "قانون الاختبار",
            "article": "المادة الأولى",
            "file": "law.txt",
            "source_url": "https://example.test/law",
            "extraction_method": "OCR",
        },
        0.2,
    )
    service = LegalRagService(Settings(auth_required=False), FakeStore([passage]), llm)

    result = await service.answer("ما حكم عقد الإيجار؟")

    assert result.declined is False
    assert DISCLAIMER in result.content
    assert result.sources[0].article == "المادة الأولى"
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_cross_domain_passages_are_removed_before_prompting():
    llm = FakeLlm()
    employment = RetrievedPassage(
        "ينتهي عقد العمل، ويجب أن يسبقه إخطار تحدد مدته القوانين الخاصة.",
        {
            "title": "القانون المدني",
            "article": "المادة 694",
            "file": "civil.txt",
        },
        0.1,
    )
    lease = RetrievedPassage(
        "إذا عقد الإيجار دون اتفاق على مدة ينتهي بعد التنبيه بالإخلاء.",
        {
            "title": "القانون المدني",
            "article": "المادة 563",
            "file": "civil.txt",
        },
        0.2,
    )
    store = FakeStore([employment, lease])
    service = LegalRagService(
        Settings(auth_required=False, retrieval_top_k=5),
        store,
        llm,
    )

    result = await service.answer("ما مدة الإخطار قبل إنهاء عقد الإيجار؟")

    assert store.requested_top_k == 15
    assert "عقد العمل" not in llm.context
    assert "عقد الإيجار" in llm.context
    assert [source.article for source in result.sources] == ["المادة 563"]
