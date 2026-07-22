import pytest

from app.config import Settings
from app.rag import DECLINE_MESSAGE, DISCLAIMER, LegalRagService
from app.vector_store import RetrievedPassage


class FakeStore:
    def __init__(self, passages):
        self.passages = passages

    async def query(self, _message, _top_k):
        return self.passages


class FakeLlm:
    def __init__(self, answer="إجابة موثقة"):
        self.result = answer
        self.calls = 0

    async def answer(self, _question, _context):
        self.calls += 1
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
