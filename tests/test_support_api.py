import json

import httpx
import pytest

from app.auth import require_user
from app.main import app
from app.services.support_rag import SupportAnswer, get_support_rag_service


class FakeSupportRag:
    def __init__(self):
        self.calls = 0

    async def answer(self, message, history=None, user_context=None):
        self.calls += 1
        return SupportAnswer(
            id="msg_support",
            content=f"إجابة الدعم: {message}",
            sources=[],
            declined=False,
        )


async def fake_user():
    return {"sub": "user-1", "role": "TENANT"}


@pytest.fixture
async def client():
    fake_rag = FakeSupportRag()
    app.dependency_overrides[require_user] = fake_user
    app.dependency_overrides[get_support_rag_service] = lambda: fake_rag
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        yield test_client, fake_rag
    app.dependency_overrides.pop(get_support_rag_service, None)
    app.dependency_overrides.pop(require_user, None)


@pytest.mark.asyncio
async def test_explicit_human_request_emits_escalation_without_calling_rag(client):
    test_client, fake_rag = client
    response = await test_client.post(
        "/support/ai-chat/stream",
        json={"message": "أريد التحدث مع موظف خدمة العملاء"},
    )

    assert response.status_code == 200
    chunks = [
        json.loads(frame.removeprefix("data: ")) for frame in response.text.strip().split("\n\n")
    ]
    assert chunks[-1] == {
        "type": "done",
        "id": chunks[-1]["id"],
        "declined": False,
        "escalated": True,
        "suggestedGuide": [],
        "escalationReason": "طلب المستخدم التحدث مع موظف دعم فني بشكل صريح",
        "priority": "HIGH",
    }
    assert fake_rag.calls == 0


@pytest.mark.asyncio
async def test_normal_support_question_still_uses_rag(client):
    test_client, fake_rag = client
    response = await test_client.post(
        "/support/ai-chat/stream",
        json={"message": "كيف أضيف عقاراً؟"},
    )

    chunks = [
        json.loads(frame.removeprefix("data: "))
        for frame in response.text.strip().split("\n\n")
    ]
    assert chunks[-1]["type"] == "done"
    assert chunks[-1]["escalated"] is False
    assert fake_rag.calls == 1
