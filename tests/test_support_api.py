import json

import httpx
import pytest

from app.auth import require_user
from app.main import app
from app.services.support_rag import SupportAnswer, get_support_rag_service


class FakeSupportRag:
    async def answer(self, message, history=None, user_context=None):
        return SupportAnswer(
            id="msg_support",
            content=f"إجابة الدعم: {message}",
            sources=[],
            declined=False,
        )


async def fake_user():
    return {"sub": "user-1", "role": "TENANT"}


async def fake_support_rag():
    return FakeSupportRag()


@pytest.fixture
async def client():
    app.dependency_overrides[require_user] = fake_user
    app.dependency_overrides[get_support_rag_service] = fake_support_rag
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_support_rag_service, None)


@pytest.mark.asyncio
async def test_support_agent_never_creates_or_announces_a_ticket(client):
    response = await client.post(
        "/support/ai-chat/stream",
        json={"message": "أريد التحدث مع موظف خدمة العملاء"},
    )

    assert response.status_code == 200
    chunks = [
        json.loads(frame.removeprefix("data: ")) for frame in response.text.strip().split("\n\n")
    ]
    assert all(chunk["type"] != "escalate" for chunk in chunks)
    assert chunks[-1] == {
        "type": "done",
        "id": "msg_support",
        "declined": False,
        "escalated": False,
        "suggestedGuide": [],
    }
