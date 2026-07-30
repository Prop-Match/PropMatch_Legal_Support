import json

import httpx
import pytest

from app.auth import require_user
from app.main import app
from app.rag import RagAnswer, get_rag_service


class FakeRag:
    async def answer(self, message):
        return RagAnswer("msg_test", f"إجابة: {message}", False, [])


async def fake_user():
    return {"sub": "user-1", "role": "tenant"}


async def fake_rag():
    return FakeRag()


app.dependency_overrides[require_user] = fake_user
app.dependency_overrides[get_rag_service] = fake_rag


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_buffered_chat_contract(client):
    response = await client.post("/legal-chat", json={"message": "عقد إيجار"})

    assert response.status_code == 200
    assert response.json() == {
        "id": "msg_test",
        "content": "إجابة: عقد إيجار",
        "declined": False,
        "sources": [],
    }


@pytest.mark.asyncio
async def test_stream_contract_matches_frontend_parser(client):
    response = await client.post("/legal-chat/stream", json={"message": "عقد إيجار"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks = [
        json.loads(frame.removeprefix("data: "))
        for frame in response.text.strip().split("\n\n")
    ]
    answer = "".join(chunk["value"] for chunk in chunks if chunk["type"] == "token")
    assert answer == "إجابة: عقد إيجار"
    assert chunks[-1] == {
        "type": "done",
        "id": "msg_test",
        "declined": False,
        "escalated": False,
        "suggestedGuide": [],
    }



@pytest.mark.asyncio
async def test_blank_message_is_a_normal_json_400(client):
    response = await client.post("/legal-chat/stream", json={"message": "   "})

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["statusCode"] == 400
