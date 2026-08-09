import json

import pytest

from app.llm import LlmProviderError
from app.models import ChatRequest
from app.routers.support_router import support_chat_stream
from app.services.support_agent import (
    AgentDecision,
    SupportEscalationAgent,
    parse_agent_decision,
)
from app.services.support_rag import SupportAnswer


def test_parse_agent_escalation_tool_decision():
    decision = parse_agent_decision(
        '{"action":"CREATE_SUPPORT_TICKET","reason":"طلب مساعدة بشرية","priority":"HIGH"}'
    )

    assert decision.should_escalate is True
    assert decision.priority == "HIGH"


def test_parse_agent_rejects_unknown_tool():
    with pytest.raises(ValueError):
        parse_agent_decision('{"action":"DELETE_ACCOUNT","reason":"","priority":"NORMAL"}')


@pytest.mark.asyncio
async def test_agent_fails_closed_when_model_is_unavailable():
    class FailingLlm:
        async def generate(self, **_kwargs):
            raise LlmProviderError("unavailable")

    decision = await SupportEscalationAgent(llm=FailingLlm()).decide("help", [])

    assert decision == AgentDecision(action="RESPOND")


@pytest.mark.asyncio
async def test_escalation_is_emitted_as_a_terminal_nestjs_intent(monkeypatch):
    class EscalatingAgent:
        async def decide(self, *_args, **_kwargs):
            return AgentDecision(
                action="CREATE_SUPPORT_TICKET",
                reason="طلب التحدث مع موظف دعم",
                priority="HIGH",
            )

    class FakeRag:
        async def answer(self, *_args, **_kwargs):
            return SupportAnswer(
                id="answer-1", content="سيتم تحويل طلبك.", sources=[], declined=False
            )

    monkeypatch.setattr("app.routers.support_router.SupportEscalationAgent", EscalatingAgent)
    response = await support_chat_stream(
        ChatRequest(message="أريد التحدث مع موظف"),
        _user={"sub": "user-1", "role": "TENANT"},
        rag=FakeRag(),
    )
    body = "".join([chunk async for chunk in response.body_iterator])
    events = [
        json.loads(frame.removeprefix("data: "))
        for frame in body.strip().split("\n\n")
    ]

    assert events[-1] == {
        "type": "done",
        "id": "answer-1",
        "declined": False,
        "escalated": True,
        "escalationReason": "طلب التحدث مع موظف دعم",
        "priority": "HIGH",
        "suggestedGuide": [],
    }
