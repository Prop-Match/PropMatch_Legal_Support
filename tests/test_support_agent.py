import pytest

from app.llm import LlmProviderError
from app.services.support_agent import (
    AgentDecision,
    SupportEscalationAgent,
    parse_agent_decision,
)


def test_agent_parses_a_ticket_tool_decision():
    decision = parse_agent_decision(
        '{"action":"CREATE_SUPPORT_TICKET","reason":"طلب مساعدة بشرية","priority":"HIGH"}'
    )

    assert decision.should_escalate is True
    assert decision.priority == "HIGH"


def test_agent_rejects_an_unknown_tool():
    with pytest.raises(ValueError):
        parse_agent_decision('{"action":"DELETE_ACCOUNT","reason":"","priority":"NORMAL"}')


def test_agent_accepts_json_wrapped_in_markdown():
    decision = parse_agent_decision(
        "```json\n"
        '{"action":"CREATE_SUPPORT_TICKET","reason":"طلب دعم","priority":"HIGH"}'
        "\n```"
    )

    assert decision.should_escalate is True


@pytest.mark.asyncio
async def test_explicit_handoff_overrides_an_inconsistent_model_response():
    class RespondingLlm:
        async def generate(self, **_kwargs):
            return '{"action":"RESPOND","reason":"","priority":"NORMAL"}'

    decision = await SupportEscalationAgent(llm=RespondingLlm()).decide(
        "أريد التحدث مع موظف دعم", []
    )

    assert decision.should_escalate is True
    assert decision.priority == "HIGH"


@pytest.mark.asyncio
async def test_agent_fails_closed_when_the_model_is_unavailable():
    class FailingLlm:
        async def generate(self, **_kwargs):
            raise LlmProviderError("unavailable")

    decision = await SupportEscalationAgent(llm=FailingLlm()).decide("help", [])

    assert decision == AgentDecision(action="RESPOND")
