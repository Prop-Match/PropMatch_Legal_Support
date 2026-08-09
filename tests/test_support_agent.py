import pytest

from app.llm import LlmProviderError
from app.services.support_agent import (
    AgentDecision,
    SupportEscalationAgent,
    parse_agent_decision,
)


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
