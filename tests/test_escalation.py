from app.services.escalation import evaluate_escalation


def test_explicit_human_request_is_high_priority():
    decision = evaluate_escalation("عايز اتكلم مع موظف دعم")

    assert decision.should_escalate is True
    assert decision.priority == "HIGH"


def test_payment_or_security_emergency_is_urgent():
    decision = evaluate_escalation("تم خصم المبلغ ولم تصل الخدمة")

    assert decision.should_escalate is True
    assert decision.priority == "URGENT"


def test_repeated_unresolved_attempts_are_escalated():
    history = [{"role": "user", "content": str(index)} for index in range(4)]

    decision = evaluate_escalation("ما زالت المشكلة موجودة", history)

    assert decision.should_escalate is True
    assert decision.priority == "NORMAL"


def test_regular_question_is_not_escalated():
    decision = evaluate_escalation("كيف أضيف عقاراً جديداً؟")

    assert decision.should_escalate is False
