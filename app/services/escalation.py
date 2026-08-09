"""Deterministic policy for requesting a human-support handoff."""

from dataclasses import dataclass
from typing import Literal

EscalationPriority = Literal["NORMAL", "HIGH", "URGENT"]


@dataclass(frozen=True)
class EscalationDecision:
    """A bounded decision consumed by the trusted NestJS gateway."""

    should_escalate: bool
    reason: str = ""
    priority: EscalationPriority = "NORMAL"


def evaluate_escalation(
    message: str, history: list[dict[str, str]] | None = None
) -> EscalationDecision:
    """Request escalation only for explicit, urgent, or repeatedly unresolved cases."""

    normalized = " ".join(message.lower().strip().split())

    human_requests = (
        "خدمة العملاء",
        "خدمة عملاء",
        "خدمه العملاء",
        "تحدث مع شخص",
        "تحدث مع موظف",
        "التحدث مع شخص",
        "التحدث مع موظف",
        "أريد التحدث",
        "اريد التحدث",
        "تحويل لموظف",
        "حولني لموظف",
        "حوّلني لموظف",
        "دعم بشري",
        "تحدث مع إنسان",
        "تحدث مع انسان",
        "كلم موظف",
        "كلمني موظف",
        "مسؤول الدعم",
        "موظف دعم",
    )
    if any(phrase in normalized for phrase in human_requests):
        return EscalationDecision(
            should_escalate=True,
            reason="طلب المستخدم التحدث مع موظف دعم فني بشكل صريح",
            priority="HIGH",
        )

    urgent_phrases = (
        "سرقة",
        "احتيال",
        "اختراق",
        "خصم بدون علم",
        "خصم دون علم",
        "عملية غير مصرح",
        "دفعة غير مصرح",
        "اتخصم المبلغ ولم",
        "تم خصم المبلغ ولم",
    )
    if any(phrase in normalized for phrase in urgent_phrases):
        return EscalationDecision(
            should_escalate=True,
            reason="تم اكتشاف حالة دفع أو أمان تحتاج إلى مراجعة بشرية عاجلة",
            priority="URGENT",
        )

    user_turns = sum(
        1 for item in history or [] if str(item.get("role", "")).lower() == "user"
    )
    if user_turns >= 4:
        return EscalationDecision(
            should_escalate=True,
            reason="تكررت محاولات المستخدم دون الوصول إلى حل",
            priority="NORMAL",
        )

    return EscalationDecision(should_escalate=False)
