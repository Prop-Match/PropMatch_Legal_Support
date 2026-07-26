"""Multi-factor escalation evaluator for human support handoff."""


def evaluate_escalation(message: str, history: list[dict] | None = None) -> dict:
    """Evaluates 3 escalation rules without relying solely on keywords or anger.

    Rules:
    1. Explicit Human Support / Customer Service Request -> HIGH Priority.
    2. Payment / Account Security Emergency -> URGENT Priority.
    3. Repeated Unresolved Attempts (history >= 4 user turns) -> NORMAL Priority.
    """
    msg_lower = message.lower().strip()

    # Rule 1: Explicit Human Support / Customer Service Request
    human_keywords = [
        "خدمة العملاء",
        "خدمة عملاء",
        "خدمه العملاء",
        "تحدث مع شخص",
        "تحدث مع موظف",
        "أريد التحدث",
        "اريد التحدث",
        "تحويل لموظف",
        "تحويل موظف",
        "دعم بشري",
        "تحدث مع إنسان",
        "كلم موظف",
        "التحدث مع شخص",
        "التحدث مع موظف",
        "التحدث مع خدمة",
        "كلمني موظف",
        "مسؤول الدعم",
    ]
    if any(phrase in msg_lower for phrase in human_keywords):
        return {
            "shouldEscalate": True,
            "reason": "طلب المستخدم التحدث مع خدمة العملاء / موظف دعم فني بشكل صريح",
            "priority": "HIGH",
        }

    # Rule 2: Emergency (Payment or Security)
    if any(kw in msg_lower for kw in ["سرقة", "احتيال", "اختراق", "خصم بدون علم"]):
        return {
            "shouldEscalate": True,
            "reason": "تم الكشف عن حالة طوارئ مالية أو أمنية",
            "priority": "URGENT",
        }

    # Rule 3: Repeated Unresolved Attempts (history >= 4 user turns)
    if history:
        user_turns = sum(1 for m in history if m.get("role") == "user")
        if user_turns >= 4:
            return {
                "shouldEscalate": True,
                "reason": "المستخدم يحاول حل المشكلة منذ عدة محاولات دون جدوى",
                "priority": "NORMAL",
            }

    return {"shouldEscalate": False}
