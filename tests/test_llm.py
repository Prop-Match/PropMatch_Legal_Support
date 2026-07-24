import pytest

from app.llm import SYSTEM_PROMPT, extract_generated_text


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"output_text": "one"}, "one"),
        ({"reply": "two"}, "two"),
        ({"content": "three"}, "three"),
        ({"choices": [{"message": {"content": "four"}}]}, "four"),
        ({"choices": [{"text": "five"}]}, "five"),
        ({"message": {"content": "six"}}, "six"),
    ],
)
def test_extract_generated_text_handles_provider_shapes(payload, expected):
    assert extract_generated_text(payload) == expected


def test_extract_generated_text_rejects_unknown_shape():
    with pytest.raises(KeyError):
        extract_generated_text({"unknown": "value"})


def test_system_prompt_forbids_cross_domain_grounding():
    assert "عقد العمل" in SYSTEM_PROMPT
    assert "لا تنسب حكم مقتطف إلى مادة أخرى" in SYSTEM_PROMPT
    assert "لا تستخدم HTML" in SYSTEM_PROMPT
    assert "كل صف في سطر مستقل" in SYSTEM_PROMPT
