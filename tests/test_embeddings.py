import httpx
import pytest

from app.config import Settings
from app.embeddings import (
    EmbeddingProviderError,
    ItiEmbeddings,
    _handle_response,
    extract_embeddings,
)


class RecordingEmbeddings(ItiEmbeddings):
    def __init__(self):
        super().__init__(Settings(auth_required=False))
        self.input_types = []

    def _embed_sync(self, texts, input_type):
        self.input_types.append((list(texts), input_type))
        return [[0.1, 0.2] for _ in texts]


def test_langchain_embedding_adapter_uses_provider_input_types():
    embeddings = RecordingEmbeddings()

    assert embeddings.embed_documents(["مادة قانونية"]) == [[0.1, 0.2]]
    assert embeddings.embed_query("سؤال قانوني") == [0.1, 0.2]
    assert embeddings.input_types == [
        (["مادة قانونية"], "search_document"),
        (["سؤال قانوني"], "search_query"),
    ]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"embeddings": [[1, 2], [3, 4]]}, [[1.0, 2.0], [3.0, 4.0]]),
        ({"vectors": [[1, 2]]}, [[1.0, 2.0]]),
        ({"data": [{"embedding": [1, 2]}]}, [[1.0, 2.0]]),
        ({"embeddings": {"vectors": [[1, 2]]}}, [[1.0, 2.0]]),
    ],
)
def test_extract_embeddings_supports_common_provider_shapes(payload, expected):
    assert extract_embeddings(payload) == expected


def test_embedding_payload_uses_configured_model():
    embeddings = ItiEmbeddings(
        Settings(auth_required=False, embedding_model_id="future-embedding-model")
    )

    assert embeddings._payload(["text"], "search_document") == {
        "model_id": "future-embedding-model",
        "texts": ["text"],
        "input_type": "search_document",
    }


def test_missing_embedding_key_fails_before_network_call():
    embeddings = ItiEmbeddings(
        Settings(auth_required=False, sbg_api_key="", embedding_api_key="")
    )

    with pytest.raises(EmbeddingProviderError):
        embeddings._headers()


def test_provider_error_includes_actionable_code_and_message():
    response = httpx.Response(
        403,
        request=httpx.Request("POST", "http://provider.test/embed"),
        json={
            "error": {
                "code": "MODEL_NOT_ALLOWED",
                "message": "This model is not allowed for the student",
                "details": {},
            }
        },
    )

    with pytest.raises(EmbeddingProviderError, match="MODEL_NOT_ALLOWED") as error:
        _handle_response(response, 1)
    assert "not allowed for the student" in str(error.value)
