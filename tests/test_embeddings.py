import httpx
import pytest

from app.config import Settings
from app.embeddings import (
    CohereEmbeddings,
    EmbeddingProviderError,
    ItiEmbeddings,
    _handle_response,
    _retry_after_seconds,
    create_embeddings,
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
        ({"embeddings": {"float": [[1, 2]]}}, [[1.0, 2.0]]),
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
    embeddings = ItiEmbeddings(Settings(auth_required=False, sbg_api_key="", embedding_api_key=""))

    with pytest.raises(EmbeddingProviderError):
        embeddings._headers()


def test_cohere_payload_uses_search_type_and_float_embeddings():
    embeddings = CohereEmbeddings(
        Settings(
            auth_required=False,
            cohere_model_id="embed-v4.0",
            cohere_output_dimension=512,
        )
    )

    assert embeddings._payload(["مادة قانونية"], "search_document") == {
        "model": "embed-v4.0",
        "texts": ["مادة قانونية"],
        "input_type": "search_document",
        "embedding_types": ["float"],
        "output_dimension": 512,
    }


def test_missing_cohere_key_fails_before_network_call():
    embeddings = CohereEmbeddings(Settings(auth_required=False, cohere_api_key=""))

    with pytest.raises(EmbeddingProviderError, match="COHERE_API_KEY"):
        embeddings._headers()


def test_embedding_factory_selects_configured_provider():
    assert isinstance(
        create_embeddings(Settings(auth_required=False, embedding_provider="iti")),
        ItiEmbeddings,
    )
    assert isinstance(
        create_embeddings(Settings(auth_required=False, embedding_provider="cohere")),
        CohereEmbeddings,
    )


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


def test_provider_error_supports_cohere_top_level_message():
    response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.cohere.com/v2/embed"),
        json={"message": "rate limit exceeded"},
    )

    with pytest.raises(EmbeddingProviderError, match="rate limit exceeded"):
        _handle_response(response, 1)


def test_retry_after_uses_header_or_configured_fallback():
    with_header = httpx.Response(429, headers={"Retry-After": "12.5"})
    without_header = httpx.Response(429)

    assert _retry_after_seconds(with_header, 60) == 12.5
    assert _retry_after_seconds(without_header, 60) == 60


def test_cohere_retries_rate_limit_response(monkeypatch):
    calls = 0
    waits = []

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={"message": "trial token rate limit exceeded"},
            )
        return httpx.Response(200, json={"embeddings": {"float": [[0.1, 0.2]]}})

    monkeypatch.setattr("app.embeddings.time.sleep", waits.append)
    embeddings = CohereEmbeddings(
        Settings(
            auth_required=False,
            cohere_api_key="test-key",
            cohere_max_retries=1,
        )
    )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = embeddings._post_with_retry(client, ["سؤال"], "search_query")

    assert response.status_code == 200
    assert calls == 2
    assert waits == [3.0]
