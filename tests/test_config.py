from app.config import Settings


def test_cors_origins_accept_comma_separated_env(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://example.test")

    assert Settings().cors_origins == ["http://localhost:3000", "https://example.test"]


def test_embedding_api_key_can_override_shared_sbg_key():
    settings = Settings(
        auth_required=False,
        sbg_api_key="shared-key",
        embedding_api_key="embedding-key",
    )

    assert settings.resolved_embedding_api_key == "embedding-key"


def test_iti_is_the_default_embedding_provider():
    assert Settings.model_fields["embedding_provider"].default == "iti"


def test_cohere_output_dimension_accepts_env_string(monkeypatch):
    monkeypatch.setenv("COHERE_OUTPUT_DIMENSION", "1024")

    assert Settings(auth_required=False).cohere_output_dimension == 1024
