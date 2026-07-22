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
