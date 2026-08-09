"""Typed environment configuration for AI providers, ChromaDB, and security."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Load and validate service configuration from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    app_name: str = "PropMatch Unified AI Service"
    log_level: str = "INFO"

    auth_required: bool = True
    internal_service_api_key: str = Field(default="", repr=False, alias="INTERNAL_SERVICE_API_KEY")
    # ITI LLM Credentials
    sbg_api_key: str = Field(default="", repr=False, alias="SBG_API_KEY")
    llm_api_url: str = Field(
        default="http://apiaccess.iti.net.eg/api/v1/student/chat",
        alias="LLM_API_URL",
    )
    llm_model_id: str = Field(default="openai.gpt-oss-120b-1:0", alias="LLM_MODEL_ID")

    # ChromaDB & Vector Collections
    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, ge=1, le=65535, alias="CHROMA_PORT")
    chroma_ssl: bool = Field(default=False, alias="CHROMA_SSL")
    chroma_legal_collection: str = Field(
        default="egypt_real_estate_laws_v1", alias="CHROMA_LEGAL_COLLECTION"
    )
    chroma_support_collection: str = Field(
        default="support_kb_v1", alias="CHROMA_SUPPORT_COLLECTION"
    )

    app_env: str = "development"
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    jwt_secret: str = Field(default="", repr=False)
    jwt_algorithm: str = "HS256"

    llm_max_tokens: int = Field(default=700, ge=100, le=4000)
    llm_timeout_seconds: float = Field(default=90, gt=0)

    embedding_provider: Literal["iti", "cohere"] = "iti"
    embedding_api_url: str = "http://apiaccess.iti.net.eg/api/v1/student/embed"
    embedding_api_key: str = Field(default="", repr=False)
    embedding_model_id: str = "amazon.titan-embed-text-v2:0:8k"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_timeout_seconds: float = Field(default=60, gt=0)
    cohere_api_url: str = "https://api.cohere.com/v2/embed"
    cohere_api_key: str = Field(default="", repr=False)
    cohere_model_id: str = "embed-v4.0"
    cohere_output_dimension: int = 1024
    cohere_max_retries: int = Field(default=6, ge=0, le=20)
    cohere_retry_wait_seconds: float = Field(default=60, gt=0)

    laws_dir: Path = Path("laws/egypt_real_estate_laws_txt_for_rag")
    chunk_size: int = Field(default=1400, ge=300, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    retrieval_top_k: int = Field(default=5, ge=1, le=15)
    relevance_max_distance: float = Field(default=0.72, ge=0, le=2)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        """Accept CORS origins as either a list or a comma-separated env value."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("cohere_output_dimension")
    @classmethod
    def validate_cohere_output_dimension(cls, value: int) -> int:
        """Reject vector sizes that Cohere Embed v4 does not support."""
        if value not in {256, 512, 1024, 1536}:
            raise ValueError("must be one of 256, 512, 1024, or 1536")
        return value

    def validate_runtime_secrets(self) -> None:
        """Fail startup when required provider or authentication secrets are absent."""
        missing: list[str] = []
        if not self.sbg_api_key:
            missing.append("SBG_API_KEY")
        if self.auth_required and not (
            self.internal_service_api_key or self.jwt_secret
        ):
            missing.append("INTERNAL_SERVICE_API_KEY or JWT_SECRET")
        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    @property
    def resolved_embedding_api_key(self) -> str:
        """Reuse the SBG credential when no dedicated ITI embedding key is set."""
        return self.embedding_api_key or self.sbg_api_key


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for FastAPI dependency injection."""
    return Settings()
