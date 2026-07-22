from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_name: str = "PropMatch Legal Support"
    app_env: str = "development"
    log_level: str = "INFO"
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]

    auth_required: bool = True
    internal_service_api_key: str = Field(default="", repr=False)
    jwt_secret: str = Field(default="", repr=False)
    jwt_algorithm: str = "HS256"

    sbg_api_key: str = Field(default="", repr=False)
    llm_api_url: str = "http://apiaccess.iti.net.eg/api/v1/student/chat"
    llm_model_id: str = "openai.gpt-oss-120b-1:0"
    llm_max_tokens: int = Field(default=700, ge=100, le=4000)
    llm_timeout_seconds: float = Field(default=90, gt=0)

    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    chroma_ssl: bool = False
    chroma_collection: str = "egypt_real_estate_laws"

    embedding_api_url: str = "http://apiaccess.iti.net.eg/api/v1/student/embed"
    embedding_api_key: str = Field(default="", repr=False)
    embedding_model_id: str = "amazon.titan-embed-text-v2:0:8k"
    embedding_batch_size: int = Field(default=32, ge=1, le=256)
    embedding_timeout_seconds: float = Field(default=60, gt=0)

    laws_dir: Path = Path("laws/egypt_real_estate_laws_txt_for_rag")
    chunk_size: int = Field(default=1400, ge=300, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    retrieval_top_k: int = Field(default=5, ge=1, le=15)
    relevance_max_distance: float = Field(default=0.72, ge=0, le=2)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    def validate_runtime_secrets(self) -> None:
        missing: list[str] = []
        if not self.sbg_api_key:
            missing.append("SBG_API_KEY")
        if self.auth_required and not (self.internal_service_api_key or self.jwt_secret):
            missing.append("INTERNAL_SERVICE_API_KEY or JWT_SECRET")
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    @property
    def resolved_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.sbg_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
