from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings(BaseSettings):
    google_api_key: str | None = None
    pinecone_api_key: str | None = None
    pinecone_index_name: str = "eldercare-kb"
    knowledge_base_dir: Path = Path("../pland")
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "models/text-embedding-004"
    enable_external_ai: bool = False
    allowed_origins_env: str = Field(
        "http://localhost:3000,http://127.0.0.1:3000",
        validation_alias="ALLOWED_ORIGINS",
    )
    trusted_hosts_env: str = Field(
        "localhost,127.0.0.1",
        validation_alias="TRUSTED_HOSTS",
    )
    internal_api_key: str | None = None
    max_request_bytes: int = 256 * 1024

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return _csv_list(self.allowed_origins_env)

    @property
    def trusted_hosts(self) -> list[str]:
        return _csv_list(self.trusted_hosts_env)


@lru_cache
def get_settings() -> Settings:
    return Settings()
