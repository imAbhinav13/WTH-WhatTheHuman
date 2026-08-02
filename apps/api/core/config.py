"""
This module provides a single, cached Settings instance for the API.
Configuration is loaded from environment variables and the repository-root=.env file.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Annotated, Self

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class AppEnvironment(StrEnum):
    """Supported application environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class ProviderMode(StrEnum):
    """Controls whether external providers are mocked or called live."""

    MOCK = "mock"
    LIVE = "live"


class LogLevel(StrEnum):
    """Supported application log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Validated application settings loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # Application
    app_name: NonEmptyString = "WTH: What The Human"
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    app_version: NonEmptyString = "0.1.0"
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    api_host: NonEmptyString = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65_535)
    api_prefix: str = "/api"
    cors_origins: Annotated[
        tuple[str, ...],
        NoDecode,
    ] = ("http://localhost:3000",)

    # Provider selection
    provider_mode: ProviderMode = ProviderMode.MOCK

    # Supabase
    supabase_url: AnyHttpUrl | None = None
    supabase_secret_key: SecretStr | None = None
    supabase_publishable_key: SecretStr | None = None

    # Groq
    groq_api_key: SecretStr | None = None
    groq_model: NonEmptyString = "openai/gpt-oss-120b"
    groq_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    groq_max_retries: int = Field(default=3, ge=0, le=10)

    # Embeddings
    google_api_key: SecretStr | None = None
    embedding_model: NonEmptyString = "gemini-embedding-2"
    embedding_dimension: int = Field(default=768, ge=128, le=4_096)
    embedding_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    embedding_max_retries: int = Field(default=3, ge=0, le=10)

    # Retrieval
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    retrieval_min_similarity: float = Field(default=0.55, ge=-1.0, le=1.0)
    concept_activation_threshold: float = Field(default=0.50, ge=-1.0, le=1.0)
    concept_ambiguity_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    max_activated_concepts: int = Field(default=3, ge=1, le=8)

    # Query validation
    question_min_length: int = Field(default=3, ge=1, le=100)
    question_max_length: int = Field(default=1_000, ge=10, le=10_000)

    # Logging and privacy
    log_full_question_text: bool = False
    query_retention_days: int = Field(default=30, ge=0, le=3_650)

    # Readiness
    readiness_check_database: bool = False
    readiness_check_providers: bool = False

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Normalize and validate the API route prefix."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("API_PREFIX must not be empty")

        if not normalized.startswith("/"):
            normalized = f"/{normalized}"

        if normalized != "/":
            normalized = normalized.rstrip("/")

        return normalized

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        """Accept comma-separated origins from environment variables."""

        if isinstance(value, str):
            origins = tuple(origin.strip() for origin in value.split(",") if origin.strip())

            if not origins:
                raise ValueError("CORS_ORIGINS must contain at least one origin")

            return origins

        return value

    @field_validator("cors_origins")
    @classmethod
    def validate_cors_origins(
        cls,
        origins: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Validate configured CORS origins."""

        if not origins:
            raise ValueError("CORS_ORIGINS must contain at least one origin")

        for origin in origins:
            if origin == "*":
                continue

            if not origin.startswith(("http://", "https://")):
                raise ValueError("Each CORS origin must begin with 'http://' or 'https://'")

            if origin.endswith("/"):
                raise ValueError("CORS origins must not include a trailing slash")

        return origins

    @model_validator(mode="after")
    def validate_length_bounds(self) -> Self:
        """Ensure query-length settings are internally consistent."""

        if self.question_min_length >= self.question_max_length:
            raise ValueError("QUESTION_MIN_LENGTH must be less than QUESTION_MAX_LENGTH")

        return self

    @model_validator(mode="after")
    def validate_live_provider_credentials(self) -> Self:
        """Require provider credentials when live mode is enabled."""

        if self.provider_mode is ProviderMode.MOCK:
            return self

        missing: list[str] = []

        if self.supabase_url is None:
            missing.append("SUPABASE_URL")

        if not self._secret_is_configured(self.supabase_secret_key):
            missing.append("SUPABASE_SECRET_KEY")

        if not self._secret_is_configured(self.groq_api_key):
            missing.append("GROQ_API_KEY")

        if not self._secret_is_configured(self.google_api_key):
            missing.append("GOOGLE_API_KEY")

        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Live provider mode requires valid values for: {joined}")

        return self

    @model_validator(mode="after")
    def validate_production_safety(self) -> Self:
        """Reject unsafe production configuration."""

        if self.app_env is AppEnvironment.PRODUCTION and self.debug:
            raise ValueError("DEBUG must be false when APP_ENV=production")

        if self.app_env is AppEnvironment.PRODUCTION and "*" in self.cors_origins:
            raise ValueError("Wildcard CORS origins are not allowed in production")

        return self

    @staticmethod
    def _secret_is_configured(secret: SecretStr | None) -> bool:
        """Return whether a secret contains a non-placeholder value."""

        if secret is None:
            return False

        value = secret.get_secret_value().strip()

        if not value:
            return False

        placeholder_fragments = (
            "replace-with",
            "your-",
            "example",
            "changeme",
        )

        lowered = value.lower()

        return not any(fragment in lowered for fragment in placeholder_fragments)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated application settings."""

    return Settings()
