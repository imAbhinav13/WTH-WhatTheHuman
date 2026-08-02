"""Abstract contracts for WTH external-service clients."""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Self

from apps.api.models.providers import (
    DatabaseProbeRequest,
    DatabaseProbeResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    GenerationRequest,
    GenerationResponse,
    ProviderMetadata,
    ProviderProbeResult,
)


class ProviderClient(ABC):
    """Common lifecycle contract for external-service clients."""

    @property
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Return provider implementation metadata."""

    @abstractmethod
    async def probe(self) -> ProviderProbeResult:
        """Check whether the provider is ready for requests."""

    async def close(self) -> None:  # noqa: B027
        """Release provider resources.

        Stateless providers may keep the default no-op implementation.
        """

    async def __aenter__(self) -> Self:
        """Enter the asynchronous provider context."""

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the asynchronous provider context."""

        await self.close()


class EmbeddingProvider(ProviderClient):
    """Contract implemented by embedding providers."""

    @abstractmethod
    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate embeddings for all texts in the request."""


class GenerationProvider(ProviderClient):
    """Contract implemented by text-generation providers."""

    @abstractmethod
    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate one normalized text response."""


class DatabaseProvider(ABC):
    """Contract implemented by the application's database client."""

    @property
    @abstractmethod
    def metadata(self) -> ProviderMetadata:
        """Return database implementation metadata."""

    @abstractmethod
    async def probe(
        self,
        request: DatabaseProbeRequest,
    ) -> DatabaseProbeResponse:
        """Verify database connectivity and required reference data."""

    async def close(self) -> None:  # noqa: B027
        """Release database resources.

        Stateless database clients may keep the default no-op implementation.
        """

    async def __aenter__(self) -> Self:
        """Enter the asynchronous database-client context."""

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the asynchronous database-client context."""

        await self.close()