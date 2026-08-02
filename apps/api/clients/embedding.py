"""Gemini and mock embedding-provider implementations."""

from __future__ import annotations

import hashlib
import math
from time import perf_counter
from typing import Final

from google import genai
from google.genai import types

from apps.api.clients.base import EmbeddingProvider
from apps.api.models.enums import ProviderKind, ProviderStatus
from apps.api.models.providers import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResult,
    ProviderMetadata,
    ProviderProbeResult,
)


DEFAULT_EMBEDDING_DIMENSIONS: Final = 768
DEFAULT_EMBEDDING_MODEL: Final = "gemini-embedding-001"
DEFAULT_EMBEDDING_TASK_TYPE: Final = "RETRIEVAL_DOCUMENT"


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings through the Google Gemini API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        """Initialize the Gemini embedding provider."""

        if not api_key.strip():
            raise ValueError("Gemini API key must not be empty")

        if not model.strip():
            raise ValueError("Embedding model must not be empty")

        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive")

        self._model = model
        self._dimensions = dimensions
        self._client = genai.Client(api_key=api_key)

    @property
    def metadata(self) -> ProviderMetadata:
        """Return Gemini provider metadata."""

        return ProviderMetadata(
            provider=ProviderKind.EMBEDDING,
            implementation=self.__class__.__name__,
            model=self._model,
            is_mock=False,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate embeddings for every input text."""

        task_type = (
            request.task_type
            if request.task_type is not None
            else DEFAULT_EMBEDDING_TASK_TYPE
        )

        response = await self._client.aio.models.embed_content(
            model=self._model,
            contents=list(request.texts),
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dimensions,
            ),
        )

        if not response.embeddings:
            raise RuntimeError(
                "Gemini returned no embeddings"
            )

        if len(response.embeddings) != len(request.texts):
            raise RuntimeError(
                "Gemini embedding count does not match input count"
            )

        results: list[EmbeddingResult] = []

        for index, content_embedding in enumerate(
            response.embeddings
        ):
            if not content_embedding.values:
                raise RuntimeError(
                    f"Gemini returned an empty embedding at index {index}"
                )

            vector = tuple(
                float(value)
                for value in content_embedding.values
            )

            if len(vector) != self._dimensions:
                raise RuntimeError(
                    "Gemini embedding dimension mismatch: "
                    f"expected {self._dimensions}, got {len(vector)}"
                )

            results.append(
                EmbeddingResult(
                    index=index,
                    embedding=vector,
                    model=self._model,
                    dimensions=self._dimensions,
                )
            )

        return EmbeddingResponse(
            results=tuple(results),
            model=self._model,
            dimensions=self._dimensions,
        )

    async def probe(self) -> ProviderProbeResult:
        """Verify that Gemini can generate an embedding."""

        started_at = perf_counter()

        try:
            response = await self.embed(
                EmbeddingRequest(
                    texts=("WTH embedding readiness probe",),
                    task_type="RETRIEVAL_QUERY",
                )
            )
        except Exception as exc:  
            return ProviderProbeResult(
                provider=ProviderKind.EMBEDDING,
                status=ProviderStatus.UNAVAILABLE,
                latency_ms=_elapsed_ms(started_at),
                detail=(
                    "Gemini embedding probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if len(response.results) != 1:
            return ProviderProbeResult(
                provider=ProviderKind.EMBEDDING,
                status=ProviderStatus.DEGRADED,
                latency_ms=_elapsed_ms(started_at),
                detail=(
                    "Gemini embedding probe returned an unexpected "
                    "number of results"
                ),
            )

        return ProviderProbeResult(
            provider=ProviderKind.EMBEDDING,
            status=ProviderStatus.READY,
            latency_ms=_elapsed_ms(started_at),
            detail=(
                f"Gemini embedding provider ready; "
                f"model={self._model}, "
                f"dimensions={self._dimensions}"
            ),
        )

    async def close(self) -> None:
        """Close the Gemini asynchronous client."""

        await self._client.aio.aclose()


class MockEmbeddingProvider(EmbeddingProvider):
    """Generate deterministic embeddings without external API calls."""

    def __init__(
        self,
        *,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
        model: str = "mock-embedding-001",
    ) -> None:
        """Initialize the deterministic mock provider."""

        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive")

        if not model.strip():
            raise ValueError("Mock embedding model must not be empty")

        self._dimensions = dimensions
        self._model = model

    @property
    def metadata(self) -> ProviderMetadata:
        """Return mock-provider metadata."""

        return ProviderMetadata(
            provider=ProviderKind.EMBEDDING,
            implementation=self.__class__.__name__,
            model=self._model,
            is_mock=True,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResponse:
        """Generate deterministic unit-length vectors."""

        results = tuple(
            EmbeddingResult(
                index=index,
                embedding=_deterministic_vector(
                    text=text,
                    dimensions=self._dimensions,
                ),
                model=self._model,
                dimensions=self._dimensions,
            )
            for index, text in enumerate(request.texts)
        )

        return EmbeddingResponse(
            results=results,
            model=self._model,
            dimensions=self._dimensions,
        )

    async def probe(self) -> ProviderProbeResult:
        """Verify the mock embedding implementation."""

        started_at = perf_counter()

        try:
            response = await self.embed(
                EmbeddingRequest(
                    texts=("WTH mock embedding readiness probe",),
                    task_type="RETRIEVAL_QUERY",
                )
            )
        except Exception as exc:  
            return ProviderProbeResult(
                provider=ProviderKind.EMBEDDING,
                status=ProviderStatus.UNAVAILABLE,
                latency_ms=_elapsed_ms(started_at),
                detail=(
                    "Mock embedding probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if (
            len(response.results) != 1
            or response.results[0].dimensions
            != self._dimensions
        ):
            return ProviderProbeResult(
                provider=ProviderKind.EMBEDDING,
                status=ProviderStatus.DEGRADED,
                latency_ms=_elapsed_ms(started_at),
                detail="Mock embedding probe returned invalid output",
            )

        return ProviderProbeResult(
            provider=ProviderKind.EMBEDDING,
            status=ProviderStatus.READY,
            latency_ms=_elapsed_ms(started_at),
            detail=(
                f"Mock embedding provider ready; "
                f"dimensions={self._dimensions}"
            ),
        )


def _deterministic_vector(
    *,
    text: str,
    dimensions: int,
) -> tuple[float, ...]:
    """Create a deterministic normalized vector from input text."""

    values: list[float] = []
    counter = 0

    while len(values) < dimensions:
        digest = hashlib.sha256(
            f"{counter}:{text}".encode()
        ).digest()

        for byte in digest:
            value = (float(byte) / 127.5) - 1.0
            values.append(value)

            if len(values) == dimensions:
                break

        counter += 1

    magnitude = math.sqrt(
        sum(value * value for value in values)
    )

    if magnitude == 0.0:
        raise RuntimeError(
            "Unable to normalize deterministic embedding"
        )

    return tuple(
        value / magnitude
        for value in values
    )


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed whole milliseconds."""

    return max(
        0,
        int((perf_counter() - started_at) * 1_000),
    )