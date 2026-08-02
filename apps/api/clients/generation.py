"""Groq and mock text-generation provider implementations."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Final, cast

from groq import AsyncGroq

from apps.api.clients.base import GenerationProvider
from apps.api.models.enums import ProviderKind, ProviderStatus
from apps.api.models.providers import (
    GenerationMessage,
    GenerationRequest,
    GenerationResponse,
    GenerationUsage,
    ProviderMetadata,
    ProviderProbeResult,
)


DEFAULT_GENERATION_MODEL: Final = "llama-3.3-70b-versatile"


class GroqGenerationProvider(GenerationProvider):
    """Generate grounded text through the Groq Chat Completions API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GENERATION_MODEL,
        timeout_seconds: float = 60.0,
    ) -> None:
        """Initialize the Groq generation provider."""

        if not api_key.strip():
            raise ValueError("Groq API key must not be empty")

        if not model.strip():
            raise ValueError("Generation model must not be empty")

        if timeout_seconds <= 0:
            raise ValueError("Timeout must be greater than zero")

        self._model = model
        self._client = AsyncGroq(
            api_key=api_key,
            timeout=timeout_seconds,
        )

    @property
    def metadata(self) -> ProviderMetadata:
        """Return Groq provider metadata."""

        return ProviderMetadata(
            provider=ProviderKind.GENERATION,
            implementation=self.__class__.__name__,
            model=self._model,
            is_mock=False,
        )

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Generate one normalized response through Groq."""

        messages = [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in request.messages
        ]

        parameters: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }

        if request.response_schema is not None:
            parameters["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_schema_name,
                    "strict": True,
                    "schema": request.response_schema,
                },
            }

        completion = await self._client.chat.completions.create(
            **parameters,
        )

        if not completion.choices:
            raise RuntimeError(
                "Groq returned no completion choices"
            )

        content = completion.choices[0].message.content

        if content is None or not content.strip():
            raise RuntimeError(
                "Groq returned an empty completion"
            )

        usage: GenerationUsage | None = None

        if completion.usage is not None:
            usage = GenerationUsage(
                input_tokens=completion.usage.prompt_tokens,
                output_tokens=completion.usage.completion_tokens,
                total_tokens=completion.usage.total_tokens,
            )

        return GenerationResponse(
            content=content,
            model=completion.model,
            usage=usage,
            provider_request_id=completion.id,
        )

    async def probe(self) -> ProviderProbeResult:
        """Verify that Groq can produce a minimal completion."""

        started_at = perf_counter()

        try:
            response = await self.generate(
                GenerationRequest(
                    messages=(
                        GenerationMessage(
                            role="user",
                            content="...",
                        ),
                    ),
                    model=self._model,
                    temperature=0.0,
                    max_output_tokens=8,
                )
            )
        except Exception as exc:  
            return ProviderProbeResult(
                provider=ProviderKind.GENERATION,
                status=ProviderStatus.UNAVAILABLE,
                latency_ms=_elapsed_ms(started_at),
                detail=(
                    "Groq generation probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not response.content.strip():
            return ProviderProbeResult(
                provider=ProviderKind.GENERATION,
                status=ProviderStatus.DEGRADED,
                latency_ms=_elapsed_ms(started_at),
                detail="Groq generation probe returned empty content",
            )

        return ProviderProbeResult(
            provider=ProviderKind.GENERATION,
            status=ProviderStatus.READY,
            latency_ms=_elapsed_ms(started_at),
            detail=(
                f"Groq generation provider ready; "
                f"model={response.model}"
            ),
        )

    async def close(self) -> None:
        """Close the Groq asynchronous HTTP client."""

        await self._client.close()


class MockGenerationProvider(GenerationProvider):
    """Generate deterministic responses without external API calls."""

    def __init__(
        self,
        *,
        model: str = "mock-generation-001",
    ) -> None:
        """Initialize the mock generation provider."""

        if not model.strip():
            raise ValueError("Mock generation model must not be empty")

        self._model = model

    @property
    def metadata(self) -> ProviderMetadata:
        """Return mock-provider metadata."""

        return ProviderMetadata(
            provider=ProviderKind.GENERATION,
            implementation=self.__class__.__name__,
            model=self._model,
            is_mock=True,
        )

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        """Return a deterministic normalized response."""

        if request.response_schema is not None:
            content = json.dumps(
                _mock_structured_output(
                    request.response_schema,
                ),
                separators=(",", ":"),
                sort_keys=True,
            )
        else:
            last_user_message = next(
                (
                    message.content
                    for message in reversed(request.messages)
                    if message.role == "user"
                ),
                request.messages[-1].content,
            )

            content = (
                "Mock generation response for: "
                f"{last_user_message}"
            )

        input_tokens = sum(
            _estimate_tokens(message.content)
            for message in request.messages
        )
        output_tokens = _estimate_tokens(content)

        return GenerationResponse(
            content=content,
            model=self._model,
            usage=GenerationUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            provider_request_id="mock-generation-request",
        )

    async def probe(self) -> ProviderProbeResult:
        """Verify the mock generation implementation."""

        started_at = perf_counter()

        try:
            response = await self.generate(
                GenerationRequest(
                    messages=(
                        GenerationMessage(
                            role="user",
                            content="WTH mock readiness probe",
                        ),
                    ),
                    model=self._model,
                    temperature=0.0,
                    max_output_tokens=32,
                )
            )
        except Exception as exc:  
            return ProviderProbeResult(
                provider=ProviderKind.GENERATION,
                status=ProviderStatus.UNAVAILABLE,
                latency_ms=_elapsed_ms(started_at),
                detail=(
                    "Mock generation probe failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
            )

        if not response.content:
            return ProviderProbeResult(
                provider=ProviderKind.GENERATION,
                status=ProviderStatus.DEGRADED,
                latency_ms=_elapsed_ms(started_at),
                detail="Mock generation probe returned empty content",
            )

        return ProviderProbeResult(
            provider=ProviderKind.GENERATION,
            status=ProviderStatus.READY,
            latency_ms=_elapsed_ms(started_at),
            detail="Mock generation provider ready",
        )


def _mock_structured_output(
    schema: dict[str, object],
) -> object:
    """Build deterministic placeholder data from a JSON schema."""

    schema_type = schema.get("type")

    if schema_type == "object":
        properties = cast(
            dict[str, dict[str, object]],
            schema.get("properties", {}),
        )

        return {
            name: _mock_structured_output(property_schema)
            for name, property_schema in properties.items()
        }

    if schema_type == "array":
        item_schema = cast(
            dict[str, object],
            schema.get("items", {}),
        )
        return [_mock_structured_output(item_schema)]

    if schema_type == "integer":
        return 0

    if schema_type == "number":
        return 0.0

    if schema_type == "boolean":
        return False

    if schema_type == "null":
        return None

    enum_values = schema.get("enum")

    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    return "mock-value"


def _estimate_tokens(text: str) -> int:
    """Estimate token usage for deterministic mock responses."""

    return max(1, len(text.split()))


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed whole milliseconds."""

    return max(
        0,
        int((perf_counter() - started_at) * 1_000),
    )