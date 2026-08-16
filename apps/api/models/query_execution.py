from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator

from apps.api.models.runtime_contracts import FinalResponse


class QueryExecutionContract(BaseModel):
    """Strict base model for runtime query orchestration."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class QueryExecutionRequest(QueryExecutionContract):
    """External input accepted by QueryOrchestrator."""

    question: str

    @field_validator("question")
    @classmethod
    def validate_question(
        cls,
        value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("question must be non-empty")

        return normalized


class QueryExecutionResult(QueryExecutionContract):
    """External output returned by QueryOrchestrator.

    ``final_response`` is the canonical structured Phase 18 result.
    ``markdown`` is the deterministic Phase 18 rendered representation.

    Intermediate Phase 14-17 objects are intentionally not copied into this
    external result contract. They remain internal orchestration state and can
    be exposed later through explicit diagnostics/observability tooling without
    changing the production response contract.
    """

    final_response: FinalResponse
    markdown: str

    @field_validator("markdown")
    @classmethod
    def validate_markdown(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError("markdown must be non-empty")

        return value


__all__ = [
    "QueryExecutionContract",
    "QueryExecutionRequest",
    "QueryExecutionResult",
]
