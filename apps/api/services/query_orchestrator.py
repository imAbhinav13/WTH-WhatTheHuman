from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel

from apps.api.models.query_execution import (
    QueryExecutionRequest,
    QueryExecutionResult,
)
from apps.api.services.coverage import (
    CoverageService,
    CoverageServiceResult,
)
from apps.api.services.domain_generation import (
    DomainGenerationService,
    DomainGenerationServiceResult,
    ProviderConfig,
)
from apps.api.services.response_assembly import (
    ResponseAssemblyResult,
    ResponseAssemblyService,
)
from apps.api.services.retrieval import (
    RetrievalService,
    RetrievalServiceResult,
)
from apps.api.services.synthesis import (
    SynthesisProviderConfig,
    SynthesisService,
    SynthesisServiceResult,
)

DOMAINS: Final[tuple[str, ...]] = (
    "science",
    "advaita",
    "samkhya",
)

QueryPhase: TypeAlias = Literal[
    "phase_14_retrieval",
    "phase_15_domain_generation",
    "phase_16_synthesis",
    "phase_17_coverage",
    "phase_18_response_assembly",
]


class QueryPipelineError(RuntimeError):
    """Base error for one in-memory WTH query execution."""


class QueryPhaseExecutionError(QueryPipelineError):
    """Wrap a service/provider failure with the owning WTH phase."""

    def __init__(
        self,
        *,
        phase: QueryPhase,
        message: str,
    ) -> None:
        self.phase = phase
        super().__init__(f"{phase} failed: {message}")


class QueryPipelineInvariantError(QueryPipelineError):
    """Raised when two otherwise-valid phases disagree on shared identity."""

    def __init__(
        self,
        *,
        phase: QueryPhase,
        invariant: str,
        detail: str,
    ) -> None:
        self.phase = phase
        self.invariant = invariant

        super().__init__(f"{phase} invariant {invariant!r} failed: {detail}")


@dataclass(frozen=True, slots=True)
class QueryPipelineServices:
    """Runtime service dependencies for the Phase 14-18 pipeline."""

    retrieval: RetrievalService
    domain_generation: DomainGenerationService
    synthesis: SynthesisService
    coverage: CoverageService
    response_assembly: ResponseAssemblyService


@dataclass(frozen=True, slots=True)
class QueryPipelineProviderConfig:
    """Provider configuration injected at application composition time."""

    domain_generation: ProviderConfig
    synthesis: SynthesisProviderConfig


@dataclass(frozen=True, slots=True)
class QueryPipelineState:
    """Internal Phase 14-17 state produced during one execution."""

    retrieval: RetrievalServiceResult
    domain_generation: DomainGenerationServiceResult
    synthesis: SynthesisServiceResult
    coverage: CoverageServiceResult


@dataclass(frozen=True, slots=True)
class QueryIdentity:
    """Question/corpus identity that every downstream phase must preserve."""

    question: str
    corpus_version: str


def _document(
    model: BaseModel,
    *,
    description: str,
) -> dict[str, object]:
    raw = model.model_dump(
        mode="python",
        by_alias=True,
    )

    if not isinstance(
        raw,
        Mapping,
    ):
        raise QueryPipelineError(f"{description} did not serialize to an object.")

    result: dict[str, object] = {}

    for key, value in raw.items():
        if not isinstance(
            key,
            str,
        ):
            raise QueryPipelineError(f"{description} contains a non-string key.")

        result[key] = value

    return result


def _mapping(
    value: object,
    *,
    phase: QueryPhase,
    invariant: str,
    description: str,
) -> dict[str, object]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant=invariant,
            detail=f"{description} must be an object.",
        )

    result: dict[str, object] = {}

    for key, nested in value.items():
        if not isinstance(
            key,
            str,
        ):
            raise QueryPipelineInvariantError(
                phase=phase,
                invariant=invariant,
                detail=(f"{description} contains a non-string key."),
            )

        result[key] = nested

    return result


def _sequence(
    value: object,
    *,
    phase: QueryPhase,
    invariant: str,
    description: str,
) -> list[object]:
    if not isinstance(
        value,
        Sequence,
    ) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant=invariant,
            detail=f"{description} must be a list.",
        )

    return list(value)


def _required_string(
    document: Mapping[str, object],
    field: str,
    *,
    phase: QueryPhase,
    invariant: str,
    description: str,
) -> str:
    value = document.get(field)

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant=invariant,
            detail=(f"{description}.{field} must be a non-empty string."),
        )

    return value.strip()


def _optional_string(
    document: Mapping[str, object],
    field: str,
) -> str | None:
    value = document.get(field)

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        return None

    normalized = value.strip()

    return normalized or None


def _assert_equal(
    *,
    phase: QueryPhase,
    invariant: str,
    description: str,
    observed: object,
    expected: object,
) -> None:
    if observed != expected:
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant=invariant,
            detail=(f"{description}: expected {expected!r}, observed {observed!r}."),
        )


def _assert_exact_domain_keys(
    domains: Mapping[str, object],
    *,
    phase: QueryPhase,
    invariant: str,
    description: str,
) -> None:
    observed = set(domains)

    expected = set(DOMAINS)

    if observed != expected:
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant=invariant,
            detail=(
                f"{description} must contain exactly "
                f"{sorted(expected)}; observed {sorted(observed)}."
            ),
        )


def _assert_exit_gate_green(
    manifest: BaseModel,
    *,
    phase: QueryPhase,
) -> None:
    """Reject any explicit false boolean in a phase exit gate.

    Stage 3 contracts differ slightly in their exit-gate field names. The
    invariant common to all of them is that an exit gate must exist and every
    boolean validation flag it publishes must be true. Numeric provenance
    fields such as ``llm_calls = 0`` are intentionally ignored here.
    """

    document = _document(
        manifest,
        description=f"{phase} manifest",
    )

    gate = _mapping(
        document.get("exit_gate"),
        phase=phase,
        invariant="exit_gate",
        description="exit_gate",
    )

    boolean_flags = {
        key: value
        for key, value in gate.items()
        if isinstance(
            value,
            bool,
        )
    }

    if not boolean_flags:
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant="exit_gate",
            detail=("exit_gate exposes no boolean validation flags."),
        )

    failed = sorted(key for key, value in boolean_flags.items() if value is not True)

    if failed:
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant="exit_gate",
            detail=("exit gate contains false validation flags: " + ", ".join(failed)),
        )


def _identity_from_phase14(
    *,
    request_question: str,
    result: RetrievalServiceResult,
) -> QueryIdentity:
    phase: QueryPhase = "phase_14_retrieval"

    evidence = _document(
        result.evidence_package,
        description="Phase 14 EvidencePackage",
    )

    manifest = _document(
        result.manifest,
        description="Phase 14 RetrievalManifest",
    )

    evidence_question = _required_string(
        evidence,
        "question",
        phase=phase,
        invariant="question_continuity",
        description="EvidencePackage",
    )

    _assert_equal(
        phase=phase,
        invariant="question_continuity",
        description="Phase 14 question",
        observed=evidence_question,
        expected=request_question,
    )

    evidence_corpus = _required_string(
        evidence,
        "corpus_version",
        phase=phase,
        invariant="corpus_continuity",
        description="EvidencePackage",
    )

    manifest_corpus = _required_string(
        manifest,
        "corpus_version",
        phase=phase,
        invariant="corpus_continuity",
        description="RetrievalManifest",
    )

    _assert_equal(
        phase=phase,
        invariant="corpus_continuity",
        description=("EvidencePackage vs RetrievalManifest corpus"),
        observed=evidence_corpus,
        expected=manifest_corpus,
    )

    retrieval_mode = _required_string(
        evidence,
        "retrieval_mode",
        phase=phase,
        invariant="retrieval_mode",
        description="EvidencePackage",
    )

    _assert_equal(
        phase=phase,
        invariant="retrieval_mode",
        description="Phase 14 retrieval mode",
        observed=retrieval_mode,
        expected="concept_aware",
    )

    manifest_status = _required_string(
        manifest,
        "status",
        phase=phase,
        invariant="phase_status",
        description="RetrievalManifest",
    )

    _assert_equal(
        phase=phase,
        invariant="phase_status",
        description="Phase 14 manifest status",
        observed=manifest_status,
        expected="evaluation_complete",
    )

    _assert_exit_gate_green(
        result.manifest,
        phase=phase,
    )

    domains = _mapping(
        evidence.get("domains"),
        phase=phase,
        invariant="domain_separation",
        description="EvidencePackage.domains",
    )

    _assert_exact_domain_keys(
        domains,
        phase=phase,
        invariant="domain_separation",
        description="EvidencePackage.domains",
    )

    for domain in DOMAINS:
        domain_payload = _mapping(
            domains[domain],
            phase=phase,
            invariant="domain_separation",
            description=(f"EvidencePackage.domains.{domain}"),
        )

        evidence_items = _sequence(
            domain_payload.get("evidence"),
            phase=phase,
            invariant="domain_separation",
            description=(f"EvidencePackage.domains.{domain}.evidence"),
        )

        declared_count = domain_payload.get("evidence_count")

        if not isinstance(
            declared_count,
            int,
        ) or isinstance(
            declared_count,
            bool,
        ):
            raise QueryPipelineInvariantError(
                phase=phase,
                invariant="domain_separation",
                detail=(f"{domain} evidence_count must be an integer."),
            )

        _assert_equal(
            phase=phase,
            invariant="domain_separation",
            description=f"{domain} evidence_count",
            observed=declared_count,
            expected=len(evidence_items),
        )

        for index, raw_item in enumerate(
            evidence_items,
            start=1,
        ):
            item = _mapping(
                raw_item,
                phase=phase,
                invariant="domain_separation",
                description=(f"{domain} evidence item {index}"),
            )

            item_domain = _required_string(
                item,
                "domain",
                phase=phase,
                invariant="domain_separation",
                description=(f"{domain} evidence item {index}"),
            )

            _assert_equal(
                phase=phase,
                invariant="domain_separation",
                description=(f"{domain} evidence item {index} domain"),
                observed=item_domain,
                expected=domain,
            )

            item_corpus = _required_string(
                item,
                "corpus_version",
                phase=phase,
                invariant="corpus_continuity",
                description=(f"{domain} evidence item {index}"),
            )

            _assert_equal(
                phase=phase,
                invariant="corpus_continuity",
                description=(f"{domain} evidence item {index} corpus"),
                observed=item_corpus,
                expected=evidence_corpus,
            )

    return QueryIdentity(
        question=request_question,
        corpus_version=evidence_corpus,
    )


def _validate_phase15(
    *,
    identity: QueryIdentity,
    result: DomainGenerationServiceResult,
) -> None:
    phase: QueryPhase = "phase_15_domain_generation"

    responses = _document(
        result.domain_responses,
        description="Phase 15 DomainResponses",
    )

    manifest = _document(
        result.manifest,
        description="Phase 15 GenerationManifest",
    )

    for description, document in (
        (
            "DomainResponses",
            responses,
        ),
        (
            "GenerationManifest",
            manifest,
        ),
    ):
        question = _required_string(
            document,
            "question",
            phase=phase,
            invariant="question_continuity",
            description=description,
        )

        _assert_equal(
            phase=phase,
            invariant="question_continuity",
            description=f"{description} question",
            observed=question,
            expected=identity.question,
        )

        corpus = _required_string(
            document,
            "corpus_version",
            phase=phase,
            invariant="corpus_continuity",
            description=description,
        )

        _assert_equal(
            phase=phase,
            invariant="corpus_continuity",
            description=f"{description} corpus",
            observed=corpus,
            expected=identity.corpus_version,
        )

    status = _required_string(
        manifest,
        "status",
        phase=phase,
        invariant="phase_status",
        description="GenerationManifest",
    )

    _assert_equal(
        phase=phase,
        invariant="phase_status",
        description="Phase 15 manifest status",
        observed=status,
        expected="domain_generation_complete",
    )

    _assert_exit_gate_green(
        result.manifest,
        phase=phase,
    )

    domains = _mapping(
        responses.get("domains"),
        phase=phase,
        invariant="domain_separation",
        description="DomainResponses.domains",
    )

    _assert_exact_domain_keys(
        domains,
        phase=phase,
        invariant="domain_separation",
        description="DomainResponses.domains",
    )

    for domain in DOMAINS:
        domain_response = _mapping(
            domains[domain],
            phase=phase,
            invariant="domain_separation",
            description=(f"DomainResponses.domains.{domain}"),
        )

        declared_domain = _optional_string(
            domain_response,
            "domain",
        )

        if declared_domain is not None:
            _assert_equal(
                phase=phase,
                invariant="domain_separation",
                description=(f"{domain} response declared domain"),
                observed=declared_domain,
                expected=domain,
            )


def _validate_phase16(
    *,
    identity: QueryIdentity,
    result: SynthesisServiceResult,
) -> None:
    phase: QueryPhase = "phase_16_synthesis"

    synthesis = _document(
        result.synthesis,
        description="Phase 16 SynthesisResult",
    )

    manifest = _document(
        result.manifest,
        description="Phase 16 SynthesisManifest",
    )

    for description, document in (
        (
            "SynthesisResult",
            synthesis,
        ),
        (
            "SynthesisManifest",
            manifest,
        ),
    ):
        question = _required_string(
            document,
            "question",
            phase=phase,
            invariant="question_continuity",
            description=description,
        )

        _assert_equal(
            phase=phase,
            invariant="question_continuity",
            description=f"{description} question",
            observed=question,
            expected=identity.question,
        )

        corpus = _required_string(
            document,
            "corpus_version",
            phase=phase,
            invariant="corpus_continuity",
            description=description,
        )

        _assert_equal(
            phase=phase,
            invariant="corpus_continuity",
            description=f"{description} corpus",
            observed=corpus,
            expected=identity.corpus_version,
        )

    status = _required_string(
        manifest,
        "status",
        phase=phase,
        invariant="phase_status",
        description="SynthesisManifest",
    )

    _assert_equal(
        phase=phase,
        invariant="phase_status",
        description="Phase 16 manifest status",
        observed=status,
        expected="synthesis_complete",
    )

    validation = _mapping(
        synthesis.get("validation"),
        phase=phase,
        invariant="synthesis_validation",
        description="SynthesisResult.validation",
    )

    _assert_equal(
        phase=phase,
        invariant="synthesis_validation",
        description="Phase 16 validation.passed",
        observed=validation.get("passed"),
        expected=True,
    )

    _assert_exit_gate_green(
        result.manifest,
        phase=phase,
    )


def _validate_phase17(
    *,
    identity: QueryIdentity,
    result: CoverageServiceResult,
) -> None:
    phase: QueryPhase = "phase_17_coverage"

    coverage = _document(
        result.coverage,
        description="Phase 17 CoverageResult",
    )

    manifest = _document(
        result.manifest,
        description="Phase 17 CoverageManifest",
    )

    # Coverage contracts may carry question/corpus in both the result and
    # manifest. Enforce continuity wherever the frozen contract publishes them.
    for description, document in (
        (
            "CoverageResult",
            coverage,
        ),
        (
            "CoverageManifest",
            manifest,
        ),
    ):
        question = _optional_string(
            document,
            "question",
        )

        if question is not None:
            _assert_equal(
                phase=phase,
                invariant="question_continuity",
                description=f"{description} question",
                observed=question,
                expected=identity.question,
            )

        corpus = _optional_string(
            document,
            "corpus_version",
        )

        if corpus is not None:
            _assert_equal(
                phase=phase,
                invariant="corpus_continuity",
                description=f"{description} corpus",
                observed=corpus,
                expected=identity.corpus_version,
            )

    _assert_exit_gate_green(
        result.manifest,
        phase=phase,
    )

    coverage_status = _optional_string(
        coverage,
        "coverage_status",
    )

    if coverage_status is None:
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant="coverage_result",
            detail=("CoverageResult.coverage_status is missing."),
        )

    # Do not require "Supported". Partially Supported and Out of Corpus are
    # legitimate deterministic Phase 17 classifications.
    allowed_statuses = {
        "Supported",
        "Partially Supported",
        "Out of Corpus",
    }

    if coverage_status not in allowed_statuses:
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant="coverage_result",
            detail=(
                "CoverageResult.coverage_status is not one of "
                f"{sorted(allowed_statuses)}: "
                f"{coverage_status!r}."
            ),
        )


def _validate_phase18(
    *,
    identity: QueryIdentity,
    result: ResponseAssemblyResult,
) -> None:
    phase: QueryPhase = "phase_18_response_assembly"

    response = _document(
        result.response,
        description="Phase 18 FinalResponse",
    )

    question = _optional_string(
        response,
        "question",
    )

    if question is not None:
        _assert_equal(
            phase=phase,
            invariant="question_continuity",
            description="FinalResponse question",
            observed=question,
            expected=identity.question,
        )

    corpus = _optional_string(
        response,
        "corpus_version",
    )

    if corpus is not None:
        _assert_equal(
            phase=phase,
            invariant="corpus_continuity",
            description="FinalResponse corpus",
            observed=corpus,
            expected=identity.corpus_version,
        )

    if not result.markdown.strip():
        raise QueryPipelineInvariantError(
            phase=phase,
            invariant="final_response",
            detail="Phase 18 markdown is empty.",
        )


class QueryOrchestrator:
    """Coordinate and guard the validated Phase 14-18 runtime pipeline."""

    def __init__(
        self,
        *,
        services: QueryPipelineServices,
        provider_config: QueryPipelineProviderConfig,
    ) -> None:
        self._services = services
        self._provider_config = provider_config

    async def execute(
        self,
        question: str,
    ) -> QueryExecutionResult:
        """Execute one complete WTH query without intermediate artifacts."""

        request = QueryExecutionRequest(
            question=question,
        )

        state, identity = await self._execute_phases_14_to_17(
            question=request.question,
        )

        assembled = self._run_phase18(
            state=state,
        )

        _validate_phase18(
            identity=identity,
            result=assembled,
        )

        return self._execution_result(assembled)

    async def _execute_phases_14_to_17(
        self,
        *,
        question: str,
    ) -> tuple[
        QueryPipelineState,
        QueryIdentity,
    ]:
        retrieval = await self._run_phase14(
            question=question,
        )

        identity = _identity_from_phase14(
            request_question=question,
            result=retrieval,
        )

        domain_generation = await self._run_phase15(
            retrieval=retrieval,
        )

        _validate_phase15(
            identity=identity,
            result=domain_generation,
        )

        synthesis = await self._run_phase16(
            domain_generation=(domain_generation),
        )

        _validate_phase16(
            identity=identity,
            result=synthesis,
        )

        coverage = self._run_phase17(
            retrieval=retrieval,
            domain_generation=(domain_generation),
            synthesis=synthesis,
        )

        _validate_phase17(
            identity=identity,
            result=coverage,
        )

        return (
            QueryPipelineState(
                retrieval=retrieval,
                domain_generation=(domain_generation),
                synthesis=synthesis,
                coverage=coverage,
            ),
            identity,
        )

    async def _run_phase14(
        self,
        *,
        question: str,
    ) -> RetrievalServiceResult:
        try:
            return await asyncio.to_thread(
                self._services.retrieval.retrieve,
                question=question,
            )

        except QueryPipelineError:
            raise

        except Exception as exc:
            raise QueryPhaseExecutionError(
                phase="phase_14_retrieval",
                message=str(exc) or exc.__class__.__name__,
            ) from exc

    async def _run_phase15(
        self,
        *,
        retrieval: RetrievalServiceResult,
    ) -> DomainGenerationServiceResult:
        try:
            return await asyncio.to_thread(
                self._services.domain_generation.generate,
                evidence_package=(retrieval.evidence_package),
                retrieval_manifest=(retrieval.manifest),
                provider_config=(self._provider_config.domain_generation),
            )

        except QueryPipelineError:
            raise

        except Exception as exc:
            raise QueryPhaseExecutionError(
                phase=("phase_15_domain_generation"),
                message=str(exc) or exc.__class__.__name__,
            ) from exc

    async def _run_phase16(
        self,
        *,
        domain_generation: DomainGenerationServiceResult,
    ) -> SynthesisServiceResult:
        try:
            return await self._services.synthesis.synthesize(
                domain_responses=(domain_generation.domain_responses),
                generation_manifest=(domain_generation.manifest),
                provider_config=(self._provider_config.synthesis),
            )

        except QueryPipelineError:
            raise

        except Exception as exc:
            raise QueryPhaseExecutionError(
                phase="phase_16_synthesis",
                message=str(exc) or exc.__class__.__name__,
            ) from exc

    def _run_phase17(
        self,
        *,
        retrieval: RetrievalServiceResult,
        domain_generation: DomainGenerationServiceResult,
        synthesis: SynthesisServiceResult,
    ) -> CoverageServiceResult:
        try:
            return self._services.coverage.classify(
                evidence_package=(retrieval.evidence_package),
                retrieval_manifest=(retrieval.manifest),
                domain_responses=(domain_generation.domain_responses),
                generation_manifest=(domain_generation.manifest),
                synthesis=(synthesis.synthesis),
                synthesis_manifest=(synthesis.manifest),
            )

        except QueryPipelineError:
            raise

        except Exception as exc:
            raise QueryPhaseExecutionError(
                phase="phase_17_coverage",
                message=str(exc) or exc.__class__.__name__,
            ) from exc

    def _run_phase18(
        self,
        *,
        state: QueryPipelineState,
    ) -> ResponseAssemblyResult:
        try:
            return self._services.response_assembly.assemble(
                evidence_package=(state.retrieval.evidence_package),
                retrieval_manifest=(state.retrieval.manifest),
                domain_responses=(state.domain_generation.domain_responses),
                generation_manifest=(state.domain_generation.manifest),
                synthesis=(state.synthesis.synthesis),
                synthesis_manifest=(state.synthesis.manifest),
                coverage=(state.coverage.coverage),
                coverage_manifest=(state.coverage.manifest),
            )

        except QueryPipelineError:
            raise

        except Exception as exc:
            raise QueryPhaseExecutionError(
                phase=("phase_18_response_assembly"),
                message=str(exc) or exc.__class__.__name__,
            ) from exc

    @staticmethod
    def _execution_result(
        assembled: ResponseAssemblyResult,
    ) -> QueryExecutionResult:
        return QueryExecutionResult(
            final_response=(assembled.response),
            markdown=assembled.markdown,
        )


__all__ = [
    "QueryIdentity",
    "QueryOrchestrator",
    "QueryPhase",
    "QueryPhaseExecutionError",
    "QueryPipelineError",
    "QueryPipelineInvariantError",
    "QueryPipelineProviderConfig",
    "QueryPipelineServices",
    "QueryPipelineState",
]
