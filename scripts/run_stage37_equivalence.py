from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, TypeAlias

from scripts.assemble_phase1_final_response import (
    run_phase18,
)
from scripts.build_phase1_domain_generation import (
    run_phase15,
)
from scripts.build_phase1_retrieval import (
    run_phase14,
)
from scripts.build_phase1_synthesis import (
    run_phase16,
)
from scripts.classify_phase1_coverage import (
    run_phase17,
)
from tests.stage3.equivalence_policy import (
    EquivalenceClass,
    PhaseName,
    classify_field,
)

from apps.api.clients.supabase_runtime import (
    get_supabase_runtime_client,
)
from apps.api.repositories.concept_repository import (
    ConceptRepository,
)
from apps.api.repositories.retrieval_repository import (
    RetrievalRepository,
)
from apps.api.services.coverage import (
    CoverageService,
)
from apps.api.services.domain_generation import (
    DomainGenerationService,
    DomainProviderConfig,
    default_domain_provider_config,
)
from apps.api.services.response_assembly import (
    ResponseAssemblyService,
)
from apps.api.services.retrieval import (
    QueryEmbeddingConfig,
    RetrievalConfig,
    RetrievalOutputPaths,
    RetrievalService,
)
from apps.api.services.synthesis import (
    DEFAULT_MAX_COMPLETION_TOKENS as SYNTHESIS_MAX_COMPLETION_TOKENS,
)
from apps.api.services.synthesis import (
    DEFAULT_MAX_PROVIDER_ATTEMPTS as SYNTHESIS_MAX_PROVIDER_ATTEMPTS,
)
from apps.api.services.synthesis import (
    DEFAULT_REASONING_EFFORT as SYNTHESIS_REASONING_EFFORT,
)
from apps.api.services.synthesis import (
    DEFAULT_SYNTHESIS_MODEL,
    SynthesisProviderConfig,
    SynthesisService,
)
from apps.api.services.synthesis import (
    DEFAULT_TEMPERATURE as SYNTHESIS_TEMPERATURE,
)
from apps.api.services.synthesis import (
    DEFAULT_TIMEOUT_SECONDS as SYNTHESIS_TIMEOUT_SECONDS,
)

LOGGER = logging.getLogger("wth.stage3.7b.equivalence")

DEFAULT_QUESTION: Final = "What is the feeling of self or personal identity?"

DEFAULT_RETRIEVAL_EVALUATION_RESULTS: Final = Path(
    "artifacts/phase1/retrieval/retrieval_evaluation_results.json"
)

DEFAULT_RETRIEVAL_REPORT: Final = Path("docs/evaluation/phase1_retrieval_report.md")

DEFAULT_COOLDOWN_SECONDS: Final = 45.0

DOMAINS: Final = (
    "science",
    "advaita",
    "samkhya",
)

_TOKEN_RE: Final = re.compile(r"[A-Za-z0-9]+(?:[''-][A-Za-z0-9]+)?")

CompareStatus: TypeAlias = Literal[
    "PASS",
    "FAIL",
    "SEMANTIC_REVIEW",
    "IGNORED",
    "UNCLASSIFIED",
]


class EquivalenceRunnerError(RuntimeError):
    """Raised when either comparison lane cannot execute safely."""


@dataclass(frozen=True, slots=True)
class ProviderBundle:
    """Identical provider configuration shared by both equivalence lanes."""

    domain_generation: DomainProviderConfig
    synthesis: SynthesisProviderConfig


@dataclass(frozen=True, slots=True)
class PipelineSnapshot:
    phase14: dict[str, object]
    phase15: dict[str, object]
    phase16: dict[str, object]
    phase17: dict[str, object]
    phase18: dict[str, object]

    def by_phase(
        self,
        phase: PhaseName,
    ) -> dict[str, object]:
        mapping = {
            "phase14": self.phase14,
            "phase15": self.phase15,
            "phase16": self.phase16,
            "phase17": self.phase17,
            "phase18": self.phase18,
        }

        return mapping[phase]


@dataclass(frozen=True, slots=True)
class FieldComparison:
    phase: PhaseName
    path: str
    comparison_class: str
    status: CompareStatus
    artifact_value: object
    runtime_value: object
    detail: str


@dataclass(frozen=True, slots=True)
class PhaseComparison:
    phase: PhaseName
    exact_failures: int
    semantic_reviews: int
    ignored_fields: int
    unclassified_fields: int
    structural_passed: bool
    structural_details: tuple[str, ...]
    field_results: tuple[FieldComparison, ...]

    @property
    def passed(
        self,
    ) -> bool:
        return self.exact_failures == 0 and self.structural_passed


@dataclass(frozen=True, slots=True)
class EquivalenceRunResult:
    question: str
    phases: tuple[PhaseComparison, ...]

    @property
    def passed(
        self,
    ) -> bool:
        return all(phase.passed for phase in self.phases)


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    root: Path
    phase14: Path
    phase15: Path
    phase16: Path
    phase17: Path
    phase18: Path

    @classmethod
    def create(
        cls,
        root: Path,
    ) -> ArtifactLayout:
        return cls(
            root=root,
            phase14=root / "phase14",
            phase15=root / "phase15",
            phase16=root / "phase16",
            phase17=root / "phase17",
            phase18=root / "phase18",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3.7B: compare one question through "
            "Artifact Mode and Runtime Mode phase by phase."
        )
    )

    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
    )

    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=DEFAULT_COOLDOWN_SECONDS,
        help=(
            "Test-only pause between Artifact and Runtime lanes. "
            "Phase 15 already uses per-model scheduling; this cooldown only "
            "reduces cross-lane quota carryover during equivalence testing."
        ),
    )

    parser.add_argument(
        "--artifact-workdir",
        type=Path,
        default=None,
        help=(
            "Optional persistent Artifact Mode working directory. "
            "If omitted, a temporary directory is deleted after the run."
        ),
    )

    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help=(
            "Optional path for a machine-readable single-question "
            "equivalence report. Stage 3.7B otherwise writes no report file."
        ),
    )

    parser.add_argument(
        "--show-differences",
        action="store_true",
        help=("Print each exact/semantic/unclassified difference."),
    )

    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
    )

    return parser.parse_args()


def configure_logging(
    level: str,
) -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            level,
        ),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )


def _load_environment_file(
    path: Path,
) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(
        encoding="utf-8",
        errors="ignore",
    ).splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, raw_value = line.split(
            "=",
            1,
        )

        key = key.strip()
        value = raw_value.strip()

        if not key or key in os.environ:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0]
            in {
                "'",
                '"',
            }
        ):
            value = value[1:-1]

        os.environ[key] = value


def load_startup_environment(
    project_root: Path,
) -> None:
    for path in (
        project_root / ".env",
        project_root / ".env.local",
        project_root / "apps" / "api" / ".env",
        project_root / "apps" / "api" / ".env.local",
    ):
        _load_environment_file(path)

    google_key = os.getenv(
        "GOOGLE_API_KEY",
        "",
    ).strip()

    gemini_key = os.getenv(
        "GEMINI_API_KEY",
        "",
    ).strip()

    if not google_key and gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key


def required_environment(
    name: str,
) -> str:
    value = os.getenv(
        name,
        "",
    ).strip()

    if not value:
        raise EquivalenceRunnerError(f"{name} is not configured.")

    return value


def build_provider_bundle(
    *,
    project_root: Path,
) -> ProviderBundle:
    """Build the exact provider configuration used by both 3.7B lanes.

    Stage 3.7B is an implementation-equivalence test, so ambient model
    environment variables must not silently alter one of the frozen starting
    configurations. Both Artifact Mode and Runtime Mode receive this same
    ProviderBundle instance.

    Phase 15:
      Science  -> GPT-OSS 20B  / medium / 2500
      Advaita  -> GPT-OSS 120B / high   / 3000
      Samkhya  -> GPT-OSS 20B  / medium / 2500

    Phase 16:
      Synthesis -> GPT-OSS 120B / high / 4500
    """

    load_startup_environment(project_root)

    groq_api_key = required_environment("GROQ_API_KEY")

    domain_generation = default_domain_provider_config(
        api_key=groq_api_key,
    )

    synthesis = SynthesisProviderConfig(
        api_key=groq_api_key,
        model=DEFAULT_SYNTHESIS_MODEL,
        temperature=(SYNTHESIS_TEMPERATURE),
        max_completion_tokens=(SYNTHESIS_MAX_COMPLETION_TOKENS),
        timeout_seconds=(SYNTHESIS_TIMEOUT_SECONDS),
        max_attempts=(SYNTHESIS_MAX_PROVIDER_ATTEMPTS),
        reasoning_effort=(SYNTHESIS_REASONING_EFFORT),
    )

    LOGGER.info(
        "Stage 3.7B Phase 15 provider lanes: science=%s/%s/%d advaita=%s/%s/%d samkhya=%s/%s/%d",
        domain_generation.science.model,
        domain_generation.science.reasoning_effort,
        domain_generation.science.max_completion_tokens,
        domain_generation.advaita.model,
        domain_generation.advaita.reasoning_effort,
        domain_generation.advaita.max_completion_tokens,
        domain_generation.samkhya.model,
        domain_generation.samkhya.reasoning_effort,
        domain_generation.samkhya.max_completion_tokens,
    )

    LOGGER.info(
        "Stage 3.7B Phase 16 provider: model=%s reasoning_effort=%s max_completion_tokens=%d",
        synthesis.model,
        synthesis.reasoning_effort,
        synthesis.max_completion_tokens,
    )

    return ProviderBundle(
        domain_generation=(domain_generation),
        synthesis=synthesis,
    )


def load_json_document(
    path: Path,
) -> dict[str, object]:
    if not path.is_file():
        raise EquivalenceRunnerError(f"Expected artifact does not exist: {path}")

    try:
        raw: object = json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except json.JSONDecodeError as exc:
        raise EquivalenceRunnerError(f"Invalid JSON artifact {path}: {exc}") from exc

    if not isinstance(
        raw,
        Mapping,
    ):
        raise EquivalenceRunnerError(f"Artifact must be a JSON object: {path}")

    return {str(key): value for key, value in raw.items()}


def combine_with_manifest(
    primary: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict[str, object]:
    result = {str(key): value for key, value in primary.items()}

    result["manifest"] = {str(key): value for key, value in manifest.items()}

    return result


def artifact_paths(
    layout: ArtifactLayout,
) -> dict[str, Path]:
    return {
        "phase14_evidence": (layout.phase14 / "evidence_package.json"),
        "phase14_manifest": (layout.phase14 / "retrieval_manifest.json"),
        "phase15_domains": (layout.phase15 / "domain_responses.json"),
        "phase15_manifest": (layout.phase15 / "generation_manifest.json"),
        "phase16_synthesis": (layout.phase16 / "synthesis.json"),
        "phase16_manifest": (layout.phase16 / "synthesis_manifest.json"),
        "phase17_coverage": (layout.phase17 / "coverage.json"),
        "phase17_manifest": (layout.phase17 / "coverage_manifest.json"),
        "phase18_response": (layout.phase18 / "final_response.json"),
    }


def run_artifact_mode(
    *,
    project_root: Path,
    question: str,
    layout: ArtifactLayout,
    providers: ProviderBundle,
) -> PipelineSnapshot:
    """Execute wrappers exactly as Artifact Mode, then reload their JSON."""

    paths = artifact_paths(layout)

    LOGGER.info("Stage 3.7B Artifact Mode starting")

    run_phase14(
        project_root=project_root,
        question=question,
        output_directory=layout.phase14,
        retrieval_evaluation_results_path=(DEFAULT_RETRIEVAL_EVALUATION_RESULTS),
        retrieval_report_path=(DEFAULT_RETRIEVAL_REPORT),
        config=RetrievalConfig(),
        embedding_timeout_seconds=45.0,
        embedding_max_attempts=4,
        replace=True,
    )

    run_phase15(
        project_root=project_root,
        evidence_package_path=(paths["phase14_evidence"]),
        retrieval_manifest_path=(paths["phase14_manifest"]),
        output_directory=layout.phase15,
        provider_config=(providers.domain_generation),
        replace=True,
    )

    run_phase16(
        project_root=project_root,
        domain_responses_path=(paths["phase15_domains"]),
        generation_manifest_path=(paths["phase15_manifest"]),
        output_directory=layout.phase16,
        provider_config=(providers.synthesis),
        replace=True,
    )

    run_phase17(
        project_root=project_root,
        evidence_package_path=(paths["phase14_evidence"]),
        retrieval_manifest_path=(paths["phase14_manifest"]),
        domain_responses_path=(paths["phase15_domains"]),
        generation_manifest_path=(paths["phase15_manifest"]),
        synthesis_path=(paths["phase16_synthesis"]),
        synthesis_manifest_path=(paths["phase16_manifest"]),
        output_directory=layout.phase17,
        replace=True,
    )

    run_phase18(
        project_root=project_root,
        evidence_package_path=(paths["phase14_evidence"]),
        retrieval_manifest_path=(paths["phase14_manifest"]),
        domain_responses_path=(paths["phase15_domains"]),
        generation_manifest_path=(paths["phase15_manifest"]),
        synthesis_path=(paths["phase16_synthesis"]),
        synthesis_manifest_path=(paths["phase16_manifest"]),
        coverage_path=(paths["phase17_coverage"]),
        coverage_manifest_path=(paths["phase17_manifest"]),
        output_directory=layout.phase18,
        replace=True,
    )

    return PipelineSnapshot(
        phase14=combine_with_manifest(
            load_json_document(paths["phase14_evidence"]),
            load_json_document(paths["phase14_manifest"]),
        ),
        phase15=combine_with_manifest(
            load_json_document(paths["phase15_domains"]),
            load_json_document(paths["phase15_manifest"]),
        ),
        phase16=combine_with_manifest(
            load_json_document(paths["phase16_synthesis"]),
            load_json_document(paths["phase16_manifest"]),
        ),
        phase17=combine_with_manifest(
            load_json_document(paths["phase17_coverage"]),
            load_json_document(paths["phase17_manifest"]),
        ),
        phase18=load_json_document(paths["phase18_response"]),
    )


async def run_runtime_mode(
    *,
    project_root: Path,
    question: str,
    providers: ProviderBundle,
    manifest_paths: ArtifactLayout,
) -> PipelineSnapshot:
    """Execute all five services directly; no artifact reads/writes occur."""

    LOGGER.info("Stage 3.7B Runtime Mode starting")

    google_api_key = required_environment("GOOGLE_API_KEY")

    client = get_supabase_runtime_client()

    retrieval_service = RetrievalService(
        retrieval_repository=(RetrievalRepository(client)),
        concept_repository=(ConceptRepository(client)),
        embedding_config=(
            QueryEmbeddingConfig(
                api_key=google_api_key,
            )
        ),
    )

    phase14_result = await asyncio.to_thread(
        retrieval_service.retrieve,
        question=question,
        config=RetrievalConfig(),
        output_paths=RetrievalOutputPaths(
            retrieval_config=(manifest_paths.phase14 / "retrieval_config.json").as_posix(),
            evidence_package=(manifest_paths.phase14 / "evidence_package.json").as_posix(),
            retrieval_evaluation_results=(
                (project_root / DEFAULT_RETRIEVAL_EVALUATION_RESULTS).resolve().as_posix()
            ),
            retrieval_report=((project_root / DEFAULT_RETRIEVAL_REPORT).resolve().as_posix()),
        ),
    )

    phase15_result = await asyncio.to_thread(
        DomainGenerationService().generate,
        evidence_package=(phase14_result.evidence_package),
        retrieval_manifest=(phase14_result.manifest),
        provider_config=(providers.domain_generation),
        output_paths={
            "science": (manifest_paths.phase15 / "science_response.json").as_posix(),
            "advaita": (manifest_paths.phase15 / "advaita_response.json").as_posix(),
            "samkhya": (manifest_paths.phase15 / "samkhya_response.json").as_posix(),
            "combined": (manifest_paths.phase15 / "domain_responses.json").as_posix(),
        },
    )

    phase16_result = await SynthesisService().synthesize(
        domain_responses=(phase15_result.domain_responses),
        generation_manifest=(phase15_result.manifest),
        provider_config=(providers.synthesis),
        synthesis_output_path=(manifest_paths.phase16 / "synthesis.json").as_posix(),
    )

    phase17_result = CoverageService().classify(
        evidence_package=(phase14_result.evidence_package),
        retrieval_manifest=(phase14_result.manifest),
        domain_responses=(phase15_result.domain_responses),
        generation_manifest=(phase15_result.manifest),
        synthesis=(phase16_result.synthesis),
        synthesis_manifest=(phase16_result.manifest),
        coverage_output_path=(manifest_paths.phase17 / "coverage.json").as_posix(),
    )

    phase18_result = ResponseAssemblyService().assemble(
        evidence_package=(phase14_result.evidence_package),
        retrieval_manifest=(phase14_result.manifest),
        domain_responses=(phase15_result.domain_responses),
        generation_manifest=(phase15_result.manifest),
        synthesis=(phase16_result.synthesis),
        synthesis_manifest=(phase16_result.manifest),
        coverage=(phase17_result.coverage),
        coverage_manifest=(phase17_result.manifest),
    )

    return PipelineSnapshot(
        phase14=combine_with_manifest(
            phase14_result.evidence_package.model_dump(
                mode="python",
                by_alias=True,
            ),
            phase14_result.manifest.model_dump(
                mode="python",
                by_alias=True,
            ),
        ),
        phase15=combine_with_manifest(
            phase15_result.domain_responses.model_dump(
                mode="python",
                by_alias=True,
            ),
            phase15_result.manifest.model_dump(
                mode="python",
                by_alias=True,
            ),
        ),
        phase16=combine_with_manifest(
            phase16_result.synthesis.model_dump(
                mode="python",
                by_alias=True,
            ),
            phase16_result.manifest.model_dump(
                mode="python",
                by_alias=True,
            ),
        ),
        phase17=combine_with_manifest(
            phase17_result.coverage.model_dump(
                mode="python",
                by_alias=True,
            ),
            phase17_result.manifest.model_dump(
                mode="python",
                by_alias=True,
            ),
        ),
        phase18=(
            phase18_result.response.model_dump(
                mode="python",
                by_alias=True,
            )
        ),
    )


def _tokenize(
    value: str,
) -> frozenset[str]:
    return frozenset(token.casefold() for token in _TOKEN_RE.findall(value))


def semantic_similarity(
    left: object,
    right: object,
) -> float | None:
    if not (
        isinstance(
            left,
            str,
        )
        and isinstance(
            right,
            str,
        )
    ):
        return None

    left_norm = " ".join(left.split()).casefold()

    right_norm = " ".join(right.split()).casefold()

    if left_norm == right_norm:
        return 1.0

    left_tokens = _tokenize(left_norm)
    right_tokens = _tokenize(right_norm)

    union = left_tokens | right_tokens

    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0

    sequence = difflib.SequenceMatcher(
        None,
        left_norm,
        right_norm,
    ).ratio()

    return 0.5 * jaccard + 0.5 * sequence


def values_exact(
    left: object,
    right: object,
    *,
    tolerance: float | None,
) -> bool:
    if (
        tolerance is not None
        and isinstance(
            left,
            (int, float),
        )
        and not isinstance(
            left,
            bool,
        )
        and isinstance(
            right,
            (int, float),
        )
        and not isinstance(
            right,
            bool,
        )
    ):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=tolerance,
        )

    return left == right


def child_path(
    parent: str,
    child: str,
) -> str:
    if not parent:
        return child

    return f"{parent}.{child}"


def list_path(
    parent: str,
    index: int,
) -> str:
    return f"{parent}[{index}]"


def compare_documents(
    *,
    phase: PhaseName,
    artifact: object,
    runtime: object,
) -> tuple[FieldComparison, ...]:
    results: list[FieldComparison] = []

    def visit(
        left: object,
        right: object,
        path: str,
    ) -> None:
        if path:
            rule = classify_field(
                phase,
                path,
            )

            if rule is not None:
                if rule.comparison == EquivalenceClass.NON_DETERMINISTIC:
                    results.append(
                        FieldComparison(
                            phase=phase,
                            path=path,
                            comparison_class=(rule.comparison.value),
                            status="IGNORED",
                            artifact_value=left,
                            runtime_value=right,
                            detail=rule.rationale,
                        )
                    )
                    return

                if rule.comparison == EquivalenceClass.EXACT:
                    passed = values_exact(
                        left,
                        right,
                        tolerance=(rule.absolute_tolerance),
                    )

                    results.append(
                        FieldComparison(
                            phase=phase,
                            path=path,
                            comparison_class=(rule.comparison.value),
                            status=("PASS" if passed else "FAIL"),
                            artifact_value=left,
                            runtime_value=right,
                            detail=rule.rationale,
                        )
                    )
                    return

                if rule.comparison == EquivalenceClass.SEMANTIC:
                    if left == right:
                        status: CompareStatus = "PASS"
                        detail = "Semantic field is also byte-equivalent."
                    else:
                        score = semantic_similarity(
                            left,
                            right,
                        )

                        status = "SEMANTIC_REVIEW"

                        detail = (
                            f"{rule.rationale} Text similarity={score:.3f}."
                            if score is not None
                            else (f"{rule.rationale} Values differ and require semantic review.")
                        )

                    results.append(
                        FieldComparison(
                            phase=phase,
                            path=path,
                            comparison_class=(rule.comparison.value),
                            status=status,
                            artifact_value=left,
                            runtime_value=right,
                            detail=detail,
                        )
                    )
                    return

        if isinstance(
            left,
            Mapping,
        ) and isinstance(
            right,
            Mapping,
        ):
            keys = sorted({str(key) for key in left} | {str(key) for key in right})

            for key in keys:
                present_left = key in left
                present_right = key in right

                next_path = child_path(
                    path,
                    key,
                )

                if not (present_left and present_right):
                    rule = classify_field(
                        phase,
                        next_path,
                    )

                    comparison_class = rule.comparison.value if rule is not None else "UNCLASSIFIED"

                    status = (
                        "FAIL"
                        if (rule is not None and rule.comparison == EquivalenceClass.EXACT)
                        else "UNCLASSIFIED"
                    )

                    results.append(
                        FieldComparison(
                            phase=phase,
                            path=next_path,
                            comparison_class=(comparison_class),
                            status=status,
                            artifact_value=(left.get(key) if present_left else "<MISSING>"),
                            runtime_value=(right.get(key) if present_right else "<MISSING>"),
                            detail=("Field is present in only one mode."),
                        )
                    )
                    continue

                visit(
                    left[key],
                    right[key],
                    next_path,
                )

            return

        if (
            isinstance(
                left,
                Sequence,
            )
            and not isinstance(
                left,
                (str, bytes, bytearray),
            )
            and isinstance(
                right,
                Sequence,
            )
            and not isinstance(
                right,
                (str, bytes, bytearray),
            )
        ):
            max_length = max(
                len(left),
                len(right),
            )

            for index in range(max_length):
                next_path = list_path(
                    path,
                    index,
                )

                if index >= len(left) or index >= len(right):
                    rule = classify_field(
                        phase,
                        next_path,
                    )

                    status = (
                        "FAIL"
                        if (rule is not None and rule.comparison == EquivalenceClass.EXACT)
                        else "UNCLASSIFIED"
                    )

                    results.append(
                        FieldComparison(
                            phase=phase,
                            path=next_path,
                            comparison_class=(
                                rule.comparison.value if rule is not None else "UNCLASSIFIED"
                            ),
                            status=status,
                            artifact_value=(left[index] if index < len(left) else "<MISSING>"),
                            runtime_value=(right[index] if index < len(right) else "<MISSING>"),
                            detail=("List element is present in only one mode."),
                        )
                    )
                    continue

                visit(
                    left[index],
                    right[index],
                    next_path,
                )

            return

        if left != right:
            results.append(
                FieldComparison(
                    phase=phase,
                    path=path or "<root>",
                    comparison_class="UNCLASSIFIED",
                    status="UNCLASSIFIED",
                    artifact_value=left,
                    runtime_value=right,
                    detail=("Differing field is not yet classified by Stage 3.7A policy."),
                )
            )

    visit(
        artifact,
        runtime,
        "",
    )

    return tuple(results)


def _mapping_value(
    value: object,
) -> Mapping[str, object]:
    if isinstance(
        value,
        Mapping,
    ):
        return value

    return {}


def _sequence_value(
    value: object,
) -> Sequence[object]:
    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return value

    return ()


def structural_signature(
    phase: PhaseName,
    document: Mapping[str, object],
) -> dict[str, object]:
    """Produce phase-specific structure/reference summaries.

    These summaries deliberately avoid LLM prose while checking the shape and
    reference topology Stage 3.7 cares about.
    """

    if phase == "phase14":
        domains = _mapping_value(document.get("domains"))

        domain_summary: dict[
            str,
            object,
        ] = {}

        for domain in DOMAINS:
            payload = _mapping_value(domains.get(domain))

            evidence = _sequence_value(payload.get("evidence"))

            chunk_ids: list[str] = []
            source_ids: list[str] = []
            token_total = 0

            for raw in evidence:
                item = _mapping_value(raw)

                chunk_id = item.get("chunk_id")
                source_id = item.get("source_id")
                estimated = item.get("estimated_tokens")

                if isinstance(
                    chunk_id,
                    str,
                ):
                    chunk_ids.append(chunk_id)

                if isinstance(
                    source_id,
                    str,
                ):
                    source_ids.append(source_id)

                if isinstance(
                    estimated,
                    int,
                ) and not isinstance(
                    estimated,
                    bool,
                ):
                    token_total += estimated

            domain_summary[domain] = {
                "evidence_count": (len(evidence)),
                "chunk_ids": (sorted(chunk_ids)),
                "unique_source_count": (len(set(source_ids))),
                "max_source_repeat": (
                    max(
                        Counter(source_ids).values(),
                        default=0,
                    )
                ),
                "estimated_tokens": (token_total),
            }

        return {
            "domain_keys": sorted(domains.keys()),
            "domains": domain_summary,
        }

    if phase == "phase15":
        domains = _mapping_value(document.get("domains"))

        summary: dict[
            str,
            object,
        ] = {}

        for domain in DOMAINS:
            payload = _mapping_value(domains.get(domain))

            claims = _sequence_value(payload.get("claims"))

            supporting_refs: list[str] = []
            citation_refs: list[str] = []

            for raw in claims:
                claim = _mapping_value(raw)

                for ref in _sequence_value(claim.get("supporting_chunk_ids")):
                    if isinstance(
                        ref,
                        str,
                    ):
                        supporting_refs.append(ref)

                for ref in _sequence_value(claim.get("citation_ids")):
                    if isinstance(
                        ref,
                        str,
                    ):
                        citation_refs.append(ref)

            summary[domain] = {
                "claim_count": len(claims),
                "supporting_chunk_ids": sorted(set(supporting_refs)),
                "citation_ids": sorted(set(citation_refs)),
            }

        return {
            "domain_keys": sorted(domains.keys()),
            "domains": summary,
        }

    if phase == "phase16":
        comparisons = _sequence_value(document.get("comparisons"))

        slot_ids: list[str] = []
        claim_refs: list[str] = []
        limitation_refs: list[str] = []

        for raw in comparisons:
            comparison = _mapping_value(raw)

            comparison_id = comparison.get("comparison_id")

            if isinstance(
                comparison_id,
                str,
            ):
                slot_ids.append(comparison_id)

            for ref in _sequence_value(comparison.get("claim_refs")):
                if isinstance(
                    ref,
                    str,
                ):
                    claim_refs.append(ref)

            for ref in _sequence_value(comparison.get("limitation_refs")):
                if isinstance(
                    ref,
                    str,
                ):
                    limitation_refs.append(ref)

        validation = _mapping_value(document.get("validation"))

        return {
            "comparison_count": len(comparisons),
            "comparison_ids": sorted(slot_ids),
            "duplicate_comparison_ids": sorted(
                slot_id for slot_id, count in Counter(slot_ids).items() if count > 1
            ),
            "claim_refs": sorted(set(claim_refs)),
            "limitation_refs": sorted(set(limitation_refs)),
            "validation_passed": (validation.get("passed")),
        }

    if phase == "phase17":
        return {
            "coverage_status": (document.get("coverage_status")),
            "coverage_score": (document.get("coverage_score")),
            "supported_concepts": (document.get("supported_concepts")),
            "partially_supported_concepts": (document.get("partially_supported_concepts")),
            "unsupported_concepts": (document.get("unsupported_concepts")),
        }

    sections = _mapping_value(document.get("sections"))

    domain_perspectives = _mapping_value(sections.get("domain_perspectives"))

    comparative = _mapping_value(sections.get("comparative_synthesis"))

    comparisons = _sequence_value(comparative.get("comparisons"))

    citations = _sequence_value(document.get("claim_level_citations"))

    fallback = _mapping_value(sections.get("general_knowledge_fallback"))

    validation = _mapping_value(document.get("validation"))

    return {
        "section_keys": sorted(sections.keys()),
        "domain_keys": sorted(domain_perspectives.keys()),
        "comparison_count": len(comparisons),
        "citation_count": len(citations),
        "fallback_shape": sorted(fallback.keys()),
        "validation_passed": (validation.get("passed")),
    }


def compare_phase(
    *,
    phase: PhaseName,
    artifact: Mapping[str, object],
    runtime: Mapping[str, object],
) -> PhaseComparison:
    field_results = compare_documents(
        phase=phase,
        artifact=artifact,
        runtime=runtime,
    )

    exact_failures = sum(1 for item in field_results if item.status == "FAIL")

    semantic_reviews = sum(1 for item in field_results if item.status == "SEMANTIC_REVIEW")

    ignored_fields = sum(1 for item in field_results if item.status == "IGNORED")

    unclassified_fields = sum(1 for item in field_results if item.status == "UNCLASSIFIED")

    artifact_structure = structural_signature(
        phase,
        artifact,
    )

    runtime_structure = structural_signature(
        phase,
        runtime,
    )

    structural_passed = artifact_structure == runtime_structure

    structural_details: tuple[
        str,
        ...,
    ] = (
        ("Artifact/runtime structural signatures match.",)
        if structural_passed
        else (
            (
                "Artifact structure: "
                + json.dumps(
                    artifact_structure,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            ),
            (
                "Runtime structure: "
                + json.dumps(
                    runtime_structure,
                    sort_keys=True,
                    ensure_ascii=False,
                )
            ),
        )
    )

    return PhaseComparison(
        phase=phase,
        exact_failures=(exact_failures),
        semantic_reviews=(semantic_reviews),
        ignored_fields=(ignored_fields),
        unclassified_fields=(unclassified_fields),
        structural_passed=(structural_passed),
        structural_details=(structural_details),
        field_results=(field_results),
    )


def compare_snapshots(
    *,
    artifact: PipelineSnapshot,
    runtime: PipelineSnapshot,
) -> tuple[
    PhaseComparison,
    ...,
]:
    phases: tuple[
        PhaseName,
        ...,
    ] = (
        "phase14",
        "phase15",
        "phase16",
        "phase17",
        "phase18",
    )

    return tuple(
        compare_phase(
            phase=phase,
            artifact=artifact.by_phase(phase),
            runtime=runtime.by_phase(phase),
        )
        for phase in phases
    )


def report_document(
    result: EquivalenceRunResult,
) -> dict[str, object]:
    return {
        "stage": ("stage_3_7b_artifact_runtime_equivalence"),
        "question": result.question,
        "passed": result.passed,
        "phases": [
            {
                "phase": phase.phase,
                "passed": phase.passed,
                "exact_failures": (phase.exact_failures),
                "semantic_reviews": (phase.semantic_reviews),
                "ignored_fields": (phase.ignored_fields),
                "unclassified_fields": (phase.unclassified_fields),
                "structural_passed": (phase.structural_passed),
                "structural_details": list(phase.structural_details),
                "differences": [
                    {
                        "path": item.path,
                        "class": (item.comparison_class),
                        "status": item.status,
                        "artifact": (item.artifact_value),
                        "runtime": (item.runtime_value),
                        "detail": item.detail,
                    }
                    for item in phase.field_results
                    if item.status
                    in {
                        "FAIL",
                        "SEMANTIC_REVIEW",
                        "UNCLASSIFIED",
                    }
                ],
            }
            for phase in result.phases
        ],
    }


def write_report(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path = path.resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def print_summary(
    result: EquivalenceRunResult,
    *,
    show_differences: bool,
) -> None:
    print()
    print("=" * 78)
    print("STAGE 3.7B — ARTIFACT / RUNTIME EQUIVALENCE")
    print("=" * 78)
    print(f"Question: {result.question}")
    print()

    for phase in result.phases:
        status = "PASS" if phase.passed else "FAIL"

        print(
            f"{phase.phase.upper():<9} "
            f"{status:<4} | "
            f"exact_failures={phase.exact_failures} "
            f"semantic_review={phase.semantic_reviews} "
            f"structural={'PASS' if phase.structural_passed else 'FAIL'} "
            f"unclassified={phase.unclassified_fields}"
        )

        if show_differences and (
            not phase.passed or phase.semantic_reviews or phase.unclassified_fields
        ):
            for item in phase.field_results:
                if item.status not in {
                    "FAIL",
                    "SEMANTIC_REVIEW",
                    "UNCLASSIFIED",
                }:
                    continue

                print(f"  - {item.status:<15} {item.path} [{item.comparison_class}]")

                if item.status == "FAIL":
                    print(f"      artifact={item.artifact_value!r}")
                    print(f"      runtime ={item.runtime_value!r}")

                print(f"      {item.detail}")

            if not phase.structural_passed:
                for detail in phase.structural_details:
                    print(f"  - STRUCTURAL: {detail}")

    print()
    print("Overall: " + ("PASS" if result.passed else "FAIL"))
    print("Semantic differences are review items and do not fail 3.7B by themselves.")
    print("=" * 78)


async def execute_equivalence(
    *,
    project_root: Path,
    question: str,
    artifact_root: Path,
    providers: ProviderBundle,
    cooldown_seconds: float,
) -> EquivalenceRunResult:
    normalized_question = question.strip()

    if not normalized_question:
        raise EquivalenceRunnerError("Question must be non-empty.")

    layout = ArtifactLayout.create(artifact_root)

    # Artifact Mode is intentionally synchronous because it exercises the
    # standalone Phase 14-18 file-I/O wrappers. Phase 16's wrapper uses
    # asyncio.run() internally for its async synthesis service, so the whole
    # Artifact Mode lane must execute off the equivalence runner's event-loop
    # thread. This preserves the real CLI wrapper behavior without nesting
    # event loops.
    artifact_snapshot = await asyncio.to_thread(
        run_artifact_mode,
        project_root=project_root,
        question=normalized_question,
        layout=layout,
        providers=providers,
    )

    if cooldown_seconds > 0.0:
        LOGGER.info(
            "Cooling down %.1f seconds before Runtime Mode",
            cooldown_seconds,
        )
        await asyncio.sleep(cooldown_seconds)

    runtime_snapshot = await run_runtime_mode(
        project_root=project_root,
        question=normalized_question,
        providers=providers,
        manifest_paths=layout,
    )

    phases = compare_snapshots(
        artifact=artifact_snapshot,
        runtime=runtime_snapshot,
    )

    return EquivalenceRunResult(
        question=normalized_question,
        phases=phases,
    )


def main() -> int:
    arguments = parse_arguments()

    configure_logging(arguments.log_level)

    project_root = Path.cwd().resolve()

    providers = build_provider_bundle(project_root=project_root)

    persistent_root: Path | None = arguments.artifact_workdir

    try:
        if persistent_root is not None:
            artifact_root = persistent_root.resolve()

            if artifact_root.exists():
                shutil.rmtree(artifact_root)

            artifact_root.mkdir(
                parents=True,
                exist_ok=True,
            )

            result = asyncio.run(
                execute_equivalence(
                    project_root=project_root,
                    question=(arguments.question),
                    artifact_root=(artifact_root),
                    providers=providers,
                    cooldown_seconds=max(
                        0.0,
                        arguments.cooldown_seconds,
                    ),
                )
            )

        else:
            with tempfile.TemporaryDirectory(prefix="wth_stage37b_") as temporary:
                artifact_root = Path(temporary)

                result = asyncio.run(
                    execute_equivalence(
                        project_root=project_root,
                        question=(arguments.question),
                        artifact_root=(artifact_root),
                        providers=providers,
                        cooldown_seconds=max(
                            0.0,
                            arguments.cooldown_seconds,
                        ),
                    )
                )

        print_summary(
            result,
            show_differences=(arguments.show_differences),
        )

        if arguments.report_json is not None:
            write_report(
                arguments.report_json,
                report_document(result),
            )

        exit_code = 0 if result.passed else 1

    except Exception:
        LOGGER.exception("Stage 3.7B equivalence run failed")
        return 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
