from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, TypeAlias

PhaseName: TypeAlias = Literal[
    "phase14",
    "phase15",
    "phase16",
    "phase17",
    "phase18",
]

_INDEX_RE: Final = re.compile(r"\[\d+\]")


class EquivalenceClass(StrEnum):
    EXACT = "EXACT"
    SEMANTIC = "SEMANTIC"
    STRUCTURAL = "STRUCTURAL"
    NON_DETERMINISTIC = "NON_DETERMINISTIC"


@dataclass(frozen=True, slots=True)
class FieldRule:
    pattern: str
    comparison: EquivalenceClass
    rationale: str
    absolute_tolerance: float | None = None


@dataclass(frozen=True, slots=True)
class StructuralRule:
    rule_id: str
    description: str


@dataclass(frozen=True, slots=True)
class PhaseEquivalencePolicy:
    phase: PhaseName
    field_rules: tuple[FieldRule, ...]
    structural_rules: tuple[StructuralRule, ...]


def normalize_equivalence_path(path: str) -> str:
    return _INDEX_RE.sub("[*]", path)


def _exact(
    pattern: str,
    rationale: str,
    *,
    tolerance: float | None = None,
) -> FieldRule:
    return FieldRule(
        pattern=pattern,
        comparison=EquivalenceClass.EXACT,
        rationale=rationale,
        absolute_tolerance=tolerance,
    )


def _semantic(
    pattern: str,
    rationale: str,
) -> FieldRule:
    return FieldRule(
        pattern=pattern,
        comparison=EquivalenceClass.SEMANTIC,
        rationale=rationale,
    )


def _nondeterministic(
    pattern: str,
    rationale: str,
) -> FieldRule:
    return FieldRule(
        pattern=pattern,
        comparison=EquivalenceClass.NON_DETERMINISTIC,
        rationale=rationale,
    )


COMMON_NON_DETERMINISTIC: Final = (
    _nondeterministic(
        "generated_at",
        "Wall-clock timestamp.",
    ),
    _nondeterministic(
        "*.generated_at",
        "Wall-clock timestamp.",
    ),
    _nondeterministic(
        "*.created_at",
        "Creation timestamp may vary.",
    ),
    _nondeterministic(
        "*.updated_at",
        "Update timestamp may vary.",
    ),
    _nondeterministic(
        "*.latency*",
        "Provider/network latency varies.",
    ),
    _nondeterministic(
        "*.elapsed*",
        "Wall-clock duration varies.",
    ),
    _nondeterministic(
        "*.duration*",
        "Wall-clock duration varies.",
    ),
    _nondeterministic(
        "*.request_id",
        "Provider request ID is per-call.",
    ),
    _nondeterministic(
        "*.response_id",
        "Provider response ID is per-call.",
    ),
    _nondeterministic(
        "*.system_fingerprint",
        "Provider infrastructure fingerprint may vary.",
    ),
    _nondeterministic(
        "*.usage.*",
        "Provider token accounting may vary.",
    ),
    _nondeterministic(
        "*.prompt_tokens",
        "Provider token accounting may vary.",
    ),
    _nondeterministic(
        "*.completion_tokens",
        "Provider token accounting may vary.",
    ),
    _nondeterministic(
        "*.total_tokens",
        "Provider token accounting may vary.",
    ),
)


PHASE14_POLICY: Final = PhaseEquivalencePolicy(
    phase="phase14",
    field_rules=(
        _exact("question", "Normalized question identity."),
        _exact("corpus_version", "Frozen corpus identity."),
        _exact("retrieval_version", "Frozen retrieval version."),
        _exact("retrieval_mode", "Both modes must remain concept-aware."),
        _exact("model_version", "Frozen embedding model/revision."),
        _exact("prototype_version", "Frozen Phase 10 prototype identity."),
        _exact("config.*", "Frozen retrieval configuration."),
        _exact("scoring.*", "Frozen Phase 14 score weights/penalties."),
        _exact(
            "query_activation.active_concepts",
            "Activated concept identity/order.",
        ),
        _exact(
            "query_activation.ambiguous",
            "Deterministic ambiguity classification.",
        ),
        _exact(
            "query_activation.unsupported",
            "Deterministic unsupported classification.",
        ),
        _exact(
            "query_activation.calibrated_weights.*",
            "Frozen calibrated concept weights.",
            tolerance=1e-9,
        ),
        _exact(
            "query_activation.raw_scores.*",
            "Frozen raw concept scores.",
            tolerance=1e-9,
        ),
        _exact("domains.*.status", "Per-domain retrieval status."),
        _exact("domains.*.evidence[*].rank", "Evidence rank."),
        _exact(
            "domains.*.evidence[*].chunk_id",
            "Retrieved chunk identity.",
        ),
        _exact(
            "domains.*.evidence[*].source_id",
            "Retrieved source identity.",
        ),
        _exact(
            "domains.*.evidence[*].domain",
            "Evidence domain identity.",
        ),
        _exact(
            "domains.*.evidence[*].citation",
            "Frozen citation identity/text.",
        ),
        _exact(
            "domains.*.evidence[*].corpus_version",
            "Evidence corpus identity.",
        ),
        _exact(
            "domains.*.evidence[*].concepts.*.human_label",
            "Frozen human-reviewed label.",
        ),
        _exact(
            "domains.*.evidence[*].concepts.*.production_active",
            "Frozen human-reviewed eligibility.",
        ),
        _exact(
            "domains.*.evidence[*].concepts.*.human_override",
            "Human override provenance.",
        ),
        _exact(
            "domains.*.evidence[*].concepts.*.calibrated_weight",
            "Frozen chunk-concept calibrated weight.",
            tolerance=1e-9,
        ),
        _exact(
            "domains.*.evidence[*].scores.*",
            "Ranking score parity with storage floating-point tolerance.",
            tolerance=1e-7,
        ),
        _exact(
            "manifest.exit_gate.*",
            "Phase 14 exit-gate booleans.",
        ),
        _exact(
            "manifest.corpus_version",
            "Manifest corpus identity.",
        ),
        _exact(
            "manifest.retrieval_version",
            "Manifest retrieval version.",
        ),
        *COMMON_NON_DETERMINISTIC,
    ),
    structural_rules=(
        StructuralRule(
            "phase14_exact_three_domains",
            "Exactly science, advaita, samkhya are present.",
        ),
        StructuralRule(
            "phase14_domain_isolation",
            "Every evidence item remains in its owning domain.",
        ),
        StructuralRule(
            "phase14_evidence_count_consistency",
            "Declared evidence_count equals list length.",
        ),
        StructuralRule(
            "phase14_unique_source_count_consistency",
            "Declared unique_source_count equals distinct sources.",
        ),
        StructuralRule(
            "phase14_top_k",
            "No domain exceeds top K=3.",
        ),
        StructuralRule(
            "phase14_source_cap",
            "No source exceeds two selected chunks/domain.",
        ),
        StructuralRule(
            "phase14_token_budget",
            "Selected evidence stays within 900 tokens/domain.",
        ),
        StructuralRule(
            "phase14_dedup",
            "Frozen same/cross-source dedup rules still hold.",
        ),
    ),
)


PHASE15_POLICY: Final = PhaseEquivalencePolicy(
    phase="phase15",
    field_rules=(
        _exact("question", "Question continuity."),
        _exact("corpus_version", "Corpus continuity."),
        _exact("generation_version", "Frozen generation version."),
        _exact("prompt_version", "Prompt version provenance."),
        _exact("provider.model", "Configured provider model."),
        _exact("provider.*model*", "Provider model/version identity."),
        _exact("domains.*.domain", "Owning domain identity."),
        _exact(
            "domains.*.claims[*].claim_id",
            "Canonical claim identity.",
        ),
        _exact(
            "domains.*.claims[*].supporting_chunk_ids",
            "Claim grounding chunk identities.",
        ),
        _exact(
            "domains.*.claims[*].citation_ids",
            "Claim citation identities.",
        ),
        _exact(
            "domains.*.validation.passed",
            "Grounding/domain-leakage validation outcome.",
        ),
        _exact(
            "domains.*.validation.issues[*].code",
            "Validation rule outcomes.",
        ),
        _exact(
            "manifest.exit_gate.*",
            "Phase 15 exit-gate booleans.",
        ),
        _exact("manifest.status", "Phase completion status."),
        _semantic(
            "domains.*.summary",
            "LLM summary wording may vary.",
        ),
        _semantic(
            "domains.*.claims[*].text",
            "LLM claim wording may vary if grounding/meaning is preserved.",
        ),
        _semantic(
            "domains.*.limitations[*]",
            "Natural-language limitation wording may vary.",
        ),
        *COMMON_NON_DETERMINISTIC,
    ),
    structural_rules=(
        StructuralRule(
            "phase15_exact_three_domains",
            "Exactly one Science, Advaita, and Samkhya response.",
        ),
        StructuralRule(
            "phase15_domain_grounding",
            "Claims cite only same-domain Phase 14 evidence.",
        ),
        StructuralRule(
            "phase15_claim_support",
            "Every substantive claim has valid support.",
        ),
        StructuralRule(
            "phase15_citation_resolution",
            "Every citation/reference resolves to Phase 14 evidence.",
        ),
        StructuralRule(
            "phase15_unsupported_propagation",
            "Unsupported aspects/limitations remain available downstream.",
        ),
        StructuralRule(
            "phase15_no_domain_leakage",
            "Frozen Science/Advaita/Samkhya boundaries remain enforced.",
        ),
    ),
)


PHASE16_POLICY: Final = PhaseEquivalencePolicy(
    phase="phase16",
    field_rules=(
        _exact("question", "Question continuity."),
        _exact("corpus_version", "Corpus continuity."),
        _exact("synthesis_version", "Frozen synthesis version."),
        _exact("prompt_version", "Prompt version provenance."),
        _exact("provider.model", "Configured synthesis model."),
        _exact(
            "comparisons[*].comparison_id",
            "Python-owned comparison slot identity.",
        ),
        _exact(
            "comparisons[*].concept",
            "Comparison concept identity.",
        ),
        _exact(
            "comparisons[*].left_domain",
            "Python-owned left domain.",
        ),
        _exact(
            "comparisons[*].right_domain",
            "Python-owned right domain.",
        ),
        _exact(
            "comparisons[*].claim_refs",
            "Canonicalized comparison claim references.",
        ),
        _exact(
            "comparisons[*].limitation_refs",
            "Python-owned valid limitation references.",
        ),
        _exact(
            "validation.passed",
            "Frozen synthesis safety validation.",
        ),
        _exact(
            "validation.issues[*].code",
            "Safety/entailment validation outcomes.",
        ),
        _exact(
            "manifest.exit_gate.*",
            "Phase 16 exit-gate booleans.",
        ),
        _exact("manifest.status", "Phase completion status."),
        _semantic(
            "comparisons[*].category",
            "Live LLM calls may choose a nearby valid relationship category.",
        ),
        _semantic(
            "comparisons[*].explanation",
            "Comparison wording is live LLM output.",
        ),
        _semantic("summary", "Synthesis prose is live LLM output."),
        *COMMON_NON_DETERMINISTIC,
    ),
    structural_rules=(
        StructuralRule(
            "phase16_complete_slot_matrix",
            "Every Python-owned concept/domain-pair slot appears exactly once.",
        ),
        StructuralRule(
            "phase16_no_duplicate_slots",
            "No comparison slot is duplicated.",
        ),
        StructuralRule(
            "phase16_claim_ref_resolution",
            "Every comparison claim_ref resolves to the correct Phase 15 claim.",
        ),
        StructuralRule(
            "phase16_limitation_ref_resolution",
            "Every limitation_ref resolves to a Python-supplied limitation.",
        ),
        StructuralRule(
            "phase16_ic_requires_limitation",
            "insufficient_corpus_coverage requires a valid limitation_ref.",
        ),
        StructuralRule(
            "phase16_atman_not_purusha",
            "Atman/Brahman is never equated with Purusha.",
        ),
        StructuralRule(
            "phase16_science_not_metaphysical_proof",
            "Science is never presented as proof/disproof of ultimate metaphysics.",
        ),
        StructuralRule(
            "phase16_advaita_samkhya_nonduality_guard",
            "Non-duality is not represented as shared Advaita/Samkhya ontology.",
        ),
    ),
)


PHASE17_POLICY: Final = PhaseEquivalencePolicy(
    phase="phase17",
    field_rules=(
        _exact("question", "Question continuity."),
        _exact("corpus_version", "Corpus continuity."),
        _exact("coverage_version", "Frozen deterministic coverage version."),
        _exact(
            "coverage_status",
            "Deterministic coverage classification.",
        ),
        _exact(
            "coverage_score",
            "Deterministic coverage score.",
            tolerance=1e-9,
        ),
        _exact(
            "supported_concepts",
            "Supported concept classification.",
        ),
        _exact(
            "partially_supported_concepts",
            "Partially-supported concept classification.",
        ),
        _exact(
            "unsupported_concepts",
            "Unsupported concept classification.",
        ),
        _exact(
            "concept_coverage.*",
            "Per-concept deterministic coverage state.",
            tolerance=1e-9,
        ),
        _exact(
            "response_policy.*",
            "Deterministic answer/fallback policy.",
        ),
        _exact(
            "knowledge_boundary.*",
            "Corpus/general-knowledge boundary.",
        ),
        _exact(
            "manifest.exit_gate.*",
            "Phase 17 deterministic exit-gate booleans.",
        ),
        _exact("manifest.status", "Phase completion status."),
        *COMMON_NON_DETERMINISTIC,
    ),
    structural_rules=(
        StructuralRule(
            "phase17_concept_partition",
            "Every active concept belongs to exactly one coverage partition.",
        ),
        StructuralRule(
            "phase17_score_range",
            "Coverage/component scores stay in valid ranges.",
        ),
        StructuralRule(
            "phase17_hard_overrides",
            "Frozen hard overrides and insufficient-evidence caps hold.",
        ),
        StructuralRule(
            "phase17_unsupported_propagation",
            "Unsupported state propagates into response policy.",
        ),
        StructuralRule(
            "phase17_no_provider_calls",
            "Phase 17 remains deterministic/provider-free.",
        ),
    ),
)


PHASE18_POLICY: Final = PhaseEquivalencePolicy(
    phase="phase18",
    field_rules=(
        _exact("question", "Final question identity."),
        _exact("corpus_version", "Final corpus identity."),
        _exact("response_version", "Frozen response assembly version."),
        _exact(
            "sections.coverage.coverage_status",
            "Final coverage classification.",
        ),
        _exact(
            "sections.coverage.coverage_score",
            "Final deterministic coverage score.",
            tolerance=1e-9,
        ),
        _exact(
            "sections.activated_concepts",
            "Activated concepts survive assembly unchanged.",
        ),
        _exact(
            "sections.domain_perspectives.*.claims[*].claim_id",
            "Final canonical claim identity.",
        ),
        _exact(
            "sections.comparative_synthesis.comparisons[*].comparison_id",
            "Final comparison slot identity.",
        ),
        _exact(
            "claim_level_citations[*].citation_id",
            "Final citation identity.",
        ),
        _exact(
            "claim_level_citations[*].chunk_id",
            "Final citation/evidence linkage.",
        ),
        _exact(
            "sections.general_knowledge_fallback.allowed",
            "Deterministic fallback permission.",
        ),
        _exact(
            "sections.general_knowledge_fallback.generated_in_phase18",
            "Phase 18 must not generate hidden fallback content.",
        ),
        _exact(
            "sections.general_knowledge_fallback.may_use_wth_corpus_citations",
            "Fallback citation boundary.",
        ),
        _exact(
            "validation.passed",
            "Final deterministic validation.",
        ),
        _exact(
            "validation.issues[*].code",
            "Final validation outcomes.",
        ),
        _semantic(
            "sections.domain_perspectives.*.claims[*].text",
            "Final response carries live Phase 15 prose.",
        ),
        _semantic(
            "sections.comparative_synthesis.comparisons[*].explanation",
            "Final response carries live Phase 16 prose.",
        ),
        _semantic(
            "sections.general_knowledge_fallback.instruction",
            "Instruction wording may vary if policy semantics remain unchanged.",
        ),
        _semantic(
            "markdown",
            "Rendered prose may vary with upstream live LLM wording.",
        ),
        *COMMON_NON_DETERMINISTIC,
    ),
    structural_rules=(
        StructuralRule(
            "phase18_required_sections",
            "All required FinalResponse sections are present.",
        ),
        StructuralRule(
            "phase18_claim_citation_validity",
            "Every rendered claim citation resolves to Phase 14 evidence.",
        ),
        StructuralRule(
            "phase18_comparison_coverage",
            "Required Phase 16 comparison slots are represented.",
        ),
        StructuralRule(
            "phase18_unsupported_propagation",
            "Unsupported/partial coverage is reflected in final policy/sections.",
        ),
        StructuralRule(
            "phase18_fallback_shape",
            "Fallback has a complete valid shape whether allowed or forbidden.",
        ),
        StructuralRule(
            "phase18_no_provider_or_retrieval_calls",
            "Phase 18 remains deterministic/provider/retrieval-free.",
        ),
    ),
)


EQUIVALENCE_POLICIES: Final[dict[PhaseName, PhaseEquivalencePolicy]] = {
    "phase14": PHASE14_POLICY,
    "phase15": PHASE15_POLICY,
    "phase16": PHASE16_POLICY,
    "phase17": PHASE17_POLICY,
    "phase18": PHASE18_POLICY,
}


def classify_field(
    phase: PhaseName,
    path: str,
) -> FieldRule | None:
    """Return the first policy rule matching ``path``.

    ``fnmatch`` treats square brackets as character-class syntax, while our
    normalized paths deliberately use the literal marker ``[*]`` for list
    elements. Replace that marker with a sentinel on both sides before
    matching so list-path rules are interpreted literally.
    """

    normalized = normalize_equivalence_path(path).replace(
        "[*]",
        "<LIST>",
    )

    for rule in EQUIVALENCE_POLICIES[phase].field_rules:
        pattern = rule.pattern.replace(
            "[*]",
            "<LIST>",
        )

        if fnmatch.fnmatchcase(
            normalized,
            pattern,
        ):
            return rule

    return None


def structural_rule_ids(
    phase: PhaseName,
) -> tuple[str, ...]:
    return tuple(rule.rule_id for rule in EQUIVALENCE_POLICIES[phase].structural_rules)


def all_phase_names() -> tuple[PhaseName, ...]:
    return (
        "phase14",
        "phase15",
        "phase16",
        "phase17",
        "phase18",
    )


__all__ = [
    "EQUIVALENCE_POLICIES",
    "PHASE14_POLICY",
    "PHASE15_POLICY",
    "PHASE16_POLICY",
    "PHASE17_POLICY",
    "PHASE18_POLICY",
    "EquivalenceClass",
    "FieldRule",
    "PhaseEquivalencePolicy",
    "PhaseName",
    "StructuralRule",
    "all_phase_names",
    "classify_field",
    "normalize_equivalence_path",
    "structural_rule_ids",
]
