"""Reclassify the broad Phase 1 corpus as candidate-only.

This script verifies that the frozen candidate-corpus snapshot and the
current working chunk artifacts describe the same corpus. It then writes
an explicit lifecycle-classification manifest.

The script does not modify chunks, embeddings, review records, or
database state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from apps.api.models.corpus import ChunkDraft

LOGGER = logging.getLogger("wth.phase1.reclassify_candidate_corpus")

DEFAULT_FROZEN_ROOT: Final = Path("artifacts/archive/phase1_candidate_corpus_v1")
DEFAULT_FROZEN_CHUNKS_ROOT: Final = DEFAULT_FROZEN_ROOT / "artifacts" / "phase1" / "chunks"
DEFAULT_WORKING_CHUNKS_ROOT: Final = Path("artifacts/phase1/chunks")
DEFAULT_FREEZE_MANIFEST: Final = DEFAULT_FROZEN_ROOT / "freeze_manifest.json"
DEFAULT_OUTPUT_ROOT: Final = Path("artifacts/phase1/candidate")
DEFAULT_EXPECTED_CHUNK_COUNT: Final = 7_469
DEFAULT_CORPUS_VERSION: Final = "phase1_candidate_corpus_v1"

CLASSIFICATION_MANIFEST_NAME: Final = "candidate_corpus_manifest.json"
STATUS_DOCUMENT_NAME: Final = "STATUS.md"
BUFFER_SIZE: Final = 1024 * 1024


class ReclassificationError(RuntimeError):
    """Raised when the corpus cannot be safely reclassified."""


@dataclass(frozen=True, slots=True)
class ChunkIdentity:
    """Stable identity fields used to compare corpus copies."""

    chunk_id: str
    source_id: str
    domain: str
    text_checksum: str


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    """Summary of one chunk-artifact directory."""

    root: str
    artifact_file_count: int
    chunk_count: int
    source_count: int
    total_token_count: int
    counts_by_domain: dict[str, int]
    counts_by_source: dict[str, int]


@dataclass(frozen=True, slots=True)
class CorpusLifecycle:
    """Lifecycle classification for the candidate corpus."""

    corpus_classification: str
    review_status: str
    phase1_activation_status: str
    production_retrieval_status: str
    evaluation_gold_status: str
    direct_activation_eligible: bool
    database_activation_permitted: bool


@dataclass(frozen=True, slots=True)
class CandidateCorpusManifest:
    """Machine-readable candidate-corpus classification."""

    manifest_version: str
    corpus_version: str
    created_at: str

    lifecycle: CorpusLifecycle

    expected_chunk_count: int
    verified_chunk_count: int
    frozen_and_working_copies_match: bool

    frozen_snapshot_root: str
    frozen_chunks_root: str
    working_chunks_root: str

    freeze_manifest_path: str
    freeze_manifest_sha256: str

    inventory: CorpusInventory

    intended_uses: tuple[str, ...]
    prohibited_uses: tuple[str, ...]
    activation_block_reasons: tuple[str, ...]
    next_required_steps: tuple[str, ...]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Verify and reclassify the broad Phase 1 corpus as candidate-only.")
    )

    parser.add_argument(
        "--frozen-root",
        type=Path,
        default=DEFAULT_FROZEN_ROOT,
    )
    parser.add_argument(
        "--frozen-chunks-root",
        type=Path,
        default=DEFAULT_FROZEN_CHUNKS_ROOT,
    )
    parser.add_argument(
        "--working-chunks-root",
        type=Path,
        default=DEFAULT_WORKING_CHUNKS_ROOT,
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=DEFAULT_FREEZE_MANIFEST,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--corpus-version",
        default=DEFAULT_CORPUS_VERSION,
    )
    parser.add_argument(
        "--expected-chunk-count",
        type=int,
        default=DEFAULT_EXPECTED_CHUNK_COUNT,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help=(
            "Replace existing classification outputs. "
            "Use only when intentionally regenerating the "
            "same corpus-version record."
        ),
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


def main() -> None:
    """Create the candidate-corpus classification record."""

    args = parse_arguments()
    configure_logging(args.log_level)

    try:
        manifest = reclassify_candidate_corpus(
            frozen_root=args.frozen_root,
            frozen_chunks_root=args.frozen_chunks_root,
            working_chunks_root=args.working_chunks_root,
            freeze_manifest_path=args.freeze_manifest,
            output_root=args.output_root,
            corpus_version=args.corpus_version,
            expected_chunk_count=args.expected_chunk_count,
            replace=args.replace,
        )
    except Exception:
        LOGGER.exception("Candidate-corpus reclassification failed")
        raise SystemExit(1) from None

    LOGGER.info("Candidate corpus reclassified successfully")
    LOGGER.info(
        "Verified chunks: %d",
        manifest.verified_chunk_count,
    )
    LOGGER.info(
        "Review status: %s",
        manifest.lifecycle.review_status,
    )
    LOGGER.info(
        "Activation status: %s",
        manifest.lifecycle.phase1_activation_status,
    )
    LOGGER.info(
        "Direct activation eligible: %s",
        manifest.lifecycle.direct_activation_eligible,
    )


def reclassify_candidate_corpus(
    *,
    frozen_root: Path,
    frozen_chunks_root: Path,
    working_chunks_root: Path,
    freeze_manifest_path: Path,
    output_root: Path,
    corpus_version: str,
    expected_chunk_count: int,
    replace: bool,
) -> CandidateCorpusManifest:
    """Verify corpus copies and write lifecycle classification."""

    if expected_chunk_count < 1:
        raise ReclassificationError("expected_chunk_count must be at least 1")

    frozen_root = frozen_root.resolve()
    frozen_chunks_root = frozen_chunks_root.resolve()
    working_chunks_root = working_chunks_root.resolve()
    freeze_manifest_path = freeze_manifest_path.resolve()
    output_root = output_root.resolve()

    validate_required_path(
        frozen_root,
        expected_kind="directory",
    )
    validate_required_path(
        frozen_chunks_root,
        expected_kind="directory",
    )
    validate_required_path(
        working_chunks_root,
        expected_kind="directory",
    )
    validate_required_path(
        freeze_manifest_path,
        expected_kind="file",
    )

    output_manifest = output_root / CLASSIFICATION_MANIFEST_NAME
    output_status = output_root / STATUS_DOCUMENT_NAME

    validate_output_paths(
        output_manifest=output_manifest,
        output_status=output_status,
        replace=replace,
    )

    freeze_manifest = load_freeze_manifest(freeze_manifest_path)

    validate_freeze_manifest(
        freeze_manifest=freeze_manifest,
        corpus_version=corpus_version,
    )

    LOGGER.info(
        "Loading frozen chunk inventory: %s",
        frozen_chunks_root,
    )
    frozen_inventory, frozen_identities = load_chunk_inventory(frozen_chunks_root)

    LOGGER.info(
        "Loading working chunk inventory: %s",
        working_chunks_root,
    )
    working_inventory, working_identities = load_chunk_inventory(working_chunks_root)

    validate_expected_count(
        inventory=frozen_inventory,
        expected_chunk_count=expected_chunk_count,
        description="frozen snapshot",
    )
    validate_expected_count(
        inventory=working_inventory,
        expected_chunk_count=expected_chunk_count,
        description="working corpus",
    )

    compare_corpus_copies(
        frozen=frozen_identities,
        working=working_identities,
    )

    compare_inventory_summaries(
        frozen=frozen_inventory,
        working=working_inventory,
    )

    lifecycle = CorpusLifecycle(
        corpus_classification="candidate_corpus",
        review_status="not_reviewed",
        phase1_activation_status="not_active",
        production_retrieval_status="not_eligible",
        evaluation_gold_status="not_gold_labeled",
        direct_activation_eligible=False,
        database_activation_permitted=False,
    )

    manifest = CandidateCorpusManifest(
        manifest_version="1.0",
        corpus_version=corpus_version,
        created_at=datetime.now(UTC).isoformat(),
        lifecycle=lifecycle,
        expected_chunk_count=expected_chunk_count,
        verified_chunk_count=(frozen_inventory.chunk_count),
        frozen_and_working_copies_match=True,
        frozen_snapshot_root=str(frozen_root),
        frozen_chunks_root=str(frozen_chunks_root),
        working_chunks_root=str(working_chunks_root),
        freeze_manifest_path=str(freeze_manifest_path),
        freeze_manifest_sha256=sha256_file(freeze_manifest_path),
        inventory=frozen_inventory,
        intended_uses=(
            ("Source-section scope analysis for the three-concept Phase 1 vertical slice."),
            (
                "Rule-based selection of consciousness, "
                "self_identity, and reality_appearance "
                "candidate passages."
            ),
            ("Selection of adjacent-concept hard negatives."),
            ("Future expansion to the remaining five canonical concept families."),
            ("Reproducible parser, chunker, and corpus quality analysis."),
        ),
        prohibited_uses=(
            ("Direct activation into the Phase 1 production retrieval corpus."),
            ("Bulk approval without passage-level human review."),
            ("Use as the held-out gold evaluation dataset."),
            ("Use for final claim-cited generation before Phase 1 relevance review."),
            ("Automatic selection based solely on the embedding or anchor model being evaluated."),
        ),
        activation_block_reasons=(
            (
                "The corpus contains broad source-derived "
                "material outside consciousness, "
                "self_identity, and reality_appearance."
            ),
            ("The corpus has not received passage-level human review."),
            ("The corpus is materially imbalanced across Science, Advaita Vedanta, and Samkhya."),
            ("Concept relevance and hard-negative labels have not been frozen."),
            ("Development and held-out evaluation splits have not been created."),
        ),
        next_required_steps=(
            ("Create the source-structure report."),
            ("Create and approve source-section scope metadata."),
            ("Run independent rule-based Phase 1 candidate selection."),
            ("Perform human review and produce a balanced 250-350 chunk vertical slice."),
            ("Freeze build, development, and held-out evaluation sets."),
            ("Generate approved-corpus embeddings and concept weights."),
            ("Activate only the reviewed Phase 1 slice."),
        ),
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_write_text(
        output_manifest,
        json.dumps(
            asdict(manifest),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )

    atomic_write_text(
        output_status,
        render_status_document(manifest),
    )

    return manifest


def load_chunk_inventory(
    root: Path,
) -> tuple[
    CorpusInventory,
    dict[str, ChunkIdentity],
]:
    """Load chunks and create an inventory and identity map."""

    artifact_paths = tuple(sorted(root.glob("*.json")))

    if not artifact_paths:
        raise ReclassificationError(f"No chunk JSON artifacts found in {root}")

    identities: dict[
        str,
        ChunkIdentity,
    ] = {}

    domain_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total_token_count = 0

    for artifact_path in artifact_paths:
        try:
            raw = json.loads(artifact_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReclassificationError(f"Invalid chunk JSON artifact: {artifact_path}") from exc

        if not isinstance(raw, list):
            raise ReclassificationError(f"Chunk artifact must contain a JSON list: {artifact_path}")

        for item_number, item in enumerate(
            raw,
            start=1,
        ):
            try:
                chunk = ChunkDraft.model_validate(item)
            except Exception as exc:
                raise ReclassificationError(
                    f"Invalid chunk record at {artifact_path}:{item_number}: {exc}"
                ) from exc

            if chunk.chunk_id in identities:
                raise ReclassificationError(f"Duplicate chunk ID detected: {chunk.chunk_id}")

            identity = ChunkIdentity(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                domain=chunk.domain.value,
                text_checksum=chunk.text_checksum,
            )

            identities[chunk.chunk_id] = identity
            domain_counts[chunk.domain.value] += 1
            source_counts[chunk.source_id] += 1
            total_token_count += chunk.token_count

    inventory = CorpusInventory(
        root=str(root),
        artifact_file_count=len(artifact_paths),
        chunk_count=len(identities),
        source_count=len(source_counts),
        total_token_count=total_token_count,
        counts_by_domain=dict(sorted(domain_counts.items())),
        counts_by_source=dict(sorted(source_counts.items())),
    )

    return inventory, identities


def compare_corpus_copies(
    *,
    frozen: Mapping[str, ChunkIdentity],
    working: Mapping[str, ChunkIdentity],
) -> None:
    """Require frozen and working chunk identities to match."""

    frozen_ids = set(frozen)
    working_ids = set(working)

    missing_from_working = frozen_ids - working_ids
    extra_in_working = working_ids - frozen_ids

    if missing_from_working:
        sample = sorted(missing_from_working)[:10]

        raise ReclassificationError(f"Working corpus is missing frozen chunk IDs. Sample: {sample}")

    if extra_in_working:
        sample = sorted(extra_in_working)[:10]

        raise ReclassificationError(
            f"Working corpus contains chunk IDs absent from the frozen snapshot. Sample: {sample}"
        )

    mismatches: list[str] = []

    for chunk_id in sorted(frozen_ids):
        frozen_identity = frozen[chunk_id]
        working_identity = working[chunk_id]

        if frozen_identity != working_identity:
            mismatches.append(chunk_id)

            if len(mismatches) >= 10:
                break

    if mismatches:
        raise ReclassificationError(
            f"Frozen and working chunk records differ for these chunk IDs: {mismatches}"
        )


def compare_inventory_summaries(
    *,
    frozen: CorpusInventory,
    working: CorpusInventory,
) -> None:
    """Require frozen and working corpus summaries to match."""

    comparable_frozen = (
        frozen.artifact_file_count,
        frozen.chunk_count,
        frozen.source_count,
        frozen.total_token_count,
        frozen.counts_by_domain,
        frozen.counts_by_source,
    )

    comparable_working = (
        working.artifact_file_count,
        working.chunk_count,
        working.source_count,
        working.total_token_count,
        working.counts_by_domain,
        working.counts_by_source,
    )

    if comparable_frozen != comparable_working:
        raise ReclassificationError("Frozen and working corpus inventory summaries do not match")


def validate_expected_count(
    *,
    inventory: CorpusInventory,
    expected_chunk_count: int,
    description: str,
) -> None:
    """Require the verified chunk count to match expectation."""

    if inventory.chunk_count != expected_chunk_count:
        raise ReclassificationError(
            f"{description} contains "
            f"{inventory.chunk_count} chunks; expected "
            f"{expected_chunk_count}"
        )


def load_freeze_manifest(
    path: Path,
) -> dict[str, object]:
    """Load the candidate-corpus freeze manifest."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReclassificationError(f"Invalid freeze manifest JSON: {path}") from exc

    if not isinstance(raw, dict):
        raise ReclassificationError("Freeze manifest must contain a JSON object")

    return {str(key): value for key, value in raw.items()}


def validate_freeze_manifest(
    *,
    freeze_manifest: Mapping[str, object],
    corpus_version: str,
) -> None:
    """Validate the snapshot classification and version."""

    status = freeze_manifest.get("corpus_status")
    version = freeze_manifest.get("corpus_version")

    if status != "candidate_only_frozen":
        raise ReclassificationError(
            "Freeze manifest does not identify a candidate-only frozen corpus"
        )

    if version != corpus_version:
        raise ReclassificationError(
            "Freeze-manifest corpus version does not match "
            f"the requested version: {version!r} != "
            f"{corpus_version!r}"
        )


def validate_required_path(
    path: Path,
    *,
    expected_kind: str,
) -> None:
    """Validate a required file or directory."""

    if not path.exists():
        raise ReclassificationError(f"Required path does not exist: {path}")

    if expected_kind == "file" and not path.is_file():
        raise ReclassificationError(f"Required path is not a file: {path}")

    if expected_kind == "directory" and not path.is_dir():
        raise ReclassificationError(f"Required path is not a directory: {path}")


def validate_output_paths(
    *,
    output_manifest: Path,
    output_status: Path,
    replace: bool,
) -> None:
    """Protect existing classification outputs."""

    existing = tuple(
        path
        for path in (
            output_manifest,
            output_status,
        )
        if path.exists()
    )

    if existing and not replace:
        raise ReclassificationError(
            "Classification output already exists. "
            "Use --replace only for intentional "
            "regeneration: " + ", ".join(str(path) for path in existing)
        )


def render_status_document(
    manifest: CandidateCorpusManifest,
) -> str:
    """Render the human-readable candidate-corpus status."""

    domain_lines = "\n".join(
        f"- **{domain}:** {count:,} chunks"
        for domain, count in (manifest.inventory.counts_by_domain.items())
    )

    source_lines = "\n".join(
        f"- `{source_id}`: {count:,} chunks"
        for source_id, count in (manifest.inventory.counts_by_source.items())
    )

    intended_uses = "\n".join(f"- {value}" for value in manifest.intended_uses)

    prohibited_uses = "\n".join(f"- {value}" for value in manifest.prohibited_uses)

    block_reasons = "\n".join(f"- {value}" for value in manifest.activation_block_reasons)

    next_steps = "\n".join(
        f"{index}. {value}"
        for index, value in enumerate(
            manifest.next_required_steps,
            start=1,
        )
    )

    return f"""# Phase 1 Candidate Corpus Status

## Classification

- **Corpus version:** `{manifest.corpus_version}`
- **Classification:** Candidate corpus
- **Review status:** Not reviewed
- **Phase 1 activation status:** Not active
- **Production retrieval status:** Not eligible
- **Evaluation gold status:** Not gold-labelled
- **Direct activation eligible:** No
- **Database activation permitted:** No
- **Verified chunk count:** {manifest.verified_chunk_count:,}
- **Frozen and working copies match:** Yes

## Authoritative status statement

The {manifest.verified_chunk_count:,} chunks in this corpus are broad,
source-derived candidate material.

They have not received passage-level Phase 1 relevance review and are
not approved as the active consciousness, self/identity, and
reality/appearance corpus.

These chunks must not be activated directly or used as production
retrieval evidence.

## Distribution by domain

{domain_lines}

## Distribution by source

{source_lines}

## Intended uses

{intended_uses}

## Prohibited uses

{prohibited_uses}

## Why direct activation is blocked

{block_reasons}

## Required next steps

{next_steps}

## Provenance

- Frozen snapshot: `{manifest.frozen_snapshot_root}`
- Frozen chunks: `{manifest.frozen_chunks_root}`
- Working chunks: `{manifest.working_chunks_root}`
- Freeze manifest: `{manifest.freeze_manifest_path}`
- Freeze-manifest SHA-256:
  `{manifest.freeze_manifest_sha256}`
- Classification created:
  `{manifest.created_at}`
"""


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 checksum of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(BUFFER_SIZE):
            digest.update(block)

    return digest.hexdigest()


def atomic_write_text(
    path: Path,
    content: str,
) -> None:
    """Write UTF-8 text using atomic replacement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(f".{path.name}.tmp")

    temporary.write_text(
        content,
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def configure_logging(level: str) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(
            logging,
            level,
        ),
        format=("%(asctime)s %(levelname)s %(name)s %(message)s"),
    )


if __name__ == "__main__":
    main()
