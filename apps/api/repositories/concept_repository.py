from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Final

from supabase import Client

from apps.api.services.concept_activation import (
    CONCEPTS,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_NORMALIZATION,
    EMBEDDING_PROVIDER,
    PROTOTYPE_VERSION,
    PrototypeBank,
    build_prototype_bank,
)


class ConceptRepositoryError(
    RuntimeError
):
    """Raised when frozen prototype data cannot be loaded safely."""


TABLE_NAME: Final = (
    "concept_prototype_embeddings"
)

PHASE1_CONCEPT_UUIDS: Final = {
    "self_identity": (
        "10000000-0000-4000-8000-000000000001"
    ),
    "consciousness": (
        "10000000-0000-4000-8000-000000000002"
    ),
    "reality_appearance": (
        "10000000-0000-4000-8000-000000000003"
    ),
}

EXPECTED_TOTAL_COUNT: Final = 63

EXPECTED_ROLE_COUNTS: Final = {
    "question": 18,
    "positive": 27,
    "hard_negative": 18,
}

EXPECTED_PER_CONCEPT_ROLE_COUNTS: Final = {
    "question": 6,
    "positive": 9,
    "hard_negative": 6,
}

QUERY_SOURCE_ARTIFACT_SHA256: Final = (
    "d4dd9c89525332bfbc492d5464e13b83432fe14096fb"
    "15d22e4247fb72ef05b4"
)

PASSAGE_SOURCE_ARTIFACT_SHA256: Final = (
    "5081c2771b5a7f173763194fad904e660b46ef337d460"
    "805ff7ae066fe6b6d6b"
)

EXPECTED_EMBEDDING_ORIGIN: Final = (
    "provider"
)


class ConceptRepository:
    """Read the immutable production prototype bank from Supabase."""

    def __init__(
        self,
        client: Client,
    ) -> None:
        self._client = client
        self._cached_bank: (
            PrototypeBank | None
        ) = None

    def get_prototype_bank(
        self,
        *,
        refresh: bool = False,
    ) -> PrototypeBank:
        """Return the frozen Phase 1 PrototypeBank.

        The first successful read validates the complete 63-row production
        representation. Subsequent calls return the same immutable in-process
        bank unless ``refresh=True`` is explicitly requested.
        """

        if (
            self._cached_bank is not None
            and not refresh
        ):
            return self._cached_bank

        concept_ids = (
            self._resolve_concept_ids()
        )

        rows = self._fetch_rows()

        bank = _build_bank_from_rows(
            rows=rows,
            concept_ids=concept_ids,
        )

        self._cached_bank = bank

        return bank

    def clear_cache(
        self,
    ) -> None:
        """Discard the in-process immutable-bank cache."""

        self._cached_bank = None

    def _resolve_concept_ids(
        self,
    ) -> dict[str, str]:
        try:
            response = (
                self._client.table(
                    "concepts"
                )
                .select(
                    "id,slug,is_active"
                )
                .execute()
            )
        except Exception as exc:
            raise ConceptRepositoryError(
                "Could not resolve the frozen Phase 1 concepts."
            ) from exc

        rows = _response_rows(
            response,
            "concept lookup",
        )

        by_slug: dict[
            str,
            Mapping[str, object],
        ] = {}

        for row in rows:
            slug = _required_string(
                row,
                "slug",
            )

            if slug in CONCEPTS:
                by_slug[
                    slug
                ] = row

        result: dict[
            str,
            str,
        ] = {}

        for concept in CONCEPTS:
            concept_row = by_slug.get(concept)

            if concept_row is None:
                raise ConceptRepositoryError(
                    f"Required Phase 1 concept {concept!r} "
                    "is missing."
                )

            if concept_row.get("is_active") is not True:
                raise ConceptRepositoryError(
                    f"Required Phase 1 concept {concept!r} "
                    "is inactive."
                )

            concept_id = _required_string(
                concept_row,
                "id",
            )

            expected_id = PHASE1_CONCEPT_UUIDS[concept]

            if concept_id != expected_id:
                raise ConceptRepositoryError(
                    f"Canonical UUID changed for concept "
                    f"{concept!r}."
                )

            result[concept] = concept_id
        return result

    def _fetch_rows(
        self,
    ) -> list[dict[str, object]]:
        columns = (
            "record_id,concept_id,prototype_version,"
            "prototype_role,record_type,embedding,"
            "provider,model,model_revision,dimensions,"
            "normalization,task_type,embedding_origin,"
            "embedding_checksum,text_checksum,"
            "source_artifact_sha256"
        )

        try:
            response = (
                self._client.table(
                    TABLE_NAME
                )
                .select(
                    columns
                )
                .eq(
                    "prototype_version",
                    PROTOTYPE_VERSION,
                )
                .execute()
            )
        except Exception as exc:
            raise ConceptRepositoryError(
                "Could not load the frozen concept prototype bank."
            ) from exc

        return _response_rows(
            response,
            "prototype bank lookup",
        )


def _build_bank_from_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    concept_ids: Mapping[str, str],
) -> PrototypeBank:
    if (
        len(rows)
        != EXPECTED_TOTAL_COUNT
    ):
        raise ConceptRepositoryError(
            "Frozen prototype row count mismatch: "
            f"expected {EXPECTED_TOTAL_COUNT}, "
            f"received {len(rows)}."
        )

    slug_by_id = {
        concept_id: slug
        for slug, concept_id
        in concept_ids.items()
    }

    question: dict[
        str,
        list[list[float]],
    ] = {
        concept: []
        for concept in CONCEPTS
    }

    passage: dict[
        str,
        list[list[float]],
    ] = {
        concept: []
        for concept in CONCEPTS
    }

    hard_negative: dict[
        str,
        list[list[float]],
    ] = {
        concept: []
        for concept in CONCEPTS
    }

    seen_record_ids: set[
        str
    ] = set()

    role_counts: Counter[
        str
    ] = Counter()

    concept_role_counts: Counter[
        tuple[str, str]
    ] = Counter()

    for index, raw in enumerate(
        rows,
        start=1,
    ):
        row = _mapping(
            raw,
            (
                "prototype row "
                f"{index}"
            ),
        )

        record_id = _required_string(
            row,
            "record_id",
        )

        if (
            record_id
            in seen_record_ids
        ):
            raise ConceptRepositoryError(
                f"Duplicate prototype row {record_id!r}."
            )

        seen_record_ids.add(
            record_id
        )

        prototype_version = (
            _required_string(
                row,
                "prototype_version",
            )
        )

        if (
            prototype_version
            != PROTOTYPE_VERSION
        ):
            raise ConceptRepositoryError(
                f"Prototype version changed for {record_id!r}."
            )

        concept_id = _required_string(
            row,
            "concept_id",
        )

        concept = slug_by_id.get(
            concept_id
        )

        if concept is None:
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} references "
                "an unknown Phase 1 concept UUID."
            )

        role = _required_string(
            row,
            "prototype_role",
        )

        record_type = (
            _required_string(
                row,
                "record_type",
            )
        )

        task_type = _required_string(
            row,
            "task_type",
        )

        if role == "question":
            if (
                record_type
                != "query_prototype"
                or task_type
                != "search_query"
            ):
                raise ConceptRepositoryError(
                    f"Prototype {record_id!r} violates "
                    "the frozen question role/type contract."
                )

            expected_artifact_hash = (
                QUERY_SOURCE_ARTIFACT_SHA256
            )

        elif role in {
            "positive",
            "hard_negative",
        }:
            if (
                record_type
                != "passage_prototype"
                or task_type
                != "search_document"
            ):
                raise ConceptRepositoryError(
                    f"Prototype {record_id!r} violates "
                    "the frozen passage role/type contract."
                )

            expected_artifact_hash = (
                PASSAGE_SOURCE_ARTIFACT_SHA256
            )

        else:
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} has "
                f"unknown role {role!r}."
            )

        _validate_embedding_identity(
            row,
            record_id=record_id,
        )

        source_hash = (
            _required_sha256(
                row,
                "source_artifact_sha256",
            )
        )

        if (
            source_hash
            != expected_artifact_hash
        ):
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} source "
                "artifact fingerprint changed."
            )

        # Both checksums are required runtime provenance even though activation
        # only consumes the vector.
        _required_sha256(
            row,
            "embedding_checksum",
        )
        _required_sha256(
            row,
            "text_checksum",
        )

        vector = _parse_vector(
            row.get(
                "embedding"
            ),
            record_id=record_id,
        )

        if role == "question":
            question[
                concept
            ].append(
                vector
            )

        elif role == "positive":
            passage[
                concept
            ].append(
                vector
            )

        else:
            hard_negative[
                concept
            ].append(
                vector
            )

        role_counts[
            role
        ] += 1

        concept_role_counts[
            (
                concept,
                role,
            )
        ] += 1

    if (
        dict(role_counts)
        != EXPECTED_ROLE_COUNTS
    ):
        raise ConceptRepositoryError(
            "Frozen prototype role distribution changed: "
            f"{dict(role_counts)}."
        )

    for concept in CONCEPTS:
        for role, expected in (
            EXPECTED_PER_CONCEPT_ROLE_COUNTS.items()
        ):
            observed = (
                concept_role_counts[
                    (
                        concept,
                        role,
                    )
                ]
            )

            if observed != expected:
                raise ConceptRepositoryError(
                    "Frozen prototype concept/role "
                    "distribution changed: "
                    f"{concept}/{role} "
                    f"expected {expected}, "
                    f"received {observed}."
                )

    # build_prototype_bank performs the same L2 normalization that historical
    # Phase 10 performed when loading the frozen JSONL records.
    return build_prototype_bank(
        question=question,
        passage=passage,
        hard_negative=hard_negative,
    )


def _validate_embedding_identity(
    row: Mapping[str, object],
    *,
    record_id: str,
) -> None:
    expected: dict[
        str,
        object,
    ] = {
        "provider": (
            EMBEDDING_PROVIDER
        ),
        "model": (
            EMBEDDING_MODEL
        ),
        "model_revision": (
            EMBEDDING_MODEL_REVISION
        ),
        "dimensions": (
            EMBEDDING_DIMENSIONS
        ),
        "normalization": (
            EMBEDDING_NORMALIZATION
        ),
        "embedding_origin": (
            EXPECTED_EMBEDDING_ORIGIN
        ),
    }

    for key, expected_value in (
        expected.items()
    ):
        observed = row.get(
            key
        )

        if observed != expected_value:
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} embedding "
                f"identity changed at {key!r}."
            )


def _parse_vector(
    value: object,
    *,
    record_id: str,
) -> list[float]:
    raw = value

    if isinstance(
        raw,
        str,
    ):
        try:
            raw = json.loads(
                raw
            )
        except json.JSONDecodeError as exc:
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} vector "
                "is not valid JSON."
            ) from exc

    if (
        not isinstance(
            raw,
            Sequence,
        )
        or isinstance(
            raw,
            (str, bytes, bytearray),
        )
    ):
        raise ConceptRepositoryError(
            f"Prototype {record_id!r} vector "
            "must be a sequence."
        )

    if (
        len(raw)
        != EMBEDDING_DIMENSIONS
    ):
        raise ConceptRepositoryError(
            f"Prototype {record_id!r} vector has "
            f"{len(raw)} dimensions; expected "
            f"{EMBEDDING_DIMENSIONS}."
        )

    result: list[
        float
    ] = []

    for index, raw_value in enumerate(
        raw
    ):
        if isinstance(
            raw_value,
            bool,
        ):
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} vector "
                f"contains a boolean at index {index}."
            )

        try:
            number = float(
                raw_value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} vector "
                f"contains a non-numeric value at "
                f"index {index}."
            ) from exc

        if not math.isfinite(
            number
        ):
            raise ConceptRepositoryError(
                f"Prototype {record_id!r} vector "
                f"contains a non-finite value at "
                f"index {index}."
            )

        result.append(
            number
        )

    # Do not normalize here. build_prototype_bank() is the single normalization
    # boundary and mirrors the historical Phase 10 vector loader.
    return result


def _response_rows(
    response: object,
    description: str,
) -> list[dict[str, object]]:
    data = getattr(
        response,
        "data",
        None,
    )

    if data is None:
        raise ConceptRepositoryError(
            f"{description} returned no data."
        )

    if isinstance(
        data,
        Mapping,
    ):
        return [
            _mapping(
                data,
                description,
            )
        ]

    if (
        not isinstance(
            data,
            Sequence,
        )
        or isinstance(
            data,
            (str, bytes, bytearray),
        )
    ):
        raise ConceptRepositoryError(
            f"{description} returned an invalid payload."
        )

    return [
        _mapping(
            row,
            description,
        )
        for row in data
    ]


def _mapping(
    value: object,
    description: str,
) -> dict[str, object]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise ConceptRepositoryError(
            f"{description} must be an object."
        )

    result: dict[
        str,
        object,
    ] = {}

    for key, nested in value.items():
        if not isinstance(
            key,
            str,
        ):
            raise ConceptRepositoryError(
                f"{description} contains a non-string key."
            )

        result[
            key
        ] = nested

    return result


def _required_string(
    row: Mapping[str, object],
    field: str,
) -> str:
    value = row.get(
        field
    )

    if (
        not isinstance(
            value,
            str,
        )
        or not value.strip()
    ):
        raise ConceptRepositoryError(
            f"Prototype DB field {field!r} "
            "is missing or invalid."
        )

    return value.strip()


def _required_sha256(
    row: Mapping[str, object],
    field: str,
) -> str:
    value = _required_string(
        row,
        field,
    ).lower()

    if (
        len(value) != 64
        or any(
            character
            not in "0123456789abcdef"
            for character
            in value
        )
    ):
        raise ConceptRepositoryError(
            f"Prototype DB field {field!r} "
            "is not a SHA-256 digest."
        )

    return value


__all__ = [
    "EXPECTED_PER_CONCEPT_ROLE_COUNTS",
    "EXPECTED_ROLE_COUNTS",
    "EXPECTED_TOTAL_COUNT",
    "PASSAGE_SOURCE_ARTIFACT_SHA256",
    "QUERY_SOURCE_ARTIFACT_SHA256",
    "TABLE_NAME",
    "ConceptRepository",
    "ConceptRepositoryError",
]
