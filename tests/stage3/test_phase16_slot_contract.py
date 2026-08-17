from __future__ import annotations

import json
from typing import Any

import pytest

from apps.api.services.synthesis import (
    CATEGORY_CODES,
    DOMAIN_PAIRS,
    ComparisonSlot,
    SynthesisError,
    parse_synthesis_draft,
    response_shape_description,
    synthesis_schema,
    synthesis_user_prompt,
)


def _slots() -> tuple[ComparisonSlot, ...]:
    concepts = (
        "self_identity",
        "reality_appearance",
        "consciousness",
    )
    slots: list[ComparisonSlot] = []
    slot_number = 1

    for concept in concepts:
        for left_domain, right_domain in DOMAIN_PAIRS:
            slots.append(
                ComparisonSlot(
                    slot=slot_number,
                    comparison_id=(
                        f"{concept}__{left_domain}__{right_domain}"
                    ),
                    concept=concept,
                    left_domain=left_domain,
                    right_domain=right_domain,
                    left_claim_refs=(f"{left_domain}:claim_1",),
                    right_claim_refs=(f"{right_domain}:claim_1",),
                    required_unsupported_refs=(),
                )
            )
            slot_number += 1

    return tuple(slots)


def _packet(slots: tuple[ComparisonSlot, ...]) -> dict[str, object]:
    return {
        "q": "What is the feeling of self or personal identity?",
        "v": "phase1_active_corpus_v1",
        "claims": {},
        "limits": {},
        "slots": {
            str(slot.slot): {
                "x": slot.concept,
                "l": slot.left_domain,
                "r": slot.right_domain,
                "lc": list(slot.left_claim_refs),
                "rc": list(slot.right_claim_refs),
                "u": [],
                "ic_allowed": False,
            }
            for slot in slots
        },
    }


def test_three_active_concepts_define_exact_nine_slots() -> None:
    slots = _slots()

    assert len(slots) == 9
    assert [slot.comparison_id for slot in slots] == [
        "self_identity__science__advaita",
        "self_identity__science__samkhya",
        "self_identity__advaita__samkhya",
        "reality_appearance__science__advaita",
        "reality_appearance__science__samkhya",
        "reality_appearance__advaita__samkhya",
        "consciousness__science__advaita",
        "consciousness__science__samkhya",
        "consciousness__advaita__samkhya",
    ]


def test_response_shape_uses_python_owned_keyed_results() -> None:
    slots = _slots()
    shape = response_shape_description(slots)

    assert set(shape) == {"results"}
    results = shape["results"]
    assert isinstance(results, dict)
    assert list(results) == [str(i) for i in range(1, 10)]
    assert "i" not in results["1"]


def test_strict_schema_requires_every_exact_result_key() -> None:
    slots = _slots()
    schema: dict[str, Any] = synthesis_schema(slots)

    assert schema["required"] == ["results"]
    results_schema = schema["properties"]["results"]
    assert results_schema["additionalProperties"] is False
    assert results_schema["required"] == [str(i) for i in range(1, 10)]
    assert set(results_schema["properties"]) == {
        str(i) for i in range(1, 10)
    }

    for item_schema in results_schema["properties"].values():
        assert item_schema["required"] == ["c", "e"]
        assert item_schema["additionalProperties"] is False
        assert item_schema["properties"]["c"]["enum"] == list(
            CATEGORY_CODES.keys()
        )

def test_prompt_explicitly_requires_all_keys_and_forbids_slots_array() -> None:
    slots = _slots()
    prompt = synthesis_user_prompt(
        packet=_packet(slots),
        slots=slots,
    )
    parsed = json.loads(prompt)
    rules = " ".join(parsed["rules"])

    assert "exactly these required keys: 1, 2, 3, 4, 5, 6, 7, 8, 9" in rules
    assert "Fill every key listed in input.slots exactly once" in rules
    assert "Do not return a 'slots' array" in rules
    assert "all supplied comparison slots are required" in rules


def test_complete_keyed_results_parse_in_python_slot_order() -> None:
    slots = _slots()
    raw = {
        "results": {
            str(slot.slot): {
                "c": "fa",
                "e": f"Grounded comparison for slot {slot.slot}.",
            }
            for slot in slots
        }
    }

    draft = parse_synthesis_draft(
        raw,
        slots=slots,
    )

    assert len(draft.comparisons) == 9
    assert [item.comparison_id for item in draft.comparisons] == [
        slot.comparison_id for slot in slots
    ]


def test_missing_key_is_rejected_not_padded() -> None:
    slots = _slots()
    raw_results = {
        str(slot.slot): {
            "c": "fa",
            "e": f"Grounded comparison for slot {slot.slot}.",
        }
        for slot in slots
    }
    del raw_results["9"]

    with pytest.raises(
        SynthesisError,
        match="incomplete result matrix",
    ):
        parse_synthesis_draft(
            {"results": raw_results},
            slots=slots,
        )


def test_unexpected_key_is_rejected() -> None:
    slots = _slots()
    raw_results = {
        str(slot.slot): {
            "c": "fa",
            "e": f"Grounded comparison for slot {slot.slot}.",
        }
        for slot in slots
    }
    raw_results["10"] = {
        "c": "fa",
        "e": "Unexpected slot.",
    }

    with pytest.raises(
        SynthesisError,
        match="unexpected",
    ):
        parse_synthesis_draft(
            {"results": raw_results},
            slots=slots,
        )


def test_ic_still_requires_python_supplied_limitation() -> None:
    slots = _slots()
    raw_results = {
        str(slot.slot): {
            "c": "fa",
            "e": f"Grounded comparison for slot {slot.slot}.",
        }
        for slot in slots
    }
    raw_results["1"] = {
        "c": "ic",
        "e": "Evidence is insufficient.",
    }

    with pytest.raises(
        SynthesisError,
        match="ic_allowed=false",
    ):
        parse_synthesis_draft(
            {"results": raw_results},
            slots=slots,
        )
