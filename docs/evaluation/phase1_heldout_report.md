# Phase 1 Held-out Concept Mapping Evaluation

- Evaluation version: `phase1-heldout-evaluation-v1`
- Generated: `2026-08-09T12:10:59.928776+00:00`
- Status: **HELD-OUT EVALUATION RECORDED**
- Held-out records: **79**
- Post-hoc threshold changes: **NONE**
- Post-hoc calibration changes: **NONE**
- Selected method changed after Held-out: **NO**

## Evaluation rule

The Phase 10 hybrid method, concept-specific thresholds, calibration parameters, ambiguity margin, prototype aggregation, lexical weight, embedding weight, negative penalty and maximum-active setting were frozen before this evaluation. Held-out is used only to measure generalization; it does not select or retune the model.

## Selected frozen method

- Method: `hybrid`
- Prototype source / aggregation: `question` / `centroid`
- Embedding / lexical weight: `0.90` / `0.20`
- Hard-negative penalty: `0.50`
- Thresholds: consciousness `0.40`, self `0.70`, reality/appearance `0.75`

## Overall Held-out metrics

- Macro F1: **0.6847**
- Micro F1: **0.7483**
- Macro average precision: **0.9193**
- Calibration error (macro soft ECE): **0.1076**
- Exact active-set accuracy: **0.2278**
- Unsupported accuracy: **0.9620**

## Development → Held-out generalization

- Development macro F1: `0.7866`
- Held-out macro F1: `0.6847`
- Macro F1 delta: `-0.1019`

## Per-concept Held-out performance

| Concept | Precision | Recall | F1 | Avg precision | Calibration error |
|---|---:|---:|---:|---:|---:|
| consciousness | 1.0000 | 0.2292 | 0.3729 | 0.8608 | 0.1134 |
| self_identity | 0.7067 | 0.9815 | 0.8217 | 0.9176 | 0.0662 |
| reality_appearance | 0.9608 | 0.7778 | 0.8596 | 0.9795 | 0.1431 |

## Baseline comparison

| System | Macro F1 | Micro F1 | Avg precision | Calibration error | Exact set |
|---|---:|---:|---:|---:|---:|
| Selected — Frozen Hybrid | 0.6847 | 0.7483 | 0.9193 | 0.1076 | 0.2278 |
| Baseline A — Plain embedding similarity | 0.6510 | 0.6873 | 0.8734 | 0.0398 | 0.1519 |
| Baseline B — Prototype centroid | 0.6460 | 0.6935 | 0.8665 | 0.0407 | 0.1519 |
| Baseline C — Prototype + hard-negative | 0.6413 | 0.7121 | 0.8602 | 0.0401 | 0.1772 |
| Baseline D — Build-trained classifier | 0.6622 | 0.6873 | 0.8673 | 0.0374 | 0.1646 |

The baseline table is descriptive. It does **not** trigger model reselection after Held-out.

## Domain-level results — selected method

| Domain | Records | Macro F1 | Micro F1 | Avg precision | Calibration error |
|---|---:|---:|---:|---:|---:|
| science | 0 | N/A | N/A | N/A | N/A |
| advaita | 50 | 0.6785 | 0.7745 | 0.9508 | 0.0912 |
| samkhya | 29 | 0.6730 | 0.6939 | 0.8557 | 0.2812 |

## Required hard-negative evaluation — selected method

| Error type | Samples | False positives | FP rate | Status |
|---|---:|---:|---:|---|
| attention_mistaken_for_consciousness | 0 | 0 | N/A | not_available_in_heldout |
| ego_mistaken_for_self | 0 | 0 | N/A | not_available_in_heldout |
| cosmology_mistaken_for_reality_appearance | 6 | 1 | 0.1667 | measured |
| purusha_collapsed_into_atman | 7 | 7 | 1.0000 | measured |
| perception_description_mistaken_for_metaphysical_appearance | 0 | 0 | N/A | not_available_in_heldout |

A required hard-negative category with zero Held-out examples is reported as unavailable rather than inferred from another category.

## Adjacent-concept confusion — selected method

### `consciousness__self_identity`

- `consciousness_mistaken_as_self_identity`: 11/13 = **0.8462**
- `self_identity_mistaken_as_consciousness`: 0/19 = **0.0000**

### `self_identity__reality_appearance`

- `self_identity_mistaken_as_reality_appearance`: 1/12 = **0.0833**
- `reality_appearance_mistaken_as_self_identity`: 19/21 = **0.9048**

### `consciousness__reality_appearance`

- `consciousness_mistaken_as_reality_appearance`: 1/9 = **0.1111**
- `reality_appearance_mistaken_as_consciousness`: 0/24 = **0.0000**

## Failure and limitation record

- `65` Held-out records are included in the error-analysis CSV because they contain an active-set mismatch and/or a reviewed hard-negative category.
- `partial` labels are treated as active for multi-label precision/recall/F1.
- Calibration error uses the original soft calibration targets negative=0, partial=0.5, positive=1 because Phase 10 calibrated weights against those targets.
- Hard-negative categories are evaluated only when the exact reviewed category is present in Held-out; zero-sample requirements remain explicit.
- The supervised baseline is deterministically reconstructed from Build using the frozen Phase 10 training algorithm. Held-out labels do not influence its training.
- If a baseline outperforms the selected method on Held-out, that is documented as a limitation; Phase 10 parameters are not changed.

## Selected-method justification

The hybrid remains the selected method because it was chosen and frozen on Development before Held-out was opened. Phase 11 reports whether that choice generalizes and compares it with frozen baselines; it does not use Held-out to make a new model-selection decision.

## Exit gate

- Held-out results recorded: **PASS**
- Post-hoc threshold changes: **NONE**
- Post-hoc calibration changes: **NONE**
- Failures and limitations explicitly documented: **PASS**
- Frozen selected method retained without Held-out reselection: **PASS**
