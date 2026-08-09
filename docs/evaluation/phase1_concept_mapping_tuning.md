# Phase 1 Concept Mapping Tuning

- Tuning version: `phase1-concept-mapping-dev-v1`
- Generated: `2026-08-09T09:01:42.304489+00:00`
- Status: **FROZEN FOR HELD-OUT EVALUATION**
- Build records used for classifier training: **159**
- Development records used for tuning: **80**
- Held-out records used: **0**

## Frozen embedding identity

- Provider: `Google Gemini API`
- Model: `gemini-embedding-2`
- Model revision: `2`
- Dimensions: `768`
- Normalization: `provider_auto_l2`

## Candidate methods evaluated

- Prototype centroid
- Maximum-example similarity
- Positive-minus-hard-negative similarity
- Lightweight soft logistic classifier trained on Build only
- Hybrid embedding similarity + lexical indicator + hard-negative penalty

## Selected configuration

- Method: **hybrid**
- Prototype source: `question`
- Prototype aggregation: `centroid`
- Negative penalty: `0.50`
- Lexical weight: `0.20`
- Embedding weight: `0.90`
- Concept activation thresholds:
  - consciousness: `0.40`
  - self_identity: `0.70`
  - reality_appearance: `0.75`
- Ambiguity margin: `0.10`
- Maximum active concepts: `3`

Raw scores are preserved separately from calibrated 0-1 weights. Weights are independent and are not normalized to sum to one.

## Development metrics

- Objective: **0.7369**
- Macro F1: **0.7866**
- Macro MAE: **0.3407**
- Macro Brier: **0.1499**
- Unsupported accuracy: **0.9625**
- Ambiguity accuracy: **0.7250**
- Hard-negative false activation rate: **0.3000**
- Exact active-set accuracy: **0.4875**

## Per-concept metrics

| Concept | F1 | MAE | Brier |
|---|---:|---:|---:|
| consciousness | 0.5763 | 0.3522 | 0.1547 |
| self_identity | 0.9065 | 0.3395 | 0.1480 |
| reality_appearance | 0.8772 | 0.3303 | 0.1469 |

## Top candidate configurations

| Rank | Method | Source | Aggregation | Neg | Lex | C Thr | S Thr | R Thr | Margin | Max | Objective | F1 | HN FP |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | hybrid | question | centroid | 0.50 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7369 | 0.7866 | 0.3000 |
| 2 | hybrid | question | centroid | 0.50 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7321 | 0.7789 | 0.3000 |
| 3 | hybrid | question | maximum | 0.75 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7308 | 0.7703 | 0.3000 |
| 4 | hybrid | combined | maximum | 0.50 | 0.20 | 0.40 | 0.70 | 0.75 | 0.08 | 3 | 0.7303 | 0.7650 | 0.3000 |
| 5 | hybrid | passage | maximum | 0.50 | 0.20 | 0.40 | 0.70 | 0.75 | 0.08 | 3 | 0.7303 | 0.7650 | 0.3000 |
| 6 | hybrid | question | maximum | 0.50 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7300 | 0.7718 | 0.3000 |
| 7 | hybrid | question | maximum | 0.50 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7300 | 0.7718 | 0.3000 |
| 8 | hybrid | question | maximum | 0.25 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7292 | 0.7609 | 0.3000 |
| 9 | hybrid | question | centroid | 0.50 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7273 | 0.7756 | 0.3000 |
| 10 | hybrid | question | maximum | 0.75 | 0.20 | 0.40 | 0.70 | 0.75 | 0.10 | 3 | 0.7272 | 0.7621 | 0.3000 |

## Frozen calibration

### `consciousness`

- Slope: `1.18105318`
- Intercept: `-0.87995013`

### `self_identity`

- Slope: `1.49534767`
- Intercept: `0.44632384`

### `reality_appearance`

- Slope: `1.75098564`
- Intercept: `0.47630048`

## Runtime behavior

- Multiple concepts may activate simultaneously.
- Weights are not forced to sum to one.
- Raw scores remain available for audit/debugging.
- Unsupported means no concept clears the frozen threshold.
- Ambiguous means the two strongest eligible weights are within the frozen ambiguity margin.
- Maximum-active limits activation decisions only; it does not discard raw scores or calibrated weights.

## Leakage controls

- Build was used only for the lightweight classifier candidate.
- Development was used for method selection, calibration, threshold, ambiguity margin, maximum active concepts, prototype aggregation, negative penalties, and lexical/embedding hybrid weights.
- Held-out was not supplied as an input and was not used for tuning.
- All selected parameters are frozen before Phase 11.

## Exit gate

**PASS** — Phase 10 parameters are frozen before held-out evaluation.

Machine-readable results: `E:/ABHINAV/Coding/Projects/WTH-WhatTheHuman/artifacts/phase1/evaluation/concept_mapping_dev_results.json`
