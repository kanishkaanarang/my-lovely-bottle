# Amazon ML Challenge 2026 — Business Entity Resolution

**Team:** [FILL team name]
**Members:** [FILL member names]
**Submission date:** [FILL date]

## 1. Executive summary

We use country-partitioned lexical retrieval followed by a hard-negative pair classifier and a calibrated per-anchor empty/set decision policy. All scored candidate IDs are exported before the final match threshold is applied. No external business lookup, geocoding or pretrained model is used.

## 2. Methodology and data

The provided Source 1 records anchor one-to-many matching against Sources 2 and 3. Training covers US and India; test additionally includes France. Unicode combining marks are preserved; Latin accent folding and suffix-reduced names are separate comparison views.

Dataset audit findings motivating the approach: large target pools, multiple true links per source, singletons, missing addresses, common-name distractors and cross-script variation. [FILL your final observed error analysis; do not copy demo results]

## 3. Candidate generation

SQLite FTS5 indexes are partitioned by observed country. Candidate channels are normalized exact name/address, suffix-reduced name, BM25 name/address and name character trigrams. Ambiguous anchors receive larger candidate budgets. The final candidates are capped per source before model inference and exported exactly as scored.

- Final retrieval settings: [FILL]
- True-link retrieval recall / complete-set retrieval / oracle macro F0.5: [FILL]
- Candidate counts and tail distribution: [FILL]

## 4. Matching and validation

The model is histogram gradient boosting on name/address similarity, digit evidence, missingness, source and retrieval context. Hard negatives come from the inference-style blocker. Anchors are divided by normalized signature into training, calibration and trust sets. Calibration is further divided for probability transformation and threshold/empty-gate tuning.

- Training sample and exact split seed/config: [FILL]
- Model settings and decision thresholds: [FILL]
- Trust macro F0.5 and per-country metrics: [FILL]
- Geographic stress tests, if run: [FILL]
- Known leakage/generalization limitations: [FILL]

## 5. Runtime and reproducibility

- Hardware, Python/packages, full index time and disk use: [FILL]
- Full test inference, validation and export time: [FILL]
- Code revision and input/model/output hashes: [FILL]

Runnable code, model, requirements and inference instructions are included under `code/business_entity_resolution/`. Both output files have passed the strict validator. The source is MIT licensed; dependencies retain their respective licenses. The newly trained model contains no pretrained foundation-model weights.

## 6. Limitations

Current retrieval preserves scripts but does not transliterate. Cross-script matching relies substantially on address evidence. Exact-signature splits do not eliminate every near-duplicate relationship. No French accuracy can be measured from the supplied labels. [FILL additional observed limitations and future work]
