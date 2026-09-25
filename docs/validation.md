# What the metrics mean

The supplied task is multi-match, per-reference entity resolution. A candidate miss is an unavoidable false negative for that blocker. Always compare model score to the oracle score for its exact final candidate set.

## Split protocol

- Reservoir sample anchors independently within each observed training country; retain the full S2/S3 target pool for search.
- Hash `(seed, country, normalized name, normalized address)` into 70% training, 15% calibration, 15% trust. Exact duplicate signatures stay together, independently of IDs.
- Train only on retrieved training candidates. Every candidate gets its label from that anchor's supplied true set. True links outside the candidate set are never injected to inflate retrieval recall.
- Split calibration anchors again by deterministic ID hash: one half fits the probability calibrator; the other selects pair threshold and empty-set gate.
- Evaluate trust anchors with the frozen model, calibrator and decision policy. Keep per-anchor candidates, probabilities, truth and selected predictions for paired analysis.

Country-balanced sampling makes the unweighted sample macro score different from the original training mixture. Both sample macro and a country-weighted score using the full Source 1 training country counts are reported. Threshold selection uses the latter. Neither is an estimate of the unseen French score. A test-mixture estimate requires a French labeled validation set that the supplied data does not provide.

The observed training data assign each target ID to a single anchor. This assumption supports grouping anchor/link supervision; if a different dataset shares targets across anchors, group the connected components before using this split. Exact signatures do not capture all near-duplicate or family relationships. Add a stricter near-duplicate split before interpreting small gains as generalization.

## Geography

`train(..., exclude_country="India")` gives India candidates zero training weight and excludes India from both probability calibration and threshold tuning. Evaluate on India trust anchors. Reverse US/India for the second stress test. Unlabeled target text remains available to retrieval in both cases.

France is automatically indexed and predicted in full test mode. It has no training labels; no French validation score is claimed. Preserve raw Unicode, separate accent folding from canonical text, and examine candidate/score distributions for drift.

## Decision policy

All candidates above the selected pair threshold are accepted unless the highest candidate probability is below the selected empty-set gate. There is no one-match-per-source constraint. This gate is a simple policy tuned on macro F0.5; it is not a trained singleton model or a general expected-utility optimizer. Missing retrieval evidence can still be mistaken for a singleton.

The fitted classifier uses anchor-normalized sample weights. Its sigmoid calibration is an empirical score transformation under that weighting, not a guarantee of perfectly calibrated marginal probabilities. Selection is evaluated end to end on the frozen trust partition.

## Honest reporting

Synthetic-demo scores only prove software behavior. Do not report them as challenge accuracy. Indexing the full target pool and testing a synthetic fixture do not establish full AWS throughput. Retain candidate-recall metrics and runtime with every real-data experiment. Do not select repeatedly on the trust set or public leaderboard.
