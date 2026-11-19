# Released Ranking and Score Records

## Goals

The release supports independent metric reproduction with local candidate labels,
fixed rankings, numerical monitor outputs, and stable benchmark record keys.

## Removed from base records

- `user_id`;
- original `candidate_ids` and `ground_truth_item_ids`;
- raw histories and raw item text;
- `raw_response`;
- prompt/history/candidate hashes;
- run IDs and local result paths;
- provider tokens, environment variables, and cache locations.

Original candidates are replaced with local labels A--T.  Because rankings are
stored as permutations of indices 0--19, this does not alter any ranking metric.

## Retained from base records

- stable `record_id` and `instance_id` needed for joining, controlled pairing,
  and cluster bootstrap;
- dataset, condition, variant, and target-source construction tags;
- local A--T candidate labels, relevance vector, and ranking indices;
- fixed-ranking metrics and Error@10 event;
- target popularity bucket for eligibility filtering;
- compact parser status/repair counts;
- public model identifier and non-thinking flag.

## UQ record transform

Every scalar uncertainty score is retained.  Components are allowlisted by
method.  Large/raw generation internals are removed, including sampled full
rankings and candidate-order permutations when not needed for an evaluator.
Separate numerical user/item response scores are retained because they are
required by the paired-control diagnostic.

## Determinism and audit

`scripts/build_record_artifacts.py` writes deterministic gzip streams with
gzip timestamp zero and sorted compact JSON keys.  The manifest stores the
SHA-256 of the uncompressed JSONL byte stream and the row count of each file.

`scripts/validate_release.py` independently streams these files, recomputes the
hashes, checks field exclusions, and verifies exact record-ID joins.


The camera-ready auxiliary Qwen rankings use the same base-record filter.
The supplemental ranking-prompt and self-consistency files contain only the
record key, method, score, schema version, and diagnostic group.
