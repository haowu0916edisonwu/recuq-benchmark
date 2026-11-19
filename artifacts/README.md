# Artifact Inventory

All record-level files are gzip-compressed JSON Lines.  They can be read with
Python's `gzip.open(path, "rt")`; the released evaluators accept both `.jsonl`
and `.jsonl.gz`.

## Base predictions

- `base_predictions/llama_fixed_rankings.jsonl.gz`: 45,163 rows.
- `base_predictions/qwen_warm_start_fixed_rankings.jsonl.gz`: 4,000 rows.

The Llama file intentionally retains non-tail S4 rows for auditability.  The
scalar evaluator filters that internal scenario to target bucket `tail`, giving
42,475 eligible records.

## UQ outputs

Each file under `uq_outputs/llama/` contains 45,163 rows; each file under
`uq_outputs/qwen/` contains 4,000 rows.

| Release filename | Family |
|---|---|
| `labelprob_max.jsonl.gz` | probability |
| `labelprob_top10mass.jsonl.gz` | ranking-aligned probability |
| `labelprob_entropy.jsonl.gz` | probability |
| `labelprob_margin.jsonl.gz` | probability |
| `temperature_scaled_labelprob.jsonl.gz` | calibration |
| `decomposed_llm_uq.jsonl.gz` | decomposition |
| `perturbation_rank_instability.jsonl.gz` | sensitivity |
| `self_consistency.jsonl.gz` | sampling |

Artifact method IDs are preserved in each row; `metadata.release_name` supplies
the stable release filename.  `metadata/method_registry.csv` records
the mapping.

## Reference results

- `paper_tables/`: exact non-rounded CSV values used to produce Tables 1--8.
- `llama/main/`: all Llama scalar aggregations and 1,000 Warm-start bootstrap.
- `qwen/warm_start/`: Qwen scalar aggregations and 1,000 Warm-start bootstrap.
- `paired_control/gap_*/`: metrics at effect-gap thresholds 0, 0.05, 0.10.
- `diagnostics/topk_alignment.csv`: Appendix TopK alignment matrix.
- `diagnostics/cluster_bootstrap/`: three 200-resample clustered checks.
- `diagnostics/sanity_checks/`: evaluator sanity references.
- `diagnostics/component_checks/`: pooled 42,475-record component checks.

Evaluator CSVs retain implementation-oriented filenames such as
`table11_error_auroc_by_method.csv` for backward compatibility.  Paper-facing
numbering is centralized under `paper_tables/` and documented in
`docs/PAPER_RESULTS_MAP.md`.

## Manifests

- `manifests/artifact_build_manifest.json` records row count and SHA-256 of the
  uncompressed byte stream for every record-level artifact.
- `manifests/release_files.csv` and root `SHA256SUMS` cover all non-generated payload files; the checksum manifests themselves are excluded to avoid recursive hashes and are generated after final validation.

## Metadata and schemas

- `metadata/model_runs.json`: public model/decoding/run metadata.
- `metadata/record_populations.json`: expected condition and diagnostic counts.
- `metadata/method_registry.csv`: stable release-name mapping.
- `schemas/*.schema.json`: machine-readable base and UQ contracts.


## Camera-ready diagnostics

- `supplement/qwen_controlled_rankings.jsonl.gz` contains 41,163 fixed Qwen
  rankings used in the auxiliary paired-control and evidence-removal analyses.
  It includes its own Warm-start rows. Use the separate frozen Warm-start file
  for the main Qwen scalar results.
- `supplement/same_prompt/` and `supplement/self_consistency/` contain the
  record-level scores used to evaluate the same-prompt and base-anchored variants.
- `reference_results/camera_ready/` contains the measured diagnostics and the
  original inference timing summary. See the [paper results map](../docs/PAPER_RESULTS_MAP.md).
