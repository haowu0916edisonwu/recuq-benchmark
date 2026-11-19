# Configuration Index

- `dataset/`: raw/processed paths, sequential construction thresholds,
  candidate sampling, and item-text fields.
- `condition/`: Warm-start, Sparse-history, frozen internal S4 tail-target
  construction, and Item-text robustness.
- `eval_subsets/`: the shared fixed 1,000-instance-per-dataset manifest.
- `experiment/`: Llama complete suite, Qwen Warm-start, and auxiliary Qwen
  paired-control rankings.
- `model/`: public model IDs and deterministic decoding settings.
- `uq/`: paper-level definitions for all eight monitors.
- `evaluation/`: scalar metrics, eligibility filtering, bootstrap units, and
  paired-control operational-label construction.

Relative paths in experiment YAML files resolve from the YAML file's parent
directory.  Tests verify that each referenced configuration and manifest
exists.
