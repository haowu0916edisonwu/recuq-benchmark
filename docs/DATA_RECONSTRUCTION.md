# Licensed Data Reconstruction

Raw dataset text is not included.  This document describes how an authorized
user can rebuild compatible local inputs.

## General rules

1. Obtain each dataset from its official source.
2. Accept and comply with provider terms.
3. Preserve the directory layout declared in `configs/dataset/*.yaml`.
4. Do not commit raw data, model caches, access tokens, or generated prompts.
5. Use the committed subset manifest to select the same 1,000 base instances
   per dataset.

## MIND-20

Expected source files are the official MINDlarge train/dev `news.tsv` and
`behaviors.tsv` files.  `scripts/prepare_dataset.py` creates the normalized
interaction representation; `scripts/build_mind_20.py` constructs the
impression-based one-positive/19-negative records.

The MIND candidate set uses official impression negatives and should not be
replaced with sampled catalog negatives.

## Amazon Books and Movies

`scripts/download_prepare_amazon_2023.py` supports the Amazon Reviews 2023
distribution and writes local parquet review/metadata files.  The exact column
selection and item-text fields are specified in the dataset YAML files.

Sequential evaluation leaves the final positive interaction out and samples
19 candidates under the configured same-category-then-global policy, excluding
the user's history where configured.

## Yelp

Place `yelp_academic_dataset_review.json` and
`yelp_academic_dataset_business.json` under `data/raw/yelp/`.  The configuration
specifies positive thresholds, sequential history construction, business text
fields, and candidate sampling.

## Fixed subset manifest

`configs/eval_subsets/aligned20_n1000_seed42.jsonl` is the shared instance
selection.  Do not independently sample a new subset for Qwen.  The manifest
is the bridge between independently reconstructed source data and the released
fixed ranking protocol.

## Expected checkpoints

After reconstruction, verify:

- 1,000 Warm-start base instances per dataset;
- 20 candidates and one positive per record;
- MIND candidates originate from the official impression;
- the selected instances follow the manifest order and the candidates follow
  the dataset's configured construction and seed;
- dataset/instance keys match the release manifest before inference;
- no source text is copied into the public Git repository.
