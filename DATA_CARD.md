# Data Card: Fixed-Ranking Recommendation UQ Artifact

## Summary

This artifact supports evaluation of uncertainty monitors over fixed LLM
recommendation rankings.  It is not a new user-interaction dataset and does not
redistribute raw user histories or raw item text.

## Source datasets

The ranking records were constructed from:

- MIND news recommendation data;
- Amazon Reviews 2023 Books and Movies/TV domains;
- Yelp Open Dataset business/review data.

Access and redistribution remain governed by each provider.  See
`DATA_LICENSES.md`.

## Unit of observation

One base artifact row is one issued 20-candidate ranking under one dataset,
underlying instance, evaluation condition, and variant.  Each UQ row is joined
to exactly one base row by `record_id`.

## Candidate construction

- MIND-20 uses impression-based candidate sets with one positive and 19
  official impression negatives.
- Amazon Books, Amazon Movies, and Yelp use sequential leave-last-positive-out
  instances with one positive and 19 sampled candidates under the policies in
  `configs/dataset/`.

## Record counts

| Dataset | Base instances | Generated Llama records | Eligible records |
|---|---:|---:|---:|
| MIND-20 | 1,000 | 11,000 | 10,870 |
| Amazon Books | 1,000 | 12,000 | 11,221 |
| Amazon Movies | 1,000 | 10,163 | 9,292 |
| Yelp | 1,000 | 12,000 | 11,092 |
| Total | 4,000 | 45,163 | 42,475 |

Amazon Movies has fewer generated ablation variants because some configured
fields were unavailable.  Qwen contributes 4,000 Warm-start ranking records. Its 41,163 auxiliary
rankings support paired controls and evidence-removal diagnostics.

## Evaluation conditions

- Warm-start is the reference condition.
- Sparse-history includes full history and truncation variants.
- Tail-target subset retains records whose target item is assigned to the
  dataset-specific tail bucket.
- Item-text robustness includes original and available item-field ablations.

## Labels and derived metrics

The base evaluator stores ranking metrics and the binary Error@10 label.  The
Error@10 label is used only by evaluation scripts, not by UQ methods.  The
paired-control operational label is derived from restoration effects inside
the evaluator and is not a source-dataset annotation.

## Released representation

To limit redistribution and privacy risk, the release:

- replaces original candidate IDs with local option labels A--T;
- retains ranking indices, binary labels, metrics, conditions, variants, and
  stable record/instance keys needed for joining and clustering;
- removes raw histories, item text, prompts, responses, original user IDs,
  original item IDs, and source hashes;
- retains only compact parser diagnostics and method-specific numerical
  components needed by the paper's checks.

## Intended uses

- reproduce paper tables and figures;
- compare additional scalar UQ monitors under the same fixed-ranking contract;
- test schema, coverage, aggregation, and selective-risk evaluators;
- study robustness of UQ ordering across the released conditions.

## Out-of-scope uses

- reconstructing individual users or original private profiles;
- treating the paired-control label as causal ground truth;
- evaluating full retrieval performance;
- claiming broad cross-model generalization from two backbones;
- training on Error@10 labels and then reporting in-sample monitor performance.

## Known limitations

- 20-candidate reranking rather than full retrieval;
- 1,000 sampled base instances per dataset;
- correlated controlled variants from the same underlying instance;
- two model backbones, with only Llama covering all conditions;
- option-label confidence depends on prompting/tokenization;
- self-consistency uses five samples as a compute-controlled setting;
- user-history and item-content are the only paired controls;
- no independent causal intervention test.

## Maintenance

See [CITATION.cff](CITATION.cff) for authorship and the paper URL. Benchmark
questions can be directed to Hao Wu at hwu522@wisc.edu.
