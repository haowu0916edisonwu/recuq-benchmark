# Artifact Contract

## 1. Fixed ranking table

Each fixed ranking row uses schema `fixed_ranking_record_v1` and is stored as
gzip-compressed JSON Lines.  Required evaluation fields are:

| Field | Meaning |
|---|---|
| `record_id` | stable key joining base and UQ records |
| `instance_id` | underlying case key used for clustered resampling |
| `backbone` | evaluated base ranker |
| `dataset` | one of four benchmark datasets |
| `scenario` | frozen internal scenario key |
| `evaluation_condition` | paper-facing condition name |
| `variant_name` | original/full/truncated/ablated variant |
| `labels` | length-20 binary relevance vector |
| `ranking` | permutation of indices 0--19 |
| `metrics` | fixed-ranking Hit/NDCG/MRR/error fields |
| `is_error_at_10` | target outside the fixed top-10 |
| `parse_success` | parser success status |
| `popularity_bucket` | target bucket where applicable |

Candidate labels are release-local A--T and do not identify source items.
The UQ evaluator never changes `ranking`.

## 2. UQ output table

Each row uses schema `uq_prediction_v1`:

| Field | Required | Meaning |
|---|---:|---|
| `schema_version` | yes | `uq_prediction_v1` |
| `record_id` | yes | joins to exactly one base row |
| `method` | yes | canonical implementation identifier |
| `method_family` | yes | family identifier from the published UQ schema, including appendix diagnostics |
| `uncertainty_score` | yes | finite scalar; larger means less reliable |
| `confidence_score` | no | optional confidence counterpart |
| `source_scores` | no | optional finite user/item and diagnostic scores |
| `components` | no | numerical method components used by diagnostics |
| `metadata` | no | dataset, condition, instance, and release name |

The Python validator is `recuq_benchmark/uq/schema.py`; the JSON schema is
`artifacts/schemas/uq_prediction.schema.json`.

## 3. Join invariants

For each backbone and method:

```text
set(UQ.record_id) == set(base.record_id)
```

Expected cardinalities:

- Llama: 45,163 rows per method before eligibility filtering;
- Qwen: 4,000 rows per method;
- duplicates: zero;
- unmatched record IDs: zero;
- schema errors: zero.

Eligibility filtering is an evaluator operation, not a destructive rewrite of
the released base/UQ files.

## 4. Scalar metrics

For score \(u_i\) and binary error \(e_i\):

- Error-AUROC: ROC AUC of `u_i` for `e_i`;
- Error-AUPRC: average precision of `u_i` for `e_i`;
- Risk@80: error rate among the floor(0.8 n) lowest-uncertainty records;
- AURC: mean selective risk over every retained-prefix coverage;
- E-AURC: AURC minus the oracle AURC at the same error prevalence.

Ties are handled by the underlying rank/ROC conventions.  Every score is
oriented so larger means more uncertain before metric computation.

## 5. UQ score definitions

Let the option-label distribution over 20 candidates be \(p(a)\).

- LabelProb-Max: `1 - max_a p(a)`.
- Top10Mass: `1 - sum_{a in fixed base top-10} p(a)`.
- Entropy: entropy of `p`, normalized by `log(20)`.
- Margin: `1 - (p_(1) - p_(2))`.
- Temperature scaling: five-fold instance-level cross-fitted scalar
  temperature; score is one minus scaled max probability.
- Decomposed UQ: normalized entropy of the mean aligned distribution across
  five candidate-order variants; recommendation and prompt components remain
  diagnostic fields.
- Perturbation instability: mean top-10 Jaccard disagreement under paired
  history/item controls.
- Self-consistency: mean pairwise top-10 Jaccard disagreement across five
  sampled rankings.

Top10Mass uses no relevance label and does not change the fixed ranking.

## 6. Paired-control contract

For each underlying instance:

```text
user restoration gain = NDCG@10(full history) - NDCG@10(truncated history)
item restoration gain = NDCG@10(original text) - NDCG@10(ablated text)
```

Available variants are averaged within each control and gains are clipped at
zero.  The larger gain defines the operational dominant control; ties are
excluded.  Scalar methods predict from clipped uncertainty increases.  A
method with finite separate user/item control scores uses those scores directly.

Prediction ties are retained and lower directional coverage.  Accuracy and F1
are computed over all operational-label cases, so a prediction tie is not a
correct class decision.

Scope statement: constructed user/item control-response agreement only; no causal attribution.

## 7. Extension contract

To evaluate a new monitor:

1. do not edit the base ranking file;
2. output exactly one `uq_prediction_v1` row per base `record_id`;
3. ensure `uncertainty_score` is finite and oriented upward;
4. use `source_scores.user` and `.item` only when they represent separate
   control responses;
5. place method-specific internals under `components`;
6. run `evaluate_uq_methods.py` and inspect coverage/schema outputs;
7. report results separately by backbone, dataset, and evaluation condition.
