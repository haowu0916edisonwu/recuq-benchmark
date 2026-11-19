# Reproducibility Guide

## CPU evaluation

Run from the repository root using Python 3.10 or 3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-eval.txt
make validate
make reproduce
```

Validation checks the release manifest, fixed populations, all 16 main UQ joins,
and the supplemental artifact manifest. Quick reproduction recomputes and
checks the point estimates from the fixed rankings and released UQ scores.
Use `make reproduce-full` to also run 1,000 dataset-stratified paired Warm-start
resamples per backbone, 200 instance-clustered resamples per non-reference Llama
condition, and the paired-control and supplementary Warm-start intervals. All
resampling uses seed 42. Results are written under `outputs/reproduced/`.

`make paper-tables` exports Tables 1–8 from the reference results. `make figures`
renders the data-derived figures and the case-study example. The camera-ready
references and matching commands are listed in the [paper results map](docs/PAPER_RESULTS_MAP.md).

## Scalar evaluator commands

### Llama full condition suite

```bash
python scripts/evaluate_uq_methods.py \
  --base_predictions artifacts/base_predictions/llama_fixed_rankings.jsonl.gz \
  --uq_predictions \
    artifacts/uq_outputs/llama/labelprob_max.jsonl.gz \
    artifacts/uq_outputs/llama/labelprob_top10mass.jsonl.gz \
    artifacts/uq_outputs/llama/labelprob_entropy.jsonl.gz \
    artifacts/uq_outputs/llama/labelprob_margin.jsonl.gz \
    artifacts/uq_outputs/llama/temperature_scaled_labelprob.jsonl.gz \
    artifacts/uq_outputs/llama/decomposed_llm_uq.jsonl.gz \
    artifacts/uq_outputs/llama/perturbation_rank_instability.jsonl.gz \
    artifacts/uq_outputs/llama/self_consistency.jsonl.gz \
  --output_dir outputs/manual/llama \
  --bootstrap_resamples 1000 \
  --bootstrap_scenario s0_warmstart \
  --bootstrap_unit record
```

Expected health checks:

- 45,163 input rows per method;
- 42,475 rows after tail-target eligibility filtering;
- no unmatched `record_id` values;
- no schema errors;
- pooled Top10Mass Error-AUROC `0.6337819441`;
- Warm-start Top10Mass Error-AUROC `0.6280276640`.

### Qwen Warm-start replication

Use `artifacts/base_predictions/qwen_warm_start_fixed_rankings.jsonl.gz` and the
eight files under `artifacts/uq_outputs/qwen/`.  The same evaluator command is
used.  Qwen contains only `s0_warmstart`; it must produce exactly 4,000 joined
rows per method.

Expected overall Error-AUROC:

- Decomposed LLM UQ: `0.5588559575`;
- Temperature-scaled LabelProb: `0.5382642372`;
- Top10Mass: `0.4466592445`.

## Eligibility and aggregation

The raw Llama artifact contains 45,163 generated records.  The evaluator keeps
all records except non-tail records whose internal scenario is `s4_longtail`.
This gives:

```text
Warm-start             4,000
Sparse-history        20,000
Tail-target subset     1,312
Item-text robustness  17,163
Total eligible        42,475
```

“Overall” within a condition means pooling all eligible records in that
condition.  The 42,475-record component check is a different population from
the 4,000-record Warm-start comparison.

## Paired-control evaluation

```bash
python scripts/evaluate_source_response_interventions.py \
  --base_predictions artifacts/base_predictions/llama_fixed_rankings.jsonl.gz \
  --uq_predictions artifacts/uq_outputs/llama/*.jsonl.gz \
  --output_dir outputs/manual/paired_control/gap_0.00 \
  --min_effect_gap 0.0 \
  --write_instance_details
```

If the shell does not expand globs consistently, list the eight files as in the
scalar command.

The evaluator performs the following operations per underlying instance:

1. pair full history with all available history truncations;
2. pair original item text with all available item-field ablations;
3. average NDCG@10 restoration gains within each control;
4. clip each mean restoration gain at zero;
5. exclude equal-gain operational-label cases;
6. compute scalar response increases or read finite separate control scores;
7. keep equal response scores as prediction ties;
8. report directional coverage, dominant-control accuracy, macro F1,
   Source-AUROC, and average control-wise Spearman correlation.

At gap 0.00, there are 1,819 cases.  Perturbation Rank Instability should give:

```text
directional coverage       0.9873556899
dominant-control accuracy  0.7091808686
macro F1                   0.7123876124
Source-AUROC               0.6748671386
rho                        0.4557862177
```

Repeat with `--min_effect_gap 0.05` and `0.10` for Table 13.
`make paired-control-positive` restricts the evaluation to 863 instances with
positive signed differences under both controls.

For signed effects, ranking distances, and the auxiliary Qwen check, run
`python scripts/reproduce_supplement.py`. Add `--full` for the reported
bootstrap intervals. This command also evaluates shared fixed temperatures,
ranking-prompt scores, base-anchored self-consistency, and evidence removal.

## Appendix diagnostics

### Error@K/TopKMass alignment

```bash
python scripts/derive_topk_alignment.py \
  --base_predictions artifacts/base_predictions/llama_fixed_rankings.jsonl.gz \
  --labelprob_predictions artifacts/uq_outputs/llama/labelprob_max.jsonl.gz \
  --output outputs/manual/topk_alignment.csv
```

This uses only the 4,000 Llama Warm-start records.

### Derived component checks

```bash
python scripts/derive_uq_component_checks.py \
  --base_predictions artifacts/base_predictions/llama_fixed_rankings.jsonl.gz \
  --labelprob_predictions artifacts/uq_outputs/llama/labelprob_max.jsonl.gz \
  --decomp_predictions artifacts/uq_outputs/llama/decomposed_llm_uq.jsonl.gz \
  --self_consistency_predictions artifacts/uq_outputs/llama/self_consistency.jsonl.gz \
  --output_root outputs/manual/components \
  --overwrite
```

Evaluate the eight resulting directories with `evaluate_uq_methods.py`.  The
evaluator applies the tail-target filter and therefore reports 42,475 records.

### Sanity references

```bash
python scripts/derive_uq_sanity_checks.py \
  --base_predictions artifacts/base_predictions/llama_fixed_rankings.jsonl.gz \
  --output_root outputs/manual/sanity \
  --overwrite
```

These are evaluator checks, not deployable UQ methods.

## Regenerate data and inference outputs

- [Build the benchmark inputs](docs/DATA_RECONSTRUCTION.md) gives the source
  downloads, exact file layout, preprocessing, and CPU checks on the frozen
  instances and prompts.
- [Regenerate rankings and uncertainty scores](docs/GPU_REGENERATION.md)
  gives the model setup and commands from fixed ranking generation through
  the eight monitors, appendix scores, and evaluation of the new predictions.

## Compute notes

Evaluation and plotting are CPU-only.  Full generation requires GPU inference.
The completed self-consistency runs reported in the artifact metadata used five
additional rankings per record, temperature 0.9, and top-p 0.95.  The Llama
controlled-suite self-consistency run took approximately 121 h 53 min; the
Qwen Warm-start run took approximately 13 h 25 min in the original execution
environment.  Hardware differences make these wall times non-portable.

## Troubleshooting

- **Missing source data:** metric reproduction does not require source data;
  use the released fixed records.
- **Hugging Face access denied:** accept the upstream model/dataset license and
  authenticate; this affects inference regeneration only.
- **Different counts:** run `validate_release.py`; most count differences come
  from failing to apply the tail-target eligibility filter.
- **0.628 vs 0.634:** the former is Warm-start (4,000); the latter is pooled
  eligible Llama records (42,475).
- **Paired-control accuracy seems low:** prediction ties are retained and count
  as no direction; inspect directional coverage jointly.
- **Bootstrap differences:** confirm the unit (`record` for Warm-start,
  `instance` for controlled conditions), strata (`dataset`), and seed 42.
