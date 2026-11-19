# Compute and Model Configuration

## Base rankers

### Llama controlled suite

- model ID: `meta-llama/Meta-Llama-3.1-8B-Instruct`;
- dtype: bfloat16;
- deterministic generation: `do_sample=false`, temperature 0;
- seed: 42;
- maximum new tokens: 128;
- score mode: parsed ranking over labels A--T;
- conditions: Warm-start, Sparse-history, tail-target subset, Item-text
  robustness.

The archived rankings identify the Llama model by its public model ID.
Record the resolved model snapshot if you regenerate rankings.

### Qwen Warm-start replication

- model ID: `Qwen/Qwen3-8B`;
- revision: `b968826d9c46dd6066d109eabc6255188de91218`;
- `enable_thinking=false`;
- dtype: bfloat16;
- deterministic generation: `do_sample=false`, temperature 0;
- seed: 42;
- scope: the shared 4,000 Warm-start instances only.

Auxiliary Qwen controlled rankings support perturbation monitoring and the
reported paired-control and evidence-removal diagnostics.

## Additional UQ inference

- Decomposed UQ: five candidate-order variants per record.
- Self-consistency: five additional rankings per record, temperature 0.9,
  top-p 0.95.
- Probability family: one-letter choice distribution over valid labels A--T.
- Temperature scaling: five instance-level cross-fitted folds.
- Perturbation instability: derived from already-issued paired fixed rankings.

## Ranking-prompt diagnostic

The cached same-prompt scores use teacher-forced reconstructions of stored
rankings. The manuscript separately records a Llama greedy regeneration
check using Transformers 5.7.0. See Tables 19 and 21 for inference settings and
logged costs.

## Recorded wall times

The complete Llama self-consistency run produced 45,163 UQ rows in
approximately 121 h 53 min.  The Qwen Warm-start self-consistency run produced
4,000 rows in approximately 13 h 25 min.  These timings describe the original
execution and should not be used as hardware-normalized efficiency claims.

## Evaluation compute

All released metric recomputation is CPU-only.  It requires neither model
weights nor raw dataset text.  The compressed record artifacts total roughly
48 MB; the full evaluation loads moderate in-memory indices (tens of thousands
of records), while validation streams each file.
