# Regenerate rankings and uncertainty scores

This workflow starts from [reconstructed source data](DATA_RECONSTRUCTION.md)
and produces new inference outputs. Use the [CPU evaluation guide](../REPRODUCIBILITY.md)
to recompute the paper's numbers from the original frozen outputs.

## 1. Set up the model environment

The recorded runs used an NVIDIA A100 80 GB. Use the checked-in CUDA 12.1,
PyTorch 2.5.1, and Transformers 4.51.3 environment for Qwen. The manuscript
identifies Transformers 5.7.0 for the original Llama generation. The separate
Llama environment below also matches the recorded teacher-forcing software
versions.

```bash
conda env create -f environment-tmlr.yml
conda activate recuq-benchmark-tmlr
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
hf auth login

conda create -n recuq-llama python=3.10 pip
conda activate recuq-llama
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-eval.txt transformers==5.7.0 accelerate==1.3.0
python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

Request access to Llama on its model page before running it. Both backbones use
bfloat16 and greedy decoding for fixed rankings. Qwen uses
`enable_thinking=false` and the pinned revision
`b968826d9c46dd6066d109eabc6255188de91218`. The archived Llama base run records
the model ID. Its exact resolved base snapshot was not logged, so regenerated
rankings can differ from the frozen rankings. Section 5 gives the snapshot
recorded for the Llama teacher-forcing diagnostic.
[Llama model card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) ·
[Qwen model snapshot](https://huggingface.co/Qwen/Qwen3-8B/tree/b968826d9c46dd6066d109eabc6255188de91218) ·
[PyTorch 2.5.1 installation](https://pytorch.org/get-started/previous-versions/#v251)

## 2. Check data and generate fixed rankings

```bash
python scripts/check_reconstructed_inputs.py

conda activate recuq-llama
python scripts/generate_fixed_rankings.py \
  --config configs/experiment/llama_controlled_suite.yaml \
  --output_dir results/phase1_base_llm_aligned20_n1000

conda activate recuq-benchmark-tmlr
python scripts/generate_fixed_rankings.py \
  --config configs/experiment/tmlr_qwen3_nonthinking_warmstart.yaml
python scripts/generate_fixed_rankings.py \
  --config configs/experiment/tmlr_qwen3_nonthinking_perturbation.yaml
```

Each output directory contains `raw_predictions.jsonl` and `run_metadata.json`.
Expect 45,163 Llama rankings, 4,000 Qwen Warm-start rankings, and 41,163
auxiliary Qwen rankings for the Warm-start, history, and item-text conditions.
The Llama table keeps all 4,000 S4 rows for provenance. Scalar evaluation uses
only the 1,312 tail items in S4, for 42,475 eligible Llama records overall.
Keep raw predictions and caches local. The UQ inference scripts use their
stored prompt and candidate checks and, for teacher forcing, the stored model
response.

## 3. Generate the eight UQ monitors

Set one of the following blocks, then run the shared commands. Repeat with
the other block to generate both backbones. The file layout also supports
the bundled public-artifact export scripts.

**Llama**

```bash
conda activate recuq-llama
config=configs/experiment/llama_controlled_suite.yaml
base=results/phase1_base_llm_aligned20_n1000/raw_predictions.jsonl
uq=results/uq
probe_dir=labelprob_llama31_8b
probe_method=labelprob_llama31_8b
decomp_dir=kweon_decomp_llama31_8b
decomp_method=kweon_decomp_llama31_8b
sc_method=self_consistency_s5
teacher=results/uq/listwise_same_inference_llama31_8b
alignment_name=llama31_8b
eval_out=results/uq_eval/rebuilt_llama
```

**Qwen**

```bash
conda activate recuq-benchmark-tmlr
config=configs/experiment/tmlr_qwen3_nonthinking_warmstart.yaml
base=results/qwen3_8b_nonthinking/warm_start/raw_predictions.jsonl
uq=results/uq/qwen3_8b_nonthinking
probe_dir=labelprob_max
probe_method=labelprob_qwen3_8b_nonthinking
decomp_dir=kweon_decomp
decomp_method=kweon_decomp_qwen3_8b_nonthinking
sc_method=self_consistency_qwen3_s5
teacher=results/uq/qwen3_8b_nonthinking/listwise_same_inference
alignment_name=qwen3_8b_nonthinking
eval_out=results/uq_eval/rebuilt_qwen
```

**Shared commands**

```bash
subset=configs/eval_subsets/aligned20_n1000_seed42.jsonl
probe="$uq/$probe_dir/uq_predictions.jsonl"
sc_dir="$uq/self_consistency_s5_t09_p095"

python scripts/run_uq_labelprob.py \
  --config "$config" --base_predictions "$base" --subset_manifest "$subset" \
  --output_dir "$uq/$probe_dir" --method "$probe_method" --batch_size 4
python scripts/derive_uq_from_labelprob.py \
  --input "$probe" --output_root "$uq" --methods entropy margin
python scripts/derive_uq_component_checks.py \
  --base_predictions "$base" --labelprob_predictions "$probe" \
  --methods derived_labelprob_base_top10_mass --output_root "$uq"
python scripts/derive_temp_scaled_labelprob.py \
  --base_predictions "$base" --labelprob_predictions "$probe" \
  --folds 5 --split_key instance_id --output_dir "$uq/labelprob_temp_scaled"
python scripts/run_uq_kweon_decomp.py \
  --config "$config" --base_predictions "$base" --subset_manifest "$subset" \
  --output_dir "$uq/$decomp_dir" --method "$decomp_method" \
  --num_prompt_variants 5 --batch_size 4 --record_batch_size 1 --seed 42
python scripts/run_uq_self_consistency.py \
  --config "$config" --base_predictions "$base" --subset_manifest "$subset" \
  --output_dir "$sc_dir" --method "$sc_method" --cache_dir "$sc_dir/cache" \
  --num_samples 5 --temperature 0.9 --top_p 0.95 --top_k 10 \
  --sampling_mode sequential --seed 42
```

One LabelProb inference pass supplies Max, Entropy, Margin, Top10Mass, and
temperature scaling. Top10Mass sums the one-letter probabilities on the
**issued** ranking's top 10 candidate indices. Temperature scaling fits on
four instance-level folds and scores the held-out fold. Decomposition uses
five permuted candidate orders. Self-consistency compares the separately
sampled rankings.

For a CPU check of regenerated raw rankings against the reconstructed inputs,
add `--validate_only` to the LabelProb, decomposition, or self-consistency
command. That mode checks record coverage and hashes without loading model
weights. For an initial GPU inference check, use `--limit` and a separate
output directory. If you change model revisions or decoding settings, use a
separate ranking cache directory. Keep published reference artifacts under
`artifacts/`.

**Perturbation Rank Instability**

Run both commands once. The Qwen auxiliary run supplies the control rankings,
and the Qwen UQ file emits scores on the 4,000 Warm-start records.

```bash
python scripts/derive_perturbation_rank_instability.py \
  --base_predictions results/phase1_base_llm_aligned20_n1000/raw_predictions.jsonl \
  --output_dir results/uq/perturbation_rank_instability --top_k 10

python scripts/derive_perturbation_rank_instability.py \
  --base_predictions results/qwen3_8b_nonthinking/perturbation_controls/raw_predictions.jsonl \
  --output_dir results/uq/qwen3_8b_nonthinking/perturbation_rank_instability \
  --top_k 10 --emit_scenario s0_warmstart
```

## 4. Evaluate the newly generated rows

With the chosen backbone variables from Section 3, evaluate the eight output
directories. The expected join coverage is 100% before applying the S4 filter.

```bash
uq_inputs=(
  "$uq/$probe_dir" "$uq/derived_labelprob_base_top10_mass"
  "$uq/labelprob_entropy" "$uq/labelprob_margin" "$uq/labelprob_temp_scaled"
  "$uq/$decomp_dir" "$uq/perturbation_rank_instability" "$sc_dir"
)
python scripts/evaluate_uq_methods.py --base_predictions "$base" \
  --uq_predictions "${uq_inputs[@]}" --output_dir "$eval_out" \
  --bootstrap_resamples 1000 --bootstrap_scenario s0_warmstart --bootstrap_unit record
```

Use the Llama variables and the same `uq_inputs` definition to compute the
instance-clustered intervals, clipped response metrics, and signed effects.

```bash
for scenario in s3_sparsity s4_longtail s5_robustness; do
  python scripts/evaluate_uq_methods.py --base_predictions "$base" \
    --uq_predictions "${uq_inputs[@]}" --output_dir "$eval_out/cluster/$scenario" \
    --bootstrap_resamples 200 --bootstrap_scenario "$scenario" --bootstrap_unit instance
done

for gap in 0.00 0.05 0.10; do
  python scripts/evaluate_source_response_interventions.py --base_predictions "$base" \
    --uq_predictions "${uq_inputs[@]}" --output_dir "outputs/rebuilt/paired_control/gap_$gap" \
    --min_effect_gap "$gap" --write_instance_details
done

python scripts/evaluate_paired_control_extended.py --base_predictions "$base" \
  --backbone llama31_8b --uq_predictions "${uq_inputs[@]}" \
  --labelprob_predictions "$probe" --output_dir outputs/rebuilt/signed/llama \
  --bootstrap_resamples 1000 --bootstrap_seed 42

python scripts/evaluate_paired_control_extended.py \
  --base_predictions results/qwen3_8b_nonthinking/perturbation_controls/raw_predictions.jsonl \
  --backbone qwen3_8b_nonthinking --output_dir outputs/rebuilt/signed/qwen \
  --bootstrap_resamples 0
```

## 5. Regenerate the appendix scores

Table 19 uses probabilities reconstructed by teacher forcing under the
ranking prompt. The Qwen run used the main environment and its pinned model
revision. The Llama run used Transformers 5.7.0, incremental KV-cache
scoring, and model revision `0e9e39f249a16976918f6564b8830bc894c89659`.
Use a separate environment for that Transformers version.

```bash
# Qwen.
conda activate recuq-benchmark-tmlr
python scripts/run_uq_listwise_same_inference.py \
  --config configs/experiment/tmlr_qwen3_nonthinking_warmstart.yaml \
  --base_predictions results/qwen3_8b_nonthinking/warm_start/raw_predictions.jsonl \
  --subset_manifest configs/eval_subsets/aligned20_n1000_seed42.jsonl \
  --output_dir results/uq/qwen3_8b_nonthinking/listwise_same_inference \
  --method listwise_same_inference_qwen3_8b_nonthinking --batch_size 1

# Llama.
conda activate recuq-llama
python scripts/run_uq_listwise_same_inference.py \
  --config configs/experiment/llama_controlled_suite.yaml \
  --base_predictions results/phase1_base_llm_aligned20_n1000/raw_predictions.jsonl \
  --subset_manifest configs/eval_subsets/aligned20_n1000_seed42.jsonl \
  --output_dir results/uq/listwise_same_inference_llama31_8b \
  --method listwise_same_inference_llama31_8b --batch_size 1 \
  --scenarios s0_warmstart --incremental \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659
conda activate recuq-benchmark-tmlr
```

For the derived rows, set the appropriate Section 3 variables again.

```bash
aligned="results/uq/revision_alignment/$alignment_name"
python scripts/derive_revision_diagnostics.py alignment \
  --base_predictions "$base" --labelprob "$probe" \
  --temp_scaled "$uq/labelprob_temp_scaled" --decomp "$uq/$decomp_dir" \
  --same_inference "$teacher" --output_root "$aligned"
python scripts/derive_sc_base_anchored.py \
  --base_predictions "$base" --sc_predictions "$sc_dir/uq_predictions.jsonl" \
  --output_root "$uq" --top_k 10
python scripts/derive_shared_temperature.py \
  --base_predictions "$base" --labelprob "$probe" --output_root "$uq/shared_temperature"

python scripts/evaluate_uq_methods.py --base_predictions "$base" \
  --uq_predictions "$uq"/shared_temperature/shared_T* "$uq/$decomp_dir" \
  --output_dir "$eval_out/shared_temperature"
python scripts/evaluate_uq_methods.py --base_predictions "$base" \
  --uq_predictions "$uq/derived_labelprob_base_top10_mass" \
    "$aligned/alignment_probe_own_top10_mass" \
    "$aligned"/same_inference_{slot1_top10mass,slot1_max,seq_nll_top10} \
  --output_dir "$eval_out/same_prompt" --bootstrap_resamples 1000 \
  --bootstrap_scenario s0_warmstart --bootstrap_unit record
python scripts/evaluate_uq_methods.py --base_predictions "$base" \
  --uq_predictions "$sc_dir" "$uq/self_consistency_base_anchored" "$uq/self_consistency_plus_base" \
  --output_dir "$eval_out/sc_base" --bootstrap_resamples 1000 \
  --bootstrap_scenario s0_warmstart --bootstrap_unit record
```

These commands export the evaluator CSVs for Tables 10, 19, and 20. The
[paper results map](PAPER_RESULTS_MAP.md) links the remaining CPU diagnostics
and their reference outputs.

## 6. Export portable records

After completing both backbones, the following commands write sanitized ranking
and UQ JSONL files and their manifests under `outputs/`.

```bash
python scripts/build_record_artifacts.py --source_root . --output_root outputs/rebuilt/artifacts
python scripts/build_supplement_artifacts.py --source_root . --output_root outputs/rebuilt/artifacts
```

These outputs are for analyzing the new inference runs. The committed artifacts
remain the source for exact CPU reproduction of the camera-ready results.
