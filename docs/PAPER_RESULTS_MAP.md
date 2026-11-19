# Camera-ready results map

The table and figure numbers follow the camera-ready manuscript. Metric and figure generation run on CPU using the released rankings and scores. Table 21 records the original inference timings.

## Main paper

| Item | Reference | Reproduction |
| --- | --- | --- |
| Table 1 · Benchmark overview | [CSV](../artifacts/reference_results/paper_tables/table1_benchmark_overview.csv) | `make paper-tables` |
| Table 2 · Base ranking quality | [CSV](../artifacts/reference_results/paper_tables/table2_base_ranking_quality.csv) | `make paper-tables` |
| Table 3 · Warm-start Error-AUROC | [CSV](../artifacts/reference_results/paper_tables/table3_warm_start_auroc.csv) | `make reproduce` |
| Table 4 · Complementary reliability metrics | [CSV](../artifacts/reference_results/paper_tables/table4_warm_start_complementary_metrics.csv) | `make reproduce` |
| Table 5 · Sparse-history Error-AUROC | [CSV](../artifacts/reference_results/paper_tables/table5_sparse_history_auroc.csv) | `make reproduce` |
| Table 6 · Tail-target Error-AUROC | [CSV](../artifacts/reference_results/paper_tables/table6_tail_target_subset_auroc.csv) | `make reproduce` |
| Table 7 · Item-text robustness Error-AUROC | [CSV](../artifacts/reference_results/paper_tables/table7_item_text_robustness_auroc.csv) | `make reproduce` |
| Table 8 · Clipped and signed paired-control agreement | [CSV](../artifacts/reference_results/paper_tables/table8_paired_control_response.csv) | `make reproduce` |

## Appendix

| Item | Reference | Reproduction |
| --- | --- | --- |
| Table 9 · UQ schema | [Artifact contract](ARTIFACT_CONTRACT.md) and [JSON schemas](../artifacts/schemas/) | `make validate` |
| Table 10 · Shared fixed temperatures | [Llama](../artifacts/reference_results/camera_ready/shared_temperature_llama_warm_ci.csv), [Qwen](../artifacts/reference_results/camera_ready/shared_temperature_qwen_warm_ci.csv) | `make reproduce` |
| Table 11 · Error@K / TopKMass alignment | [CSV](../artifacts/reference_results/diagnostics/topk_alignment.csv) | `make reproduce` |
| Table 12 · Instance-clustered bootstrap | [Reference intervals](../artifacts/reference_results/diagnostics/cluster_bootstrap/) | `make reproduce-full` |
| Table 13 · Paired-control gap sensitivity | [Results by threshold](../artifacts/reference_results/paired_control/) | `make reproduce` |
| Table 14 · Error-AUROC sanity checks | [Reference results](../artifacts/reference_results/diagnostics/sanity_checks/) | `make reproduce` |
| Table 15 · Ranking alignment and component checks | [Reference results](../artifacts/reference_results/diagnostics/component_checks/) | `make reproduce` |
| Table 16 · Signed restoration differences and clipping | [Llama](../artifacts/reference_results/camera_ready/restoration_effect_distribution_llama.csv), [Qwen](../artifacts/reference_results/camera_ready/restoration_effect_distribution_qwen.csv) | `make reproduce` |
| Table 17 · Supplementary paired-control results | [Llama](../artifacts/reference_results/camera_ready/paired_control_extended_llama.csv), [Qwen](../artifacts/reference_results/camera_ready/paired_control_extended_qwen.csv) | `make reproduce-full` |
| Table 18 · Evidence-removal diagnostics | [Positional outputs](../artifacts/reference_results/camera_ready/positional_by_variant.csv), [field availability](../artifacts/reference_results/camera_ready/itemtext_variant_composition.csv) | `make reproduce` |
| Table 19 · Same-prompt and cross-elicitation scores | [Llama](../artifacts/reference_results/camera_ready/alignment_and_same_inference_llama_warm_ci.csv), [Qwen](../artifacts/reference_results/camera_ready/alignment_and_same_inference_qwen_warm_ci.csv) | `make reproduce-full` |
| Table 20 · Base-anchored self-consistency | [Llama by condition](../artifacts/reference_results/camera_ready/sc_base_anchored_llama_by_condition.csv), [Llama Warm-start](../artifacts/reference_results/camera_ready/sc_base_anchored_llama_warm_ci.csv), [Qwen Warm-start](../artifacts/reference_results/camera_ready/sc_base_anchored_qwen_warm_ci.csv) | `make reproduce-full` |
| Table 21 · Inference calls and timings | [Original timings](../artifacts/reference_results/camera_ready/cost.csv) | Archived GPU measurements |

## Figures

| Item | Camera-ready PDF | Reproduction |
| --- | --- | --- |
| Figure 1 · Benchmark pipeline | [PDF](../figures/reference/fig_benchmark_pipeline.pdf) | Static diagram |
| Figure 2 · Ranking-failure detection | [PDF](../figures/reference/fig_backbone_view_reliability.pdf) | `make figures` |
| Figure 3 · Item-content case | [PDF](../figures/reference/fig_case_study_item_ablation.pdf) | `make figures` renders the released case values |
| Figure 4 · Paired-control agreement | [PDF](../figures/reference/fig_paired_control_response.pdf) | `make figures` |
| Figure 5 · Signed NDCG@10 differences | [PDF](../figures/reference/fig_signed_restoration_effects.pdf) | `make figures` |
