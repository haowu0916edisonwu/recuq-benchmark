#!/usr/bin/env python
"""Assemble paper-facing Tables 1--8 from released record/evaluation files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.utils.io import open_text


METHOD_ORDER = [
    "labelprob_max", "labelprob_top10mass", "labelprob_entropy",
    "labelprob_margin", "temperature_scaled_labelprob", "decomposed_llm_uq",
    "perturbation_rank_instability", "self_consistency",
]
DISPLAY = {
    "labelprob_max": "LabelProb-Max",
    "labelprob_top10mass": "LabelProb-Top10Mass",
    "labelprob_entropy": "LabelProb-Entropy",
    "labelprob_margin": "LabelProb-Margin",
    "temperature_scaled_labelprob": "Temperature-scaled LabelProb",
    "decomposed_llm_uq": "Decomposed LLM UQ",
    "perturbation_rank_instability": "Perturbation Rank Instability",
    "self_consistency": "Self-consistency",
}
CANONICAL_METHOD = {
    "labelprob_llama31_8b": "labelprob_max",
    "labelprob_qwen3_8b_nonthinking": "labelprob_max",
    "derived_labelprob_base_top10_mass": "labelprob_top10mass",
    "labelprob_entropy": "labelprob_entropy",
    "labelprob_margin": "labelprob_margin",
    "labelprob_temp_scaled": "temperature_scaled_labelprob",
    "kweon_decomp_llama31_8b": "decomposed_llm_uq",
    "kweon_decomp_qwen3_8b_nonthinking": "decomposed_llm_uq",
    "perturbation_rank_instability": "perturbation_rank_instability",
    "self_consistency_s5": "self_consistency",
    "self_consistency_qwen3_s5": "self_consistency",
}
CONDITION = {
    "s0_warmstart": "Warm-start",
    "s3_sparsity": "Sparse-history",
    "s4_longtail": "Tail-target subset",
    "s5_robustness": "Item-text robustness",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open_text(path, "r") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eligible(row: dict[str, Any]) -> bool:
    return row.get("scenario") != "s4_longtail" or row.get("popularity_bucket") == "tail"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table1(base: list[dict[str, Any]]) -> list[dict[str, Any]]:
    datasets = ["mind_20", "amazon_books", "amazon_movies", "yelp"]
    rows = []
    for dataset in datasets:
        items = [row for row in base if row["dataset"] == dataset]
        warm = [row for row in items if row["scenario"] == "s0_warmstart"]
        rows.append({
            "dataset": dataset,
            "base_instances": len({row["instance_id"] for row in warm}),
            "generated_records": len(items),
            "evaluation_eligible_records": sum(eligible(row) for row in items),
        })
    return rows


def table2(base: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in base:
        if eligible(row):
            groups[(row["dataset"], row["scenario"])].append(row)
    rows = []
    for (dataset, scenario), items in groups.items():
        rows.append({
            "dataset": dataset,
            "evaluation_condition": CONDITION[scenario],
            "records": len(items),
            "hit_at_10": sum(float(row["metrics"]["hit@10"]) for row in items) / len(items),
            "ndcg_at_10": sum(float(row["metrics"]["ndcg@10"]) for row in items) / len(items),
        })
    return sorted(rows, key=lambda row: (
        ["mind_20", "amazon_books", "amazon_movies", "yelp"].index(row["dataset"]),
        list(CONDITION.values()).index(row["evaluation_condition"]),
    ))


def normalize_eval(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        method = CANONICAL_METHOD.get(row["method"])
        if method is None:
            continue
        value = dict(row)
        value["release_method"] = method
        value["display_method"] = DISPLAY[method]
        output.append(value)
    return output


def condition_table(
    rows: list[dict[str, Any]], overall: list[dict[str, Any]], scenario: str,
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {
        dataset: {} for dataset in ("mind_20", "amazon_books", "amazon_movies", "yelp", "overall")
    }
    for row in rows:
        if row["scenario"] == scenario:
            groups[row["dataset"]][row["release_method"]] = row
    for row in overall:
        if row["scenario"] == scenario:
            groups["overall"][row["release_method"]] = row

    output = []
    for dataset, matches in groups.items():
        row = {"dataset": dataset}
        if scenario == "s4_longtail":
            counts = {method["n"] for method in matches.values()}
            if len(counts) != 1:
                raise ValueError(f"Expected one shared tail-target population for {dataset}")
            (row["n"],) = counts
        row.update({method: matches[method]["error_auroc"] for method in METHOD_ORDER})
        output.append(row)
    return output


RANKING_DISTANCES = {
    "Jaccard@5": "Top-5 Jaccard distance",
    "Kendall distance": "Kendall distance",
    "RBO distance (p=0.9)": "RBO distance (p=0.9)",
    "Majority class": "Majority class",
}


def paired_control_table(clipped_path: Path, extended_path: Path) -> list[dict[str, Any]]:
    clipped_rows = read_csv(clipped_path)
    extended = {
        (row["label_mode"], row["group"], row["signal"]): row
        for row in read_csv(extended_path)
        if row["dataset"] == "all"
    }
    rows = []
    for row in clipped_rows:
        method = CANONICAL_METHOD[row["method"]]
        direct = method == "perturbation_rank_instability"
        signed = extended[
            "signed", "source-specific UQ" if direct else "scalar response (signed)", row["method"]
        ]
        rows.append({
            "method": DISPLAY[method],
            "signal": "source scores" if direct else "scalar response",
            "clipped_n": row["n"],
            "directional_coverage": row["directional_coverage"],
            "dominant_control_accuracy": row["dominant_control_accuracy"],
            "macro_f1": row["macro_control_f1"],
            "source_auroc": row["source_auroc"],
            "rho": row["control_response_rho"],
            "signed_n": signed["n"],
            "signed_dominant_control_accuracy": signed["d_acc"],
            "signed_source_auroc": signed["source_auroc"],
        })
    order = [DISPLAY[name] for name in (
        "decomposed_llm_uq", "labelprob_max", "labelprob_top10mass", "labelprob_entropy",
        "labelprob_margin", "temperature_scaled_labelprob", "self_consistency",
        "perturbation_rank_instability",
    )]
    rows.sort(key=lambda row: order.index(row["method"]))
    for name, label in RANKING_DISTANCES.items():
        group = "reference" if name == "Majority class" else "non-UQ listwise distance"
        clipped = extended["clipped", group, name]
        signed = extended["signed", group, name]
        rows.append({
            "method": label,
            "signal": group,
            "clipped_n": clipped["n"],
            "directional_coverage": clipped["dircov"],
            "dominant_control_accuracy": clipped["d_acc"],
            "macro_f1": clipped["macro_f1"],
            "source_auroc": clipped["source_auroc"],
            "rho": clipped["rho"],
            "signed_n": signed["n"],
            "signed_dominant_control_accuracy": signed["d_acc"],
            "signed_source_auroc": signed["source_auroc"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    references = REPO_ROOT / "artifacts/reference_results"
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs/paper_tables")
    parser.add_argument("--llama_results", type=Path, default=references / "llama/main")
    parser.add_argument("--qwen_results", type=Path, default=references / "qwen/warm_start")
    parser.add_argument("--paired_clipped", type=Path, default=references / "paired_control/gap_0.00/control_response_metrics.csv")
    parser.add_argument("--paired_extended", type=Path, default=references / "camera_ready/paired_control_extended_llama.csv")
    args = parser.parse_args()
    base = read_jsonl(REPO_ROOT / "artifacts/base_predictions/llama_fixed_rankings.jsonl.gz")
    write_csv(args.output_dir / "table1_benchmark_overview.csv", table1(base))
    write_csv(args.output_dir / "table2_base_ranking_quality.csv", table2(base))

    datasets_csv = "table11_error_auroc_by_method_dataset_scenario.csv"
    overall_csv = "table11_error_auroc_by_method_scenario.csv"
    llama_dataset = normalize_eval(read_csv(args.llama_results / datasets_csv))
    qwen_dataset = normalize_eval(read_csv(args.qwen_results / datasets_csv))
    llama_overall = normalize_eval(read_csv(args.llama_results / overall_csv))
    qwen_overall = normalize_eval(read_csv(args.qwen_results / overall_csv))
    backbones = (
        ("Llama-3.1-8B-Instruct", llama_dataset, llama_overall),
        ("Qwen3-8B non-thinking", qwen_dataset, qwen_overall),
    )
    table3 = []
    for backbone, rows, overall in backbones:
        selected = [row for row in rows if row["scenario"] == "s0_warmstart"]
        for dataset in ["mind_20", "amazon_books", "amazon_movies", "yelp"]:
            matches = {row["release_method"]: row for row in selected if row["dataset"] == dataset}
            table3.append({"backbone": backbone, "dataset": dataset, **{
                method: matches[method]["error_auroc"] for method in METHOD_ORDER
            }})
        matches = {row["release_method"]: row for row in overall if row["scenario"] == "s0_warmstart"}
        table3.append({"backbone": backbone, "dataset": "overall", **{
            method: matches[method]["error_auroc"] for method in METHOD_ORDER
        }})
    write_csv(args.output_dir / "table3_warm_start_auroc.csv", table3)

    table4 = []
    for backbone, _, rows in backbones:
        for row in rows:
            if row["scenario"] == "s0_warmstart":
                table4.append({
                    "backbone": backbone,
                    "method": row["display_method"],
                    "error_auprc": row["error_auprc"],
                    "risk_at_80": row["risk_at_80"],
                    "aurc": row["aurc"],
                    "eaurc": row["eaurc"],
                })
    table4.sort(key=lambda row: (["Llama-3.1-8B-Instruct", "Qwen3-8B non-thinking"].index(row["backbone"]), METHOD_ORDER.index(next(key for key, value in DISPLAY.items() if value == row["method"]))))
    write_csv(args.output_dir / "table4_warm_start_complementary_metrics.csv", table4)
    write_csv(args.output_dir / "table5_sparse_history_auroc.csv", condition_table(llama_dataset, llama_overall, "s3_sparsity"))
    write_csv(args.output_dir / "table6_tail_target_subset_auroc.csv", condition_table(llama_dataset, llama_overall, "s4_longtail"))
    write_csv(args.output_dir / "table7_item_text_robustness_auroc.csv", condition_table(llama_dataset, llama_overall, "s5_robustness"))

    write_csv(
        args.output_dir / "table8_paired_control_response.csv",
        paired_control_table(args.paired_clipped, args.paired_extended),
    )
    print(f"Wrote Tables 1--8 to {args.output_dir}")


if __name__ == "__main__":
    main()
