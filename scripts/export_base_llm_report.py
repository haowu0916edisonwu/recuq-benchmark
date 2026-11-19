#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DATASET_NAMES = {
    "mind_20": "MIND-20",
    "amazon_books": "Amazon Books",
    "amazon_movies": "Amazon Movies",
    "yelp": "Yelp",
}

SCENARIO_NAMES = {
    "s0_warmstart": "S0 Warm",
    "s3_sparsity": "S3 Sparse",
    "s4_longtail": "S4 Long-tail",
    "s5_robustness": "S5 Robust",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a human-readable base LLM run report.")
    parser.add_argument("result_dir", type=str, help="Directory containing fixed LLM ranking outputs.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Markdown output path. Defaults to <result_dir>/base_llm_report.md.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_path = Path(args.output) if args.output else result_dir / "base_llm_report.md"

    run_summary = read_json(result_dir / "run_summary.json")
    main_rows = read_csv(result_dir / "main_metrics_by_dataset_scenario.csv")
    s3_rows = read_csv(result_dir / "s3_metrics_by_history_variant.csv")
    s4_rows = read_csv(result_dir / "s4_metrics_by_popularity_bucket.csv")
    s5_rows = read_csv(result_dir / "s5_metrics_by_ablated_field.csv")
    parse_rows = read_csv(result_dir / "parse_quality_by_dataset_scenario.csv")

    lines: list[str] = []
    lines.extend(render_header(run_summary))
    lines.extend(render_base_main_table(main_rows))
    lines.extend(render_s3_table(s3_rows))
    lines.extend(render_s4_table(s4_rows))
    lines.extend(render_s5_table(s5_rows))
    lines.extend(render_parse_table(parse_rows, run_summary))
    lines.extend(render_proposal_mapping())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote report to {output_path}")


def render_header(summary: dict[str, Any]) -> list[str]:
    return [
        "# Base LLM Benchmark Report",
        "",
        "## Run Summary",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Model | Llama-3.1-8B-Instruct |",
        f"| Result directory | `{summary.get('result_dir', '')}` |",
        f"| Total records | {int(summary.get('total_records', 0)):,} |",
        f"| Strict parse success | {fmt(summary.get('strict_parse_success_rate', 0.0))} |",
        f"| Deterministic repair rate | {fmt(summary.get('repair_rate', 0.0))} |",
        f"| Fallback rate | {fmt(summary.get('fallback_rate', 0.0))} |",
        "",
        "This run is the base LLM ranking substrate. It provides ranking metrics, parse diagnostics, "
        "error labels, scenario metadata, and unique `record_id`s for later UQ methods.",
        "",
        "It is not yet the final UQ table: Error-AUROC, paired-control response, and matched-restoration "
        "results require additional UQ method outputs joined by `record_id`.",
        "",
    ]


def render_base_main_table(rows: list[dict[str, str]]) -> list[str]:
    by_dataset: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["scenario"]] = row

    lines = [
        "## Base Ranking Main Table",
        "",
        "Values are base LLM recommendation quality on the fixed evaluation manifest.",
        "",
        "| Dataset | S0 Hit@10 | S3 Hit@10 | S4 Hit@10 | S5 Hit@10 | S0 NDCG | S3 NDCG | S4 NDCG | S5 NDCG | Parse min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in ["mind_20", "amazon_books", "amazon_movies", "yelp"]:
        d = by_dataset[dataset]
        parse_min = min(float(row["parse_success_rate"]) for row in d.values())
        lines.append(
            f"| {DATASET_NAMES[dataset]} "
            f"| {metric(d, 's0_warmstart', 'hit@10')} "
            f"| {metric(d, 's3_sparsity', 'hit@10')} "
            f"| {metric(d, 's4_longtail', 'hit@10')} "
            f"| {metric(d, 's5_robustness', 'hit@10')} "
            f"| {metric(d, 's0_warmstart', 'ndcg@10')} "
            f"| {metric(d, 's3_sparsity', 'ndcg@10')} "
            f"| {metric(d, 's4_longtail', 'ndcg@10')} "
            f"| {metric(d, 's5_robustness', 'ndcg@10')} "
            f"| {fmt(parse_min)} |"
        )
    lines.append("")
    return lines


def render_s3_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## S3 Sparsity Dose Response",
        "",
        "| Dataset | Variant | n | Hit@10 | NDCG@10 | MRR@10 | Top1 | Parse |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"truncate_1": 0, "truncate_3": 1, "truncate_5": 2, "truncate_10": 3, "full": 4}
    rows = sorted(rows, key=lambda r: (r["dataset"], order.get(r["variant_name"], 99)))
    for row in rows:
        lines.append(render_metric_row(row, "variant_name"))
    lines.append("")
    return lines


def render_s4_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## S4 Long-tail Analysis",
        "",
        "| Dataset | Bucket | n | Hit@10 | NDCG@10 | MRR@10 | Top1 | Parse |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"head": 0, "mid": 1, "tail": 2}
    rows = sorted(rows, key=lambda r: (r["dataset"], order.get(r["popularity_bucket"], 99)))
    for row in rows:
        lines.append(render_metric_row(row, "popularity_bucket"))
    lines.append("")
    return lines


def render_s5_table(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "## S5 Robustness Ablations",
        "",
        "| Dataset | Ablated field | n | Hit@10 | NDCG@10 | MRR@10 | Top1 | Parse |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = sorted(rows, key=lambda r: (r["dataset"], r["ablated_field"]))
    for row in rows:
        lines.append(render_metric_row(row, "ablated_field"))
    lines.append("")
    return lines


def render_parse_table(rows: list[dict[str, str]], summary: dict[str, Any]) -> list[str]:
    lines = [
        "## Parse Quality",
        "",
        f"Overall strict parse success is {fmt(summary.get('strict_parse_success_rate', 0.0))}. "
        f"All strict failures repaired deterministically: fallback rate is {fmt(summary.get('fallback_rate', 0.0))}.",
        "",
        "| Dataset | Scenario | n | Strict parse | Repair rate | Fallback rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    rows = sorted(rows, key=lambda r: (r["dataset"], r["scenario"]))
    for row in rows:
        lines.append(
            f"| {DATASET_NAMES.get(row['dataset'], row['dataset'])} "
            f"| {SCENARIO_NAMES.get(row['scenario'], row['scenario'])} "
            f"| {int(float(row['n'])):,} "
            f"| {fmt(row['strict_parse_success_rate'])} "
            f"| {fmt(row['repair_rate'])} "
            f"| {fmt(row['fallback_rate'])} |"
        )
    lines.append("")
    return lines


def render_proposal_mapping() -> list[str]:
    return [
        "## Mapping to Proposal Tables",
        "",
        "| Proposal table | Current status | What is still needed |",
        "|---|---|---|",
        "| Table 11: Error-AUROC by scenario | Base error labels are ready: `is_error_at_10`, `error_at_10` | UQ methods must output `uncertainty_score`; compute AUROC by `record_id` |",
        "| Table 12: Paired-control response | Scenario/source metadata are ready | UQ methods may output separate user/item control scores |",
        "| Table 13: Matched intervention | Base records and scenario metadata are ready | Need interventions conditioned on diagnosed source and compare NDCG lift |",
        "",
        "Available current scenarios: S0 Warm, S3 Sparse, S4 Long-tail, S5 Robust. "
        "Proposal S1 User Cold and S2 Item Cold are not implemented in the current run.",
        "",
    ]


def render_metric_row(row: dict[str, str], group_key: str) -> str:
    return (
        f"| {DATASET_NAMES.get(row['dataset'], row['dataset'])} "
        f"| {row[group_key]} "
        f"| {int(float(row['n'])):,} "
        f"| {fmt(row['hit@10'])} "
        f"| {fmt(row['ndcg@10'])} "
        f"| {fmt(row['mrr@10'])} "
        f"| {fmt(row['top1_accuracy'])} "
        f"| {fmt(row['parse_success_rate'])} |"
    )


def metric(rows_by_scenario: dict[str, dict[str, str]], scenario: str, key: str) -> str:
    return fmt(rows_by_scenario[scenario][key])


def fmt(value: Any) -> str:
    return f"{float(value):.3f}"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
