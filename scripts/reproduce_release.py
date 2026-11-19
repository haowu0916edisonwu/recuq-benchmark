#!/usr/bin/env python
"""One-command CPU reproduction from the released fixed record artifacts."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
LLAMA_UQ = [
    "labelprob_max", "labelprob_top10mass", "labelprob_entropy",
    "labelprob_margin", "temperature_scaled_labelprob", "decomposed_llm_uq",
    "perturbation_rank_instability", "self_consistency",
]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def uq_paths(backbone: str) -> list[str]:
    return [str(ROOT / f"artifacts/uq_outputs/{backbone}/{name}.jsonl.gz") for name in LLAMA_UQ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_csv(
    actual: Path,
    reference: Path,
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
    *,
    actual_filter: Callable[[dict[str, str]], bool] | None = None,
    reference_subset: bool = False,
) -> None:
    actual_rows = {tuple(row[key] for key in keys): row for row in read_csv(actual)
                   if actual_filter is None or actual_filter(row)}
    reference_rows = {tuple(row[key] for key in keys): row for row in read_csv(reference)}
    if not actual_rows or (not reference_subset and actual_rows.keys() != reference_rows.keys()):
        raise AssertionError(f"Row-key mismatch: {actual} vs {reference}")
    if actual_rows.keys() - reference_rows.keys():
        raise AssertionError(f"Unexpected rows in {actual}: {sorted(actual_rows.keys() - reference_rows.keys())}")
    for key, row in actual_rows.items():
        for metric in metrics:
            left = row.get(metric, "")
            right = reference_rows[key].get(metric, "")
            if left == right:
                continue
            if not left or not right or not math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12):
                raise AssertionError(f"Mismatch {key} {metric}: {left} vs {right} ({actual})")


def compare_paper_tables(actual_dir: Path, reference_dir: Path) -> None:
    key_columns = {"backbone", "dataset", "evaluation_condition", "method", "signal"}
    for reference in sorted(reference_dir.glob("table*.csv")):
        actual = actual_dir / reference.name
        columns = tuple(read_csv(reference)[0])
        actual_rows = read_csv(actual)
        if not actual_rows or tuple(actual_rows[0]) != columns:
            raise AssertionError(f"Unexpected table columns in {actual}")
        keys = tuple(column for column in columns if column in key_columns)
        metrics = tuple(column for column in columns if column not in key_columns)
        compare_csv(actual, reference, keys, metrics)


def scalar(backbone: str, output: Path, bootstrap: int) -> None:
    base = (
        "llama_fixed_rankings.jsonl.gz" if backbone == "llama"
        else "qwen_warm_start_fixed_rankings.jsonl.gz"
    )
    command = [
        PYTHON, "scripts/evaluate_uq_methods.py",
        "--base_predictions", str(ROOT / "artifacts/base_predictions" / base),
        "--uq_predictions", *uq_paths(backbone),
        "--output_dir", str(output),
    ]
    if bootstrap:
        command.extend([
            "--bootstrap_resamples", str(bootstrap),
            "--bootstrap_scenario", "s0_warmstart",
            "--bootstrap_unit", "record",
        ])
    run(command)


def checks(output: Path) -> None:
    base = str(ROOT / "artifacts/base_predictions/llama_fixed_rankings.jsonl.gz")
    for name, script, other_inputs in (
        ("sanity_checks", "derive_uq_sanity_checks.py", []),
        ("component_checks", "derive_uq_component_checks.py", [
            "--labelprob_predictions", uq_paths("llama")[0],
            "--decomp_predictions", str(ROOT / "artifacts/uq_outputs/llama/decomposed_llm_uq.jsonl.gz"),
            "--self_consistency_predictions", str(ROOT / "artifacts/uq_outputs/llama/self_consistency.jsonl.gz"),
        ]),
    ):
        directory = output / name
        scores = directory / "scores"
        run([
            PYTHON, f"scripts/{script}", "--base_predictions", base,
            "--output_root", str(scores), "--overwrite", *other_inputs,
        ])
        run([
            PYTHON, "scripts/evaluate_uq_methods.py", "--base_predictions", base,
            "--uq_predictions", *map(str, sorted(scores.glob("*/uq_predictions.jsonl"))),
            "--output_dir", str(directory),
        ])
        reference = ROOT / "artifacts/reference_results/diagnostics" / name
        compare_csv(
            directory / "table11_error_auroc_by_method.csv", reference / "table11_error_auroc_by_method.csv",
            ("backbone", "method"), ("n", "error_auroc"),
        )


def compare_bootstrap(actual: Path, reference: Path) -> None:
    columns = ("error_auroc", "error_auprc", "risk_at_80", "aurc", "eaurc")
    bounds = tuple(f"{metric}_ci_{bound}" for metric in columns for bound in ("low", "high"))
    compare_csv(actual, reference, ("backbone", "method", "scenario"), ("n", *columns, *bounds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["quick", "full"], default="quick",
        help="quick recomputes point estimates; full also runs all paper bootstrap resamples",
    )
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/reproduced")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    run([PYTHON, "scripts/validate_release.py"])
    bootstrap = 1000 if args.mode == "full" else 0
    scalar("llama", output / "llama", bootstrap)
    scalar("qwen", output / "qwen", bootstrap)

    for gap in (0.0, 0.05, 0.10):
        command = [
            PYTHON, "scripts/evaluate_source_response_interventions.py",
            "--base_predictions", str(ROOT / "artifacts/base_predictions/llama_fixed_rankings.jsonl.gz"),
            "--uq_predictions", *uq_paths("llama"),
            "--output_dir", str(output / f"paired_control/gap_{gap:.2f}"),
            "--min_effect_gap", str(gap),
        ]
        if gap == 0.0:
            command.append("--write_instance_details")
        run(command)

    run([
        PYTHON, "scripts/derive_topk_alignment.py",
        "--base_predictions", str(ROOT / "artifacts/base_predictions/llama_fixed_rankings.jsonl.gz"),
        "--labelprob_predictions", str(ROOT / "artifacts/uq_outputs/llama/labelprob_max.jsonl.gz"),
        "--output", str(output / "topk_alignment.csv"),
    ])

    if args.mode == "full":
        for scenario, name in (
            ("s3_sparsity", "sparse_history"),
            ("s4_longtail", "tail_target_subset"),
            ("s5_robustness", "item_text_robustness"),
        ):
            run([
                PYTHON, "scripts/evaluate_uq_methods.py",
                "--base_predictions", str(ROOT / "artifacts/base_predictions/llama_fixed_rankings.jsonl.gz"),
                "--uq_predictions", *uq_paths("llama"),
                "--output_dir", str(output / f"cluster_bootstrap/{name}"),
                "--bootstrap_resamples", "200",
                "--bootstrap_scenario", scenario,
                "--bootstrap_unit", "instance",
            ])

    compare_csv(
        output / "llama/table11_error_auroc_by_method_scenario.csv",
        ROOT / "artifacts/reference_results/llama/main/table11_error_auroc_by_method_scenario.csv",
        ("backbone", "method", "scenario"),
        ("n", "error_auroc", "error_auprc", "risk_at_80", "aurc", "eaurc"),
    )
    compare_csv(
        output / "qwen/table11_error_auroc_by_method_scenario.csv",
        ROOT / "artifacts/reference_results/qwen/warm_start/table11_error_auroc_by_method_scenario.csv",
        ("backbone", "method", "scenario"),
        ("n", "error_auroc", "error_auprc", "risk_at_80", "aurc", "eaurc"),
    )
    for gap in (0.0, 0.05, 0.10):
        relative = f"paired_control/gap_{gap:.2f}/control_response_metrics.csv"
        compare_csv(
            output / relative, ROOT / "artifacts/reference_results" / relative, ("method",),
            ("n", "directional_coverage", "dominant_control_accuracy", "macro_control_f1",
             "source_auroc", "control_response_rho"),
        )
    compare_csv(
        output / "topk_alignment.csv",
        ROOT / "artifacts/reference_results/diagnostics/topk_alignment.csv",
        ("score",),
        ("n", "error_at_5_auroc", "error_at_10_auroc", "error_at_15_auroc"),
    )
    if args.mode == "full":
        references = ROOT / "artifacts/reference_results"
        for location, reference in (
            ("llama", "llama/main"), ("qwen", "qwen/warm_start"),
            *[(f"cluster_bootstrap/{name}", f"diagnostics/cluster_bootstrap/{name}") for name in (
                "sparse_history", "tail_target_subset", "item_text_robustness",
            )],
        ):
            filename = "bootstrap_metrics_ci.csv"
            compare_bootstrap(output / location / filename, references / reference / filename)
    checks(output / "diagnostics")
    supplement = [PYTHON, "scripts/reproduce_supplement.py", "--output_dir", str(output / "camera_ready")]
    if args.mode == "full":
        supplement.append("--full")
    run(supplement)
    run([
        PYTHON, "scripts/assemble_paper_tables.py", "--output_dir", str(output / "paper_tables"),
        "--llama_results", str(output / "llama"), "--qwen_results", str(output / "qwen"),
        "--paired_clipped", str(output / "paired_control/gap_0.00/control_response_metrics.csv"),
        "--paired_extended", str(output / "camera_ready/paired_control/llama/paired_control_extended.csv"),
    ])
    compare_paper_tables(output / "paper_tables", ROOT / "artifacts/reference_results/paper_tables")
    print(f"Reproduction passed; outputs are in {output}")


if __name__ == "__main__":
    main()
