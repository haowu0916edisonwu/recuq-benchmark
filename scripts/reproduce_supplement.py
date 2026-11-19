#!/usr/bin/env python3
"""Recompute and check camera-ready diagnostics from the released artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from reproduce_release import ROOT, PYTHON, compare_csv, read_csv, run, uq_paths


ARTIFACTS = ROOT / "artifacts"
REF = ARTIFACTS / "reference_results/camera_ready"
LLAMA = ARTIFACTS / "base_predictions/llama_fixed_rankings.jsonl.gz"
QWEN = ARTIFACTS / "base_predictions/qwen_warm_start_fixed_rankings.jsonl.gz"
QWEN_CONTROLS = ARTIFACTS / "supplement/qwen_controlled_rankings.jsonl.gz"


def paired(output: Path, full: bool) -> None:
    keys = ("label_mode", "group", "signal", "dataset")
    metrics = ("n", "dircov", "d_acc", "macro_f1", "source_auroc", "rho")
    intervals = ("d_acc_ci_low", "d_acc_ci_high", "source_auroc_ci_low", "source_auroc_ci_high")
    for backbone, name, base in (("llama", "llama31_8b", LLAMA), ("qwen", "qwen3_8b_nonthinking", QWEN_CONTROLS)):
        folder = output / "paired_control" / backbone
        command = [
            PYTHON, "scripts/evaluate_paired_control_extended.py",
            "--backbone", name, "--base_predictions", str(base),
            "--output_dir", str(folder), "--bootstrap_resamples", "1000" if full and backbone == "llama" else "0",
        ]
        if backbone == "llama":
            command += ["--uq_predictions", *uq_paths("llama"), "--labelprob_predictions", uq_paths("llama")[0]]
        run(command)
        compare_csv(
            folder / "paired_control_extended.csv", REF / f"paired_control_extended_{backbone}.csv",
            keys, metrics + (intervals if full and backbone == "llama" else ()),
        )
        actual = folder / "restoration_effect_distribution.csv"
        compare_csv(
            actual, REF / f"restoration_effect_distribution_{backbone}.csv",
            ("dataset",), tuple(k for k in read_csv(actual)[0] if k not in {"dataset", "backbone"}),
        )


def evaluate(base: Path, inputs: list[Path], output: Path, full: bool = False) -> None:
    run([
        PYTHON, "scripts/evaluate_uq_methods.py", "--base_predictions", str(base),
        "--uq_predictions", *map(str, inputs), "--output_dir", str(output),
        "--bootstrap_resamples", "1000" if full else "0", "--bootstrap_scenario", "s0_warmstart",
    ])


def scalar_diagnostics(output: Path, full: bool) -> None:
    keys = ("method",)
    for backbone, base in (("llama", LLAMA), ("qwen", QWEN)):
        reference_model = "llama31_8b" if backbone == "llama" else "qwen3_8b_nonthinking"
        uq = ARTIFACTS / "uq_outputs" / backbone
        temperatures = output / "shared_temperature" / backbone
        run([
            PYTHON, "scripts/derive_shared_temperature.py", "--base_predictions", str(base),
            "--labelprob", str(uq / "labelprob_max.jsonl.gz"), "--output_root", str(temperatures / "scores"),
        ])
        for name, scores, reference in (
            ("shared_temperature", sorted((temperatures / "scores").glob("*/uq_predictions.jsonl")),
             f"shared_temperature_{backbone}_warm_ci.csv"),
            ("same_prompt", sorted((ARTIFACTS / "supplement/same_prompt" / backbone).glob("*.gz"))
             + [uq / "labelprob_top10mass.jsonl.gz"], f"alignment_and_same_inference_{backbone}_warm_ci.csv"),
            ("self_consistency", sorted((ARTIFACTS / "supplement/self_consistency" / backbone).glob("*.gz"))
             + [uq / "self_consistency.jsonl.gz"], f"sc_base_anchored_{backbone}_warm_ci.csv"),
        ):
            folder = output / name / backbone / "metrics"
            evaluate(base, scores, folder, full=full)
            actual_path = folder / ("bootstrap_warm_metrics_ci.csv" if full else "table11_error_auroc_by_method_scenario.csv")
            values = ("n", "error_auroc") + (("error_auroc_ci_low", "error_auroc_ci_high") if full else ())
            compare_csv(
                actual_path, REF / reference, keys, values,
                actual_filter=lambda row: row["scenario"] == "s0_warmstart",
                reference_subset=True,
            )
            if name == "self_consistency" and backbone == "llama":
                actual = {(r["method"], r["scenario"]): r for r in read_csv(folder / "table11_error_auroc_by_method_scenario.csv")}
                for row in read_csv(REF / "sc_base_anchored_llama_by_condition.csv"):
                    for field, scenario in (("warm", "s0_warmstart"), ("sparse", "s3_sparsity"),
                                            ("long_tail", "s4_longtail"), ("robust", "s5_robustness")):
                        got = actual[row["method"], scenario]
                        if got["n"] != row[f"{field}_n"] or abs(float(got["error_auroc"]) - float(row[field])) > 1e-12:
                            raise AssertionError(f"{reference_model} self-consistency mismatch for {row['method']} / {scenario}")


def evidence_removal(output: Path) -> None:
    folder = output / "evidence_removal"
    run([
        PYTHON, "scripts/derive_revision_diagnostics.py", "positional",
        "--backbone", f"llama31_8b,{LLAMA}", "--backbone", f"qwen3_8b_nonthinking,{QWEN_CONTROLS}",
        "--stratify", ",".join(["llama31_8b", str(LLAMA), *uq_paths("llama"),
                                *map(str, sorted((ARTIFACTS / "supplement/same_prompt/llama").glob("*.gz")))]),
        "--stratify", ",".join(["qwen3_8b_nonthinking", str(QWEN), *uq_paths("qwen"),
                                *map(str, sorted((ARTIFACTS / "supplement/same_prompt/qwen").glob("*.gz")))]),
        "--output_dir", str(folder),
    ])
    for filename, keys in (
        ("positional_by_variant.csv", ("backbone", "dataset", "scenario", "variant")),
        ("itemtext_variant_composition.csv", ("backbone", "dataset", "variant")),
        ("warm_error_auroc_by_positional_fallback.csv", ("backbone", "method")),
    ):
        path = folder / filename
        compare_csv(path, REF / filename, keys, tuple(k for k in read_csv(path)[0] if k not in keys), reference_subset=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "outputs/reproduced/camera_ready")
    parser.add_argument("--full", action="store_true", help="also check reported bootstrap intervals")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    paired(output, args.full)
    scalar_diagnostics(output, args.full)
    evidence_removal(output)
    print(f"Camera-ready diagnostics passed; outputs are in {output}")


if __name__ == "__main__":
    main()
