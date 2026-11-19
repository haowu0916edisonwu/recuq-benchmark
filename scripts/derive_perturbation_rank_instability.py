#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION, normalize_uq_prediction
from recuq_benchmark.utils.logging import get_logger


METHOD = "perturbation_rank_instability"
METHOD_FAMILY = "sensitivity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a benchmark-native perturbation-instability UQ baseline from "
            "existing base LLM rankings. This does not rerun the LLM; it reuses "
            "paired S3/S5 challenge variants already present in the benchmark."
        )
    )
    parser.add_argument(
        "--base_predictions",
        type=str,
        required=True,
        help="Base LLM raw_predictions.jsonl containing rankings for all challenge variants.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/uq/perturbation_rank_instability",
        help="Output directory for uq_predictions.jsonl and run_metadata.json.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-k ranking prefix used for Jaccard instability.",
    )
    parser.add_argument(
        "--emit_scenario",
        action="append",
        default=None,
        help=(
            "Only emit rows for this scenario after computing case statistics from "
            "all input records. Repeat to retain multiple scenarios. By default, "
            "rows for every input scenario are emitted."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing uq_predictions.jsonl.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("derive_perturbation_rank_instability")
    base_path = Path(args.base_predictions)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "uq_predictions.jsonl"

    if args.top_k < 1:
        raise ValueError("--top_k must be positive")
    if not base_path.exists():
        raise FileNotFoundError(f"Base predictions not found: {base_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite.")

    logger.info("Reading base predictions from %s", base_path)
    records = _read_base_records(base_path)
    logger.info("Loaded %d base records", len(records))
    groups = _group_by_instance(records)
    case_stats = _build_case_stats(groups, top_k=args.top_k)
    emit_scenarios = set(args.emit_scenario or [])
    rows, summary = _derive_rows(
        records,
        case_stats,
        top_k=args.top_k,
        emit_scenarios=emit_scenarios or None,
    )

    with output_path.open("w", encoding="utf-8") as output_file:
        for row_number, row in enumerate(rows, start=1):
            normalize_uq_prediction(row, row_number=row_number)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    _write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": UQ_SCHEMA_VERSION,
            "method": METHOD,
            "method_family": METHOD_FAMILY,
            "base_predictions": str(base_path),
            "input_rows": len(records),
            "output_path": str(output_path),
            "rows": len(rows),
            "top_k": args.top_k,
            "emit_scenarios": sorted(emit_scenarios) if emit_scenarios else "all",
            "definition": (
                "Benchmark-native perturbation UQ derived from existing paired "
                "challenge variants. The scalar uncertainty score is the "
                "instance-level mean top-k Jaccard instability between restored "
                "controls and source-stressed variants. Source scores store the "
                "user-history and item-content components separately."
            ),
            "score_orientation": "larger uncertainty_score means more unstable / more uncertain",
            "literature_basis": (
                "Kweon et al. WWW 2025 for LLM-rec uncertainty decomposition; "
                "Inv-Entropy NeurIPS 2025 for perturbations as UQ signal; "
                "BrittleBench 2026 for prompt/input sensitivity benchmarks"
            ),
            "does_not_rerun_llm": True,
            "paired_controls": {
                "user_history": "s3_sparsity/full vs s3_sparsity/truncate_*",
                "item_content": "s5_robustness/original vs s5_robustness/ablate_*",
            },
            "summary": summary,
        },
    )
    logger.info("Wrote %d perturbation UQ rows to %s", len(rows), output_path)


def _read_base_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{line_number}: missing record_id")
            if record_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate record_id {record_id}")
            _ranked_candidate_ids(row, row_number=line_number)
            seen.add(record_id)
            records.append(row)
    return records


def _group_by_instance(records: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        dataset = str(record.get("dataset"))
        instance_id = str(record.get("instance_id"))
        groups[(dataset, instance_id)].append(record)
    return dict(groups)


def _build_case_stats(
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    top_k: int,
) -> dict[tuple[str, str], dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key, records in groups.items():
        by_variant = {
            (str(record.get("scenario")), str(record.get("variant_name"))): record
            for record in records
        }
        s3_full = by_variant.get(("s3_sparsity", "full"))
        s3_truncates = [
            record
            for record in records
            if str(record.get("scenario")) == "s3_sparsity"
            and str(record.get("variant_name")).startswith("truncate_")
        ]
        s5_original = by_variant.get(("s5_robustness", "original"))
        s5_ablations = [
            record
            for record in records
            if str(record.get("scenario")) == "s5_robustness"
            and str(record.get("variant_name")).startswith("ablate_")
        ]

        user_distances = {}
        if s3_full is not None:
            for record in s3_truncates:
                user_distances[str(record["record_id"])] = _topk_jaccard_distance(
                    s3_full,
                    record,
                    top_k=top_k,
                )

        item_distances = {}
        if s5_original is not None:
            for record in s5_ablations:
                item_distances[str(record["record_id"])] = _topk_jaccard_distance(
                    s5_original,
                    record,
                    top_k=top_k,
                )

        user_mean = _mean(list(user_distances.values()))
        item_mean = _mean(list(item_distances.values()))
        all_distances = [*user_distances.values(), *item_distances.values()]
        stats[key] = {
            "user_control_record_id": str(s3_full["record_id"]) if s3_full is not None else None,
            "item_control_record_id": str(s5_original["record_id"]) if s5_original is not None else None,
            "user_distances": user_distances,
            "item_distances": item_distances,
            "user_mean_instability": user_mean,
            "item_mean_instability": item_mean,
            "case_mean_instability": _mean(all_distances),
            "num_user_perturbations": len(user_distances),
            "num_item_perturbations": len(item_distances),
        }
    return stats


def _derive_rows(
    records: list[dict[str, Any]],
    case_stats: dict[tuple[str, str], dict[str, Any]],
    *,
    top_k: int,
    emit_scenarios: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    score_sources: Counter[str] = Counter()
    for record in records:
        if emit_scenarios is not None and str(record.get("scenario")) not in emit_scenarios:
            continue
        key = (str(record.get("dataset")), str(record.get("instance_id")))
        stats = case_stats.get(key, {})
        record_id = str(record["record_id"])
        score, score_source, paired_control, variant_distance = _record_uncertainty(record, stats)
        score = _clip01(score)
        score_sources[score_source] += 1
        user_score = _optional_clip(stats.get("user_mean_instability"))
        item_score = _optional_clip(stats.get("item_mean_instability"))
        row = {
            "schema_version": UQ_SCHEMA_VERSION,
            "record_id": record_id,
            "method": METHOD,
            "method_family": METHOD_FAMILY,
            "uncertainty_score": score,
            "confidence_score": _clip01(1.0 - score),
            "source_scores": {
                "user": user_score,
                "item": item_score,
                "candidate": None,
                "prompt": None,
                "model": None,
            },
            "components": {
                "score_definition": "case_mean_top_k_jaccard_instability_over_paired_controls",
                "top_k": top_k,
                "rank_distance": score,
                "variant_distance_to_paired_control": variant_distance,
                "score_source": score_source,
                "paired_control_record_id": paired_control,
                "user_mean_instability": user_score,
                "item_mean_instability": item_score,
                "case_mean_instability": _optional_clip(stats.get("case_mean_instability")),
                "num_user_perturbations": int(stats.get("num_user_perturbations", 0)),
                "num_item_perturbations": int(stats.get("num_item_perturbations", 0)),
                "user_control_record_id": stats.get("user_control_record_id"),
                "item_control_record_id": stats.get("item_control_record_id"),
                "literature_basis": (
                    "benchmark-native paired-variant adaptation of perturbation/sensitivity UQ"
                ),
            },
            "metadata": {
                "dataset": record.get("dataset"),
                "scenario": record.get("scenario"),
                "variant_name": record.get("variant_name"),
                "instance_id": record.get("instance_id"),
                "challenge": record.get("challenge"),
                "target_source": record.get("target_source"),
                "derived_from": "fixed_llm_rankings",
                "does_not_rerun_llm": True,
            },
        }
        rows.append(row)
    return rows, {"score_source_counts": dict(sorted(score_sources.items()))}


def _record_uncertainty(
    record: dict[str, Any],
    stats: dict[str, Any],
) -> tuple[float, str, str | None, float | None]:
    record_id = str(record["record_id"])
    scenario = str(record.get("scenario"))
    variant_name = str(record.get("variant_name"))
    user_distances = stats.get("user_distances") if isinstance(stats.get("user_distances"), dict) else {}
    item_distances = stats.get("item_distances") if isinstance(stats.get("item_distances"), dict) else {}
    case_mean = stats.get("case_mean_instability")
    if isinstance(case_mean, (int, float)) and math.isfinite(float(case_mean)):
        score = float(case_mean)
        if record_id in user_distances:
            return score, "case_mean_with_user_history_perturbation_component", stats.get("user_control_record_id"), float(user_distances[record_id])
        if record_id in item_distances:
            return score, "case_mean_with_item_content_perturbation_component", stats.get("item_control_record_id"), float(item_distances[record_id])
        if scenario == "s3_sparsity" and variant_name == "full":
            return score, "case_mean_user_history_restored_control", None, 0.0
        if scenario == "s5_robustness" and variant_name == "original":
            return score, "case_mean_item_content_restored_control", None, 0.0
        return score, "case_mean_perturbation_instability", None, None

    if record_id in user_distances:
        distance = float(user_distances[record_id])
        return distance, "user_history_perturbation_no_case_mean", stats.get("user_control_record_id"), distance
    if record_id in item_distances:
        distance = float(item_distances[record_id])
        return distance, "item_content_perturbation_no_case_mean", stats.get("item_control_record_id"), distance
    if scenario == "s3_sparsity" and variant_name == "full":
        return 0.0, "user_history_restored_control_no_case_mean", None, 0.0
    if scenario == "s5_robustness" and variant_name == "original":
        return 0.0, "item_content_restored_control_no_case_mean", None, 0.0
    return 0.0, "missing_paired_controls", None, None


def _ranked_candidate_ids(record: dict[str, Any], *, row_number: int | None = None) -> list[str]:
    context = f"row {row_number}: " if row_number is not None else ""
    candidate_ids = record.get("candidate_ids")
    ranking = record.get("ranking")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError(f"{context}candidate_ids must be a non-empty list")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError(f"{context}ranking must be a non-empty list")
    ranked: list[str] = []
    for raw_idx in ranking:
        if isinstance(raw_idx, bool):
            raise ValueError(f"{context}ranking contains bool index")
        idx = int(raw_idx)
        if idx < 0 or idx >= len(candidate_ids):
            raise ValueError(f"{context}ranking index out of range: {idx}")
        ranked.append(str(candidate_ids[idx]))
    return ranked


def _topk_jaccard_distance(
    control: dict[str, Any],
    perturbed: dict[str, Any],
    *,
    top_k: int,
) -> float:
    control_top = set(_ranked_candidate_ids(control)[:top_k])
    perturbed_top = set(_ranked_candidate_ids(perturbed)[:top_k])
    union = control_top | perturbed_top
    if not union:
        return 0.0
    return _clip01(1.0 - (len(control_top & perturbed_top) / len(union)))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _optional_clip(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return _clip01(float(value))
    return None


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False, sort_keys=True)
        output_file.write("\n")


if __name__ == "__main__":
    main()
