#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_DIR = REPO_ROOT / "recuq_benchmark" / "uq"
if str(SCHEMA_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEMA_DIR))
from schema import normalize_uq_prediction
from recuq_benchmark.utils.io import open_text, resolve_jsonl
from recuq_benchmark.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UQ methods against base LLM errors.")
    parser.add_argument(
        "--base_predictions",
        type=str,
        required=True,
        help="Path to a plain or gzip-compressed base-prediction JSONL file.",
    )
    parser.add_argument(
        "--uq_predictions",
        type=str,
        nargs="+",
        required=True,
        help="One or more UQ JSONL files, or directories containing plain/gzipped UQ JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for scalar UQ evaluation outputs.",
    )
    parser.add_argument(
        "--label_key",
        type=str,
        default="is_error_at_10",
        help="Binary error label in the base prediction records.",
    )
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=0,
        help="If positive, write paired dataset-stratified bootstrap CIs for the selected scenario.",
    )
    parser.add_argument(
        "--bootstrap_scenario",
        type=str,
        default="s0_warmstart",
        help="Scenario to bootstrap; defaults to the fixed Warm-start view.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
        help="Random seed for paired bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap_unit",
        choices=["record", "instance"],
        default="record",
        help=(
            "Resampling unit for paired bootstrap. Use instance to keep all controlled "
            "variants of one underlying case in the same resampled cluster."
        ),
    )
    parser.add_argument(
        "--write_joined",
        action="store_true",
        help="Also write joined_uq_predictions.jsonl for audit/debugging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("evaluate_uq_methods")
    base_path = Path(args.base_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_records = _read_jsonl(base_path)
    base_by_id = _index_base_records(base_records)
    uq_paths = _resolve_uq_paths(args.uq_predictions)
    logger.info(
        "Evaluating UQ methods: base_records=%d uq_files=%d label_key=%s",
        len(base_records),
        len(uq_paths),
        args.label_key,
    )

    joined_records: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    schema_errors: list[dict[str, Any]] = []
    duplicate_keys: Counter[tuple[str, str]] = Counter()
    seen_method_record_ids: set[tuple[str, str]] = set()

    for uq_path in uq_paths:
        rows, file_errors = _read_uq_predictions(uq_path)
        schema_errors.extend(file_errors)
        joined_count = 0
        unmatched_count = 0
        methods = sorted({row["method"] for row in rows})
        for row in rows:
            method_record_key = (row["method"], row["record_id"])
            if method_record_key in seen_method_record_ids:
                duplicate_keys[method_record_key] += 1
                continue
            seen_method_record_ids.add(method_record_key)
            base = base_by_id.get(row["record_id"])
            if base is None:
                unmatched_count += 1
                continue
            joined_records.append(_join_record(base, row, args.label_key))
            joined_count += 1
        coverage_rows.append(
            {
                "uq_path": str(uq_path),
                "methods": ",".join(methods),
                "input_rows": len(rows),
                "joined_rows": joined_count,
                "unmatched_record_ids": unmatched_count,
                "schema_errors": len(file_errors),
            }
        )

    duplicate_rows = [
        {"method": method, "record_id": record_id, "duplicate_count": count}
        for (method, record_id), count in sorted(duplicate_keys.items())
    ]
    if schema_errors:
        _write_csv(output_dir / "schema_errors.csv", schema_errors)
        raise SystemExit(f"Found {len(schema_errors)} UQ schema errors; see {output_dir / 'schema_errors.csv'}")
    if duplicate_rows:
        _write_csv(output_dir / "duplicate_uq_predictions.csv", duplicate_rows)
        raise SystemExit(
            f"Found {len(duplicate_rows)} duplicate method/record_id pairs; "
            f"see {output_dir / 'duplicate_uq_predictions.csv'}"
        )

    # S4 is the tail-target subset: Warm-start cases filtered by target-item
    # popularity. Source artifacts retain all buckets for auditability.
    base_eval_records = _tail_target_records(base_records)
    joined_eval_records = _tail_target_records(joined_records)
    base_backbone_counts = _base_group_counts(base_eval_records, ["backbone"])
    base_scenario_counts = _base_group_counts(base_eval_records, ["backbone", "scenario"])
    base_dataset_scenario_counts = _base_group_counts(
        base_eval_records, ["backbone", "dataset", "scenario"]
    )
    base_challenge_counts = _base_group_counts(base_eval_records, ["backbone", "challenge"])

    method_rows = _aggregate_uq(
        joined_eval_records,
        ["backbone", "method"],
        base_count_lookup=base_backbone_counts,
        base_lookup_keys=["backbone"],
    )
    scenario_rows = _aggregate_uq(
        joined_eval_records,
        ["backbone", "method", "scenario"],
        base_count_lookup=base_scenario_counts,
        base_lookup_keys=["backbone", "scenario"],
    )
    dataset_scenario_rows = _aggregate_uq(
        joined_eval_records,
        ["backbone", "method", "dataset", "scenario"],
        base_count_lookup=base_dataset_scenario_counts,
        base_lookup_keys=["backbone", "dataset", "scenario"],
    )
    challenge_rows = _aggregate_uq(
        joined_eval_records,
        ["backbone", "method", "challenge"],
        base_count_lookup=base_challenge_counts,
        base_lookup_keys=["backbone", "challenge"],
    )
    base_longtail_records = [record for record in base_eval_records if record.get("scenario") == "s4_longtail"]
    joined_longtail_records = [record for record in joined_eval_records if record.get("scenario") == "s4_longtail"]
    base_longtail_bucket_counts = _base_group_counts(base_longtail_records, ["backbone", "popularity_bucket"])
    base_dataset_longtail_bucket_counts = _base_group_counts(
        base_longtail_records, ["backbone", "dataset", "popularity_bucket"]
    )
    longtail_bucket_rows = _aggregate_uq(
        joined_longtail_records,
        ["backbone", "method", "popularity_bucket"],
        base_count_lookup=base_longtail_bucket_counts,
        base_lookup_keys=["backbone", "popularity_bucket"],
    )
    longtail_dataset_bucket_rows = _aggregate_uq(
        joined_longtail_records,
        ["backbone", "method", "dataset", "popularity_bucket"],
        base_count_lookup=base_dataset_longtail_bucket_counts,
        base_lookup_keys=["backbone", "dataset", "popularity_bucket"],
    )

    _write_csv(output_dir / "uq_coverage_by_file.csv", coverage_rows)
    _write_csv(output_dir / "table11_error_auroc_by_method.csv", method_rows)
    _write_csv(output_dir / "table11_error_auroc_by_method_scenario.csv", scenario_rows)
    _write_csv(output_dir / "table11_error_auroc_by_method_dataset_scenario.csv", dataset_scenario_rows)
    _write_csv(output_dir / "table11_error_auroc_by_method_challenge.csv", challenge_rows)
    _write_csv(output_dir / "table11_error_auroc_by_method_longtail_bucket.csv", longtail_bucket_rows)
    _write_csv(
        output_dir / "table11_error_auroc_by_method_dataset_longtail_bucket.csv",
        longtail_dataset_bucket_rows,
    )
    _write_csv(
        output_dir / "table11_error_auroc_by_method_challenge_pivot.csv",
        _pivot_challenge_rows(challenge_rows),
    )
    if args.write_joined:
        _write_jsonl(output_dir / "joined_uq_predictions.jsonl", joined_records)

    bootstrap_metadata: dict[str, Any] | None = None
    if args.bootstrap_resamples:
        if args.bootstrap_resamples < 1:
            raise ValueError("--bootstrap_resamples must be positive when supplied")
        bootstrap_rows = _paired_dataset_bootstrap(
            base_eval_records,
            joined_eval_records,
            scenario=args.bootstrap_scenario,
            num_resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
            unit=args.bootstrap_unit,
        )
        _write_csv(output_dir / "bootstrap_metrics_ci.csv", bootstrap_rows)
        if args.bootstrap_scenario == "s0_warmstart" and args.bootstrap_unit == "record":
            _write_csv(output_dir / "bootstrap_warm_metrics_ci.csv", bootstrap_rows)
        bootstrap_metadata = {
            "scenario": args.bootstrap_scenario,
            "strata": ["dataset"],
            "unit": args.bootstrap_unit,
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "paired_across_methods": True,
            "metrics": ["error_auroc", "error_auprc", "risk_at_80", "aurc", "eaurc"],
        }
        logger.info("Wrote paired dataset-stratified bootstrap CIs to %s", output_dir)

    _write_json(
        output_dir / "run_metadata.json",
        {
            "base_predictions": str(base_path),
            "uq_predictions": [str(path) for path in uq_paths],
            "label_key": args.label_key,
            "base_records": len(base_records),
            "base_records_after_tail_target_filter": len(base_eval_records),
            "joined_records": len(joined_records),
            "joined_records_after_tail_target_filter": len(joined_eval_records),
            "tail_target_included_buckets": ["tail"],
            "bootstrap": bootstrap_metadata,
            "metric_definitions": {
                "error_auprc": "average_precision_score(is_error_at_10, uncertainty_score)",
                "risk_at_c": "error rate after retaining the lowest-uncertainty floor(c*n) records",
                "aurc": "mean selective risk over every retained-prefix coverage from 1/n through 1",
                "oracle_aurc": "minimum AURC obtainable for the observed number of errors",
                "eaurc": "aurc minus oracle_aurc; lower is better",
            },
            "output_dir": str(output_dir),
        },
    )
    logger.info("Wrote UQ evaluation outputs to %s", output_dir)


def _resolve_uq_paths(paths: list[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            resolved.append(resolve_jsonl(path))
        elif path.exists():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"UQ prediction path not found: {path}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open_text(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


def _read_uq_predictions(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with open_text(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw_record = json.loads(stripped)
                if not isinstance(raw_record, dict):
                    raise ValueError("expected JSON object")
                records.append(normalize_uq_prediction(raw_record, row_number=line_number))
            except (json.JSONDecodeError, ValueError) as exc:
                errors.append(
                    {
                        "uq_path": str(path),
                        "row_number": line_number,
                        "error": str(exc),
                    }
                )
    return records, errors


def _index_base_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for record in records:
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Base prediction record missing non-empty record_id")
        if record_id in by_id:
            duplicates.append(record_id)
        by_id[record_id] = record
    if duplicates:
        raise ValueError(f"Base predictions contain duplicate record_id values: {duplicates[:5]}")
    return by_id


def _join_record(base: dict[str, Any], uq: dict[str, Any], label_key: str) -> dict[str, Any]:
    label = _base_error_label(base, label_key)
    return {
        "record_id": uq["record_id"],
        "method": uq["method"],
        "method_family": uq["method_family"],
        "backbone": base.get("backbone") or base.get("model_name") or "",
        "dataset": base.get("dataset"),
        "scenario": base.get("scenario"),
        "challenge": base.get("challenge"),
        "variant_name": base.get("variant_name"),
        "target_source": base.get("target_source"),
        "popularity_bucket": _record_value(base, "popularity_bucket"),
        "error_label": int(label),
        "uncertainty_score": uq["uncertainty_score"],
        "confidence_score": uq["confidence_score"],
        "source_scores": uq["source_scores"],
        "components": uq["components"],
        "uq_metadata": uq["metadata"],
        "base_metrics": base.get("metrics", {}),
    }


def _base_error_label(base: dict[str, Any], label_key: str) -> bool:
    if label_key in base:
        return bool(base[label_key])
    prefix = "is_error_at_"
    if label_key.startswith(prefix) and label_key[len(prefix):].isdigit():
        k = int(label_key[len(prefix):])
        ranking = base.get("ranking")
        labels = base.get("labels")
        if not isinstance(ranking, list) or not isinstance(labels, list):
            raise KeyError(f"Base prediction cannot derive {label_key!r}: {base.get('record_id')}")
        positive_indices = {idx for idx, value in enumerate(labels) if bool(value)}
        return not any(index in positive_indices for index in ranking[:k])
    raise KeyError(f"Base prediction missing label_key {label_key!r}: {base.get('record_id')}")

def _record_value(record: dict[str, Any], key: str) -> Any:
    if key == "backbone":
        model_metadata = record.get("model_metadata", {})
        metadata_backbone = model_metadata.get("backbone") if isinstance(model_metadata, dict) else None
        return record.get("backbone") or metadata_backbone or record.get("model_name") or ""
    if key in record:
        return record.get(key)
    metadata = record.get("metadata", {})
    return metadata.get(key) if isinstance(metadata, dict) else None


def _tail_target_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("scenario") != "s4_longtail" or _record_value(record, "popularity_bucket") == "tail"
    ]


def _base_group_counts(records: list[dict[str, Any]], group_keys: list[str]) -> dict[tuple[Any, ...], int]:
    counts: Counter[tuple[Any, ...]] = Counter()
    for record in records:
        counts[tuple(_record_value(record, key) for key in group_keys)] += 1
    return dict(counts)


def _aggregate_uq(
    records: list[dict[str, Any]],
    group_keys: list[str],
    *,
    base_count_lookup: dict[tuple[Any, ...], int],
    base_lookup_keys: list[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record.get(key) for key in group_keys)].append(record)

    rows: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda pair: tuple(str(v) for v in pair[0])):
        group = {group_key: value for group_key, value in zip(group_keys, key)}
        if base_lookup_keys == ["all"]:
            base_key = ("all",)
        else:
            base_key = tuple(group[key] for key in base_lookup_keys)
        base_n = base_count_lookup.get(base_key, 0)
        labels = [int(item["error_label"]) for item in items]
        scores = [float(item["uncertainty_score"]) for item in items]
        record_ids = [str(item["record_id"]) for item in items]
        method_families = sorted({str(item["method_family"]) for item in items})
        positives = sum(labels)
        negatives = len(labels) - positives
        error_auroc = _roc_auc_score(labels, scores)
        error_auprc = _error_auprc_score(labels, scores)
        selective_risk = _selective_risk_metrics(labels, scores, record_ids)
        row: dict[str, Any] = {
            **group,
            "method_family": method_families[0] if len(method_families) == 1 else ",".join(method_families),
            "n": len(items),
            "base_n": base_n,
            "coverage_rate": len(items) / base_n if base_n else 0.0,
            "positives": positives,
            "negatives": negatives,
            "error_rate": positives / len(items) if items else 0.0,
            "error_auroc": error_auroc if error_auroc is not None else "",
            "auroc_defined": error_auroc is not None,
            "error_auprc": error_auprc if error_auprc is not None else "",
            "auprc_defined": error_auprc is not None,
            **selective_risk,
            "mean_uncertainty": sum(scores) / len(scores) if scores else 0.0,
            "mean_uncertainty_on_errors": _mean_for_label(scores, labels, 1),
            "mean_uncertainty_on_correct": _mean_for_label(scores, labels, 0),
        }
        rows.append(row)
    return rows


def _roc_auc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda pair: pair[0])
    rank_sum_positive = 0.0
    rank = 1
    idx = 0
    while idx < len(pairs):
        j = idx + 1
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        average_rank = (rank + rank + (j - idx) - 1) / 2.0
        positive_in_tie = sum(label for _, label in pairs[idx:j])
        rank_sum_positive += positive_in_tie * average_rank
        rank += j - idx
        idx = j
    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def _error_auprc_score(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    return float(average_precision_score(labels, scores))


def _selective_risk_metrics(
    labels: list[int],
    scores: list[float],
    record_ids: list[str],
) -> dict[str, float | bool | str]:
    """Compute the discrete full-coverage risk curve after rejecting high uncertainty."""
    if not labels:
        return {
            "risk_at_100": "",
            "risk_at_90": "",
            "risk_at_80": "",
            "risk_at_70": "",
            "aurc": "",
            "aurc_defined": False,
            "oracle_aurc": "",
            "eaurc": "",
            "eaurc_defined": False,
        }
    ordered = sorted(range(len(labels)), key=lambda idx: (scores[idx], record_ids[idx]))
    cumulative_errors = 0
    risks: list[float] = []
    for retained, index in enumerate(ordered, start=1):
        cumulative_errors += labels[index]
        risks.append(cumulative_errors / retained)

    def risk_at(coverage: float) -> float:
        retained = max(1, int(len(labels) * coverage))
        return risks[retained - 1]

    aurc = sum(risks) / len(risks)
    oracle_aurc = _oracle_aurc(len(labels), sum(labels))
    return {
        "risk_at_100": risk_at(1.0),
        "risk_at_90": risk_at(0.9),
        "risk_at_80": risk_at(0.8),
        "risk_at_70": risk_at(0.7),
        "aurc": aurc,
        "aurc_defined": True,
        "oracle_aurc": oracle_aurc,
        "eaurc": max(0.0, aurc - oracle_aurc),
        "eaurc_defined": True,
    }


def _oracle_aurc(num_records: int, num_errors: int) -> float:
    """Discrete AURC of an oracle that retains every correct record before errors."""
    if num_records <= 0:
        return 0.0
    num_correct = num_records - num_errors
    return sum(
        error_rank / (num_correct + error_rank)
        for error_rank in range(1, num_errors + 1)
    ) / num_records


def _mean_for_label(scores: list[float], labels: list[int], target_label: int) -> float | str:
    values = [score for score, label in zip(scores, labels) if label == target_label]
    if not values:
        return ""
    return sum(values) / len(values)


def _paired_dataset_bootstrap(
    base_records: list[dict[str, Any]],
    joined_records: list[dict[str, Any]],
    *,
    scenario: str,
    num_resamples: int,
    seed: int,
    unit: str,
) -> list[dict[str, Any]]:
    """Paired record- or instance-level bootstrap, stratified by dataset."""
    target_base = [record for record in base_records if record.get("scenario") == scenario]
    if not target_base:
        raise ValueError(f"No base records found for bootstrap scenario {scenario!r}")
    ids_by_backbone: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in target_base:
        backbone = str(_record_value(record, "backbone"))
        dataset = str(record.get("dataset") or "")
        record_id = str(record["record_id"])
        ids_by_backbone[backbone][dataset].append(record_id)
    for datasets in ids_by_backbone.values():
        for record_ids in datasets.values():
            record_ids.sort()

    clusters_by_backbone: dict[str, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for record in target_base:
        backbone = str(_record_value(record, "backbone"))
        dataset = str(record.get("dataset") or "")
        record_id = str(record["record_id"])
        instance_id = str(record.get("instance_id") or record_id)
        cluster_id = record_id if unit == "record" else instance_id
        clusters_by_backbone[backbone][dataset][cluster_id].append(record_id)

    uq_maps: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in joined_records:
        if record.get("scenario") != scenario:
            continue
        key = (str(record.get("backbone") or ""), str(record["method"]))
        record_id = str(record["record_id"])
        if record_id in uq_maps[key]:
            raise ValueError(f"Duplicate joined bootstrap record for {key}: {record_id}")
        uq_maps[key][record_id] = record

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    metric_names = ["error_auroc", "error_auprc", "risk_at_80", "aurc", "eaurc"]
    for backbone in sorted(ids_by_backbone):
        datasets = ids_by_backbone[backbone]
        dataset_order = sorted(datasets)
        ordered_ids = [record_id for dataset in dataset_order for record_id in datasets[dataset]]
        expected_ids = set(ordered_ids)
        methods = sorted(method for candidate_backbone, method in uq_maps if candidate_backbone == backbone)
        if not methods:
            raise ValueError(f"No UQ records found for bootstrap backbone {backbone!r}")
        for method in methods:
            observed_ids = set(uq_maps[(backbone, method)])
            missing = expected_ids - observed_ids
            extra = observed_ids - expected_ids
            if missing or extra:
                raise ValueError(
                    f"Bootstrap coverage mismatch for {backbone}/{method}: "
                    f"missing={len(missing)} extra={len(extra)}"
                )

        point_values = {
            method: _uq_metric_values([uq_maps[(backbone, method)][record_id] for record_id in ordered_ids])
            for method in methods
        }
        samples: list[list[str]] = []
        for _ in range(num_resamples):
            sample_ids: list[str] = []
            for dataset in dataset_order:
                clusters = clusters_by_backbone[backbone][dataset]
                cluster_ids = sorted(clusters)
                for cluster_id in rng.choices(cluster_ids, k=len(cluster_ids)):
                    sample_ids.extend(clusters[cluster_id])
            samples.append(sample_ids)
        bootstrap_values: dict[str, dict[str, list[float]]] = {
            method: {metric: [] for metric in metric_names} for method in methods
        }
        for sample_ids in samples:
            for method in methods:
                values = _uq_metric_values([uq_maps[(backbone, method)][record_id] for record_id in sample_ids])
                for metric in metric_names:
                    value = values[metric]
                    if value is not None:
                        bootstrap_values[method][metric].append(float(value))

        stratum_sizes = ",".join(f"{dataset}:{len(datasets[dataset])}" for dataset in dataset_order)
        cluster_sizes = ",".join(
            f"{dataset}:{len(clusters_by_backbone[backbone][dataset])}"
            for dataset in dataset_order
        )
        for method in methods:
            records = [uq_maps[(backbone, method)][record_id] for record_id in ordered_ids]
            families = sorted({str(record["method_family"]) for record in records})
            row: dict[str, Any] = {
                "backbone": backbone,
                "method": method,
                "method_family": families[0] if len(families) == 1 else ",".join(families),
                "scenario": scenario,
                "n": len(ordered_ids),
                "dataset_strata": stratum_sizes,
                "bootstrap_unit": unit,
                "cluster_strata": cluster_sizes,
                "bootstrap_resamples": num_resamples,
                "bootstrap_seed": seed,
                "error_rate": point_values[method]["error_rate"],
            }
            for metric in metric_names:
                row[metric] = point_values[method][metric]
                lower, upper = _bootstrap_interval(bootstrap_values[method][metric])
                row[f"{metric}_ci_low"] = lower
                row[f"{metric}_ci_high"] = upper
            rows.append(row)
    return rows


def _uq_metric_values(records: list[dict[str, Any]]) -> dict[str, float | None]:
    labels = [int(record["error_label"]) for record in records]
    scores = [float(record["uncertainty_score"]) for record in records]
    record_ids = [str(record["record_id"]) for record in records]
    selective = _selective_risk_metrics(labels, scores, record_ids)
    return {
        "error_rate": sum(labels) / len(labels) if labels else None,
        "error_auroc": _roc_auc_score(labels, scores),
        "error_auprc": _error_auprc_score(labels, scores),
        "risk_at_80": selective["risk_at_80"] if selective["risk_at_80"] != "" else None,
        "aurc": selective["aurc"] if selective["aurc"] != "" else None,
        "eaurc": selective["eaurc"] if selective["eaurc"] != "" else None,
    }


def _bootstrap_interval(values: list[float]) -> tuple[float | str, float | str]:
    if not values:
        return "", ""
    ordered = sorted(values)
    return _linear_quantile(ordered, 0.025), _linear_quantile(ordered, 0.975)


def _linear_quantile(ordered_values: list[float], q: float) -> float:
    if len(ordered_values) == 1:
        return ordered_values[0]
    position = (len(ordered_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = position - lower_index
    return ordered_values[lower_index] * (1.0 - fraction) + ordered_values[upper_index] * fraction

def _pivot_challenge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    challenge_order = ["warm", "sparse", "long_tail", "robust"]
    by_method: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        backbone = str(row.get("backbone", ""))
        method = str(row["method"])
        key = (backbone, method)
        by_method[key]["backbone"] = backbone
        by_method[key]["method"] = method
        by_method[key]["method_family"] = row.get("method_family", "")
        challenge = str(row["challenge"])
        by_method[key][challenge] = row["error_auroc"]
        by_method[key][f"{challenge}_n"] = row["n"]
    output: list[dict[str, Any]] = []
    for backbone, method in sorted(by_method):
        row = by_method[(backbone, method)]
        output.append(
            {
                "backbone": backbone,
                "method": method,
                **{challenge: row.get(challenge, "") for challenge in challenge_order},
                **{f"{challenge}_n": row.get(f"{challenge}_n", "") for challenge in challenge_order},
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
