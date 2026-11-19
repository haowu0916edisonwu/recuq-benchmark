#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.uq.schema import normalize_uq_prediction
from recuq_benchmark.utils.io import open_text, resolve_jsonl
from recuq_benchmark.utils.logging import get_logger


CONTROL_ORDER = ["user", "item"]
CONTROL_TO_BASE_TARGET = {"user": "U_user", "item": "U_item"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate paired-control response agreement and descriptive restoration comparisons "
            "for scalar UQ methods using the constructed S3/S5 controls."
        )
    )
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
        help="Directory for paired-control response outputs.",
    )
    parser.add_argument(
        "--min_effect_gap",
        type=float,
        default=0.0,
        help="Minimum absolute user/item effect gap required to assign a diagnostic label.",
    )
    parser.add_argument(
        "--write_instance_details",
        action="store_true",
        help="Write per-method per-instance source-response details for auditing.",
    )
    parser.add_argument(
        "--require_both_positive_effects",
        action="store_true",
        help=(
            "Retain only cases whose raw user-history and item-content restoration "
            "effects are both strictly positive before clipping."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("evaluate_source_response_interventions")
    base_path = Path(args.base_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_records = _read_jsonl(base_path)
    uq_paths = _resolve_uq_paths(args.uq_predictions)
    uq_by_method = _read_uq_by_method(uq_paths)
    logger.info(
        "Evaluating source response: base_records=%d uq_methods=%d",
        len(base_records),
        len(uq_by_method),
    )

    instance_groups = _group_base_instances(base_records)
    source_cases = _build_source_cases(
        instance_groups,
        args.min_effect_gap,
        require_both_positive_effects=args.require_both_positive_effects,
    )
    if not source_cases:
        raise SystemExit("No source cases available. Need S3 full/truncate and S5 original/ablate variants.")

    detail_rows: list[dict[str, Any]] = []
    table12_rows: list[dict[str, Any]] = []
    table13_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for method, scores_by_record_id in sorted(uq_by_method.items()):
        method_details, missing = _score_method_source_responses(source_cases, scores_by_record_id, method)
        detail_rows.extend(method_details)
        coverage_rows.append(
            {
                "method": method,
                "source_cases": len(source_cases),
                "scored_cases": len(method_details),
                "missing_cases": missing,
                "coverage_rate": len(method_details) / len(source_cases) if source_cases else 0.0,
            }
        )
        table12_rows.append(_table12_row(method, method_details))
        table13_rows.extend(_table13_rows(method, method_details))
        confusion_rows.extend(_confusion_rows(method, method_details))

    _write_csv(output_dir / "control_response_coverage.csv", coverage_rows)
    _write_csv(output_dir / "control_response_metrics.csv", table12_rows)
    _write_csv(output_dir / "matched_restoration.csv", table13_rows)
    _write_csv(output_dir / "control_response_confusion.csv", confusion_rows)
    if args.write_instance_details:
        _write_csv(output_dir / "control_response_instance_details.csv", detail_rows)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "base_predictions": str(base_path),
            "uq_predictions": [str(path) for path in uq_paths],
            "output_dir": str(output_dir),
            "source_case_count": len(source_cases),
            "method_count": len(uq_by_method),
            "controls": CONTROL_ORDER,
            "operational_label_definition": (
                "Operational dominant control is whichever restoration has the larger clipped "
                "NDCG@10 loss reduction: S3 full history vs truncated history for the user-history control, "
                "and S5 original item text vs ablated item text for the item-content control. This is a constructed diagnostic label, not a causal source."
            ),
            "response_prediction_definition": (
                "If a UQ method provides finite source_scores.user and source_scores.item, "
                "those source scores are used directly. Otherwise, predicted control-response scores "
                "are scalar uncertainty increases under S3 history truncation and S5 "
                "item-field ablation relative to their corresponding restored controls."
            ),
            "intervention_definition": (
                "Matched restoration uses the restoration delta for the method-predicted control; "
                "mismatched restoration uses the other control restoration delta."
            ),
        },
    )
    logger.info("Wrote source-response outputs to %s", output_dir)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open_text(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


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


def _read_uq_by_method(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicates: list[tuple[str, str]] = []
    for path in paths:
        with open_text(path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw = json.loads(stripped)
                    if not isinstance(raw, dict):
                        raise ValueError("expected JSON object")
                    row = normalize_uq_prediction(raw, row_number=line_number)
                except (json.JSONDecodeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid UQ prediction: {exc}") from exc
                method = row["method"]
                record_id = row["record_id"]
                if record_id in by_method[method]:
                    duplicates.append((method, record_id))
                    continue
                by_method[method][record_id] = {
                    "uncertainty_score": float(row["uncertainty_score"]),
                    "source_scores": row.get("source_scores") if isinstance(row.get("source_scores"), dict) else {},
                }
    if duplicates:
        sample = ", ".join(f"{method}:{record_id}" for method, record_id in duplicates[:5])
        raise ValueError(f"Duplicate method/record_id pairs in UQ predictions: {sample}")
    return dict(by_method)


def _group_base_instances(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        dataset = str(record.get("dataset"))
        instance_id = str(record.get("instance_id"))
        scenario = str(record.get("scenario"))
        variant_name = str(record.get("variant_name"))
        groups[(dataset, instance_id)][f"{scenario}/{variant_name}"] = record
    return dict(groups)


def _build_source_cases(
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]],
    min_effect_gap: float,
    require_both_positive_effects: bool = False,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for (dataset, instance_id), records in sorted(groups.items()):
        s3_full = records.get("s3_sparsity/full")
        s3_truncates = [
            record
            for key, record in records.items()
            if key.startswith("s3_sparsity/truncate_")
        ]
        s5_original = records.get("s5_robustness/original")
        s5_ablations = [
            record
            for key, record in records.items()
            if key.startswith("s5_robustness/ablate_")
        ]
        if not s3_full or not s3_truncates or not s5_original or not s5_ablations:
            continue

        raw_user_effect = _mean([_restoration_delta(record, s3_full) for record in s3_truncates])
        raw_item_effect = _mean([_restoration_delta(record, s5_original) for record in s5_ablations])
        if require_both_positive_effects and not (raw_user_effect > 0.0 and raw_item_effect > 0.0):
            continue
        clipped_user_effect = max(raw_user_effect, 0.0)
        clipped_item_effect = max(raw_item_effect, 0.0)
        effect_gap = abs(clipped_user_effect - clipped_item_effect)
        if effect_gap <= min_effect_gap:
            continue
        if clipped_user_effect == clipped_item_effect:
            continue
        dominant_control = "user" if clipped_user_effect > clipped_item_effect else "item"
        cases.append(
            {
                "dataset": dataset,
                "instance_id": instance_id,
                "dominant_control": dominant_control,
                "user_effect": clipped_user_effect,
                "item_effect": clipped_item_effect,
                "s3_full_record_id": s3_full["record_id"],
                "s3_truncate_record_ids": [record["record_id"] for record in s3_truncates],
                "s5_original_record_id": s5_original["record_id"],
                "s5_ablate_record_ids": [record["record_id"] for record in s5_ablations],
                "s3_truncate_count": len(s3_truncates),
                "s5_ablate_count": len(s5_ablations),
            }
        )
    return cases


def _ndcg_loss(record: dict[str, Any]) -> float:
    metrics = record.get("metrics") if isinstance(record.get("metrics"), dict) else {}
    ndcg = float(metrics.get("ndcg@10", 0.0))
    return 1.0 - ndcg


def _restoration_delta(stressed: dict[str, Any], restored: dict[str, Any]) -> float:
    return _ndcg_loss(stressed) - _ndcg_loss(restored)


def _score_method_source_responses(
    cases: list[dict[str, Any]],
    uq_by_record_id: dict[str, dict[str, Any]],
    method: str,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    missing = 0
    for case in cases:
        try:
            s3_full_score = _uncertainty_score(uq_by_record_id, str(case["s3_full_record_id"]))
            s3_truncate_scores = [
                _uncertainty_score(uq_by_record_id, str(record_id))
                for record_id in case["s3_truncate_record_ids"]
            ]
            s5_original_score = _uncertainty_score(uq_by_record_id, str(case["s5_original_record_id"]))
            s5_ablate_scores = [
                _uncertainty_score(uq_by_record_id, str(record_id))
                for record_id in case["s5_ablate_record_ids"]
            ]
        except KeyError:
            missing += 1
            continue

        direct_source_scores = _direct_source_scores(uq_by_record_id, case)
        if direct_source_scores is not None:
            user_score = direct_source_scores["user"]
            item_score = direct_source_scores["item"]
            prediction_signal = "source_scores"
        else:
            user_score = max(_mean(s3_truncate_scores) - s3_full_score, 0.0)
            item_score = max(_mean(s5_ablate_scores) - s5_original_score, 0.0)
            prediction_signal = "scalar_response"
        predicted_control = _dominant_control({"user": user_score, "item": item_score})
        matched_delta = case[f"{predicted_control}_effect"] if predicted_control in CONTROL_ORDER else ""
        other_control = _other_control(predicted_control)
        mismatched_delta = case[f"{other_control}_effect"] if other_control in CONTROL_ORDER else ""
        rows.append(
            {
                "method": method,
                "dataset": case["dataset"],
                "instance_id": case["instance_id"],
                "dominant_control": case["dominant_control"],
                "predicted_control": predicted_control,
                "correct": predicted_control == case["dominant_control"],
                "user_score": user_score,
                "item_score": item_score,
                "prediction_signal": prediction_signal,
                "user_effect": case["user_effect"],
                "item_effect": case["item_effect"],
                "matched_delta_ndcg@10": matched_delta,
                "mismatched_delta_ndcg@10": mismatched_delta,
                "matched_wins": (
                    bool(matched_delta > mismatched_delta)
                    if isinstance(matched_delta, float) and isinstance(mismatched_delta, float)
                    else ""
                ),
                "s3_truncate_count": case["s3_truncate_count"],
                "s5_ablate_count": case["s5_ablate_count"],
            }
        )
    return rows, missing


def _uncertainty_score(uq_by_record_id: dict[str, dict[str, Any]], record_id: str) -> float:
    return float(uq_by_record_id[record_id]["uncertainty_score"])


def _direct_source_scores(
    uq_by_record_id: dict[str, dict[str, Any]],
    case: dict[str, Any],
) -> dict[str, float] | None:
    record_ids = [
        str(case["s3_full_record_id"]),
        *[str(record_id) for record_id in case["s3_truncate_record_ids"]],
        str(case["s5_original_record_id"]),
        *[str(record_id) for record_id in case["s5_ablate_record_ids"]],
    ]
    source_values: dict[str, list[float]] = {control: [] for control in CONTROL_ORDER}
    for record_id in record_ids:
        row = uq_by_record_id.get(record_id)
        if row is None:
            continue
        source_scores = row.get("source_scores")
        if not isinstance(source_scores, dict):
            continue
        for control in CONTROL_ORDER:
            value = source_scores.get(control)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                source_values[control].append(float(value))
    if all(source_values[control] for control in CONTROL_ORDER):
        return {control: _mean(source_values[control]) for control in CONTROL_ORDER}
    return None


def _dominant_control(scores: dict[str, float]) -> str:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return "tie"
    return ordered[0][0]


def _other_control(source: str) -> str:
    if source == "user":
        return "item"
    if source == "item":
        return "user"
    return ""


def _table12_row(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(row["dominant_control"]) for row in rows]
    predictions = [str(row["predicted_control"]) for row in rows]
    source_aurocs = []
    source_rhos = []
    for control in CONTROL_ORDER:
        binary_labels = [1 if label == control else 0 for label in labels]
        source_scores = [float(row[f"{control}_score"]) for row in rows]
        auroc = _roc_auc_score(binary_labels, source_scores)
        if auroc is not None:
            source_aurocs.append(auroc)
        control_rho = _spearman(
            [float(row[f"{control}_effect"]) for row in rows],
            source_scores,
        )
        if control_rho is not None:
            source_rhos.append(control_rho)

    top2_hits = [
        row["dominant_control"] in _top_k_sources(row, k=2)
        for row in rows
    ]
    return {
        "method": method,
        "control_set": ",".join(CONTROL_ORDER),
        "n": len(rows),
        "directional_coverage": (sum(prediction != "tie" for prediction in predictions) / len(predictions) if predictions else 0.0),
        "dominant_control_accuracy": _accuracy(labels, predictions),
        "macro_control_f1": _macro_f1(labels, predictions, CONTROL_ORDER),
        "source_auroc": _mean(source_aurocs) if source_aurocs else "",
        "control_response_rho": _mean(source_rhos) if source_rhos else "",
        "top2_recall": sum(top2_hits) / len(top2_hits) if top2_hits else "",
        "confusion": _confusion_summary(labels, predictions),
    }


def _table13_rows(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in [*CONTROL_ORDER, "all"]:
        items = rows if source == "all" else [row for row in rows if row["predicted_control"] == source]
        matched = [
            float(row["matched_delta_ndcg@10"])
            for row in items
            if isinstance(row["matched_delta_ndcg@10"], float)
        ]
        mismatched = [
            float(row["mismatched_delta_ndcg@10"])
            for row in items
            if isinstance(row["mismatched_delta_ndcg@10"], float)
        ]
        wins = [
            bool(row["matched_wins"])
            for row in items
            if isinstance(row["matched_wins"], bool)
        ]
        output.append(
            {
                "method": method,
                "predicted_control": source,
                "n": len(items),
                "delta_matched_ndcg@10": _mean(matched) if matched else "",
                "delta_mismatched_ndcg@10": _mean(mismatched) if mismatched else "",
                "win_rate": sum(wins) / len(wins) if wins else "",
            }
        )
    return output


def _confusion_rows(method: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counter[(str(row["dominant_control"]), str(row["predicted_control"]))] += 1
    labels = CONTROL_ORDER
    output: list[dict[str, Any]] = []
    for control_label in labels:
        total = sum(count for (source, _), count in counter.items() if source == control_label)
        row: dict[str, Any] = {"method": method, "dominant_control": control_label, "n": total}
        for predicted in [*labels, "tie"]:
            row[f"pred_{predicted}"] = counter[(control_label, predicted)]
        output.append(row)
    return output


def _top_k_sources(row: dict[str, Any], k: int) -> list[str]:
    scores = [(control, float(row[f"{control}_score"])) for control in CONTROL_ORDER]
    return [control for control, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:k]]


def _accuracy(labels: list[str], predictions: list[str]) -> float:
    if not labels:
        return 0.0
    return sum(label == prediction for label, prediction in zip(labels, predictions)) / len(labels)


def _macro_f1(labels: list[str], predictions: list[str], classes: list[str]) -> float:
    scores: list[float] = []
    for klass in classes:
        tp = sum(label == klass and prediction == klass for label, prediction in zip(labels, predictions))
        fp = sum(label != klass and prediction == klass for label, prediction in zip(labels, predictions))
        fn = sum(label == klass and prediction != klass for label, prediction in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        scores.append(f1)
    return _mean(scores) if scores else 0.0


def _confusion_summary(labels: list[str], predictions: list[str]) -> str:
    counter: Counter[tuple[str, str]] = Counter(zip(labels, predictions))
    parts = [
        f"{true}->{pred}:{counter[(true, pred)]}"
        for true in CONTROL_ORDER
        for pred in [*CONTROL_ORDER, "tie"]
        if counter[(true, pred)]
    ]
    return "; ".join(parts)


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


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    rank = 1
    idx = 0
    while idx < len(indexed):
        j = idx + 1
        while j < len(indexed) and indexed[j][1] == indexed[idx][1]:
            j += 1
        average_rank = (rank + rank + (j - idx) - 1) / 2.0
        for original_idx, _ in indexed[idx:j]:
            ranks[original_idx] = average_rank
        rank += j - idx
        idx = j
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(x * x for x in centered_x))
    denom_y = math.sqrt(sum(y * y for y in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return None
    return sum(x * y for x, y in zip(centered_x, centered_y)) / (denom_x * denom_y)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
