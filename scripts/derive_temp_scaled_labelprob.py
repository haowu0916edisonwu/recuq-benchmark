#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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
from recuq_benchmark.utils.io import open_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive a temperature-scaled LabelProb UQ baseline from existing "
            "LabelProb option probabilities and base LLM labels. This does not "
            "rerun the LLM."
        )
    )
    parser.add_argument(
        "--labelprob_predictions",
        type=str,
        default="results/uq/labelprob_llama31_8b/uq_predictions.jsonl",
        help="Input LabelProb-Max UQ JSONL containing components.option_probabilities.",
    )
    parser.add_argument(
        "--base_predictions",
        type=str,
        required=True,
        help="Base LLM raw_predictions.jsonl containing record_id and labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/uq/labelprob_temp_scaled",
        help="Output directory for uq_predictions.jsonl and run_metadata.json.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-fitting folds used to avoid train/eval leakage.",
    )
    parser.add_argument(
        "--split_key",
        choices=["record_id", "instance_id"],
        default="instance_id",
        help=(
            "Key used to assign folds. instance_id keeps paired scenario variants "
            "for the same underlying case in the same fold when available."
        ),
    )
    parser.add_argument(
        "--min_log_temperature",
        type=float,
        default=-4.0,
        help="Lower bound for scalar log-temperature search.",
    )
    parser.add_argument(
        "--max_log_temperature",
        type=float,
        default=4.0,
        help="Upper bound for scalar log-temperature search.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_dir if uq_predictions.jsonl already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("derive_temp_scaled_labelprob")
    labelprob_path = Path(args.labelprob_predictions)
    base_path = Path(args.base_predictions)
    output_dir = Path(args.output_dir)
    output_path = output_dir / "uq_predictions.jsonl"

    if args.folds < 2:
        raise ValueError("--folds must be at least 2 for cross-fitting")
    if not labelprob_path.exists():
        raise FileNotFoundError(f"LabelProb predictions not found: {labelprob_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"Base predictions not found: {base_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite.")

    logger.info("Reading base predictions from %s", base_path)
    base_by_id = _read_base_records(base_path)
    logger.info("Reading LabelProb predictions from %s", labelprob_path)
    records = _read_labelprob_records(labelprob_path, base_by_id)
    logger.info("Loaded %d joined records", len(records))

    fold_to_indices: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        fold = _stable_fold(record["split_value"], args.folds)
        record["fold"] = fold
        fold_to_indices[fold].append(index)

    source_methods = sorted({str(record["source_method"]) for record in records})
    fold_temperatures: dict[int, float] = {}
    for fold in range(args.folds):
        train_records = [record for record in records if record["fold"] != fold]
        if not train_records:
            raise ValueError(f"Fold {fold} has no training records")
        temperature = _fit_temperature(
            train_records,
            min_log_temperature=args.min_log_temperature,
            max_log_temperature=args.max_log_temperature,
        )
        fold_temperatures[fold] = temperature
        logger.info(
            "Fold %d: train=%d heldout=%d temperature=%.6f",
            fold,
            len(train_records),
            len(fold_to_indices.get(fold, [])),
            temperature,
        )

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            temperature = fold_temperatures[int(record["fold"])]
            scaled = _temperature_scaled_distribution(record["probabilities"], temperature)
            max_index = max(range(len(scaled)), key=lambda idx: scaled[idx])
            confidence = _clip01(scaled[max_index])
            uncertainty = _clip01(1.0 - confidence)
            labels = record["labels"]
            derived = {
                "schema_version": UQ_SCHEMA_VERSION,
                "record_id": record["record_id"],
                "method": "labelprob_temp_scaled",
                "method_family": "calibration",
                "uncertainty_score": uncertainty,
                "confidence_score": confidence,
                "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
                "components": {
                    "source_method": record["source_method"],
                    "source_uncertainty_score": record["source_uncertainty_score"],
                    "source_confidence_score": record["source_confidence_score"],
                    "temperature": temperature,
                    "fold": int(record["fold"]),
                    "split_key": args.split_key,
                    "score_definition": "uncertainty_score = 1 - max(softmax(log(option_probability) / T))",
                    "temperature_objective": "-log probability mass assigned to positive candidate labels",
                    "max_option_probability": confidence,
                    "top_letter": labels[max_index],
                    "positive_letters": [labels[idx] for idx in record["positive_indices"]],
                    "option_probabilities": dict(zip(labels, scaled)),
                    "source_option_probabilities": dict(zip(labels, record["probabilities"])),
                },
                "metadata": {
                    **record["metadata"],
                    "derived_from": record["source_method"],
                    "derived_score": "temperature_scaled_max_probability",
                    "calibration_folds": args.folds,
                },
            }
            output_file.write(json.dumps(derived, ensure_ascii=False) + "\n")

    _write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": UQ_SCHEMA_VERSION,
            "method": "labelprob_temp_scaled",
            "method_family": "calibration",
            "source_method": ",".join(source_methods),
            "labelprob_predictions": str(labelprob_path),
            "base_predictions": str(base_path),
            "output_path": str(output_path),
            "rows": len(records),
            "folds": args.folds,
            "split_key": args.split_key,
            "fold_temperatures": {str(key): value for key, value in sorted(fold_temperatures.items())},
            "definition": "Cross-fitted scalar temperature scaling over option-label probabilities; uncertainty is 1 - max scaled option probability.",
            "score_orientation": "larger uncertainty_score means more uncertain / more likely erroneous",
            "literature_basis": "Guo et al. (2017), temperature scaling for neural network calibration",
        },
    )
    logger.info("Wrote %d temp-scaled LabelProb rows to %s", len(records), output_path)


def _read_base_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with open_text(path, "r") as input_file:
        for row_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{row_number}: expected JSON object")
            record_id = raw.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{row_number}: missing record_id")
            if record_id in records:
                raise ValueError(f"{path}:{row_number}: duplicate record_id {record_id}")
            records[record_id] = raw
    return records


def _read_labelprob_records(path: Path, base_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open_text(path, "r") as input_file:
        for row_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{row_number}: expected JSON object")
            uq = normalize_uq_prediction(raw, row_number=row_number)
            base = base_by_id.get(uq["record_id"])
            if base is None:
                raise ValueError(f"{path}:{row_number}: no matching base record for {uq['record_id']}")

            labels, probabilities = _normalized_option_distribution(uq, row_number)
            positive_indices = _positive_indices(base, labels, row_number)
            metadata = uq.get("metadata") if isinstance(uq.get("metadata"), dict) else {}
            split_value = _split_value(base, metadata)
            records.append(
                {
                    "record_id": uq["record_id"],
                    "source_method": uq["method"],
                    "source_uncertainty_score": uq["uncertainty_score"],
                    "source_confidence_score": uq["confidence_score"],
                    "metadata": metadata,
                    "labels": labels,
                    "probabilities": probabilities,
                    "positive_indices": positive_indices,
                    "split_value": split_value,
                }
            )
    return records


def _normalized_option_distribution(
    uq: dict[str, Any],
    row_number: int,
) -> tuple[list[str], list[float]]:
    components = uq.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"row {row_number}: components must be an object")
    option_probabilities = components.get("option_probabilities")
    if not isinstance(option_probabilities, dict) or not option_probabilities:
        raise ValueError(f"row {row_number}: components.option_probabilities must be a non-empty object")

    metadata = uq.get("metadata") if isinstance(uq.get("metadata"), dict) else {}
    candidate_letters = metadata.get("candidate_letters")
    if isinstance(candidate_letters, list) and candidate_letters:
        labels = [str(label) for label in candidate_letters]
    else:
        labels = sorted(str(label) for label in option_probabilities)

    probabilities: list[float] = []
    for label in labels:
        if label not in option_probabilities:
            raise ValueError(f"row {row_number}: missing option probability for {label!r}")
        value = option_probabilities[label]
        if isinstance(value, bool):
            raise ValueError(f"row {row_number}: probability for {label!r} must be numeric")
        probability = float(value)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(f"row {row_number}: probability for {label!r} must be finite and non-negative")
        probabilities.append(probability)
    total = sum(probabilities)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError(f"row {row_number}: option probability mass must be positive and finite")
    return labels, [probability / total for probability in probabilities]


def _positive_indices(base: dict[str, Any], labels: list[str], row_number: int) -> list[int]:
    raw_labels = base.get("labels")
    if not isinstance(raw_labels, list) or len(raw_labels) != len(labels):
        raise ValueError(f"row {row_number}: base labels must align with option labels")
    positive_indices = [idx for idx, value in enumerate(raw_labels) if bool(value)]
    if not positive_indices:
        raise ValueError(f"row {row_number}: base record has no positive labels")
    return positive_indices


def _split_value(base: dict[str, Any], metadata: dict[str, Any]) -> str:
    dataset = str(base.get("dataset") or metadata.get("dataset") or "")
    instance_id = str(base.get("instance_id") or metadata.get("instance_id") or base.get("record_id"))
    return f"{dataset}::{instance_id}"


def _stable_fold(value: str, folds: int) -> int:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % folds


def _fit_temperature(
    records: list[dict[str, Any]],
    *,
    min_log_temperature: float,
    max_log_temperature: float,
) -> float:
    # One-dimensional convex-ish search over log(T). This avoids requiring scipy.
    lower = min_log_temperature
    upper = max_log_temperature
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - golden * (upper - lower)
    right = lower + golden * (upper - lower)
    left_value = _mean_nll(records, math.exp(left))
    right_value = _mean_nll(records, math.exp(right))
    for _ in range(80):
        if abs(upper - lower) < 1e-6:
            break
        if left_value > right_value:
            lower = left
            left = right
            left_value = right_value
            right = lower + golden * (upper - lower)
            right_value = _mean_nll(records, math.exp(right))
        else:
            upper = right
            right = left
            right_value = left_value
            left = upper - golden * (upper - lower)
            left_value = _mean_nll(records, math.exp(left))
    return math.exp((lower + upper) / 2.0)


def _mean_nll(records: list[dict[str, Any]], temperature: float) -> float:
    total = 0.0
    for record in records:
        total += _positive_mass_nll(record["probabilities"], record["positive_indices"], temperature)
    return total / len(records)


def _positive_mass_nll(
    probabilities: list[float],
    positive_indices: list[int],
    temperature: float,
) -> float:
    log_probs = [math.log(max(probability, 1e-12)) / temperature for probability in probabilities]
    denominator = _logsumexp(log_probs)
    numerator = _logsumexp([log_probs[idx] for idx in positive_indices])
    return denominator - numerator


def _temperature_scaled_distribution(probabilities: list[float], temperature: float) -> list[float]:
    log_probs = [math.log(max(probability, 1e-12)) / temperature for probability in probabilities]
    denominator = _logsumexp(log_probs)
    return [math.exp(value - denominator) for value in log_probs]


def _logsumexp(values: list[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")


if __name__ == "__main__":
    main()
