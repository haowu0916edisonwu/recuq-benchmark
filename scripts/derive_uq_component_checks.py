#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION, normalize_uq_prediction
from recuq_benchmark.utils.logging import get_logger
from recuq_benchmark.utils.io import open_text


METHOD_DEFINITIONS = {
    "derived_labelprob_base_top1": (
        "probability_derived",
        "uncertainty_score = 1 - p(option selected as the base ranking top-1 item)",
    ),
    "derived_labelprob_base_top10_mass": (
        "probability_derived",
        "uncertainty_score = 1 - sum p(options appearing in the base ranking top-10 set)",
    ),
    "derived_decomp_recommendation": (
        "decomposition_derived",
        "uncertainty_score = recommendation_uncertainty component from decomposed LLM UQ",
    ),
    "derived_decomp_prompt": (
        "decomposition_derived",
        "uncertainty_score = prompt_uncertainty component from decomposed LLM UQ",
    ),
    "derived_decomp_prompt_ratio": (
        "decomposition_derived",
        "uncertainty_score = prompt_uncertainty_ratio component from decomposed LLM UQ",
    ),
    "derived_selfcons_top1_disagreement": (
        "sampling_derived",
        "uncertainty_score = 1 - top1_agreement_confidence from self-consistency samples",
    ),
    "derived_selfcons_top1_entropy": (
        "sampling_derived",
        "uncertainty_score = top1_entropy_uncertainty from self-consistency samples",
    ),
    "derived_selfcons_top1_fsd": (
        "sampling_derived",
        "uncertainty_score = top1_fsd_uncertainty from self-consistency samples",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive low-cost scalar UQ component checks from fixed-ranking artifacts. "
            "This script does not rerun the LLM; it only reads completed base, LabelProb, "
            "decomposition, and self-consistency JSONL files and writes standard UQ JSONL rows."
        )
    )
    parser.add_argument(
        "--base_predictions",
        type=str,
        required=True,
        help="Path to base LLM raw_predictions.jsonl.",
    )
    parser.add_argument(
        "--labelprob_predictions",
        type=str,
        required=True,
        help="Path to LabelProb-Max uq_predictions.jsonl.",
    )
    parser.add_argument(
        "--decomp_predictions",
        type=str,
        default=None,
        help="Path to decomposed LLM UQ uq_predictions.jsonl.",
    )
    parser.add_argument(
        "--self_consistency_predictions",
        type=str,
        default=None,
        help="Path to self-consistency uq_predictions.jsonl.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/uq",
        help="Root directory under which derived method subdirectories are written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_DEFINITIONS),
        choices=sorted(METHOD_DEFINITIONS),
        help="Subset of derived methods to write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing derived uq_predictions.jsonl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("derive_uq_component_checks")
    methods = list(dict.fromkeys(args.methods))
    output_root = Path(args.output_root)

    needs_decomp = any(method.startswith("derived_decomp_") for method in methods)
    needs_selfcons = any(method.startswith("derived_selfcons_") for method in methods)
    if needs_decomp and not args.decomp_predictions:
        raise ValueError("--decomp_predictions is required for the selected decomposition methods")
    if needs_selfcons and not args.self_consistency_predictions:
        raise ValueError("--self_consistency_predictions is required for the selected self-consistency methods")

    base_by_id = _read_base_by_id(Path(args.base_predictions))
    labelprob_rows = _read_uq_by_id(Path(args.labelprob_predictions))
    decomp_rows = _read_uq_by_id(Path(args.decomp_predictions)) if needs_decomp else {}
    selfcons_rows = _read_uq_by_id(Path(args.self_consistency_predictions)) if needs_selfcons else {}

    logger.info(
        "Loaded artifacts: base=%d labelprob=%d decomp=%d self_consistency=%d",
        len(base_by_id),
        len(labelprob_rows),
        len(decomp_rows),
        len(selfcons_rows),
    )

    builders: dict[str, Callable[[str], dict[str, Any]]] = {
        "derived_labelprob_base_top1": lambda record_id: _derive_labelprob_base_top1(
            record_id, base_by_id, labelprob_rows
        ),
        "derived_labelprob_base_top10_mass": lambda record_id: _derive_labelprob_base_top10_mass(
            record_id, base_by_id, labelprob_rows
        ),
        "derived_decomp_recommendation": lambda record_id: _derive_component(
            record_id,
            decomp_rows,
            method="derived_decomp_recommendation",
            component_key="recommendation_uncertainty",
        ),
        "derived_decomp_prompt": lambda record_id: _derive_component(
            record_id,
            decomp_rows,
            method="derived_decomp_prompt",
            component_key="prompt_uncertainty",
        ),
        "derived_decomp_prompt_ratio": lambda record_id: _derive_component(
            record_id,
            decomp_rows,
            method="derived_decomp_prompt_ratio",
            component_key="prompt_uncertainty_ratio",
        ),
        "derived_selfcons_top1_disagreement": lambda record_id: _derive_selfcons_top1_disagreement(
            record_id, selfcons_rows
        ),
        "derived_selfcons_top1_entropy": lambda record_id: _derive_component(
            record_id,
            selfcons_rows,
            method="derived_selfcons_top1_entropy",
            component_key="top1_entropy_uncertainty",
        ),
        "derived_selfcons_top1_fsd": lambda record_id: _derive_component(
            record_id,
            selfcons_rows,
            method="derived_selfcons_top1_fsd",
            component_key="top1_fsd_uncertainty",
        ),
    }

    for method in methods:
        output_dir = output_root / method
        output_path = output_dir / "uq_predictions.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite.")

        count = 0
        with output_path.open("w", encoding="utf-8") as output_file:
            for record_id in base_by_id:
                row = builders[method](record_id)
                normalize_uq_prediction(row, row_number=count + 1)
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1

        family, definition = METHOD_DEFINITIONS[method]
        _write_json(
            output_dir / "run_metadata.json",
            {
                "schema_version": UQ_SCHEMA_VERSION,
                "method": method,
                "method_family": family,
                "definition": definition,
                "score_orientation": "larger uncertainty_score means more uncertain / more likely erroneous",
                "rows": count,
                "base_predictions": args.base_predictions,
                "labelprob_predictions": args.labelprob_predictions,
                "decomp_predictions": args.decomp_predictions,
                "self_consistency_predictions": args.self_consistency_predictions,
                "output_path": str(output_path),
                "does_not_rerun_llm": True,
            },
        )
        logger.info("Wrote %d rows to %s", count, output_path)


def _read_base_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open_text(path, "r") as input_file:
        for row_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{row_number}: expected JSON object")
            record_id = row.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"{path}:{row_number}: missing record_id")
            if record_id in rows:
                raise ValueError(f"{path}:{row_number}: duplicate record_id {record_id}")
            _require_ranking(row, row_number=row_number)
            rows[record_id] = row
    return rows


def _read_uq_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open_text(path, "r") as input_file:
        for row_number, line in enumerate(input_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            raw = json.loads(stripped)
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{row_number}: expected JSON object")
            row = normalize_uq_prediction(raw, row_number=row_number)
            record_id = row["record_id"]
            if record_id in rows:
                raise ValueError(f"{path}:{row_number}: duplicate record_id {record_id}")
            rows[record_id] = row
    return rows


def _derive_labelprob_base_top1(
    record_id: str,
    base_by_id: dict[str, dict[str, Any]],
    labelprob_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = _get_row(base_by_id, record_id, "base")
    source = _get_row(labelprob_rows, record_id, "labelprob")
    labels, probabilities = _option_distribution(source)
    ranking = _require_ranking(base)
    top1_index = ranking[0]
    top1_letter = _letter_for_index(labels, top1_index)
    confidence = _clip01(probabilities[top1_letter])
    uncertainty = _clip01(1.0 - confidence)
    return _derived_row(
        record_id=record_id,
        method="derived_labelprob_base_top1",
        uncertainty=uncertainty,
        confidence=confidence,
        source=source,
        components={
            "source_method": source["method"],
            "score_definition": METHOD_DEFINITIONS["derived_labelprob_base_top1"][1],
            "base_top1_index": top1_index,
            "base_top1_letter": top1_letter,
            "base_top1_probability": confidence,
        },
    )


def _derive_labelprob_base_top10_mass(
    record_id: str,
    base_by_id: dict[str, dict[str, Any]],
    labelprob_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = _get_row(base_by_id, record_id, "base")
    source = _get_row(labelprob_rows, record_id, "labelprob")
    labels, probabilities = _option_distribution(source)
    ranking = _require_ranking(base)
    top_k = min(10, len(ranking), len(labels))
    top_letters = [_letter_for_index(labels, index) for index in ranking[:top_k]]
    confidence = _clip01(sum(probabilities[letter] for letter in top_letters))
    uncertainty = _clip01(1.0 - confidence)
    return _derived_row(
        record_id=record_id,
        method="derived_labelprob_base_top10_mass",
        uncertainty=uncertainty,
        confidence=confidence,
        source=source,
        components={
            "source_method": source["method"],
            "score_definition": METHOD_DEFINITIONS["derived_labelprob_base_top10_mass"][1],
            "top_k": top_k,
            "base_top_k_letters": top_letters,
            "base_top_k_probability_mass": confidence,
        },
    )


def _derive_component(
    record_id: str,
    source_rows: dict[str, dict[str, Any]],
    *,
    method: str,
    component_key: str,
) -> dict[str, Any]:
    source = _get_row(source_rows, record_id, method)
    components = source.get("components")
    if not isinstance(components, dict) or component_key not in components:
        raise ValueError(f"{record_id}: source row missing components.{component_key}")
    uncertainty = _require_finite_float(components[component_key], f"{record_id}:{component_key}")
    return _derived_row(
        record_id=record_id,
        method=method,
        uncertainty=uncertainty,
        confidence=_confidence_from_uncertainty(uncertainty),
        source=source,
        components={
            "source_method": source["method"],
            "source_component_key": component_key,
            "score_definition": METHOD_DEFINITIONS[method][1],
            component_key: uncertainty,
        },
    )


def _derive_selfcons_top1_disagreement(
    record_id: str,
    selfcons_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source = _get_row(selfcons_rows, record_id, "self_consistency")
    components = source.get("components")
    if not isinstance(components, dict) or "top1_agreement_confidence" not in components:
        raise ValueError(f"{record_id}: source row missing components.top1_agreement_confidence")
    agreement = _clip01(
        _require_finite_float(
            components["top1_agreement_confidence"],
            f"{record_id}:top1_agreement_confidence",
        )
    )
    uncertainty = _clip01(1.0 - agreement)
    return _derived_row(
        record_id=record_id,
        method="derived_selfcons_top1_disagreement",
        uncertainty=uncertainty,
        confidence=agreement,
        source=source,
        components={
            "source_method": source["method"],
            "score_definition": METHOD_DEFINITIONS["derived_selfcons_top1_disagreement"][1],
            "top1_agreement_confidence": agreement,
            "top1_disagreement_uncertainty": uncertainty,
        },
    )


def _derived_row(
    *,
    record_id: str,
    method: str,
    uncertainty: float,
    confidence: float | None,
    source: dict[str, Any],
    components: dict[str, Any],
) -> dict[str, Any]:
    family = METHOD_DEFINITIONS[method][0]
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    return {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": record_id,
        "method": method,
        "method_family": family,
        "uncertainty_score": float(uncertainty),
        "confidence_score": confidence,
        "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
        "components": {
            **components,
            "source_uncertainty_score": source.get("uncertainty_score"),
            "source_confidence_score": source.get("confidence_score"),
        },
        "metadata": {
            "dataset": metadata.get("dataset"),
            "scenario": metadata.get("scenario"),
            "variant_name": metadata.get("variant_name"),
            "instance_id": metadata.get("instance_id"),
            "derived_from": source.get("method"),
            "does_not_rerun_llm": True,
        },
    }


def _option_distribution(row: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    components = row.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"{row['record_id']}: components must be an object")
    raw_probs = components.get("option_probabilities")
    if not isinstance(raw_probs, dict) or not raw_probs:
        raise ValueError(f"{row['record_id']}: missing components.option_probabilities")

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    candidate_letters = metadata.get("candidate_letters")
    if isinstance(candidate_letters, list) and candidate_letters:
        labels = [str(label) for label in candidate_letters]
    else:
        labels = sorted(str(label) for label in raw_probs)

    probabilities: dict[str, float] = {}
    total = 0.0
    for label in labels:
        if label not in raw_probs:
            raise ValueError(f"{row['record_id']}: missing option probability for {label}")
        value = _require_finite_float(raw_probs[label], f"{row['record_id']}:{label}")
        if value < 0.0:
            raise ValueError(f"{row['record_id']}: probability for {label} is negative")
        probabilities[label] = value
        total += value
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError(f"{row['record_id']}: option probabilities have non-positive mass")
    return labels, {label: probabilities[label] / total for label in labels}


def _require_ranking(row: dict[str, Any], *, row_number: int | None = None) -> list[int]:
    context = f"row {row_number}: " if row_number is not None else ""
    ranking = row.get("ranking")
    candidate_ids = row.get("candidate_ids")
    if not isinstance(candidate_ids, list) or not candidate_ids:
        raise ValueError(f"{context}candidate_ids must be a non-empty list")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError(f"{context}ranking must be a non-empty list")
    parsed: list[int] = []
    seen: set[int] = set()
    for raw_index in ranking:
        if isinstance(raw_index, bool):
            raise ValueError(f"{context}ranking contains bool index")
        index = int(raw_index)
        if index < 0 or index >= len(candidate_ids):
            raise ValueError(f"{context}ranking index out of range: {index}")
        if index in seen:
            raise ValueError(f"{context}ranking contains duplicate index: {index}")
        seen.add(index)
        parsed.append(index)
    return parsed


def _letter_for_index(labels: list[str], index: int) -> str:
    if index < 0 or index >= len(labels):
        raise ValueError(f"ranking index {index} out of range for {len(labels)} labels")
    return labels[index]


def _get_row(rows: dict[str, dict[str, Any]], record_id: str, name: str) -> dict[str, Any]:
    row = rows.get(record_id)
    if row is None:
        raise KeyError(f"{name} predictions missing record_id: {record_id}")
    return row


def _require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not bool")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _confidence_from_uncertainty(value: float) -> float:
    return _clip01(1.0 - _clip01(value))


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False, sort_keys=True)
        output_file.write("\n")


if __name__ == "__main__":
    main()
