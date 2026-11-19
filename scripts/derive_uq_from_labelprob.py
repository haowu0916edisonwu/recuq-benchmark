#!/usr/bin/env python
from __future__ import annotations

import argparse
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
            "Derive LabelProb-Entropy and LabelProb-Margin UQ predictions "
            "from an existing LabelProb-Max uq_predictions.jsonl."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/uq/labelprob_llama31_8b/uq_predictions.jsonl",
        help="Input LabelProb-Max UQ JSONL containing option probabilities and entropy components.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/uq",
        help="Root directory under which method subdirectories are written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["entropy", "margin"],
        choices=["entropy", "margin"],
        help="Derived methods to write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing derived uq_predictions.jsonl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("derive_uq_from_labelprob")
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    methods = list(dict.fromkeys(args.methods))

    if not input_path.exists():
        raise FileNotFoundError(f"Input LabelProb UQ file not found: {input_path}")

    output_paths = {
        "entropy": output_root / "labelprob_entropy" / "uq_predictions.jsonl",
        "margin": output_root / "labelprob_margin" / "uq_predictions.jsonl",
    }
    for method in methods:
        output_paths[method].parent.mkdir(parents=True, exist_ok=True)
        if output_paths[method].exists() and not args.overwrite:
            raise FileExistsError(f"{output_paths[method]} already exists. Use --overwrite.")

    logger.info("Deriving %s from %s", ",".join(methods), input_path)
    writers = {
        method: output_paths[method].open("w", encoding="utf-8")
        for method in methods
    }
    counts = {method: 0 for method in methods}
    source_methods: set[str] = set()
    try:
        with open_text(input_path, "r") as input_file:
            for row_number, line in enumerate(input_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                raw = json.loads(stripped)
                if not isinstance(raw, dict):
                    raise ValueError(f"{input_path}:{row_number}: expected JSON object")
                # Validate the source row with the same contract used by the evaluator.
                normalize_uq_prediction(raw, row_number=row_number)
                source_methods.add(str(raw["method"]))
                for method in methods:
                    derived = _derive_record(raw, method=method, row_number=row_number)
                    writers[method].write(json.dumps(derived, ensure_ascii=False) + "\n")
                    counts[method] += 1
    finally:
        for writer in writers.values():
            writer.close()

    for method in methods:
        _write_metadata(
            output_paths[method].parent / "run_metadata.json",
            {
                "schema_version": UQ_SCHEMA_VERSION,
                "method": _method_name(method),
                "method_family": "probability",
                "source_method": ",".join(sorted(source_methods)),
                "source_uq_predictions": str(input_path),
                "output_path": str(output_paths[method]),
                "rows": counts[method],
                "definition": _method_definition(method),
                "score_orientation": "larger uncertainty_score means more uncertain / more likely erroneous",
                "literature_basis": _method_literature_basis(method),
            },
        )
        logger.info("Wrote %d rows to %s", counts[method], output_paths[method])


def _derive_record(raw: dict[str, Any], *, method: str, row_number: int) -> dict[str, Any]:
    components = raw.get("components")
    if not isinstance(components, dict):
        raise ValueError(f"row {row_number}: components must be an object")
    labels, probabilities = _normalized_option_distribution(raw, components, row_number)

    if method == "entropy":
        ent = _entropy(probabilities)
        uncertainty = ent / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        confidence = 1.0 - uncertainty
        extra_components = {
            "entropy": ent,
            "normalized_entropy": uncertainty,
            "max_option_probability": max(probabilities),
            "num_options": len(probabilities),
            "score_definition": "normalized_entropy(option_label_distribution)",
            "literature_basis": (
                "Shannon entropy; entropy sampling in active learning; "
                "predictive entropy in LLM recommendation UQ"
            ),
        }
    elif method == "margin":
        sorted_pairs = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
        top_letter, top_probability = sorted_pairs[0]
        second_letter, second_probability = sorted_pairs[1] if len(sorted_pairs) > 1 else ("", 0.0)
        margin = top_probability - second_probability
        # The benchmark convention is larger uncertainty_score => more likely wrong.
        # Classical margin sampling queries small top-1/top-2 gaps, so we invert it.
        uncertainty = 1.0 - margin
        confidence = margin
        extra_components = {
            "probability_margin": margin,
            "max_option_probability": top_probability,
            "second_option_probability": second_probability,
            "top_letter": top_letter,
            "second_letter": second_letter,
            "num_options": len(probabilities),
            "score_definition": "1 - (top1_option_probability - top2_option_probability)",
            "literature_basis": "margin-based uncertainty sampling",
        }
    else:
        raise ValueError(f"Unknown derived method: {method}")

    uncertainty = _clip_unit_interval(uncertainty)
    confidence = _clip_unit_interval(confidence)
    derived_components = {
        "source_method": raw.get("method"),
        "source_uncertainty_score": raw.get("uncertainty_score"),
        "source_confidence_score": raw.get("confidence_score"),
        **extra_components,
    }
    if method != "margin" and "top_letter" in components:
        derived_components["top_letter"] = components["top_letter"]
    derived_components["option_probabilities"] = dict(zip(labels, probabilities))

    return {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": raw["record_id"],
        "method": _method_name(method),
        "method_family": "probability",
        "uncertainty_score": uncertainty,
        "confidence_score": confidence,
        "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
        "components": derived_components,
        "metadata": {
            **(raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}),
            "derived_from": raw.get("method"),
            "derived_score": method,
        },
    }


def _normalized_option_distribution(
    raw: dict[str, Any],
    components: dict[str, Any],
    row_number: int,
) -> tuple[list[str], list[float]]:
    option_probabilities = components.get("option_probabilities")
    if not isinstance(option_probabilities, dict) or not option_probabilities:
        raise ValueError(f"row {row_number}: components.option_probabilities must be a non-empty object")

    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    candidate_letters = metadata.get("candidate_letters")
    if isinstance(candidate_letters, list) and candidate_letters:
        labels = [str(label) for label in candidate_letters]
    else:
        labels = sorted(str(label) for label in option_probabilities)

    probabilities: list[float] = []
    for label in labels:
        if label not in option_probabilities:
            raise ValueError(
                f"row {row_number}: components.option_probabilities is missing candidate label {label!r}"
            )
        value = option_probabilities[label]
        if isinstance(value, bool):
            raise ValueError(
                f"row {row_number}: components.option_probabilities.{label} must be numeric, not bool"
            )
        try:
            probability = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"row {row_number}: components.option_probabilities.{label} must be numeric"
            ) from exc
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(
                f"row {row_number}: components.option_probabilities.{label} must be finite and non-negative"
            )
        probabilities.append(probability)

    total = sum(probabilities)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError(f"row {row_number}: option probabilities must have positive finite mass")

    normalized = [probability / total for probability in probabilities]
    return labels, normalized


def _entropy(probabilities: list[float]) -> float:
    return -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)


def _clip_unit_interval(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _method_name(method: str) -> str:
    if method == "entropy":
        return "labelprob_entropy"
    if method == "margin":
        return "labelprob_margin"
    raise ValueError(f"Unknown method: {method}")


def _method_definition(method: str) -> str:
    if method == "entropy":
        return "uncertainty_score = normalized entropy over the valid option-label probability distribution"
    if method == "margin":
        return "uncertainty_score = 1 - (top1 option probability - top2 option probability)"
    raise ValueError(f"Unknown method: {method}")


def _method_literature_basis(method: str) -> str:
    if method == "entropy":
        return "Shannon (1948); Settles (2009); Kweon et al. (2025)"
    if method == "margin":
        return "Scheffer et al. (2001); Settles (2009)"
    raise ValueError(f"Unknown method: {method}")


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
