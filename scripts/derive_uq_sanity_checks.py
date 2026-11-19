#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
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
    "sanity_random_score": (
        "sanity",
        "deterministic pseudo-random score from record_id; expected Error-AUROC is near 0.5",
    ),
    "sanity_oracle_error": (
        "sanity",
        "oracle score equal to is_error_at_10; diagnostic upper reference, not deployable UQ",
    ),
    "sanity_parse_failure": (
        "sanity",
        "score equal to 1 when the base ranking required deterministic parse repair",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive low-cost sanity-reference UQ files from frozen fixed-ranking records. "
            "These outputs are diagnostic references for appendix tables, not deployable UQ methods."
        )
    )
    parser.add_argument(
        "--base_predictions",
        type=str,
        required=True,
        help="Path to base LLM raw_predictions.jsonl.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="results/uq",
        help="Root directory under which sanity method subdirectories are written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHOD_DEFINITIONS),
        choices=sorted(METHOD_DEFINITIONS),
        help="Subset of sanity methods to write.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sanity uq_predictions.jsonl files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("derive_uq_sanity_checks")
    methods = list(dict.fromkeys(args.methods))
    output_root = Path(args.output_root)
    base_rows = _read_base_rows(Path(args.base_predictions))

    builders: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "sanity_random_score": _random_row,
        "sanity_oracle_error": _oracle_row,
        "sanity_parse_failure": _parse_failure_row,
    }

    logger.info("Loaded base artifact: rows=%d", len(base_rows))
    for method in methods:
        output_dir = output_root / method
        output_path = output_dir / "uq_predictions.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite.")

        count = 0
        with output_path.open("w", encoding="utf-8") as output_file:
            for base_row in base_rows:
                row = builders[method](base_row)
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
                "output_path": str(output_path),
                "does_not_rerun_llm": True,
                "not_deployable_uq_method": method == "sanity_oracle_error",
            },
        )
        logger.info("Wrote %d rows to %s", count, output_path)


def _read_base_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
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
            if record_id in seen:
                raise ValueError(f"{path}:{row_number}: duplicate record_id {record_id}")
            seen.add(record_id)
            rows.append(row)
    return rows


def _random_row(base_row: dict[str, Any]) -> dict[str, Any]:
    record_id = str(base_row["record_id"])
    digest = hashlib.sha256(record_id.encode("utf-8")).digest()
    raw_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
    uncertainty = raw_int / float(2**64 - 1)
    return _sanity_row(
        base_row,
        method="sanity_random_score",
        uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
        components={"hash_prefix_uint64": raw_int},
    )


def _oracle_row(base_row: dict[str, Any]) -> dict[str, Any]:
    is_error = bool(base_row.get("is_error_at_10"))
    uncertainty = 1.0 if is_error else 0.0
    return _sanity_row(
        base_row,
        method="sanity_oracle_error",
        uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
        components={"uses_ground_truth_error_label": True},
    )


def _parse_failure_row(base_row: dict[str, Any]) -> dict[str, Any]:
    parse_success = bool(base_row.get("parse_success"))
    diagnostics = _parse_diagnostics(base_row)
    parse_repaired = bool(diagnostics.get("parse_repaired")) if diagnostics else not parse_success
    uncertainty = 1.0 if parse_repaired or not parse_success else 0.0
    return _sanity_row(
        base_row,
        method="sanity_parse_failure",
        uncertainty=uncertainty,
        confidence=1.0 - uncertainty,
        components={
            "parse_success": parse_success,
            "parse_repaired": parse_repaired,
        },
    )


def _sanity_row(
    base_row: dict[str, Any],
    *,
    method: str,
    uncertainty: float,
    confidence: float,
    components: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": base_row["record_id"],
        "method": method,
        "method_family": METHOD_DEFINITIONS[method][0],
        "uncertainty_score": float(uncertainty),
        "confidence_score": float(confidence),
        "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
        "components": {
            **components,
            "score_definition": METHOD_DEFINITIONS[method][1],
        },
        "metadata": {
            "dataset": base_row.get("dataset"),
            "scenario": base_row.get("scenario"),
            "challenge": base_row.get("challenge"),
            "variant_name": base_row.get("variant_name"),
            "instance_id": base_row.get("instance_id"),
            "does_not_rerun_llm": True,
            "sanity_reference_only": True,
        },
    }


def _parse_diagnostics(base_row: dict[str, Any]) -> dict[str, Any] | None:
    model_metadata = base_row.get("model_metadata")
    if not isinstance(model_metadata, dict):
        return None
    diagnostics = model_metadata.get("parse_diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False, sort_keys=True)
        output_file.write("\n")


if __name__ == "__main__":
    main()
