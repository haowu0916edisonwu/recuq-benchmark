from __future__ import annotations

import math
from typing import Any


UQ_SCHEMA_VERSION = "uq_prediction_v1"
SOURCE_SCORE_KEYS = ("user", "item", "candidate", "prompt", "model")


def normalize_uq_prediction(record: dict[str, Any], *, row_number: int | None = None) -> dict[str, Any]:
    """Validate and normalize one UQ prediction row.

    The evaluator intentionally keeps this contract small: every method must
    expose one scalar uncertainty score, while optional internals stay in
    components/metadata for method-specific analysis.
    """
    context = f"row {row_number}: " if row_number is not None else ""
    schema_version = record.get("schema_version")
    if schema_version != UQ_SCHEMA_VERSION:
        raise ValueError(f"{context}schema_version must be {UQ_SCHEMA_VERSION!r}")

    normalized = {
        "schema_version": schema_version,
        "record_id": _require_nonempty_string(record, "record_id", context),
        "method": _require_nonempty_string(record, "method", context),
        "method_family": _require_nonempty_string(record, "method_family", context),
        "uncertainty_score": _require_finite_float(record, "uncertainty_score", context),
        "confidence_score": _optional_finite_float(record.get("confidence_score"), context),
        "source_scores": _normalize_source_scores(record.get("source_scores"), context),
        "components": _optional_dict(record.get("components"), "components", context),
        "metadata": _optional_dict(record.get("metadata"), "metadata", context),
    }
    return normalized


def _require_nonempty_string(record: dict[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}{key} must be a non-empty string")
    return value


def _require_finite_float(record: dict[str, Any], key: str, context: str) -> float:
    if key not in record:
        raise ValueError(f"{context}{key} is required")
    value = _coerce_float(record[key], key, context)
    if not math.isfinite(value):
        raise ValueError(f"{context}{key} must be finite")
    return value


def _optional_finite_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    coerced = _coerce_float(value, "confidence_score", context)
    if not math.isfinite(coerced):
        raise ValueError(f"{context}confidence_score must be finite when provided")
    return coerced


def _coerce_float(value: Any, key: str, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context}{key} must be numeric, not bool")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}{key} must be numeric") from exc


def _normalize_source_scores(value: Any, context: str) -> dict[str, float | None]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"{context}source_scores must be an object")
    normalized: dict[str, float | None] = {}
    for key in SOURCE_SCORE_KEYS:
        raw_value = value.get(key)
        if raw_value is None:
            normalized[key] = None
        else:
            score = _coerce_float(raw_value, f"source_scores.{key}", context)
            if not math.isfinite(score):
                raise ValueError(f"{context}source_scores.{key} must be finite when provided")
            normalized[key] = score
    return normalized


def _optional_dict(value: Any, key: str, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context}{key} must be an object")
    return value
