from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_source_response_interventions import _build_source_cases


def _record(record_id: str, ndcg: float) -> dict[str, Any]:
    return {"record_id": record_id, "metrics": {"ndcg@10": ndcg}}


def _group(
    instance_id: str,
    *,
    full_history: float,
    truncated_history: float,
    original_item: float,
    ablated_item: float,
) -> tuple[tuple[str, str], dict[str, dict[str, Any]]]:
    return (
        ("dataset", instance_id),
        {
            "s3_sparsity/full": _record(f"{instance_id}-full", full_history),
            "s3_sparsity/truncate_1": _record(f"{instance_id}-truncated", truncated_history),
            "s5_robustness/original": _record(f"{instance_id}-original", original_item),
            "s5_robustness/ablate_1": _record(f"{instance_id}-ablated", ablated_item),
        },
    )


def test_both_positive_filter_preserves_default_and_restricts_raw_effects() -> None:
    groups = dict(
        [
            _group(
                "both-positive",
                full_history=0.8,
                truncated_history=0.6,
                original_item=0.7,
                ablated_item=0.6,
            ),
            _group(
                "non-positive-user",
                full_history=0.6,
                truncated_history=0.6,
                original_item=0.7,
                ablated_item=0.5,
            ),
            _group(
                "non-positive-item",
                full_history=0.8,
                truncated_history=0.5,
                original_item=0.5,
                ablated_item=0.6,
            ),
        ]
    )

    default_cases = _build_source_cases(groups, min_effect_gap=0.0)
    restricted_cases = _build_source_cases(
        groups,
        min_effect_gap=0.0,
        require_both_positive_effects=True,
    )

    assert {case["instance_id"] for case in default_cases} == {
        "both-positive",
        "non-positive-user",
        "non-positive-item",
    }
    assert [case["instance_id"] for case in restricted_cases] == ["both-positive"]
