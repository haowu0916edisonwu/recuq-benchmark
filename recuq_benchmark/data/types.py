from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items() if v is not None)
    return str(value)


@dataclass
class Item:
    item_id: str
    dataset: str
    text_fields: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_text(
        self,
        fields: list[str] | None = None,
        ablate_fields: set[str] | None = None,
        max_chars_per_field: int = 500,
    ) -> str:
        selected_fields = fields or list(self.text_fields.keys())
        ablated = ablate_fields or set()
        chunks: list[str] = []
        for field_name in selected_fields:
            if field_name in ablated:
                continue
            value = _format_value(self.text_fields.get(field_name, ""))
            if not value:
                continue
            if max_chars_per_field > 0:
                value = value[:max_chars_per_field]
            label = field_name.replace("_", " ").title()
            chunks.append(f"{label}: {value}")
        return "\n".join(chunks) if chunks else "No item text available."


@dataclass
class HistoryEvent:
    item_id: str
    timestamp: int | str
    text_fields: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def render_text(self, fields: list[str] | None = None, max_chars_per_field: int = 300) -> str:
        item = Item(
            item_id=self.item_id,
            dataset=str(self.metadata.get("dataset", "")),
            text_fields=self.text_fields,
            metadata=self.metadata,
        )
        return item.render_text(fields=fields, max_chars_per_field=max_chars_per_field)


@dataclass
class RecommendationInstance:
    instance_id: str
    dataset: str
    user_id: str
    timestamp: int | str
    history: list[HistoryEvent]
    candidate_ids: list[str]
    labels: list[int]
    ground_truth_item_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if len(self.candidate_ids) != len(self.labels):
            raise ValueError(
                f"candidate_ids and labels length mismatch for instance {self.instance_id}: "
                f"{len(self.candidate_ids)} vs {len(self.labels)}"
            )
        if not self.candidate_ids:
            raise ValueError(f"instance {self.instance_id} has no candidates")
        if not self.ground_truth_item_ids:
            raise ValueError(f"instance {self.instance_id} has no ground-truth positives")


@dataclass
class DatasetBundle:
    name: str
    train_instances: list[RecommendationInstance]
    val_instances: list[RecommendationInstance]
    test_instances: list[RecommendationInstance]
    item_catalog: dict[str, Item]
    train_item_counts: dict[str, int]
    dataset_stats: dict[str, Any] = field(default_factory=dict)

    def all_instances(self) -> list[RecommendationInstance]:
        return self.train_instances + self.val_instances + self.test_instances
