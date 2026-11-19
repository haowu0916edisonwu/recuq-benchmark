from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from recuq_benchmark.data.types import Item, RecommendationInstance


@dataclass
class Variant:
    variant_id: str
    instance_id: str
    dataset: str
    user_id: str
    scenario: str
    variant_name: str
    target_source: str
    history_text: list[str]
    candidate_ids: list[str]
    candidate_texts: list[str]
    labels: list[int]
    ground_truth_item_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class ScenarioBuilder(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name = str(config["name"])
        self.target_source = str(config.get("target_source", "none"))

    @abstractmethod
    def build(
        self,
        instance: RecommendationInstance,
        item_catalog: dict[str, Item],
        train_item_counts: dict[str, int],
        dataset_config: dict[str, Any],
    ) -> list[Variant]:
        """Build scenario variants for one recommendation instance."""

    def _history_text(
        self,
        instance: RecommendationInstance,
        dataset_config: dict[str, Any],
        history_limit: int | None = None,
    ) -> list[str]:
        fields = dataset_config.get("item_text_fields")
        events = instance.history[-history_limit:] if history_limit is not None else instance.history
        return [event.render_text(fields=fields) for event in events]

    def _candidate_texts(
        self,
        candidate_ids: list[str],
        item_catalog: dict[str, Item],
        dataset_config: dict[str, Any],
        ablate_fields: set[str] | None = None,
    ) -> list[str]:
        fields = dataset_config.get("item_text_fields")
        texts: list[str] = []
        for item_id in candidate_ids:
            item = item_catalog.get(item_id)
            if item is None:
                texts.append("No item text available.")
                continue
            texts.append(item.render_text(fields=fields, ablate_fields=ablate_fields))
        return texts

    def _variant(
        self,
        instance: RecommendationInstance,
        variant_name: str,
        target_source: str,
        history_text: list[str],
        candidate_texts: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> Variant:
        return Variant(
            variant_id=f"{instance.instance_id}:{self.name}:{variant_name}",
            instance_id=instance.instance_id,
            dataset=instance.dataset,
            user_id=instance.user_id,
            scenario=self.name,
            variant_name=variant_name,
            target_source=target_source,
            history_text=history_text,
            candidate_ids=list(instance.candidate_ids),
            candidate_texts=candidate_texts,
            labels=list(instance.labels),
            ground_truth_item_ids=list(instance.ground_truth_item_ids),
            metadata=metadata or {},
        )
