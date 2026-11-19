from __future__ import annotations

from typing import Any

from recuq_benchmark.data.types import Item, RecommendationInstance
from recuq_benchmark.scenarios.base import ScenarioBuilder, Variant


class S5RobustnessBuilder(ScenarioBuilder):
    def build(
        self,
        instance: RecommendationInstance,
        item_catalog: dict[str, Item],
        train_item_counts: dict[str, int],
        dataset_config: dict[str, Any],
    ) -> list[Variant]:
        max_history = int(dataset_config.get("max_history", 50))
        history_text = self._history_text(instance, dataset_config, history_limit=max_history)
        variants = [
            self._variant(
                instance=instance,
                variant_name="original",
                target_source="none",
                history_text=history_text,
                candidate_texts=self._candidate_texts(instance.candidate_ids, item_catalog, dataset_config),
                metadata={"ablated_field": None},
            )
        ]
        fields = dataset_config.get("s5_ablate_fields") or self.config.get("ablate_fields", [])
        for field_name in fields:
            if not self._field_available(field_name, instance.candidate_ids, item_catalog):
                continue
            candidate_texts = self._candidate_texts(
                instance.candidate_ids,
                item_catalog,
                dataset_config,
                ablate_fields={field_name},
            )
            variants.append(
                self._variant(
                    instance=instance,
                    variant_name=f"ablate_{field_name}",
                    target_source=self.target_source,
                    history_text=history_text,
                    candidate_texts=candidate_texts,
                    metadata={"ablated_field": field_name},
                )
            )
        return variants

    def _field_available(
        self,
        field_name: str,
        candidate_ids: list[str],
        item_catalog: dict[str, Item],
    ) -> bool:
        for item_id in candidate_ids:
            item = item_catalog.get(item_id)
            if item and item.text_fields.get(field_name):
                return True
        return False
