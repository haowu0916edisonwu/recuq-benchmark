from __future__ import annotations

from typing import Any

from recuq_benchmark.data.types import Item, RecommendationInstance
from recuq_benchmark.scenarios.base import ScenarioBuilder, Variant


class S0WarmstartBuilder(ScenarioBuilder):
    def build(
        self,
        instance: RecommendationInstance,
        item_catalog: dict[str, Item],
        train_item_counts: dict[str, int],
        dataset_config: dict[str, Any],
    ) -> list[Variant]:
        max_history = int(dataset_config.get("max_history", 50))
        history_text = self._history_text(instance, dataset_config, history_limit=max_history)
        candidate_texts = self._candidate_texts(instance.candidate_ids, item_catalog, dataset_config)
        return [
            self._variant(
                instance=instance,
                variant_name="original",
                target_source="none",
                history_text=history_text,
                candidate_texts=candidate_texts,
                metadata={"history_length": len(history_text)},
            )
        ]
