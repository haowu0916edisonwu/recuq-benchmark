from __future__ import annotations

from typing import Any

from recuq_benchmark.data.types import Item, RecommendationInstance
from recuq_benchmark.scenarios.base import ScenarioBuilder, Variant


class S3SparsityBuilder(ScenarioBuilder):
    def build(
        self,
        instance: RecommendationInstance,
        item_catalog: dict[str, Item],
        train_item_counts: dict[str, int],
        dataset_config: dict[str, Any],
    ) -> list[Variant]:
        variants: list[Variant] = []
        candidate_texts = self._candidate_texts(instance.candidate_ids, item_catalog, dataset_config)
        lengths = [int(length) for length in self.config.get("truncate_lengths", [1, 3, 5, 10])]
        for length in lengths:
            if len(instance.history) < length:
                continue
            history_text = self._history_text(instance, dataset_config, history_limit=length)
            variants.append(
                self._variant(
                    instance=instance,
                    variant_name=f"truncate_{length}",
                    target_source=self.target_source,
                    history_text=history_text,
                    candidate_texts=candidate_texts,
                    metadata={"history_length": length, "truncation": True},
                )
            )
        if self.config.get("include_full", True):
            max_history = int(dataset_config.get("max_history", 50))
            history_text = self._history_text(instance, dataset_config, history_limit=max_history)
            variants.append(
                self._variant(
                    instance=instance,
                    variant_name="full",
                    target_source="none",
                    history_text=history_text,
                    candidate_texts=candidate_texts,
                    metadata={"history_length": len(history_text), "truncation": False},
                )
            )
        return variants
