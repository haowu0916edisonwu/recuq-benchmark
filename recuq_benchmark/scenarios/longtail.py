from __future__ import annotations

from typing import Any

from recuq_benchmark.data.types import Item, RecommendationInstance
from recuq_benchmark.scenarios.base import ScenarioBuilder, Variant


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = round((percentile / 100.0) * (len(ordered) - 1))
    return float(ordered[int(idx)])


class S4LongtailBuilder(ScenarioBuilder):
    def build(
        self,
        instance: RecommendationInstance,
        item_catalog: dict[str, Item],
        train_item_counts: dict[str, int],
        dataset_config: dict[str, Any],
    ) -> list[Variant]:
        values = list(train_item_counts.values())
        tail_cut = _percentile(values, float(self.config.get("tail_percentile", 25)))
        head_cut = _percentile(values, float(self.config.get("head_percentile", 75)))
        positive_counts = [train_item_counts.get(item_id, 0) for item_id in instance.ground_truth_item_ids]
        positive_count = max(positive_counts) if positive_counts else 0
        if positive_count <= tail_cut:
            bucket = "tail"
        elif positive_count >= head_cut:
            bucket = "head"
        else:
            bucket = "mid"
        include_buckets = {str(value) for value in self.config.get("include_buckets", ["tail"])}
        unknown_buckets = include_buckets.difference({"tail", "mid", "head"})
        if unknown_buckets:
            raise ValueError(f"Unknown long-tail popularity buckets: {sorted(unknown_buckets)}")
        if bucket not in include_buckets:
            return []
        max_history = int(dataset_config.get("max_history", 50))
        history_text = self._history_text(instance, dataset_config, history_limit=max_history)
        candidate_texts = self._candidate_texts(instance.candidate_ids, item_catalog, dataset_config)
        return [
            self._variant(
                instance=instance,
                variant_name="original",
                target_source=self.target_source,
                history_text=history_text,
                candidate_texts=candidate_texts,
                metadata={
                    "popularity_bucket": bucket,
                    "positive_train_count": positive_count,
                    "tail_cut": tail_cut,
                    "head_cut": head_cut,
                },
            )
        ]
