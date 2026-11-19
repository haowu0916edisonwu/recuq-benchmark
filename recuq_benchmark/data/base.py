from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
import pickle
from typing import Any

from recuq_benchmark.data.types import DatasetBundle, HistoryEvent, Item, RecommendationInstance


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, config: dict[str, Any]) -> DatasetBundle:
        """Load and prepare a dataset bundle from raw local files."""

    def save_processed(self, bundle: DatasetBundle, output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(bundle, f)

    def load_processed(self, processed_path: str | Path) -> DatasetBundle:
        path = Path(processed_path)
        with path.open("rb") as f:
            return pickle.load(f)


def load_processed_bundle(processed_path: str | Path) -> DatasetBundle:
    path = Path(processed_path)
    with path.open("rb") as f:
        return pickle.load(f)


def build_toy_bundle(name: str = "toy") -> DatasetBundle:
    items = {
        f"{name}_item_{idx}": Item(
            item_id=f"{name}_item_{idx}",
            dataset=name,
            text_fields={
                "title": f"{name.title()} item {idx}",
                "abstract": f"Short description for item {idx}.",
                "category": "Books" if idx % 2 == 0 else "News",
                "description": f"Metadata description for item {idx}.",
                "author": "Synthetic Author",
                "name": f"Synthetic business {idx}",
                "location": "Synthetic City",
                "train_review_summary": f"Train-only review summary for item {idx}.",
            },
        )
        for idx in range(1, 9)
    }
    instances: list[RecommendationInstance] = []
    for user_idx in range(1, 4):
        history_ids = [f"{name}_item_{idx}" for idx in range(1, 5)]
        candidate_ids = [f"{name}_item_{idx}" for idx in range(4, 9)]
        labels = [1, 0, 0, 0, 0]
        history = [
            HistoryEvent(
                item_id=item_id,
                timestamp=idx,
                text_fields=items[item_id].text_fields,
                metadata={"dataset": name},
            )
            for idx, item_id in enumerate(history_ids)
        ]
        instances.append(
            RecommendationInstance(
                instance_id=f"{name}_toy_{user_idx}",
                dataset=name,
                user_id=f"{name}_user_{user_idx}",
                timestamp=user_idx,
                history=history,
                candidate_ids=candidate_ids,
                labels=labels,
                ground_truth_item_ids=[candidate_ids[0]],
                metadata={"source": "dry_run"},
            )
        )
    counts = Counter()
    for instance in instances:
        for event in instance.history:
            counts[event.item_id] += 1
    return DatasetBundle(
        name=name,
        train_instances=instances[:1],
        val_instances=instances[1:2],
        test_instances=instances[2:],
        item_catalog=items,
        train_item_counts=dict(counts),
        dataset_stats={"dry_run": True, "num_items": len(items), "num_instances": len(instances)},
    )
