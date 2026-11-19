from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from recuq_benchmark.data.base import DatasetLoader
from recuq_benchmark.data.interaction_builder import (
    PositiveEvent,
    build_category_index,
    build_recommendation_instance,
    clean_text_value,
    format_location,
    normalize_categories,
)
from recuq_benchmark.data.types import DatasetBundle, Item


class YelpLoader(DatasetLoader):
    """Yelp Open Dataset loader for review interactions and business metadata."""

    def load(self, config: dict[str, Any]) -> DatasetBundle:
        raw_path = Path(config["raw_path"])
        review_path = raw_path / config.get("review_file", "yelp_academic_dataset_review.json")
        business_path = raw_path / config.get("business_file", "yelp_academic_dataset_business.json")
        missing = [p for p in [review_path, business_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Yelp raw files: "
                + ", ".join(str(p) for p in missing)
                + ". Expected review and business metadata files locally; use --dry_run before data is downloaded."
            )
        item_catalog = self._read_businesses(business_path, config)
        user_events = self._read_positive_review_events(review_path, item_catalog, config)
        item_pool = list(item_catalog)
        category_index = build_category_index(item_catalog)
        train_instances, val_instances, test_instances, train_item_counts, train_review_ids = self._build_instances(
            user_events=user_events,
            item_catalog=item_catalog,
            item_pool=item_pool,
            category_index=category_index,
            config=config,
        )
        self._attach_train_review_summaries(review_path, item_catalog, train_review_ids, config)
        return DatasetBundle(
            name=config["name"],
            train_instances=train_instances,
            val_instances=val_instances,
            test_instances=test_instances,
            item_catalog=item_catalog,
            train_item_counts=dict(train_item_counts),
            dataset_stats={
                "num_train": len(train_instances),
                "num_val": len(val_instances),
                "num_test": len(test_instances),
                "num_items": len(item_catalog),
                "num_positive_reviews": sum(len(events) for events in user_events.values()),
                "max_instances_per_split": config.get("max_instances_per_split"),
            },
        )

    def _read_businesses(self, path: Path, config: dict[str, Any]) -> dict[str, Item]:
        catalog: dict[str, Item] = {}
        text_fields = config.get("item_text_fields", ["name", "categories", "location", "train_review_summary"])
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                item_id = str(row.get("business_id", "")).strip()
                if not item_id:
                    continue
                categories = normalize_categories(row.get("categories"))
                location = format_location(
                    row.get("address"),
                    row.get("city"),
                    row.get("state"),
                    row.get("postal_code"),
                )
                values = {
                    "name": clean_text_value(row.get("name")),
                    "categories": categories,
                    "location": location,
                    "train_review_summary": "",
                }
                catalog[item_id] = Item(
                    item_id=item_id,
                    dataset=config["name"],
                    text_fields={field: values.get(field, clean_text_value(row.get(field))) for field in text_fields},
                    metadata={
                        "categories": categories,
                        "city": clean_text_value(row.get("city")),
                        "state": clean_text_value(row.get("state")),
                        "postal_code": clean_text_value(row.get("postal_code")),
                        "stars": clean_text_value(row.get("stars")),
                        "review_count": clean_text_value(row.get("review_count")),
                        "is_open": clean_text_value(row.get("is_open")),
                    },
                )
        return catalog

    def _read_positive_review_events(
        self,
        path: Path,
        item_catalog: dict[str, Item],
        config: dict[str, Any],
    ) -> dict[str, list[PositiveEvent]]:
        user_col = config.get("user_id_col", "user_id")
        item_col = config.get("item_id_col", "business_id")
        rating_col = config.get("rating_col", "stars")
        timestamp_col = config.get("timestamp_col", "date")
        threshold = float(config.get("positive_threshold", 4.0))
        user_events: dict[str, list[PositiveEvent]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                item_id = str(row.get(item_col, "")).strip()
                user_id = str(row.get(user_col, "")).strip()
                if not item_id or not user_id or item_id not in item_catalog:
                    continue
                rating = _to_float(row.get(rating_col))
                if rating is None or rating < threshold:
                    continue
                timestamp_raw = str(row.get(timestamp_col, "")).strip()
                sort_key = _parse_yelp_datetime(timestamp_raw)
                if sort_key is None:
                    continue
                user_events[user_id].append(
                    PositiveEvent(
                        user_id=user_id,
                        item_id=item_id,
                        timestamp=timestamp_raw,
                        sort_key=sort_key,
                        rating=rating,
                        event_id=clean_text_value(row.get("review_id")),
                    )
                )
        return user_events

    def _build_instances(
        self,
        *,
        user_events: dict[str, list[PositiveEvent]],
        item_catalog: dict[str, Item],
        item_pool: list[str],
        category_index: dict[str, list[str]],
        config: dict[str, Any],
    ):
        max_history = int(config.get("max_history", 50))
        min_history = int(config.get("min_history_for_instance", 1))
        min_user_interactions = max(int(config.get("min_user_interactions", 5)), min_history + 2)
        max_per_split = config.get("max_instances_per_split")
        max_per_split = int(max_per_split) if max_per_split is not None else None
        sampling_cfg = config.get("candidate_sampling", {})
        num_negatives = int(sampling_cfg.get("num_negatives", 19))
        negative_strategy = str(sampling_cfg.get("strategy", "global"))
        exclude_user_history = bool(sampling_cfg.get("exclude_user_history", True))
        seed = int(sampling_cfg.get("seed", 42))

        train_item_counts: Counter[str] = Counter()
        train_review_ids: set[str] = set()
        train_instances = []
        val_instances = []
        test_instances = []
        split_lists = {"train": train_instances, "val": val_instances, "test": test_instances}

        for user_id, events in user_events.items():
            events = [event for event in events if event.item_id in item_catalog]
            events.sort(key=lambda event: (event.sort_key, event.item_id, event.event_id or ""))
            if len(events) < min_user_interactions:
                continue

            train_end = max(0, len(events) - 2)
            for event in events[:train_end]:
                train_item_counts[event.item_id] += 1
                if event.event_id:
                    train_review_ids.add(event.event_id)

            target_indices = {
                "train": min_history if train_end > min_history else None,
                "val": len(events) - 2,
                "test": len(events) - 1,
            }
            for split, target_idx in target_indices.items():
                if target_idx is None or target_idx < min_history:
                    continue
                if max_per_split is not None and len(split_lists[split]) >= max_per_split:
                    continue
                instance = build_recommendation_instance(
                    dataset_name=config["name"],
                    split=split,
                    source="yelp_open_dataset",
                    user_id=user_id,
                    target_idx=target_idx,
                    events=events,
                    item_catalog=item_catalog,
                    item_pool=item_pool,
                    category_index=category_index,
                    max_history=max_history,
                    num_negatives=num_negatives,
                    seed=seed,
                    negative_strategy=negative_strategy,
                    exclude_user_history=exclude_user_history,
                )
                if instance is not None:
                    split_lists[split].append(instance)

        return train_instances, val_instances, test_instances, train_item_counts, train_review_ids

    def _attach_train_review_summaries(
        self,
        path: Path,
        item_catalog: dict[str, Item],
        train_review_ids: set[str],
        config: dict[str, Any],
    ) -> None:
        if not train_review_ids:
            return
        max_reviews = int(config.get("review_summary_max_reviews", 3))
        max_chars = int(config.get("review_summary_max_chars_per_review", 240))
        summaries: dict[str, list[str]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                review_id = clean_text_value(row.get("review_id"))
                if review_id not in train_review_ids:
                    continue
                item_id = str(row.get(config.get("item_id_col", "business_id"), "")).strip()
                if item_id not in item_catalog or len(summaries[item_id]) >= max_reviews:
                    continue
                text = " ".join(str(row.get("text", "")).split())
                if not text:
                    continue
                summaries[item_id].append(text[:max_chars])

        for item_id, snippets in summaries.items():
            item_catalog[item_id].text_fields["train_review_summary"] = " | ".join(snippets)


def _parse_yelp_datetime(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
