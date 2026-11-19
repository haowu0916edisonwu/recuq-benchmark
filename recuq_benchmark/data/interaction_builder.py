from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
import math
import random
from typing import Any

from recuq_benchmark.data.sampling import deterministic_negative_sample, stable_seed
from recuq_benchmark.data.types import HistoryEvent, Item, RecommendationInstance


@dataclass(frozen=True)
class PositiveEvent:
    user_id: str
    item_id: str
    timestamp: int | str
    sort_key: Any
    rating: float | None = None
    event_id: str | None = None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def clean_text_value(value: Any) -> Any:
    if is_missing(value):
        return ""
    if isinstance(value, list):
        return [str(v).strip() for v in value if not is_missing(v) and str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if not is_missing(v) and str(v).strip()]
    if hasattr(value, "tolist"):
        return clean_text_value(value.tolist())
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items() if not is_missing(v)}
    text = str(value).strip()
    return "" if text.lower() in {"none", "nan", "null", "<na>", "nat"} else text


def normalize_categories(value: Any, fallback: Any = None) -> list[str]:
    raw = clean_text_value(value)
    categories: list[str]
    if isinstance(raw, list):
        categories = raw
    elif isinstance(raw, str) and raw:
        categories = [part.strip() for part in raw.split(",")]
    else:
        categories = []

    if not categories and fallback is not None:
        fallback_value = clean_text_value(fallback)
        if isinstance(fallback_value, list):
            categories = fallback_value
        elif isinstance(fallback_value, str) and fallback_value:
            categories = [fallback_value]

    return [category for category in categories if category]


def format_location(*parts: Any) -> str:
    values = [str(clean_text_value(part)).strip() for part in parts]
    return ", ".join(value for value in values if value)


def build_category_index(item_catalog: dict[str, Item]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = defaultdict(list)
    for item_id, item in item_catalog.items():
        for category in item.metadata.get("categories", []):
            index[str(category)].append(item_id)
    return dict(index)


def build_recommendation_instance(
    *,
    dataset_name: str,
    split: str,
    source: str,
    user_id: str,
    target_idx: int,
    events: Sequence[PositiveEvent],
    item_catalog: dict[str, Item],
    item_pool: Sequence[str],
    category_index: dict[str, list[str]],
    max_history: int,
    num_negatives: int,
    seed: int,
    negative_strategy: str,
    exclude_user_history: bool,
) -> RecommendationInstance | None:
    if target_idx <= 0 or target_idx >= len(events):
        return None
    target = events[target_idx]
    if target.item_id not in item_catalog:
        return None
    history_events = [event for event in events[:target_idx] if event.item_id in item_catalog][-max_history:]
    if not history_events:
        return None

    user_positive_items = {event.item_id for event in events}
    excluded = set(user_positive_items) if exclude_user_history else set()
    positives = {target.item_id}
    preferred_pool: list[str] = []
    if negative_strategy == "same_category_then_global":
        for category in item_catalog[target.item_id].metadata.get("categories", []):
            preferred_pool.extend(category_index.get(str(category), []))

    instance_seed = stable_seed(seed, dataset_name, split, user_id, target.timestamp, target.item_id)
    negatives = deterministic_negative_sample(
        item_pool=item_pool,
        positives=positives,
        excluded=excluded,
        num_negatives=num_negatives,
        seed=instance_seed,
        preferred_pool=preferred_pool,
    )
    if len(negatives) < num_negatives:
        return None

    candidate_ids = [target.item_id] + negatives
    shuffle_seed = stable_seed(seed, "shuffle", dataset_name, split, user_id, target.timestamp, target.item_id)
    random.Random(shuffle_seed).shuffle(candidate_ids)
    labels = [1 if item_id == target.item_id else 0 for item_id in candidate_ids]
    history = [
        HistoryEvent(
            item_id=event.item_id,
            timestamp=event.timestamp,
            text_fields=item_catalog[event.item_id].text_fields,
            metadata={
                "dataset": dataset_name,
                "rating": event.rating,
                "event_id": event.event_id,
            },
        )
        for event in history_events
    ]
    instance = RecommendationInstance(
        instance_id=f"{split}_{stable_seed(seed, source, user_id, target.timestamp, target.item_id):08x}",
        dataset=dataset_name,
        user_id=user_id,
        timestamp=target.timestamp,
        history=history,
        candidate_ids=candidate_ids,
        labels=labels,
        ground_truth_item_ids=[target.item_id],
        metadata={
            "split": split,
            "source": source,
            "target_item_id": target.item_id,
            "target_rating": target.rating,
            "target_event_id": target.event_id,
            "history_length": len(history),
            "num_negatives": num_negatives,
            "negative_strategy": negative_strategy,
        },
    )
    instance.validate()
    return instance
