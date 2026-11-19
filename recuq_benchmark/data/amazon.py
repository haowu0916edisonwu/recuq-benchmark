from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from recuq_benchmark.data.base import DatasetLoader
from recuq_benchmark.data.interaction_builder import (
    PositiveEvent,
    build_category_index,
    build_recommendation_instance,
    clean_text_value,
    normalize_categories,
)
from recuq_benchmark.data.types import DatasetBundle, Item


class AmazonLoader(DatasetLoader):
    """Shared Amazon Reviews 2023 loader for Books and Movies & TV.

    Review tables are used only for user behavior, ratings, and timestamps.
    Candidate item text comes from the metadata table joined by parent_asin,
    which avoids using held-out review text as candidate evidence.
    """

    def load(self, config: dict[str, Any]) -> DatasetBundle:
        raw_path = Path(config["raw_path"])
        review_path = raw_path / config.get("review_file", "reviews.parquet")
        meta_path = raw_path / config.get("meta_file", "meta.parquet")
        missing = [p for p in [review_path, meta_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing Amazon raw files: "
                + ", ".join(str(p) for p in missing)
                + ". Expected Review and Meta tables locally; use --dry_run before data is downloaded."
            )

        review_df = self._read_reviews(review_path, config)
        meta_df = self._read_metadata(meta_path, config)
        item_catalog = self._build_item_catalog(meta_df, config)
        item_pool = list(item_catalog)
        category_index = build_category_index(item_catalog)
        train_instances, val_instances, test_instances, train_item_counts = self._build_instances(
            review_df=review_df,
            item_catalog=item_catalog,
            item_pool=item_pool,
            category_index=category_index,
            config=config,
        )
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
                "num_positive_reviews": int(len(review_df)),
                "max_instances_per_split": config.get("max_instances_per_split"),
            },
        )

    def _read_reviews(self, path: Path, config: dict[str, Any]):
        pd = _import_pandas()
        user_col = config.get("user_id_col", "user_id")
        item_col = config.get("item_id_col", "parent_asin")
        rating_col = config.get("rating_col", "rating")
        timestamp_col = config.get("timestamp_col", "timestamp")
        columns = [user_col, item_col, rating_col, timestamp_col]
        df = pd.read_parquet(path, columns=columns)
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise ValueError(f"Amazon review table is missing columns: {missing}")
        df = df.dropna(subset=[user_col, item_col, rating_col, timestamp_col])
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
        df[timestamp_col] = pd.to_numeric(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=[rating_col, timestamp_col])
        threshold = float(config.get("positive_threshold", 4.0))
        df = df[df[rating_col] >= threshold].copy()
        df[user_col] = df[user_col].astype(str)
        df[item_col] = df[item_col].astype(str)
        df[timestamp_col] = df[timestamp_col].astype("int64")
        return df

    def _read_metadata(self, path: Path, config: dict[str, Any]):
        pd = _import_pandas()
        item_col = config.get("item_id_col", "parent_asin")
        requested = set(config.get("item_text_fields", []))
        metadata_columns = {
            item_col,
            "main_category",
            "title",
            "subtitle",
            "author",
            "store",
            "categories",
            "description",
            "average_rating",
            "rating_number",
            "price",
        }
        try:
            df = pd.read_parquet(path, columns=sorted(metadata_columns | requested))
        except Exception:
            df = pd.read_parquet(path)
        if item_col not in df.columns:
            raise ValueError(f"Amazon metadata table is missing item id column: {item_col}")
        df = df.dropna(subset=[item_col]).copy()
        df[item_col] = df[item_col].astype(str)
        return df

    def _build_item_catalog(self, meta_df, config: dict[str, Any]) -> dict[str, Item]:
        item_col = config.get("item_id_col", "parent_asin")
        text_fields = config.get("item_text_fields", ["title", "categories", "description"])
        catalog: dict[str, Item] = {}
        for row in meta_df.to_dict(orient="records"):
            item_id = str(row.get(item_col, "")).strip()
            if not item_id:
                continue
            categories = normalize_categories(row.get("categories"), fallback=row.get("main_category"))
            row_values = {
                "title": clean_text_value(row.get("title")),
                "subtitle": clean_text_value(row.get("subtitle")),
                "author": clean_text_value(row.get("author")),
                "store": clean_text_value(row.get("store")),
                "categories": categories,
                "description": clean_text_value(row.get("description")),
            }
            catalog[item_id] = Item(
                item_id=item_id,
                dataset=config["name"],
                text_fields={field: row_values.get(field, clean_text_value(row.get(field))) for field in text_fields},
                metadata={
                    "main_category": clean_text_value(row.get("main_category")),
                    "categories": categories,
                    "average_rating": clean_text_value(row.get("average_rating")),
                    "rating_number": clean_text_value(row.get("rating_number")),
                    "price": clean_text_value(row.get("price")),
                },
            )
        return catalog

    def _build_instances(
        self,
        *,
        review_df,
        item_catalog: dict[str, Item],
        item_pool: list[str],
        category_index: dict[str, list[str]],
        config: dict[str, Any],
    ):
        user_col = config.get("user_id_col", "user_id")
        item_col = config.get("item_id_col", "parent_asin")
        rating_col = config.get("rating_col", "rating")
        timestamp_col = config.get("timestamp_col", "timestamp")
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

        review_df = review_df[review_df[item_col].isin(item_catalog)].copy()
        review_df = review_df.sort_values([user_col, timestamp_col], kind="mergesort")
        train_item_counts: Counter[str] = Counter()
        train_instances = []
        val_instances = []
        test_instances = []
        split_lists = {"train": train_instances, "val": val_instances, "test": test_instances}

        for user_id, group in review_df.groupby(user_col, sort=False):
            if len(group) < min_user_interactions:
                continue
            events = [
                PositiveEvent(
                    user_id=str(user_id),
                    item_id=str(row[item_col]),
                    timestamp=int(row[timestamp_col]),
                    sort_key=int(row[timestamp_col]),
                    rating=float(row[rating_col]),
                )
                for row in group.to_dict(orient="records")
                if str(row[item_col]) in item_catalog
            ]
            events.sort(key=lambda event: (event.sort_key, event.item_id))
            if len(events) < min_user_interactions:
                continue

            train_end = max(0, len(events) - 2)
            for event in events[:train_end]:
                train_item_counts[event.item_id] += 1

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
                    source="amazon_reviews_2023",
                    user_id=str(user_id),
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

        return train_instances, val_instances, test_instances, train_item_counts


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to parse Amazon parquet files.") from exc
    return pd
