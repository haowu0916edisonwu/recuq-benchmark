from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from recuq_benchmark.data.base import DatasetLoader
from recuq_benchmark.data.types import DatasetBundle, HistoryEvent, Item, RecommendationInstance


class MINDLoader(DatasetLoader):
    """Loader for local MIND files.

    This loader expects MIND files under the configured raw_path. It does not
    download data; use --dry_run in scripts before raw data is available.
    """

    NEWS_COLUMNS = [
        "item_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]

    def load(self, config: dict[str, Any]) -> DatasetBundle:
        raw_path = Path(config["raw_path"])
        train_news = raw_path / config["train_news_file"]
        dev_news = raw_path / config["dev_news_file"]
        train_behaviors = raw_path / config["train_behaviors_file"]
        dev_behaviors = raw_path / config["dev_behaviors_file"]
        missing = [p for p in [train_news, dev_news, train_behaviors, dev_behaviors] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing MIND raw files: "
                + ", ".join(str(p) for p in missing)
                + ". Put raw files under data/raw or run scripts with --dry_run."
            )

        item_catalog = self._read_news([train_news, dev_news], config)
        validation_date = self._latest_behavior_date(train_behaviors)
        train_instances, val_instances, counts = self._read_train_behaviors(
            train_behaviors,
            item_catalog,
            config,
            validation_date=validation_date,
        )
        dev_instances = self._read_behaviors(dev_behaviors, "test", item_catalog, config)
        return DatasetBundle(
            name=config["name"],
            train_instances=train_instances,
            val_instances=val_instances,
            test_instances=dev_instances,
            item_catalog=item_catalog,
            train_item_counts=dict(counts),
            dataset_stats={
                "num_train": len(train_instances),
                "num_val": len(val_instances),
                "num_test": len(dev_instances),
                "num_items": len(item_catalog),
                "validation_date": validation_date.isoformat() if validation_date else None,
                "max_instances_per_split": config.get("max_instances_per_split"),
            },
        )

    def _read_news(self, paths: list[Path], config: dict[str, Any]) -> dict[str, Item]:
        item_catalog: dict[str, Item] = {}
        text_fields = config.get("item_text_fields", ["title", "abstract", "category"])
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < len(self.NEWS_COLUMNS):
                        parts.extend([""] * (len(self.NEWS_COLUMNS) - len(parts)))
                    row = dict(zip(self.NEWS_COLUMNS, parts))
                    item_id = row["item_id"]
                    item_catalog[item_id] = Item(
                        item_id=item_id,
                        dataset=config["name"],
                        text_fields={field: row.get(field, "") for field in text_fields},
                        metadata={"subcategory": row.get("subcategory", "")},
                    )
        return item_catalog

    def _read_train_behaviors(
        self,
        path: Path,
        item_catalog: dict[str, Item],
        config: dict[str, Any],
        validation_date: date | None,
    ) -> tuple[list[RecommendationInstance], list[RecommendationInstance], Counter[str]]:
        max_per_split = config.get("max_instances_per_split")
        max_per_split = int(max_per_split) if max_per_split is not None else None
        min_history = int(config.get("min_history_for_instance", 1))
        train_instances: list[RecommendationInstance] = []
        val_instances: list[RecommendationInstance] = []
        counts: Counter[str] = Counter()

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                timestamp = parts[2]
                parsed_date = _parse_mind_time(timestamp)
                split = "val" if validation_date and parsed_date and parsed_date.date() == validation_date else "train"
                instance = self._parse_behavior_parts(parts, split, item_catalog, config)
                if instance is None:
                    continue
                if split == "train":
                    for event in instance.history:
                        counts[event.item_id] += 1
                    for item_id, label in zip(instance.candidate_ids, instance.labels):
                        if label:
                            counts[item_id] += 1
                if len(instance.history) < min_history:
                    continue
                target = val_instances if split == "val" else train_instances
                if max_per_split is None or len(target) < max_per_split:
                    target.append(instance)

        if validation_date is None and train_instances and not val_instances:
            val_size = max(1, int(0.1 * len(train_instances)))
            val_instances = train_instances[-val_size:]
            train_instances = train_instances[:-val_size]
        return train_instances, val_instances, counts

    def _read_behaviors(
        self,
        path: Path,
        split: str,
        item_catalog: dict[str, Item],
        config: dict[str, Any],
    ) -> list[RecommendationInstance]:
        instances: list[RecommendationInstance] = []
        max_per_split = config.get("max_instances_per_split")
        max_per_split = int(max_per_split) if max_per_split is not None else None
        min_history = int(config.get("min_history_for_instance", 1))
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                instance = self._parse_behavior_parts(parts, split, item_catalog, config)
                if instance is None or len(instance.history) < min_history:
                    continue
                if max_per_split is not None and len(instances) >= max_per_split:
                    break
                instances.append(instance)
        return instances

    def _parse_behavior_parts(
        self,
        parts: list[str],
        split: str,
        item_catalog: dict[str, Item],
        config: dict[str, Any],
    ) -> RecommendationInstance | None:
        if len(parts) < 5:
            return None
        max_history = int(config.get("max_history", 50))
        impression_id, user_id, timestamp, history_raw, impressions_raw = parts[:5]
        history_ids = [item_id for item_id in history_raw.split() if item_id in item_catalog][-max_history:]
        candidate_ids: list[str] = []
        labels: list[int] = []
        for token in impressions_raw.split():
            if "-" not in token:
                continue
            item_id, label_raw = token.rsplit("-", 1)
            if item_id not in item_catalog:
                continue
            try:
                label = int(label_raw)
            except ValueError:
                continue
            candidate_ids.append(item_id)
            labels.append(label)
        positives = [item_id for item_id, label in zip(candidate_ids, labels) if label == 1]
        if not history_ids or not candidate_ids or not positives:
            return None
        history = [
            HistoryEvent(
                item_id=item_id,
                timestamp=timestamp,
                text_fields=item_catalog[item_id].text_fields,
                metadata={"dataset": config["name"]},
            )
            for item_id in history_ids
        ]
        instance = RecommendationInstance(
            instance_id=f"{split}_{impression_id}",
            dataset=config["name"],
            user_id=user_id,
            timestamp=timestamp,
            history=history,
            candidate_ids=candidate_ids,
            labels=labels,
            ground_truth_item_ids=positives,
            metadata={"split": split, "source": "mind_impression"},
        )
        instance.validate()
        return instance

    def _latest_behavior_date(self, path: Path) -> date | None:
        latest: date | None = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                parsed = _parse_mind_time(parts[2])
                if parsed is None:
                    continue
                current = parsed.date()
                latest = current if latest is None or current > latest else latest
        return latest


def _parse_mind_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        return None
