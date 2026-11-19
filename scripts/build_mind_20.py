#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import pickle
import random
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.data.sampling import stable_seed
from recuq_benchmark.data.types import DatasetBundle, HistoryEvent, Item, RecommendationInstance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive an aligned MIND-20 pool from the official MIND processed bundle. "
            "Each output instance has 1 clicked positive and 19 official impression negatives."
        )
    )
    parser.add_argument("--input", type=str, default="data/processed/mind.pkl")
    parser.add_argument("--output", type=str, default="data/processed/mind_20.pkl")
    parser.add_argument("--name", type=str, default="mind_20")
    parser.add_argument("--num_negatives", type=int, default=19)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_per_split",
        type=int,
        default=None,
        help="Optional cap after filtering. Default keeps all eligible instances.",
    )
    parser.add_argument("--summary_out", type=str, default="results/mind_20_build_summary.json")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists. Pass --force to overwrite it.")

    with input_path.open("rb") as f:
        source: DatasetBundle = pickle.load(f)

    item_catalog = {
        item_id: replace(item, dataset=args.name)
        for item_id, item in source.item_catalog.items()
    }

    build_stats: dict[str, Any] = {}
    train_instances, build_stats["train"] = derive_split(
        source.train_instances,
        split="train",
        output_dataset=args.name,
        seed=args.seed,
        num_negatives=args.num_negatives,
        max_instances=args.max_per_split,
    )
    val_instances, build_stats["val"] = derive_split(
        source.val_instances,
        split="val",
        output_dataset=args.name,
        seed=args.seed,
        num_negatives=args.num_negatives,
        max_instances=args.max_per_split,
    )
    test_instances, build_stats["test"] = derive_split(
        source.test_instances,
        split="test",
        output_dataset=args.name,
        seed=args.seed,
        num_negatives=args.num_negatives,
        max_instances=args.max_per_split,
    )

    bundle = DatasetBundle(
        name=args.name,
        train_instances=train_instances,
        val_instances=val_instances,
        test_instances=test_instances,
        item_catalog=item_catalog,
        train_item_counts=dict(source.train_item_counts),
        dataset_stats={
            **source.dataset_stats,
            "source_dataset": source.name,
            "source_processed_path": str(input_path),
            "candidate_policy": f"1 positive + {args.num_negatives} official impression negatives",
            "num_negatives": args.num_negatives,
            "seed": args.seed,
            "max_per_split": args.max_per_split,
            "num_train": len(train_instances),
            "num_val": len(val_instances),
            "num_test": len(test_instances),
            "num_items": len(item_catalog),
            "build_stats": build_stats,
        },
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(bundle, f)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "name": args.name,
        "num_negatives": args.num_negatives,
        "seed": args.seed,
        "max_per_split": args.max_per_split,
        "splits": build_stats,
        "dataset_stats": bundle.dataset_stats,
    }
    if args.summary_out:
        summary_path = Path(args.summary_out)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def derive_split(
    instances: list[RecommendationInstance],
    *,
    split: str,
    output_dataset: str,
    seed: int,
    num_negatives: int,
    max_instances: int | None,
) -> tuple[list[RecommendationInstance], dict[str, Any]]:
    output: list[RecommendationInstance] = []
    skip_reasons: Counter[str] = Counter()
    original_candidate_sizes: list[int] = []
    original_positive_counts: list[int] = []

    for instance in instances:
        if max_instances is not None and len(output) >= max_instances:
            break
        positives = [
            (idx, item_id)
            for idx, (item_id, label) in enumerate(zip(instance.candidate_ids, instance.labels))
            if label > 0
        ]
        negatives = [
            item_id
            for item_id, label in zip(instance.candidate_ids, instance.labels)
            if label <= 0
        ]
        original_candidate_sizes.append(len(instance.candidate_ids))
        original_positive_counts.append(len(positives))

        if not positives:
            skip_reasons["no_positive"] += 1
            continue
        if len(negatives) < num_negatives:
            skip_reasons["insufficient_official_negatives"] += 1
            continue

        instance_seed = stable_seed(seed, output_dataset, split, instance.instance_id)
        rng = random.Random(instance_seed)
        positive_original_idx, positive_item_id = rng.choice(positives)
        sampled_negatives = rng.sample(negatives, num_negatives)
        candidate_ids = [positive_item_id] + sampled_negatives
        rng.shuffle(candidate_ids)
        labels = [1 if item_id == positive_item_id else 0 for item_id in candidate_ids]

        derived = RecommendationInstance(
            instance_id=f"{output_dataset}_{instance.instance_id}",
            dataset=output_dataset,
            user_id=instance.user_id,
            timestamp=instance.timestamp,
            history=retag_history(instance.history, output_dataset),
            candidate_ids=candidate_ids,
            labels=labels,
            ground_truth_item_ids=[positive_item_id],
            metadata={
                **instance.metadata,
                "split": split,
                "source": "mind_official_to_20",
                "source_dataset": instance.dataset,
                "source_instance_id": instance.instance_id,
                "source_candidate_size": len(instance.candidate_ids),
                "source_positive_count": len(positives),
                "selected_positive_item_id": positive_item_id,
                "selected_positive_original_index": positive_original_idx,
                "num_negatives": num_negatives,
                "candidate_policy": f"1 positive + {num_negatives} official impression negatives",
                "sampling_seed": instance_seed,
            },
        )
        derived.validate()
        output.append(derived)

    stats = {
        "source_instances": len(instances),
        "derived_instances": len(output),
        "skipped_instances": sum(skip_reasons.values()),
        "skip_reasons": dict(skip_reasons),
        "source_candidate_size": describe(original_candidate_sizes),
        "source_positive_count": describe(original_positive_counts),
        "output_candidate_size": 1 + num_negatives,
        "output_positive_count": 1,
    }
    return output, stats


def retag_history(history: list[HistoryEvent], dataset: str) -> list[HistoryEvent]:
    retagged: list[HistoryEvent] = []
    for event in history:
        metadata = dict(event.metadata)
        metadata["dataset"] = dataset
        retagged.append(
            HistoryEvent(
                item_id=event.item_id,
                timestamp=event.timestamp,
                text_fields=event.text_fields,
                metadata=metadata,
            )
        )
    return retagged


def describe(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p25": percentile(ordered, 25),
        "median": percentile(ordered, 50),
        "mean": sum(ordered) / len(ordered),
        "p75": percentile(ordered, 75),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
    }


def percentile(ordered: list[int], pct: float) -> float:
    if not ordered:
        return 0.0
    idx = (pct / 100.0) * (len(ordered) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


if __name__ == "__main__":
    main()
