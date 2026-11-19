#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import statistics
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.data.types import DatasetBundle, RecommendationInstance
from recuq_benchmark.utils.config import load_config


DATASET_CONFIGS = {
    "mind": REPO_ROOT / "configs/dataset/mind.yaml",
    "mind_20": REPO_ROOT / "configs/dataset/mind_20.yaml",
    "amazon_books": REPO_ROOT / "configs/dataset/amazon_books.yaml",
    "amazon_movies": REPO_ROOT / "configs/dataset/amazon_movies.yaml",
    "yelp": REPO_ROOT / "configs/dataset/yelp.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a processed fixed-ranking UQ benchmark DatasetBundle.")
    parser.add_argument("processed_path", type=str, help="Path to data/processed/*.pkl")
    parser.add_argument("--dataset_config", type=str, default=None, help="Optional dataset YAML config path.")
    parser.add_argument("--json_out", type=str, default=None, help="Optional path to write JSON summary.")
    parser.add_argument("--top_items", type=int, default=10, help="Number of popular train items to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.processed_path)
    with path.open("rb") as f:
        bundle: DatasetBundle = pickle.load(f)
    dataset_config = load_dataset_config(bundle.name, args.dataset_config)

    summary = {
        "path": str(path),
        "name": bundle.name,
        "dataset_stats": bundle.dataset_stats,
        "num_items": len(bundle.item_catalog),
        "splits": {
            "train": summarize_instances(bundle.train_instances, bundle.train_item_counts),
            "val": summarize_instances(bundle.val_instances, bundle.train_item_counts),
            "test": summarize_instances(bundle.test_instances, bundle.train_item_counts),
        },
        "train_item_popularity": summarize_counter(bundle.train_item_counts),
        "text_field_coverage": summarize_text_fields(bundle),
        "scenario_variant_estimate_on_test": estimate_variants(bundle.test_instances, dataset_config),
    }

    print_summary(summary, top_items=args.top_items, train_item_counts=bundle.train_item_counts)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nWrote JSON summary: {out_path}")


def load_dataset_config(dataset_name: str, config_path: str | None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DATASET_CONFIGS.get(dataset_name)
    if path is None or not path.exists():
        return {}
    return load_config(path)


def summarize_instances(
    instances: list[RecommendationInstance],
    train_item_counts: dict[str, int],
) -> dict[str, Any]:
    history_lengths = [len(instance.history) for instance in instances]
    candidate_sizes = [len(instance.candidate_ids) for instance in instances]
    positive_counts = [sum(1 for label in instance.labels if label > 0) for instance in instances]
    positive_popularities = [
        max((train_item_counts.get(item_id, 0) for item_id in instance.ground_truth_item_ids), default=0)
        for instance in instances
    ]
    return {
        "n": len(instances),
        "history_length": describe_numeric(history_lengths),
        "candidate_size": describe_numeric(candidate_sizes),
        "positive_label_count": describe_numeric(positive_counts),
        "positive_train_popularity": describe_numeric(positive_popularities),
        "history_length_buckets": bucket_counts(history_lengths, [1, 3, 5, 10, 20, 50]),
        "candidate_size_buckets": bucket_counts(candidate_sizes, [5, 10, 20, 50, 100]),
        "positive_label_count_distribution": dict(sorted(Counter(positive_counts).items())),
    }


def describe_numeric(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0],
        "p25": percentile(ordered, 25),
        "median": percentile(ordered, 50),
        "mean": sum(ordered) / len(ordered),
        "p75": percentile(ordered, 75),
        "p90": percentile(ordered, 90),
        "p95": percentile(ordered, 95),
        "p99": percentile(ordered, 99),
        "max": ordered[-1],
        "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def percentile(ordered: list[int | float], pct: float) -> float:
    if not ordered:
        return 0.0
    idx = (pct / 100.0) * (len(ordered) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(ordered) - 1)
    weight = idx - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def bucket_counts(values: list[int], cutoffs: list[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in values:
        placed = False
        previous = 0
        for cutoff in cutoffs:
            if value <= cutoff:
                label = f"{previous + 1}-{cutoff}" if previous else f"<= {cutoff}"
                counts[label] += 1
                placed = True
                break
            previous = cutoff
        if not placed:
            counts[f"> {cutoffs[-1]}"] += 1
    return dict(counts)


def summarize_counter(counter: dict[str, int]) -> dict[str, Any]:
    values = list(counter.values())
    return {
        "num_observed_items": len(counter),
        "count_distribution": describe_numeric(values),
        "zero_count_items_not_included": True,
    }


def summarize_text_fields(bundle: DatasetBundle) -> dict[str, Any]:
    field_counts: Counter[str] = Counter()
    nonempty_counts: Counter[str] = Counter()
    for item in bundle.item_catalog.values():
        for field_name, value in item.text_fields.items():
            field_counts[field_name] += 1
            if has_value(value):
                nonempty_counts[field_name] += 1
    return {
        field_name: {
            "items_with_field": field_counts[field_name],
            "items_with_nonempty_value": nonempty_counts[field_name],
            "coverage": nonempty_counts[field_name] / field_counts[field_name] if field_counts[field_name] else 0.0,
        }
        for field_name in sorted(field_counts)
    }


def has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def estimate_variants(instances: list[RecommendationInstance], dataset_config: dict[str, Any]) -> dict[str, Any]:
    s3_counts = Counter()
    for instance in instances:
        for length in [1, 3, 5, 10]:
            if len(instance.history) >= length:
                s3_counts[f"truncate_{length}"] += 1
        s3_counts["full"] += 1

    n = len(instances)
    s5_fields = list(dataset_config.get("s5_ablate_fields", []))
    s5_estimate = {"original": n}
    for field_name in s5_fields:
        s5_estimate[f"ablate_{field_name}"] = n
    s5_estimate["total_if_all_configured_fields_available"] = n * (1 + len(s5_fields))

    return {
        "s0_warmstart": n,
        "s3_sparsity": dict(s3_counts),
        "s3_sparsity_total": sum(s3_counts.values()),
        "s4_longtail": n,
        "s5_robustness_upper_bound": s5_estimate,
        "total_upper_bound_for_s0_s3_s4_s5": n + sum(s3_counts.values()) + n + s5_estimate[
            "total_if_all_configured_fields_available"
        ],
    }


def print_summary(summary: dict[str, Any], top_items: int, train_item_counts: dict[str, int]) -> None:
    print(f"Dataset: {summary['name']}")
    print(f"Path: {summary['path']}")
    print(f"Items: {summary['num_items']}")
    print(f"Dataset stats: {summary['dataset_stats']}")

    for split_name, split in summary["splits"].items():
        print(f"\n[{split_name}] n={split['n']}")
        print_compact("history_length", split["history_length"])
        print_compact("candidate_size", split["candidate_size"])
        print_compact("positive_label_count", split["positive_label_count"])
        print_compact("positive_train_popularity", split["positive_train_popularity"])
        print(f"history_length_buckets: {split['history_length_buckets']}")
        print(f"candidate_size_buckets: {split['candidate_size_buckets']}")
        print(f"positive_label_count_distribution: {split['positive_label_count_distribution']}")

    print("\n[train item popularity]")
    print_compact("count_distribution", summary["train_item_popularity"]["count_distribution"])
    print(f"num_observed_items: {summary['train_item_popularity']['num_observed_items']}")
    print(f"top_{top_items}: {Counter(train_item_counts).most_common(top_items)}")

    print("\n[text field coverage]")
    for field_name, stats in summary["text_field_coverage"].items():
        print(
            f"{field_name}: "
            f"{stats['items_with_nonempty_value']}/{stats['items_with_field']} "
            f"({stats['coverage']:.2%})"
        )

    print("\n[scenario variant estimate on test]")
    print(json.dumps(summary["scenario_variant_estimate_on_test"], indent=2, ensure_ascii=False))


def print_compact(name: str, stats: dict[str, Any]) -> None:
    if not stats or stats.get("n", 0) == 0:
        print(f"{name}: n=0")
        return
    keys = ["min", "p25", "median", "mean", "p75", "p90", "p95", "p99", "max"]
    parts = [f"{key}={stats[key]:.2f}" if isinstance(stats[key], float) else f"{key}={stats[key]}" for key in keys]
    print(f"{name}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
