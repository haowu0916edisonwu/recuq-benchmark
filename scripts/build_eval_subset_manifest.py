#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.data.base import load_processed_bundle
from recuq_benchmark.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed evaluation subset manifest from processed test splits."
    )
    parser.add_argument("--config", type=str, default="configs/experiment/llama_controlled_suite.yaml")
    parser.add_argument("--subset_size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path, e.g. configs/eval_subsets/aligned20_n1000_seed42.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)["experiment"]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema_version": "eval_subset_manifest_v1",
        "config": args.config,
        "subset_size": args.subset_size,
        "seed": args.seed,
        "sampling_policy": "uniform_random_without_replacement_per_dataset",
        "datasets": {},
    }

    for dataset_config in config["datasets"]:
        dataset_name = str(dataset_config["name"])
        bundle = load_processed_bundle(dataset_config["processed_path"])
        test_instances = list(bundle.test_instances)
        sample_size = min(args.subset_size, len(test_instances))
        rng = random.Random(_dataset_seed(args.seed, dataset_name))
        selected = rng.sample(test_instances, sample_size)

        summary["datasets"][dataset_name] = {
            "source_test_size": len(test_instances),
            "sample_size": sample_size,
            "processed_path": dataset_config["processed_path"],
        }

        for sample_index, instance in enumerate(selected):
            rows.append(
                {
                    "schema_version": "eval_subset_manifest_v1",
                    "dataset": dataset_name,
                    "split": "test",
                    "instance_id": instance.instance_id,
                    "sample_index": sample_index,
                    "seed": args.seed,
                    "sampling_policy": "uniform_random_without_replacement_per_dataset",
                    "source_test_size": len(test_instances),
                }
            )

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(rows)} manifest rows to {output_path}")
    print(f"Wrote summary to {summary_path}")


def _dataset_seed(seed: int, dataset_name: str) -> int:
    payload = f"{seed}:{dataset_name}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


if __name__ == "__main__":
    main()
