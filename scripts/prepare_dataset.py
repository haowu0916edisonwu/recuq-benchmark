#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.data.amazon import AmazonLoader
from recuq_benchmark.data.base import DatasetLoader, build_toy_bundle
from recuq_benchmark.data.mind import MINDLoader
from recuq_benchmark.data.yelp import YelpLoader
from recuq_benchmark.utils.config import load_config
from recuq_benchmark.utils.logging import get_logger


DATASET_CONFIGS = {
    "mind": REPO_ROOT / "configs/dataset/mind.yaml",
    "amazon_books": REPO_ROOT / "configs/dataset/amazon_books.yaml",
    "amazon_movies": REPO_ROOT / "configs/dataset/amazon_movies.yaml",
    "yelp": REPO_ROOT / "configs/dataset/yelp.yaml",
}

LOADER_REGISTRY: dict[str, type[DatasetLoader]] = {
    "mind": MINDLoader,
    "amazon_books": AmazonLoader,
    "amazon_movies": AmazonLoader,
    "yelp": YelpLoader,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for the fixed-ranking UQ benchmark.")
    parser.add_argument("--dataset", choices=["mind", "amazon_books", "amazon_movies", "yelp", "all"], required=True)
    parser.add_argument("--config", type=str, default=None, help="Optional single dataset config path.")
    parser.add_argument("--output_dir", type=Path, help="Override the processed bundle directory.")
    parser.add_argument("--dry_run", action="store_true", help="Write synthetic bundles instead of reading raw data.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("prepare_dataset")
    dataset_names = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        config_path = Path(args.config) if args.config and args.dataset != "all" else DATASET_CONFIGS[dataset_name]
        config = load_config(config_path)
        if args.dry_run:
            output_path = (args.output_dir or Path("outputs/dry_run/data")) / f"{dataset_name}.pkl"
            bundle = build_toy_bundle(dataset_name)
            DatasetLoader.save_processed(DatasetLoader, bundle, output_path)
            logger.info("Wrote dry-run bundle for %s to %s", dataset_name, output_path)
            continue
        loader_cls = LOADER_REGISTRY[dataset_name]
        loader = loader_cls()
        output_path = (
            args.output_dir / f"{dataset_name}.pkl" if args.output_dir
            else Path(config.get("processed_path", f"data/processed/{dataset_name}.pkl"))
        )
        bundle = loader.load(config)
        loader.save_processed(bundle, output_path)
        logger.info("Prepared %s: %s", dataset_name, bundle.dataset_stats)


if __name__ == "__main__":
    main()
