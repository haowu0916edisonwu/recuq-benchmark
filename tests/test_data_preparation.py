from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recuq_benchmark.data.base import load_processed_bundle
from recuq_benchmark.utils.config import load_config


def test_amazon_conversion_and_preparation_preserve_ids_and_output_override(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    from scripts.download_prepare_amazon_2023 import convert_meta_jsonl, convert_reviews_csv

    reviews = tmp_path / "reviews.csv"
    # Aliased, reordered fields exercise the Hub CSV ingestion and ID preservation.
    reviews.write_text("time,asin,overall,user\n" + "".join(
        f"{i},{i:06},5,0001\n" for i in range(13)
    ) + "100,000020,2,0001\n")
    items = tmp_path / "meta.jsonl"
    items.write_text("".join(json.dumps({
        "parent_asin": f"{i:06}",
        "title": f"Item {i}",
        "categories": ["sample"],
    }) + "\n" for i in range(35)))
    convert_reviews_csv(reviews, tmp_path / "reviews.parquet", 3, force=False)
    convert_meta_jsonl(items, tmp_path / "meta.parquet", 10, force=False)

    config = load_config(ROOT / "configs/dataset/amazon_books.yaml")
    config["raw_path"] = str(tmp_path)
    configured_output = tmp_path / "default.pkl"
    config["processed_path"] = str(configured_output)
    config_path = tmp_path / "book-config.yaml"
    config_path.write_text(json.dumps(config))
    processed = tmp_path / "custom"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/prepare_dataset.py"), "--dataset", "amazon_books",
        "--config", str(config_path), "--output_dir", str(processed),
    ], cwd=tmp_path, check=True)

    assert not configured_output.exists()
    bundle = load_processed_bundle(processed / "amazon_books.pkl")
    val, test = bundle.val_instances[0], bundle.test_instances[0]
    assert test.user_id == "0001"
    assert val.ground_truth_item_ids == ["000011"]
    assert test.ground_truth_item_ids == ["000012"]
    assert test.history[-1].item_id == "000011"
    assert len(set(test.candidate_ids)) == 20
    assert sum(test.labels) == 1
    assert set(test.candidate_ids) & {f"{i:06}" for i in range(13)} == {"000012"}
    assert "000011" not in bundle.train_item_counts
