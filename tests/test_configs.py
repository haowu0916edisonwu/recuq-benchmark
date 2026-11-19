from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_experiment_config_references_resolve() -> None:
    for config_path in (ROOT / "configs/experiment").glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text())["experiment"]
        for key in ("model",):
            assert (config_path.parent / config[key]).resolve().exists()
        for key in ("datasets", "scenarios"):
            for relative in config[key]:
                assert (config_path.parent / relative).resolve().exists()
        if "subset_manifest" in config:
            assert (config_path.parent / config["subset_manifest"]).resolve().exists()


def test_final_protocol_sizes() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiment/llama_controlled_suite.yaml").read_text()
    )["experiment"]
    assert config["evaluation"] == {"subset_size": 1000, "k": 10, "seed": 42}
    qwen = yaml.safe_load(
        (ROOT / "configs/model/qwen3-8b-nonthinking.yaml").read_text()
    )
    assert qwen["chat_template_kwargs"]["enable_thinking"] is False
    assert qwen["do_sample"] is False


def test_tail_target_config_preserves_generated_population() -> None:
    s4 = yaml.safe_load((ROOT / "configs/condition/s4_longtail.yaml").read_text())
    assert set(s4["include_buckets"]) == {"tail", "mid", "head"}


def test_ranking_dry_run_uses_toy_instances(tmp_path: Path) -> None:
    config = ROOT / "configs/experiment/llama_controlled_suite.yaml"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_fixed_rankings.py"), "--dry_run", "--config", str(config)],
        cwd=tmp_path,
        check=True,
    )
    output = tmp_path / "outputs/dry_run/llama_controlled_suite"
    metadata = json.loads((output / "run_metadata.json").read_text())
    assert metadata["dry_run"] is True
    assert metadata["subset_manifest"] is None
    rows = [json.loads(line) for line in (output / "raw_predictions.jsonl").read_text().splitlines()]
    assert {row["dataset"] for row in rows} == {"mind_20", "amazon_books", "amazon_movies", "yelp"}
    assert {row["scenario"] for row in rows} == {"s0_warmstart", "s3_sparsity", "s4_longtail", "s5_robustness"}
    assert (output / "cache").is_dir()
    assert not (tmp_path / "results").exists()
