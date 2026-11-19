from __future__ import annotations

from pathlib import Path

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
