from __future__ import annotations

import json
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recuq_benchmark.data.base import build_toy_bundle


@pytest.fixture
def ranking_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    bundle = build_toy_bundle("toy")
    processed = tmp_path / "toy.pkl"
    with processed.open("wb") as handle:
        pickle.dump(bundle, handle)
    manifest = tmp_path / "subset.jsonl"
    manifest.write_text(json.dumps({"dataset": "toy", "instance_id": bundle.test_instances[0].instance_id}) + "\n")
    config = tmp_path / "experiment.yaml"
    config.write_text(json.dumps({
        "experiment": {
            "model": {"model_name": "toy"},
            "datasets": [{"name": "toy", "processed_path": str(processed)}],
            "scenarios": [{"name": "s0_warmstart"}],
            "evaluation": {"k": 3, "seed": 42, "subset_size": 1},
        }
    }))
    base_dir = tmp_path / "base"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/generate_fixed_rankings.py"),
        "--dry_run", "--config", str(config), "--subset_manifest", str(manifest),
        "--output_dir", str(base_dir),
    ], cwd=tmp_path, check=True)
    return config, manifest, base_dir / "raw_predictions.jsonl"


@pytest.mark.parametrize("script", (
    "run_uq_labelprob.py", "run_uq_kweon_decomp.py",
    "run_uq_self_consistency.py", "run_uq_listwise_same_inference.py",
))
def test_validate_only_preserves_predictions_and_checks_coverage(
    tmp_path: Path, ranking_inputs: tuple[Path, Path, Path], script: str,
) -> None:
    config, manifest, base = ranking_inputs
    output_dir = tmp_path / "uq"
    output_dir.mkdir()
    predictions = output_dir / "uq_predictions.jsonl"
    original = "existing local prediction\n"
    predictions.write_text(original)
    command = [
        sys.executable, str(ROOT / "scripts" / script),
        "--config", str(config), "--base_predictions", str(base), "--subset_manifest", str(manifest),
        "--output_dir", str(output_dir), "--method", "toy", "--validate_only", "--overwrite",
    ]
    subprocess.run(command, cwd=tmp_path, check=True)
    assert predictions.read_text() == original

    row = json.loads(base.read_text())
    row["record_id"] = "additional_base_record"
    with base.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Reconstructed 1 of 2 base records" in result.stderr
    assert predictions.read_text() == original
