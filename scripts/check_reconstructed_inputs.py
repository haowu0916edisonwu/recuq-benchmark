#!/usr/bin/env python3
"""Check local processed data against the frozen camera-ready inference inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_fixed_rankings import _load_subset_manifest, _select_manifest_instances, record_input_hashes
from recuq_benchmark.data.base import load_processed_bundle
from recuq_benchmark.scenarios import SCENARIO_REGISTRY
from recuq_benchmark.utils.config import load_config

CONFIG = ROOT / "configs/experiment/llama_controlled_suite.yaml"
SUBSET = ROOT / "configs/eval_subsets/aligned20_n1000_seed42.jsonl"
REFERENCE = ROOT / "artifacts/metadata/input_reconstruction.json"
HASH_FIELDS = ("candidate_hash", "history_hash", "prompt_hash")


def summarize(records: dict[str, dict[str, str]]) -> dict[str, int | str]:
    result: dict[str, int | str] = {"rows": len(records)}
    for field in HASH_FIELDS:
        digest = hashlib.sha256()
        for record_id in sorted(records):
            line = json.dumps([record_id, records[record_id][field]], separators=(",", ":"))
            digest.update((line + "\n").encode("utf-8"))
        result[f"{field}_sha256"] = digest.hexdigest()
    return result


def dataset_inputs(
    dataset_config: dict[str, Any],
    scenario_configs: list[dict[str, Any]],
    manifest: dict[str, list[str]],
    processed_dir: Path,
) -> dict[str, Any]:
    dataset = dataset_config["name"]
    bundle = load_processed_bundle(processed_dir / Path(dataset_config["processed_path"]).name)
    instances = _select_manifest_instances(dataset, bundle.test_instances, manifest, subset_size=None)
    scenarios = {}
    for config in scenario_configs:
        scenario = config["name"]
        builder = SCENARIO_REGISTRY[scenario](config)
        records = {}
        for instance in instances:
            for variant in builder.build(instance, bundle.item_catalog, bundle.train_item_counts, dataset_config):
                key = "::".join((dataset, scenario, variant.variant_name, instance.instance_id))
                if key in records:
                    raise ValueError(f"{dataset}/{scenario}: duplicate reconstructed record")
                records[key] = record_input_hashes(variant)
        scenarios[scenario] = summarize(records)
    return {"test_pool": len(bundle.test_instances), "selected_instances": len(instances), "scenarios": scenarios}


def main() -> None:
    config = load_config(CONFIG)["experiment"]
    names = [dataset["name"] for dataset in config["datasets"]]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=names, action="append", help="Repeat to check specific datasets.")
    parser.add_argument("--processed_dir", type=Path, default=ROOT / "data/processed")
    args = parser.parse_args()
    manifest = _load_subset_manifest(str(SUBSET))
    expected = json.loads(REFERENCE.read_text())["datasets"]
    errors = []
    for dataset_config in config["datasets"]:
        dataset = dataset_config["name"]
        if args.dataset and dataset not in args.dataset:
            continue
        print(f"Checking {dataset}...", flush=True)
        actual = dataset_inputs(dataset_config, config["scenarios"], manifest, args.processed_dir)
        target = expected[dataset]
        differences = [key for key in ("test_pool", "selected_instances") if actual[key] != target[key]]
        for scenario, checks in target["scenarios"].items():
            differences.extend(
                f"{scenario}/{key}" for key, value in checks.items()
                if actual["scenarios"][scenario][key] != value
            )
        if differences:
            errors.append(f"{dataset}: {', '.join(differences)}")
            continue
        total = sum(scenario["rows"] for scenario in actual["scenarios"].values())
        print(f"{dataset}: {actual['selected_instances']:,}/{actual['test_pool']:,} test instances, "
              f"{total:,} reconstructed ranking inputs match", flush=True)
    if errors:
        raise SystemExit("Reconstruction mismatch:\n" + "\n".join(errors))


if __name__ == "__main__":
    main()
