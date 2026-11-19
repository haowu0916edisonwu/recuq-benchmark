#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.data.base import build_toy_bundle, load_processed_bundle
from recuq_benchmark.metrics.aggregate import aggregate_records
from recuq_benchmark.metrics.ranking import compute_ranking_metrics
from recuq_benchmark.models.base import DryRunRecommender, RankingOutput
from recuq_benchmark.models.llama_instruct import CausalLMInstructRecommender
from recuq_benchmark.scenarios import SCENARIO_REGISTRY
from recuq_benchmark.scenarios.base import Variant
from recuq_benchmark.utils.cache import CacheManager
from recuq_benchmark.utils.config import load_config
from recuq_benchmark.utils.logging import get_logger
from recuq_benchmark.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed LLM ranking records.")
    parser.add_argument("--config", type=str, default="configs/experiment/llama_controlled_suite.yaml")
    parser.add_argument("--subset_size", type=int, default=None)
    parser.add_argument(
        "--subset_manifest",
        type=str,
        default=None,
        help="Optional JSONL manifest listing dataset/test instance IDs to evaluate.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=50,
        help="Log progress after this many completed records.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Use toy bundles and DryRunRecommender.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("generate_fixed_rankings")
    config = load_config(args.config)["experiment"]
    eval_cfg = config.get("evaluation", {})
    set_global_seed(int(eval_cfg.get("seed", 42)))
    eval_k = int(eval_cfg.get("k", 10))
    subset_size = args.subset_size if args.subset_size is not None else int(eval_cfg.get("subset_size", 50))
    subset_manifest_path = args.subset_manifest
    if subset_manifest_path is None and not args.dry_run and config.get("subset_manifest"):
        subset_manifest_path = str((Path(args.config).resolve().parent / config["subset_manifest"]).resolve())
    subset_manifest = _load_subset_manifest(subset_manifest_path) if subset_manifest_path else None
    manifest_counts = {dataset: len(instance_ids) for dataset, instance_ids in subset_manifest.items()} if subset_manifest else {}
    if args.dry_run:
        default_output = Path("outputs/dry_run") / Path(args.config).stem
        output_dir = Path(args.output_dir or default_output)
        cache = CacheManager(output_dir / "cache")
    else:
        output_dir = Path(args.output_dir or config.get("results_dir", "results/fixed_rankings"))
        cache = CacheManager(config.get("cache_dir", "cache/fixed_rankings"))
    output_dir.mkdir(parents=True, exist_ok=True)
    recommender = DryRunRecommender() if args.dry_run else CausalLMInstructRecommender(config["model"])
    model_name = "dry_run" if args.dry_run else str(config["model"]["model_name"])
    backbone = "dry_run" if args.dry_run else str(config["model"].get("backbone", model_name))
    run_id = str(config.get("name", output_dir.name))
    method = "dry_run" if args.dry_run else str(config.get("method", "llama_base_rank"))
    reuse_equivalent_prompts = bool(config.get("reuse_equivalent_prompts", False))
    shared_prompt_outputs: dict[str, dict[str, Any]] = {}
    progress_interval = max(1, int(args.progress_interval))
    run_start = time.monotonic()
    estimated_total_records = _estimate_run_records_from_config(config, subset_size)
    if subset_manifest:
        estimated_total_records = _estimate_run_records_from_manifest_config(
            config=config,
            manifest_counts=manifest_counts,
            subset_size=subset_size if args.subset_size is not None else None,
        )
    completed_records = 0
    records: list[dict[str, Any]] = []
    dataset_rows: list[dict[str, Any]] = []

    logger.info(
        "Run plan: datasets=%d scenarios=%d subset_size=%d subset_manifest=%s estimated_records<=%d progress_interval=%d",
        len(config["datasets"]),
        len(config["scenarios"]),
        subset_size,
        subset_manifest_path or "none",
        estimated_total_records,
        progress_interval,
    )

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        logger.info("Loading dataset %s", dataset_name)
        bundle = build_toy_bundle(dataset_name) if args.dry_run else load_processed_bundle(dataset_config["processed_path"])
        dataset_rows.append(
            {
                "dataset": dataset_name,
                "num_train": len(bundle.train_instances),
                "num_val": len(bundle.val_instances),
                "num_test": len(bundle.test_instances),
                "num_items": len(bundle.item_catalog),
                **bundle.dataset_stats,
            }
        )
        if subset_manifest:
            instances = _select_manifest_instances(
                dataset_name=dataset_name,
                test_instances=bundle.test_instances,
                manifest=subset_manifest,
                subset_size=subset_size if args.subset_size is not None else None,
            )
        else:
            instances = bundle.test_instances[:subset_size]
        dataset_estimate = sum(
            _estimate_scenario_records(
                instances=instances,
                scenario_config=scenario_config,
                dataset_config=dataset_config,
                item_catalog=bundle.item_catalog,
                train_item_counts=bundle.train_item_counts,
            )
            for scenario_config in config["scenarios"]
        )
        dataset_records = 0
        dataset_start = time.monotonic()
        logger.info(
            "Dataset plan: %s test_instances=%d estimated_records=%d",
            dataset_name,
            len(instances),
            dataset_estimate,
        )
        for scenario_config in config["scenarios"]:
            scenario_name = scenario_config["name"]
            builder_cls = SCENARIO_REGISTRY[scenario_name]
            builder = builder_cls(scenario_config)
            scenario_estimate = _estimate_scenario_records(
                instances=instances,
                scenario_config=scenario_config,
                dataset_config=dataset_config,
                item_catalog=bundle.item_catalog,
                train_item_counts=bundle.train_item_counts,
            )
            scenario_records = 0
            scenario_start = time.monotonic()
            logger.info(
                "Running %s / %s on %d instances; estimated_records=%d",
                dataset_name,
                scenario_name,
                len(instances),
                scenario_estimate,
            )
            for instance in instances:
                variants = builder.build(
                    instance=instance,
                    item_catalog=bundle.item_catalog,
                    train_item_counts=bundle.train_item_counts,
                    dataset_config=dataset_config,
                )
                for variant in variants:
                    input_hashes = record_input_hashes(variant)
                    prompt_hash = input_hashes["prompt_hash"]
                    key = cache.build_key(
                        model_name=model_name,
                        dataset=variant.dataset,
                        scenario=variant.scenario,
                        instance_id=variant.instance_id,
                        variant_name=variant.variant_name,
                        prompt_hash=prompt_hash,
                    )
                    shared_key = f"{model_name}::{prompt_hash}"
                    cached = shared_prompt_outputs.get(shared_key) if reuse_equivalent_prompts else None
                    if cached is None:
                        cached = cache.get(key)
                    if cached is None:
                        output = recommender.rank(variant.history_text, variant.candidate_texts)
                        cached = asdict(output)
                        cache.set(key, cached)
                    if reuse_equivalent_prompts:
                        shared_prompt_outputs[shared_key] = cached
                    output = RankingOutput(**cached)
                    metrics = compute_ranking_metrics(variant.labels, output.ranking, k=eval_k)
                    record_id = "::".join(
                        [variant.dataset, variant.scenario, variant.variant_name, variant.instance_id]
                    )
                    record = {
                        "schema_version": "fixed_ranking_v1",
                        "record_id": record_id,
                        "run_id": run_id,
                        "method": method,
                        "model_name": model_name,
                        "backbone": backbone,
                        "dataset": variant.dataset,
                        "scenario": variant.scenario,
                        "challenge": _challenge_name(variant.scenario),
                        "variant_name": variant.variant_name,
                        "target_source": variant.target_source,
                        "instance_id": variant.instance_id,
                        "user_id": variant.user_id,
                        "candidate_ids": variant.candidate_ids,
                        "labels": variant.labels,
                        "ground_truth_item_ids": variant.ground_truth_item_ids,
                        **input_hashes,
                        "eval_k": eval_k,
                        f"is_error_at_{eval_k}": bool(metrics.get(f"error@{eval_k}", 0.0) >= 1.0),
                        f"error_at_{eval_k}": float(metrics.get(f"error@{eval_k}", 0.0)),
                        "ranking": output.ranking,
                        "scores": output.scores,
                        "raw_response": output.raw_response,
                        "parse_success": output.parse_success,
                        "metrics": metrics,
                        "metadata": variant.metadata,
                        "model_metadata": output.metadata,
                    }
                    records.append(record)
                    completed_records += 1
                    dataset_records += 1
                    scenario_records += 1
                    if completed_records == 1 or completed_records % progress_interval == 0:
                        _log_progress(
                            logger=logger,
                            completed_records=completed_records,
                            estimated_total_records=estimated_total_records,
                            dataset_name=dataset_name,
                            scenario_name=scenario_name,
                            scenario_records=scenario_records,
                            scenario_estimate=scenario_estimate,
                            run_start=run_start,
                        )
            logger.info(
                "Completed %s / %s: records=%d estimated_records=%d elapsed=%s",
                dataset_name,
                scenario_name,
                scenario_records,
                scenario_estimate,
                _format_duration(time.monotonic() - scenario_start),
            )
        logger.info(
            "Completed dataset %s: records=%d estimated_records=%d elapsed=%s",
            dataset_name,
            dataset_records,
            dataset_estimate,
            _format_duration(time.monotonic() - dataset_start),
        )

    _write_jsonl(output_dir / "raw_predictions.jsonl", records)
    _write_csv(output_dir / "dataset_summary.csv", dataset_rows)
    _write_csv(output_dir / "overall_scenario_summary.csv", aggregate_records(records, ["dataset", "scenario"]))
    _write_csv(
        output_dir / "s3_history_length_summary.csv",
        aggregate_records(
            [r for r in records if r["scenario"] == "s3_sparsity"],
            ["dataset", "variant_name"],
        ),
    )
    _write_csv(
        output_dir / "s4_popularity_summary.csv",
        aggregate_records(
            _flatten_metadata(records, "popularity_bucket", scenario="s4_longtail"),
            ["dataset", "popularity_bucket"],
        ),
    )
    _write_csv(
        output_dir / "s5_ablation_summary.csv",
        aggregate_records(
            _flatten_metadata(records, "ablated_field", scenario="s5_robustness"),
            ["dataset", "ablated_field"],
        ),
    )
    _write_csv(output_dir / "parse_success_summary.csv", aggregate_records(records, ["dataset", "scenario"]))
    _write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": "fixed_ranking_v1",
            "run_id": run_id,
            "method": method,
            "model_name": model_name,
            "backbone": backbone,
            "eval_k": eval_k,
            "config": config,
            "dry_run": args.dry_run,
            "reuse_equivalent_prompts": reuse_equivalent_prompts,
            "subset_manifest": subset_manifest_path,
            "manifest_counts": manifest_counts,
            "num_records": len(records),
        },
    )
    logger.info("Wrote %d records to %s", len(records), output_dir)


def _estimate_run_records_from_config(config: dict[str, Any], subset_size: int) -> int:
    total = 0
    for dataset_config in config["datasets"]:
        for scenario_config in config["scenarios"]:
            total += subset_size * _estimate_variants_per_instance_upper_bound(
                scenario_config=scenario_config,
                dataset_config=dataset_config,
            )
    return total


def _estimate_run_records_from_manifest_config(
    config: dict[str, Any],
    manifest_counts: dict[str, int],
    subset_size: int | None,
) -> int:
    total = 0
    for dataset_config in config["datasets"]:
        dataset_name = str(dataset_config["name"])
        n = manifest_counts.get(dataset_name, 0)
        if subset_size is not None:
            n = min(n, subset_size)
        for scenario_config in config["scenarios"]:
            total += n * _estimate_variants_per_instance_upper_bound(
                scenario_config=scenario_config,
                dataset_config=dataset_config,
            )
    return total


def _load_subset_manifest(path: str | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    manifest_path = Path(path)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            dataset = str(row["dataset"])
            instance_id = str(row["instance_id"])
            key = (dataset, instance_id)
            if key in seen:
                raise ValueError(f"Duplicate manifest entry at line {line_number}: {dataset}/{instance_id}")
            seen.add(key)
            by_dataset[dataset].append(instance_id)
    return dict(by_dataset)


def _select_manifest_instances(
    dataset_name: str,
    test_instances: list[Any],
    manifest: dict[str, list[str]],
    subset_size: int | None,
) -> list[Any]:
    instance_ids = manifest.get(dataset_name)
    if not instance_ids:
        raise ValueError(f"Subset manifest has no entries for dataset {dataset_name}")
    if subset_size is not None:
        instance_ids = instance_ids[:subset_size]
    by_id = {instance.instance_id: instance for instance in test_instances}
    missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Subset manifest references missing {dataset_name} instances: {preview}")
    return [by_id[instance_id] for instance_id in instance_ids]


def _estimate_variants_per_instance_upper_bound(
    scenario_config: dict[str, Any],
    dataset_config: dict[str, Any],
) -> int:
    scenario_name = scenario_config["name"]
    if scenario_name in {"s0_warmstart", "s4_longtail"}:
        return 1
    if scenario_name == "s3_sparsity":
        count = len(scenario_config.get("truncate_lengths", [1, 3, 5, 10]))
        return count + int(bool(scenario_config.get("include_full", True)))
    if scenario_name == "s5_robustness":
        fields = dataset_config.get("s5_ablate_fields") or scenario_config.get("ablate_fields", [])
        return 1 + len(fields)
    return 1


def _estimate_scenario_records(
    instances: list[Any],
    scenario_config: dict[str, Any],
    dataset_config: dict[str, Any],
    item_catalog: dict[str, Any],
    train_item_counts: dict[str, int],
) -> int:
    scenario_name = scenario_config["name"]
    if scenario_name == "s0_warmstart":
        return len(instances)
    if scenario_name == "s4_longtail":
        builder = SCENARIO_REGISTRY[scenario_name](scenario_config)
        return sum(
            len(
                builder.build(
                    instance=instance,
                    item_catalog=item_catalog,
                    train_item_counts=train_item_counts,
                    dataset_config=dataset_config,
                )
            )
            for instance in instances
        )
    if scenario_name == "s3_sparsity":
        lengths = [int(length) for length in scenario_config.get("truncate_lengths", [1, 3, 5, 10])]
        include_full = bool(scenario_config.get("include_full", True))
        total = 0
        for instance in instances:
            total += sum(1 for length in lengths if len(instance.history) >= length)
            total += int(include_full)
        return total
    if scenario_name == "s5_robustness":
        fields = dataset_config.get("s5_ablate_fields") or scenario_config.get("ablate_fields", [])
        total = 0
        for instance in instances:
            total += 1
            for field_name in fields:
                if _field_available(field_name, instance.candidate_ids, item_catalog):
                    total += 1
        return total
    return len(instances)


def _field_available(field_name: str, candidate_ids: list[str], item_catalog: dict[str, Any]) -> bool:
    for item_id in candidate_ids:
        item = item_catalog.get(item_id)
        if item and item.text_fields.get(field_name):
            return True
    return False


def _log_progress(
    logger: Any,
    completed_records: int,
    estimated_total_records: int,
    dataset_name: str,
    scenario_name: str,
    scenario_records: int,
    scenario_estimate: int,
    run_start: float,
) -> None:
    elapsed = max(time.monotonic() - run_start, 1e-9)
    records_per_second = completed_records / elapsed
    remaining = max(estimated_total_records - completed_records, 0)
    eta_seconds = remaining / max(records_per_second, 1e-9)
    overall_pct = completed_records / estimated_total_records * 100.0 if estimated_total_records else 0.0
    scenario_pct = scenario_records / scenario_estimate * 100.0 if scenario_estimate else 0.0
    logger.info(
        "Progress: %5.1f%%|%s| %d/%d [%s<%s, %.2frec/s] current=%s/%s %d/%d (%.1f%%)",
        overall_pct,
        _progress_bar(overall_pct),
        completed_records,
        estimated_total_records,
        _format_duration(elapsed),
        _format_duration(eta_seconds),
        records_per_second,
        dataset_name,
        scenario_name,
        scenario_records,
        scenario_estimate,
        scenario_pct,
    )


def _progress_bar(percent: float, width: int = 10) -> str:
    bounded = min(max(percent, 0.0), 100.0)
    filled = int(width * bounded / 100.0)
    return "#" * filled + "-" * (width - filled)


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _challenge_name(scenario: str) -> str:
    mapping = {
        "s0_warmstart": "warm",
        "s3_sparsity": "sparse",
        "s4_longtail": "long_tail",
        "s5_robustness": "robust",
    }
    return mapping.get(scenario, scenario)


def record_input_hashes(variant: Variant) -> dict[str, str]:
    return {
        "candidate_hash": _stable_hash({
            "candidate_ids": variant.candidate_ids,
            "labels": variant.labels,
            "ground_truth_item_ids": variant.ground_truth_item_ids,
        }),
        "history_hash": _stable_hash(variant.history_text),
        "prompt_hash": CacheManager.prompt_hash(variant.history_text, variant.candidate_texts),
    }


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _flatten_metadata(records: list[dict[str, Any]], key: str, scenario: str) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for record in records:
        if record["scenario"] != scenario:
            continue
        copy = dict(record)
        value = copy.get("metadata", {}).get(key)
        copy[key] = "original" if value is None else value
        flattened.append(copy)
    return flattened


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
