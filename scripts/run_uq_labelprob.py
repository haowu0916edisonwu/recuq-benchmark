#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.data.base import load_processed_bundle
from recuq_benchmark.scenarios import SCENARIO_REGISTRY
from recuq_benchmark.uq.labelprob import LabelProbabilityScorer
from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION
from recuq_benchmark.utils.cache import CacheManager
from recuq_benchmark.utils.config import load_config
from recuq_benchmark.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MaxProb/LabelProb UQ scoring for fixed base LLM records."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/llama_controlled_suite.yaml",
        help="Experiment config used for the base LLM run.",
    )
    parser.add_argument(
        "--base_predictions",
        type=str,
        required=True,
        help="Path to base raw_predictions.jsonl.",
    )
    parser.add_argument(
        "--subset_manifest",
        type=str,
        required=True,
        help="Fixed eval manifest used by the base LLM run.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/uq/labelprob_llama31_8b",
        help="Output directory for uq_predictions.jsonl.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="labelprob_llama31_8b",
        help="Method name written into UQ prediction rows.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Forward-pass batch size. Defaults to model.batch_size from config.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of reconstructed base records to score.",
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=100,
        help="Log progress after this many scored records.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing uq_predictions.jsonl.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append only missing record_ids when uq_predictions.jsonl already exists.",
    )
    parser.add_argument(
        "--skip_hash_check",
        action="store_true",
        help="Skip candidate/history/prompt hash checks against the base raw predictions.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Reconstruct variants and validate hashes without loading the model or writing UQ rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("run_uq_labelprob")
    config = load_config(args.config)["experiment"]
    base_predictions_path = Path(args.base_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "uq_predictions.jsonl"

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"{output_path} already exists. Use --resume or --overwrite."
        )
    if args.overwrite and output_path.exists():
        output_path.unlink()

    base_by_id = _load_base_predictions(base_predictions_path)
    existing_ids = _load_existing_ids(output_path) if args.resume and output_path.exists() else set()
    manifest = _load_subset_manifest(args.subset_manifest)
    batch_size = int(args.batch_size or config["model"].get("batch_size", 1))
    progress_interval = max(1, int(args.progress_interval))
    estimated_total = min(len(base_by_id), args.limit) if args.limit is not None else len(base_by_id)
    estimated_to_score = max(estimated_total - len(existing_ids), 0)

    logger.info(
        "LabelProb run plan: base_records=%d existing=%d to_score<=%d batch_size=%d limit=%s",
        len(base_by_id),
        len(existing_ids),
        estimated_to_score,
        batch_size,
        args.limit if args.limit is not None else "none",
    )
    cache = CacheManager(config.get("cache_dir", "cache/llama_controlled_suite"))
    if args.validate_only:
        visited, hash_mismatches = _validate_reconstruction(
            config=config,
            manifest=manifest,
            cache=cache,
            base_by_id=base_by_id,
            limit=args.limit,
            skip_hash_check=args.skip_hash_check,
        )
        logger.info(
            "Validation complete: visited=%d hash_mismatches=%d base_records=%d",
            visited,
            hash_mismatches,
            len(base_by_id),
        )
        if hash_mismatches:
            raise RuntimeError(f"Encountered {hash_mismatches} hash mismatches.")
        return

    scorer = LabelProbabilityScorer(config["model"])

    scored = 0
    visited = 0
    skipped_existing = 0
    hash_mismatches = 0
    batch: list[dict[str, Any]] = []
    start = time.monotonic()

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for item in _iter_reconstructed_records(config, manifest, cache, base_by_id):
            if args.limit is not None and visited >= args.limit:
                break
            visited += 1
            record_id = item["record_id"]
            if record_id in existing_ids:
                skipped_existing += 1
                continue
            if not args.skip_hash_check and not _hashes_match(item["hashes"], item["base"]):
                hash_mismatches += 1
                continue
            batch.append(item)
            if len(batch) >= batch_size:
                scored += _score_and_write_batch(
                    scorer=scorer,
                    batch=batch,
                    output_file=output_file,
                    method=args.method,
                )
                batch.clear()
                if scored == 1 or scored % progress_interval == 0:
                    _log_progress(logger, scored, estimated_to_score, start)

        if batch:
            scored += _score_and_write_batch(
                scorer=scorer,
                batch=batch,
                output_file=output_file,
                method=args.method,
            )
            batch.clear()
            _log_progress(logger, scored, estimated_to_score, start)

    if hash_mismatches:
        raise RuntimeError(
            f"Encountered {hash_mismatches} hash mismatches. "
            "The reconstructed variants do not match the base run."
        )
    _write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": UQ_SCHEMA_VERSION,
            "method": args.method,
            "method_family": "probability",
            "config": args.config,
            "base_predictions": args.base_predictions,
            "subset_manifest": args.subset_manifest,
            "output_path": str(output_path),
            "base_records": len(base_by_id),
            "limit": args.limit,
            "visited_records": visited,
            "scored_records": scored,
            "skipped_existing": skipped_existing,
            "hash_mismatches": hash_mismatches,
            "batch_size": batch_size,
            "model_name": str(config["model"]["model_name"]),
        },
    )
    logger.info(
        "Wrote %d LabelProb UQ rows to %s; skipped_existing=%d elapsed=%s",
        scored,
        output_path,
        skipped_existing,
        _format_duration(time.monotonic() - start),
    )


def _validate_reconstruction(
    config: dict[str, Any],
    manifest: dict[str, list[str]],
    cache: CacheManager,
    base_by_id: dict[str, dict[str, Any]],
    limit: int | None,
    skip_hash_check: bool,
) -> tuple[int, int]:
    visited = 0
    hash_mismatches = 0
    for item in _iter_reconstructed_records(config, manifest, cache, base_by_id):
        if limit is not None and visited >= limit:
            break
        visited += 1
        if not skip_hash_check and not _hashes_match(item["hashes"], item["base"]):
            hash_mismatches += 1
    return visited, hash_mismatches


def _score_and_write_batch(
    scorer: LabelProbabilityScorer,
    batch: list[dict[str, Any]],
    output_file: Any,
    method: str,
) -> int:
    scored = scorer.score_batch(
        [item["variant"].history_text for item in batch],
        [item["variant"].candidate_texts for item in batch],
    )
    for item, score in zip(batch, scored):
        output_file.write(json.dumps(_uq_record(item, score, method), ensure_ascii=False) + "\n")
    output_file.flush()
    return len(scored)


def _uq_record(item: dict[str, Any], score: dict[str, Any], method: str) -> dict[str, Any]:
    max_probability = float(score["max_option_probability"])
    uncertainty = 1.0 - max_probability
    variant = item["variant"]
    return {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": item["record_id"],
        "method": method,
        "method_family": "probability",
        "uncertainty_score": uncertainty,
        "confidence_score": max_probability,
        "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
        "components": {
            "max_option_probability": max_probability,
            "second_option_probability": float(score["second_option_probability"]),
            "probability_margin": float(score["probability_margin"]),
            "entropy": float(score["entropy"]),
            "normalized_entropy": float(score["normalized_entropy"]),
            "top_letter": score["top_letter"],
            "option_probabilities": score["option_probabilities"],
        },
        "metadata": {
            "dataset": variant.dataset,
            "scenario": variant.scenario,
            "variant_name": variant.variant_name,
            "instance_id": variant.instance_id,
            "num_candidates": len(variant.candidate_ids),
            "candidate_letters": score["candidate_letters"],
            "selected_token_ids": score["selected_token_ids"],
            "base_prompt_hash": item["base"]["prompt_hash"],
        },
    }


def _iter_reconstructed_records(
    config: dict[str, Any],
    manifest: dict[str, list[str]],
    cache: CacheManager,
    base_by_id: dict[str, dict[str, Any]],
) -> Any:
    for dataset_config in config["datasets"]:
        dataset_name = str(dataset_config["name"])
        bundle = load_processed_bundle(dataset_config["processed_path"])
        instances = _select_manifest_instances(
            dataset_name=dataset_name,
            test_instances=bundle.test_instances,
            manifest=manifest,
        )
        for scenario_config in config["scenarios"]:
            builder_cls = SCENARIO_REGISTRY[scenario_config["name"]]
            builder = builder_cls(scenario_config)
            for instance in instances:
                variants = builder.build(
                    instance=instance,
                    item_catalog=bundle.item_catalog,
                    train_item_counts=bundle.train_item_counts,
                    dataset_config=dataset_config,
                )
                for variant in variants:
                    record_id = "::".join(
                        [variant.dataset, variant.scenario, variant.variant_name, variant.instance_id]
                    )
                    base = base_by_id.get(record_id)
                    if base is None:
                        raise KeyError(f"Base predictions missing reconstructed record_id: {record_id}")
                    hashes = {
                        "candidate_hash": _stable_hash(
                            {
                                "candidate_ids": variant.candidate_ids,
                                "labels": variant.labels,
                                "ground_truth_item_ids": variant.ground_truth_item_ids,
                            }
                        ),
                        "history_hash": _stable_hash(variant.history_text),
                        "prompt_hash": cache.prompt_hash(variant.history_text, variant.candidate_texts),
                    }
                    yield {
                        "record_id": record_id,
                        "variant": variant,
                        "base": base,
                        "hashes": hashes,
                    }


def _hashes_match(hashes: dict[str, str], base: dict[str, Any]) -> bool:
    return all(hashes[key] == base.get(key) for key in ("candidate_hash", "history_hash", "prompt_hash"))


def _load_base_predictions(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["record_id"])
            if record_id in by_id:
                raise ValueError(f"Duplicate base record_id at line {line_number}: {record_id}")
            by_id[record_id] = row
    return by_id


def _load_existing_ids(path: Path) -> set[str]:
    existing: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing.add(str(row["record_id"]))
    return existing


def _load_subset_manifest(path: str) -> dict[str, list[str]]:
    by_dataset: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as f:
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
            by_dataset.setdefault(dataset, []).append(instance_id)
    return by_dataset


def _select_manifest_instances(
    dataset_name: str,
    test_instances: list[Any],
    manifest: dict[str, list[str]],
) -> list[Any]:
    instance_ids = manifest.get(dataset_name)
    if not instance_ids:
        raise ValueError(f"Subset manifest has no entries for dataset {dataset_name}")
    by_id = {instance.instance_id: instance for instance in test_instances}
    missing = [instance_id for instance_id in instance_ids if instance_id not in by_id]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Subset manifest references missing {dataset_name} instances: {preview}")
    return [by_id[instance_id] for instance_id in instance_ids]


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _log_progress(logger: Any, scored: int, estimated_total: int, start: float) -> None:
    elapsed = max(time.monotonic() - start, 1e-9)
    rate = scored / elapsed
    remaining = max(estimated_total - scored, 0)
    eta = remaining / max(rate, 1e-9)
    pct = scored / estimated_total * 100.0 if estimated_total else 0.0
    logger.info(
        "Progress: %5.1f%%|%s| %d/%d [%s<%s, %.2frec/s]",
        pct,
        _progress_bar(pct),
        scored,
        estimated_total,
        _format_duration(elapsed),
        _format_duration(eta),
        rate,
    )


def _progress_bar(percent: float, width: int = 10) -> str:
    bounded = min(max(percent, 0.0), 100.0)
    filled = int(width * bounded / 100.0)
    return "#" * filled + "-" * (width - filled)


def _format_duration(seconds: float) -> str:
    seconds = max(int(math.ceil(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    main()
