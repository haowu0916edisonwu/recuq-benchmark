#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
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
from recuq_benchmark.models.base import RankingOutput
from recuq_benchmark.models.llama_instruct import LlamaInstructRecommender
from recuq_benchmark.scenarios import SCENARIO_REGISTRY
from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION, normalize_uq_prediction
from recuq_benchmark.utils.cache import CacheManager
from recuq_benchmark.utils.config import load_config
from recuq_benchmark.utils.logging import get_logger


DEFAULT_METHOD_PREFIX = "self_consistency"
METHOD_FAMILY = "sampling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sample-consistency UQ baseline for fixed LLM recommendation records. "
            "The script reconstructs the frozen benchmark prompts, samples multiple "
            "rankings per record, and writes one UQ row per base record."
        )
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
        default=None,
        help="Output directory. Defaults to results/uq/self_consistency_s{num_samples}.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Method name written into UQ rows. Defaults to self_consistency_s{num_samples}.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of stochastic ranking samples per record.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for stochastic ranking generations.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p value.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Ranking prefix used for pairwise Jaccard disagreement.",
    )
    parser.add_argument(
        "--sampling_mode",
        choices=("sequential", "multi_return"),
        default="sequential",
        help=(
            "How to obtain multiple samples. sequential calls generate once per sample "
            "and preserves the validated smoke path. multi_return requests all samples "
            "in one generate call and should be smoke-tested separately."
        ),
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
        help="Log progress after this many newly scored records.",
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
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory for sampled rankings. Defaults depend on num_samples and sampling_mode.",
    )
    parser.add_argument(
        "--store_sample_raw_responses",
        action="store_true",
        help="Store raw sampled LLM responses in components. Off by default to keep JSONL compact.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for per-record stochastic samples. Defaults to model.seed from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("run_uq_self_consistency")
    _validate_args(args)

    config = load_config(args.config)["experiment"]
    method = args.method or f"{DEFAULT_METHOD_PREFIX}_s{args.num_samples}"
    output_dir = Path(args.output_dir or f"results/uq/{method}")
    output_path = output_dir / "uq_predictions.jsonl"
    base_predictions_path = Path(args.base_predictions)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if output_path.exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"{output_path} already exists. Use --resume or --overwrite.")
    if args.overwrite and output_path.exists():
        output_path.unlink()

    base_by_id = _load_base_predictions(base_predictions_path)
    manifest = _load_subset_manifest(args.subset_manifest)
    existing_ids = _load_existing_ids(output_path) if args.resume and output_path.exists() else set()
    estimated_total = min(len(base_by_id), args.limit) if args.limit is not None else len(base_by_id)
    estimated_to_score = max(estimated_total - len(existing_ids), 0)

    reconstruction_cache = CacheManager(config.get("cache_dir", "cache/llama_controlled_suite"))
    sample_cache_dir = _default_sample_cache_dir(args)
    sample_cache = CacheManager(sample_cache_dir)

    logger.info(
        (
            "Self-consistency run plan: base_records=%d existing=%d to_score<=%d "
            "num_samples=%d temperature=%.3f top_p=%.3f top_k=%d sampling_mode=%s limit=%s"
        ),
        len(base_by_id),
        len(existing_ids),
        estimated_to_score,
        args.num_samples,
        args.temperature,
        args.top_p,
        args.top_k,
        args.sampling_mode,
        args.limit if args.limit is not None else "none",
    )

    if args.validate_only:
        visited, hash_mismatches = _validate_reconstruction(
            config=config,
            manifest=manifest,
            cache=reconstruction_cache,
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

    model_config = dict(config["model"])
    model_config["do_sample"] = True
    model_config["temperature"] = args.temperature
    model_config["top_p"] = args.top_p
    base_seed = int(args.seed if args.seed is not None else model_config.get("seed", 42))
    model_config["seed"] = base_seed
    recommender = LlamaInstructRecommender(model_config)
    model_name = str(model_config["model_name"])

    scored = 0
    visited = 0
    skipped_existing = 0
    hash_mismatches = 0
    start = time.monotonic()
    progress_interval = max(1, int(args.progress_interval))

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as output_file:
        for item in _iter_reconstructed_records(config, manifest, reconstruction_cache, base_by_id):
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

            samples = _sample_rankings(
                recommender=recommender,
                item=item,
                sample_cache=sample_cache,
                model_name=model_name,
                num_samples=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                base_seed=base_seed,
                sampling_mode=args.sampling_mode,
            )
            row = _uq_record(
                item=item,
                samples=samples,
                method=method,
                top_k=args.top_k,
                temperature=args.temperature,
                top_p=args.top_p,
                sampling_mode=args.sampling_mode,
                store_raw_responses=args.store_sample_raw_responses,
            )
            normalize_uq_prediction(row, row_number=scored + 1)
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_file.flush()

            scored += 1
            if scored == 1 or scored % progress_interval == 0:
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
            "method": method,
            "method_family": METHOD_FAMILY,
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
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "sampling_mode": args.sampling_mode,
            "seed": base_seed,
            "sample_cache_dir": str(sample_cache_dir),
            "model_name": model_name,
            "uncertainty_definition": (
                "Mean pairwise top-k Jaccard disagreement among stochastic sampled rankings "
                "for the same reconstructed recommendation record."
            ),
            "score_orientation": "larger uncertainty_score means less self-consistent / more uncertain",
            "literature_basis": (
                "Wang et al. self-consistency; Lyu et al. AAAI 2025 sample-consistency "
                "calibration; Xiao et al. UAI 2025 consistency hypothesis"
            ),
        },
    )
    logger.info(
        "Wrote %d self-consistency UQ rows to %s; skipped_existing=%d elapsed=%s",
        scored,
        output_path,
        skipped_existing,
        _format_duration(time.monotonic() - start),
    )


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 2:
        raise ValueError("--num_samples must be at least 2")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p must be in (0, 1]")
    if args.top_k < 1:
        raise ValueError("--top_k must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when provided")


def _default_sample_cache_dir(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir)
    if args.sampling_mode == "sequential":
        return Path(f"cache/uq_self_consistency_s{args.num_samples}")
    return Path(f"cache/uq_self_consistency_s{args.num_samples}_multi_return")


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


def _sample_rankings(
    *,
    recommender: LlamaInstructRecommender,
    item: dict[str, Any],
    sample_cache: CacheManager,
    model_name: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    base_seed: int,
    sampling_mode: str,
) -> list[dict[str, Any]]:
    if sampling_mode == "multi_return":
        return _sample_rankings_multi_return(
            recommender=recommender,
            item=item,
            sample_cache=sample_cache,
            model_name=model_name,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            base_seed=base_seed,
        )
    if sampling_mode != "sequential":
        raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
    return _sample_rankings_sequential(
        recommender=recommender,
        item=item,
        sample_cache=sample_cache,
        model_name=model_name,
        num_samples=num_samples,
        temperature=temperature,
        top_p=top_p,
        base_seed=base_seed,
    )


def _sample_rankings_sequential(
    *,
    recommender: LlamaInstructRecommender,
    item: dict[str, Any],
    sample_cache: CacheManager,
    model_name: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    variant = item["variant"]
    prompt_hash = item["hashes"]["prompt_hash"]
    for sample_index in range(num_samples):
        cache_key = "::".join(
            [
                model_name,
                item["record_id"],
                f"sample={sample_index}",
                f"seed={base_seed}",
                f"temperature={temperature:.6f}",
                f"top_p={top_p:.6f}",
                "sampling_mode=sequential",
                prompt_hash,
            ]
        )
        cached = sample_cache.get(cache_key)
        if cached is None:
            _set_sample_seed(item["record_id"], sample_index=sample_index, base_seed=base_seed)
            output = recommender.rank(variant.history_text, variant.candidate_texts)
            cached = asdict(output)
            sample_cache.set(cache_key, cached)
        ranking_output = RankingOutput(**cached)
        samples.append(
            {
                "sample_index": sample_index,
                "ranking": list(ranking_output.ranking),
                "parse_success": bool(ranking_output.parse_success),
                "raw_response": ranking_output.raw_response,
                "metadata": ranking_output.metadata,
            }
        )
    return samples


def _sample_rankings_multi_return(
    *,
    recommender: LlamaInstructRecommender,
    item: dict[str, Any],
    sample_cache: CacheManager,
    model_name: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    base_seed: int,
) -> list[dict[str, Any]]:
    variant = item["variant"]
    prompt_hash = item["hashes"]["prompt_hash"]
    cache_key = "::".join(
        [
            model_name,
            item["record_id"],
            f"num_samples={num_samples}",
            f"seed={base_seed}",
            f"temperature={temperature:.6f}",
            f"top_p={top_p:.6f}",
            "sampling_mode=multi_return",
            prompt_hash,
        ]
    )
    cached_outputs = sample_cache.get(cache_key)
    if cached_outputs is None:
        _set_sample_seed(item["record_id"], sample_index=0, base_seed=base_seed)
        outputs = recommender.rank_many(
            variant.history_text,
            variant.candidate_texts,
            num_return_sequences=num_samples,
        )
        cached_outputs = [asdict(output) for output in outputs]
        sample_cache.set(cache_key, cached_outputs)

    samples: list[dict[str, Any]] = []
    for sample_index, cached in enumerate(cached_outputs):
        ranking_output = RankingOutput(**cached)
        samples.append(
            {
                "sample_index": sample_index,
                "ranking": list(ranking_output.ranking),
                "parse_success": bool(ranking_output.parse_success),
                "raw_response": ranking_output.raw_response,
                "metadata": ranking_output.metadata,
            }
        )
    return samples


def _set_sample_seed(record_id: str, *, sample_index: int, base_seed: int) -> None:
    seed_payload = f"{base_seed}::{record_id}::{sample_index}"
    seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
    try:
        from transformers import set_seed

        set_seed(seed)
    except ImportError:
        pass


def _uq_record(
    *,
    item: dict[str, Any],
    samples: list[dict[str, Any]],
    method: str,
    top_k: int,
    temperature: float,
    top_p: float,
    sampling_mode: str,
    store_raw_responses: bool,
) -> dict[str, Any]:
    variant = item["variant"]
    rankings = [sample["ranking"] for sample in samples]
    top_k = min(top_k, len(variant.candidate_ids))
    pairwise_distances = _pairwise_topk_jaccard_distances(rankings, top_k=top_k)
    uncertainty = _mean(pairwise_distances)
    top1_counts = _top1_counts(rankings)
    top1_entropy = _normalized_counter_entropy(top1_counts)
    top1_agreement = max(top1_counts.values()) / len(rankings) if top1_counts else 0.0
    top1_fsd = _first_second_distance(top1_counts, total=len(rankings))
    parse_success_rate = sum(1 for sample in samples if sample["parse_success"]) / len(samples)
    sampled_rankings = [_ranking_to_candidate_ids(ranking, variant.candidate_ids) for ranking in rankings]

    components: dict[str, Any] = {
        "score_definition": "mean_pairwise_top_k_jaccard_disagreement",
        "top_k": top_k,
        "num_samples": len(samples),
        "temperature": temperature,
        "top_p": top_p,
        "sampling_mode": sampling_mode,
        "pairwise_top_k_jaccard_distances": pairwise_distances,
        "top_k_jaccard_uncertainty": uncertainty,
        "top1_agreement_confidence": top1_agreement,
        "top1_entropy_uncertainty": top1_entropy,
        "top1_fsd_confidence": top1_fsd,
        "top1_fsd_uncertainty": 1.0 - top1_fsd,
        "num_unique_top1": len(top1_counts),
        "sampled_top1_counts": dict(sorted(top1_counts.items())),
        "parse_success_rate": parse_success_rate,
        "sampled_rankings": sampled_rankings,
        "literature_basis": (
            "sample-consistency confidence adapted from answer distributions "
            "to 20-candidate recommendation ranking distributions"
        ),
    }
    if store_raw_responses:
        components["raw_sample_responses"] = [sample["raw_response"] for sample in samples]

    return {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": item["record_id"],
        "method": method,
        "method_family": METHOD_FAMILY,
        "uncertainty_score": _clip01(uncertainty),
        "confidence_score": _clip01(1.0 - uncertainty),
        "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
        "components": components,
        "metadata": {
            "dataset": variant.dataset,
            "scenario": variant.scenario,
            "variant_name": variant.variant_name,
            "instance_id": variant.instance_id,
            "num_candidates": len(variant.candidate_ids),
            "candidate_ids": variant.candidate_ids,
            "base_prompt_hash": item["base"]["prompt_hash"],
            "base_record_error_at_10": item["base"].get("is_error_at_10"),
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


def _pairwise_topk_jaccard_distances(rankings: list[list[int]], *, top_k: int) -> list[float]:
    distances: list[float] = []
    for left_index in range(len(rankings)):
        left = set(rankings[left_index][:top_k])
        for right_index in range(left_index + 1, len(rankings)):
            right = set(rankings[right_index][:top_k])
            union = left | right
            if not union:
                distances.append(0.0)
            else:
                distances.append(1.0 - (len(left & right) / len(union)))
    return distances


def _ranking_to_candidate_ids(ranking: list[int], candidate_ids: list[str]) -> list[str]:
    output: list[str] = []
    for index in ranking:
        if 0 <= int(index) < len(candidate_ids):
            output.append(candidate_ids[int(index)])
    return output


def _top1_counts(rankings: list[list[int]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ranking in rankings:
        if ranking:
            counts[str(ranking[0])] += 1
    return counts


def _normalized_counter_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log(probability)
    return _clip01(entropy / math.log(len(counts)))


def _first_second_distance(counts: Counter[str], *, total: int) -> float:
    if total <= 0:
        return 0.0
    ordered = sorted(counts.values(), reverse=True)
    first = ordered[0] if ordered else 0
    second = ordered[1] if len(ordered) > 1 else 0
    return _clip01((first - second) / total)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _clip01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(max(float(value), 0.0), 1.0)


def _hashes_match(hashes: dict[str, str], base: dict[str, Any]) -> bool:
    return all(hashes[key] == base.get(key) for key in ("candidate_hash", "history_hash", "prompt_hash"))


def _load_base_predictions(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
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
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            existing.add(str(row["record_id"]))
    return existing


def _load_subset_manifest(path: str) -> dict[str, list[str]]:
    by_dataset: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
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
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, ensure_ascii=False)


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
