#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_uq_labelprob import (  # noqa: E402
    _format_duration,
    _hashes_match,
    _iter_reconstructed_records,
    _load_base_predictions,
    _load_existing_ids,
    _load_subset_manifest,
    _log_progress,
    _write_json,
)
from recuq_benchmark.uq.labelprob import LabelProbabilityScorer, option_letters  # noqa: E402
from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION  # noqa: E402
from recuq_benchmark.utils.cache import CacheManager  # noqa: E402
from recuq_benchmark.utils.config import load_config  # noqa: E402
from recuq_benchmark.utils.logging import get_logger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Kweon-style decomposed UQ for fixed base LLM records. "
            "This adapts WWW'25 LLM-rec UQ by scoring multiple candidate-order "
            "prompt variants and decomposing predictive uncertainty into total, "
            "recommendation, and prompt uncertainty."
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
        default="results/uq/kweon_decomp_llama31_8b",
        help="Output directory for uq_predictions.jsonl.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="kweon_decomp_llama31_8b",
        help="Method name written into UQ prediction rows.",
    )
    parser.add_argument(
        "--num_prompt_variants",
        "--nper",
        dest="num_prompt_variants",
        type=int,
        default=5,
        help="Number of candidate-order prompt variants per base record.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Forward-pass prompt batch size. Defaults to model.batch_size from config.",
    )
    parser.add_argument(
        "--record_batch_size",
        type=int,
        default=1,
        help="Number of base records grouped before scoring flattened prompt variants.",
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
        help="Log progress after this many scored base records.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic per-record candidate-order permutations.",
    )
    parser.add_argument(
        "--include_original_order",
        action="store_true",
        help=(
            "Use the original candidate order as the first prompt variant. "
            "The remaining variants are random permutations."
        ),
    )
    parser.add_argument(
        "--store_variant_probabilities",
        action="store_true",
        help="Store aligned option probabilities for every prompt variant in components.",
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
    logger = get_logger("run_uq_kweon_decomp")
    config = load_config(args.config)["experiment"]
    base_predictions_path = Path(args.base_predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "uq_predictions.jsonl"

    if args.num_prompt_variants < 1:
        raise ValueError("--num_prompt_variants must be at least 1")
    if args.record_batch_size < 1:
        raise ValueError("--record_batch_size must be at least 1")
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
    cache = CacheManager(config.get("cache_dir", "cache/llama_controlled_suite"))
    prompt_batch_size = int(args.batch_size or config["model"].get("batch_size", 1))
    progress_interval = max(1, int(args.progress_interval))
    estimated_total, estimated_to_score = _estimate_reconstructed_records(
        config=config,
        manifest=manifest,
        cache=cache,
        base_by_id=base_by_id,
        existing_ids=existing_ids,
        limit=args.limit,
    )

    logger.info(
        (
            "Kweon decomposition run plan: base_records=%d manifest_records=%d "
            "existing_in_manifest=%d to_score=%d prompt_batch_size=%d record_batch_size=%d "
            "num_prompt_variants=%d limit=%s"
        ),
        len(base_by_id),
        estimated_total,
        estimated_total - estimated_to_score,
        estimated_to_score,
        prompt_batch_size,
        args.record_batch_size,
        args.num_prompt_variants,
        args.limit if args.limit is not None else "none",
    )

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
    record_batch: list[dict[str, Any]] = []
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
            record_batch.append(item)
            if len(record_batch) >= args.record_batch_size:
                scored += _score_and_write_record_batch(
                    scorer=scorer,
                    record_batch=record_batch,
                    output_file=output_file,
                    method=args.method,
                    prompt_batch_size=prompt_batch_size,
                    num_prompt_variants=args.num_prompt_variants,
                    seed=args.seed,
                    include_original_order=args.include_original_order,
                    store_variant_probabilities=args.store_variant_probabilities,
                )
                record_batch.clear()
                if scored == 1 or scored % progress_interval == 0:
                    _log_progress(logger, scored, estimated_to_score, start)

        if record_batch:
            scored += _score_and_write_record_batch(
                scorer=scorer,
                record_batch=record_batch,
                output_file=output_file,
                method=args.method,
                prompt_batch_size=prompt_batch_size,
                num_prompt_variants=args.num_prompt_variants,
                seed=args.seed,
                include_original_order=args.include_original_order,
                store_variant_probabilities=args.store_variant_probabilities,
            )
            record_batch.clear()
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
            "method_family": "decomposition",
            "literature_basis": "Kweon et al., WWW 2025, decomposed LLM recommendation uncertainty",
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
            "prompt_batch_size": prompt_batch_size,
            "record_batch_size": args.record_batch_size,
            "num_prompt_variants": args.num_prompt_variants,
            "include_original_order": args.include_original_order,
            "store_variant_probabilities": args.store_variant_probabilities,
            "seed": args.seed,
            "model_name": str(config["model"]["model_name"]),
            "uncertainty_definition": (
                "total_uncertainty = normalized entropy of mean aligned option-label "
                "distribution across candidate-order prompt variants; "
                "recommendation_uncertainty = mean normalized entropy per variant; "
                "prompt_uncertainty = total_uncertainty - recommendation_uncertainty"
            ),
        },
    )
    logger.info(
        "Wrote %d Kweon decomposition UQ rows to %s; skipped_existing=%d elapsed=%s",
        scored,
        output_path,
        skipped_existing,
        _format_duration(time.monotonic() - start),
    )


def _estimate_reconstructed_records(
    config: dict[str, Any],
    manifest: dict[str, list[str]],
    cache: CacheManager,
    base_by_id: dict[str, dict[str, Any]],
    existing_ids: set[str],
    limit: int | None,
) -> tuple[int, int]:
    total = 0
    to_score = 0
    for item in _iter_reconstructed_records(config, manifest, cache, base_by_id):
        if limit is not None and total >= limit:
            break
        total += 1
        if item["record_id"] not in existing_ids:
            to_score += 1
    return total, to_score


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


def _score_and_write_record_batch(
    scorer: LabelProbabilityScorer,
    record_batch: list[dict[str, Any]],
    output_file: Any,
    method: str,
    prompt_batch_size: int,
    num_prompt_variants: int,
    seed: int,
    include_original_order: bool,
    store_variant_probabilities: bool,
) -> int:
    flattened_history_texts: list[list[str]] = []
    flattened_candidate_texts: list[list[str]] = []
    groups: list[dict[str, Any]] = []

    for item in record_batch:
        variant = item["variant"]
        permutations = _candidate_order_permutations(
            record_id=item["record_id"],
            num_candidates=len(variant.candidate_texts),
            num_prompt_variants=num_prompt_variants,
            seed=seed,
            include_original_order=include_original_order,
        )
        start_index = len(flattened_history_texts)
        for permutation in permutations:
            flattened_history_texts.append(variant.history_text)
            flattened_candidate_texts.append([variant.candidate_texts[idx] for idx in permutation])
        groups.append(
            {
                "item": item,
                "permutations": permutations,
                "start_index": start_index,
                "end_index": len(flattened_history_texts),
            }
        )

    flat_scores = _score_prompt_batches(
        scorer=scorer,
        history_texts=flattened_history_texts,
        candidate_texts=flattened_candidate_texts,
        prompt_batch_size=prompt_batch_size,
    )

    for group in groups:
        item = group["item"]
        group_scores = flat_scores[group["start_index"] : group["end_index"]]
        record = _decomposed_uq_record(
            item=item,
            scores=group_scores,
            permutations=group["permutations"],
            method=method,
            store_variant_probabilities=store_variant_probabilities,
        )
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_file.flush()
    return len(groups)


def _score_prompt_batches(
    scorer: LabelProbabilityScorer,
    history_texts: list[list[str]],
    candidate_texts: list[list[str]],
    prompt_batch_size: int,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for start in range(0, len(history_texts), prompt_batch_size):
        end = start + prompt_batch_size
        scores.extend(scorer.score_batch(history_texts[start:end], candidate_texts[start:end]))
    return scores


def _decomposed_uq_record(
    item: dict[str, Any],
    scores: list[dict[str, Any]],
    permutations: list[list[int]],
    method: str,
    store_variant_probabilities: bool,
) -> dict[str, Any]:
    variant = item["variant"]
    num_candidates = len(variant.candidate_texts)
    letters = option_letters(num_candidates)
    aligned_probabilities = [
        _align_option_probabilities(score, permutation, letters)
        for score, permutation in zip(scores, permutations)
    ]
    mean_probabilities = _mean_distribution(aligned_probabilities)
    per_variant_entropies = [_normalized_entropy(row) for row in aligned_probabilities]
    total_uncertainty = _clamp01(_normalized_entropy(mean_probabilities))
    recommendation_uncertainty = _clamp01(
        float(np.mean(per_variant_entropies)) if per_variant_entropies else 0.0
    )
    prompt_uncertainty = _clamp01(max(total_uncertainty - recommendation_uncertainty, 0.0))
    confidence = _clamp01(1.0 - total_uncertainty)
    prompt_ratio = prompt_uncertainty / total_uncertainty if total_uncertainty > 0.0 else 0.0
    top_index = max(range(len(mean_probabilities)), key=mean_probabilities.__getitem__)

    components: dict[str, Any] = {
        "total_uncertainty": total_uncertainty,
        "recommendation_uncertainty": recommendation_uncertainty,
        "prompt_uncertainty": prompt_uncertainty,
        "prompt_uncertainty_ratio": prompt_ratio,
        "num_prompt_variants": len(scores),
        "entropy_normalization": "divide_by_log_num_candidates",
        "prompt_variant_policy": "candidate_order_permutation",
        "mean_option_probabilities": dict(zip(letters, mean_probabilities)),
        "per_variant_normalized_entropy": per_variant_entropies,
        "top_letter": letters[top_index],
    }
    if store_variant_probabilities:
        components["aligned_option_probabilities_by_variant"] = [
            dict(zip(letters, row)) for row in aligned_probabilities
        ]

    return {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": item["record_id"],
        "method": method,
        "method_family": "decomposition",
        "uncertainty_score": total_uncertainty,
        "confidence_score": confidence,
        "source_scores": _source_scores(prompt_uncertainty),
        "components": components,
        "metadata": {
            "dataset": variant.dataset,
            "scenario": variant.scenario,
            "variant_name": variant.variant_name,
            "instance_id": variant.instance_id,
            "num_candidates": num_candidates,
            "candidate_letters": letters,
            "candidate_order_permutations": permutations,
            "base_prompt_hash": item["base"]["prompt_hash"],
            "literature_basis": "Kweon et al. WWW 2025: total/recommendation/prompt uncertainty decomposition",
        },
    }


def _candidate_order_permutations(
    record_id: str,
    num_candidates: int,
    num_prompt_variants: int,
    seed: int,
    include_original_order: bool,
) -> list[list[int]]:
    rng = np.random.default_rng(_record_seed(seed, record_id))
    permutations: list[list[int]] = []
    if include_original_order:
        permutations.append(list(range(num_candidates)))
    while len(permutations) < num_prompt_variants:
        permutation = [int(idx) for idx in rng.permutation(num_candidates).tolist()]
        permutations.append(permutation)
    return permutations


def _record_seed(seed: int, record_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _align_option_probabilities(
    score: dict[str, Any],
    permutation: list[int],
    letters: list[str],
) -> list[float]:
    option_probabilities = score["option_probabilities"]
    aligned = [0.0 for _ in letters]
    for prompt_position, original_index in enumerate(permutation):
        prompt_letter = letters[prompt_position]
        aligned[original_index] = float(option_probabilities[prompt_letter])
    return _normalize_distribution(aligned)


def _mean_distribution(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    matrix = np.asarray(rows, dtype=float)
    return _normalize_distribution(matrix.mean(axis=0).tolist())


def _normalize_distribution(probabilities: list[float]) -> list[float]:
    total = float(sum(probabilities))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Option probability distribution has non-positive or non-finite mass")
    return [float(probability / total) for probability in probabilities]


def _normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    normalized = _normalize_distribution(probabilities)
    entropy = -sum(prob * math.log(prob) for prob in normalized if prob > 0.0)
    return float(entropy / math.log(len(normalized)))


def _source_scores(prompt_uncertainty: float) -> dict[str, float | None]:
    scores: dict[str, float | None] = {key: None for key in SOURCE_SCORE_KEYS}
    scores["prompt"] = prompt_uncertainty
    return scores


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


if __name__ == "__main__":
    main()
