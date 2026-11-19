#!/usr/bin/env python
"""Build the compact, public fixed rankings and record-aligned UQ scores.

This script is intentionally separate from model inference.  It streams the
canonical local outputs, removes raw prompts/responses and source identifiers,
and writes deterministic gzip-compressed JSONL files.  The resulting records
retain every field needed by the released evaluators.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


LLAMA_UQ = {
    "labelprob_max": "results/uq/labelprob_llama31_8b/uq_predictions.jsonl",
    "labelprob_top10mass": "results/uq/derived_labelprob_base_top10_mass/uq_predictions.jsonl",
    "labelprob_entropy": "results/uq/labelprob_entropy/uq_predictions.jsonl",
    "labelprob_margin": "results/uq/labelprob_margin/uq_predictions.jsonl",
    "temperature_scaled_labelprob": "results/uq/labelprob_temp_scaled/uq_predictions.jsonl",
    "decomposed_llm_uq": "results/uq/kweon_decomp_llama31_8b/uq_predictions.jsonl",
    "perturbation_rank_instability": "results/uq/perturbation_rank_instability/uq_predictions.jsonl",
    "self_consistency": "results/uq/self_consistency_s5_t09_p095/uq_predictions.jsonl",
}

QWEN_UQ = {
    "labelprob_max": "labelprob_max/uq_predictions.jsonl",
    "labelprob_top10mass": "derived_labelprob_base_top10_mass/uq_predictions.jsonl",
    "labelprob_entropy": "labelprob_entropy/uq_predictions.jsonl",
    "labelprob_margin": "labelprob_margin/uq_predictions.jsonl",
    "temperature_scaled_labelprob": "labelprob_temp_scaled/uq_predictions.jsonl",
    "decomposed_llm_uq": "kweon_decomp/uq_predictions.jsonl",
    "perturbation_rank_instability": "perturbation_rank_instability/uq_predictions.jsonl",
    "self_consistency": "self_consistency_s5_t09_p095/uq_predictions.jsonl",
}

PUBLIC_COMPONENTS = {
    "labelprob_max": {
        "entropy", "max_option_probability", "normalized_entropy",
        "option_probabilities", "probability_margin",
        "second_option_probability", "top_letter",
    },
    "labelprob_top10mass": {
        "base_top_k", "base_top_k_letters", "base_top_k_probability_mass",
        "option_probabilities", "score_definition",
    },
    "labelprob_entropy": {"normalized_entropy", "entropy", "option_probabilities"},
    "labelprob_margin": {
        "probability_margin", "max_option_probability",
        "second_option_probability", "option_probabilities",
    },
    "temperature_scaled_labelprob": {
        "temperature", "fold", "max_option_probability",
        "temperature_scaled_option_probabilities", "score_definition",
    },
    "decomposed_llm_uq": {
        "entropy_normalization", "num_prompt_variants", "prompt_uncertainty",
        "prompt_uncertainty_ratio", "prompt_variant_policy",
        "recommendation_uncertainty", "total_uncertainty", "top_letter",
    },
    "perturbation_rank_instability": {
        "case_mean_instability", "item_mean_instability", "literature_basis",
        "num_item_perturbations", "num_user_perturbations", "rank_distance",
        "score_definition", "score_source", "top_k", "user_mean_instability",
        "variant_distance_to_paired_control",
    },
    "self_consistency": {
        "literature_basis", "num_samples", "num_unique_top1",
        "parse_success_rate", "sampling_mode", "score_definition",
        "temperature", "top1_agreement_confidence", "top1_entropy_uncertainty",
        "top1_fsd_confidence", "top1_fsd_uncertainty", "top_k",
        "top_k_jaccard_uncertainty", "top_p",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            yield value


def condition_name(scenario: Any) -> str:
    return {
        "s0_warmstart": "warm_start",
        "s3_sparsity": "sparse_history",
        "s4_longtail": "tail_target_subset",
        "s5_robustness": "item_text_robustness",
    }.get(str(scenario), str(scenario))


def clean_parse_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    allowed = {
        "num_found_letters", "num_unique_found_letters", "parse_repaired",
        "repair_policy", "missing_count", "duplicate_count", "invalid_count",
    }
    result = {key: value[key] for key in allowed if key in value}
    result.setdefault("missing_count", len(value.get("missing_letters", [])))
    result.setdefault("duplicate_count", len(value.get("duplicate_letters", [])))
    result.setdefault("invalid_count", len(value.get("invalid_letters", [])))
    return result


def clean_base(row: dict[str, Any], backbone: str) -> dict[str, Any]:
    model_metadata = row.get("model_metadata") if isinstance(row.get("model_metadata"), dict) else {}
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    result = {
        "schema_version": "fixed_ranking_record_v1",
        "record_id": row["record_id"],
        "instance_id": row.get("instance_id"),
        "backbone": backbone,
        "dataset": row.get("dataset"),
        "scenario": row.get("scenario"),
        "evaluation_condition": condition_name(row.get("scenario")),
        "challenge": row.get("challenge"),
        "variant_name": row.get("variant_name"),
        "target_source": row.get("target_source"),
        "candidate_ids": [chr(65 + idx) for idx in range(len(row.get("labels", [])))],
        "labels": row.get("labels"),
        "ranking": row.get("ranking"),
        "metrics": row.get("metrics"),
        "is_error_at_10": bool(row.get("is_error_at_10")),
        "error_at_10": int(bool(row.get("is_error_at_10"))),
        "eval_k": int(row.get("eval_k", 10)),
        "parse_success": bool(row.get("parse_success")),
        "model_metadata": {
            "model": model_metadata.get("model") or row.get("model_name"),
            "score_mode": model_metadata.get("score_mode"),
            "non_thinking": backbone == "qwen3_8b_nonthinking",
            "parse_diagnostics": clean_parse_diagnostics(model_metadata.get("parse_diagnostics")),
        },
        "metadata": {"history_length": metadata.get("history_length")},
    }
    popularity = metadata.get("popularity_bucket") or row.get("popularity_bucket")
    if popularity is not None:
        result["metadata"]["popularity_bucket"] = popularity
        result["popularity_bucket"] = popularity
    return result


def clean_uq(row: dict[str, Any], release_name: str, backbone: str) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    components = row.get("components") if isinstance(row.get("components"), dict) else {}
    allowed = PUBLIC_COMPONENTS[release_name]
    clean_components = {key: components[key] for key in sorted(allowed) if key in components}
    source_scores = row.get("source_scores") if isinstance(row.get("source_scores"), dict) else {}
    clean_sources = {
        key: source_scores[key]
        for key in ("user", "item", "candidate", "prompt", "model")
        if isinstance(source_scores.get(key), (int, float))
    }
    return {
        "schema_version": "uq_prediction_v1",
        "record_id": row["record_id"],
        "method": row["method"],
        "method_family": row["method_family"],
        "uncertainty_score": row["uncertainty_score"],
        "confidence_score": row.get("confidence_score"),
        "source_scores": clean_sources,
        "components": clean_components,
        "metadata": {
            "release_name": release_name,
            "backbone": backbone,
            "dataset": metadata.get("dataset"),
            "scenario": metadata.get("scenario"),
            "evaluation_condition": condition_name(metadata.get("scenario")),
            "variant_name": metadata.get("variant_name"),
            "instance_id": metadata.get("instance_id"),
        },
    }


def write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]], overwrite: bool) -> dict[str, Any]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                payload = (json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
                compressed.write(payload)
                digest.update(payload)
                count += 1
    return {"path": str(path), "rows": count, "jsonl_sha256": digest.hexdigest()}


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    output = args.output_root.resolve()
    manifest: list[dict[str, Any]] = []

    llama_base = source / "results/phase1_base_llm_aligned20_n1000/raw_predictions.jsonl"
    qwen_base = source / "results/qwen3_8b_nonthinking/warm_start/raw_predictions.jsonl"
    manifest.append(write_gzip_jsonl(
        output / "base_predictions/llama_fixed_rankings.jsonl.gz",
        (clean_base(row, "llama3.1_8b_instruct") for row in iter_jsonl(llama_base)),
        args.overwrite,
    ))
    manifest.append(write_gzip_jsonl(
        output / "base_predictions/qwen_warm_start_fixed_rankings.jsonl.gz",
        (clean_base(row, "qwen3_8b_nonthinking") for row in iter_jsonl(qwen_base)),
        args.overwrite,
    ))

    for release_name, relative in LLAMA_UQ.items():
        manifest.append(write_gzip_jsonl(
            output / f"uq_outputs/llama/{release_name}.jsonl.gz",
            (clean_uq(row, release_name, "llama3.1_8b_instruct") for row in iter_jsonl(source / relative)),
            args.overwrite,
        ))
    qwen_root = source / "results/uq/qwen3_8b_nonthinking"
    for release_name, relative in QWEN_UQ.items():
        manifest.append(write_gzip_jsonl(
            output / f"uq_outputs/qwen/{release_name}.jsonl.gz",
            (clean_uq(row, release_name, "qwen3_8b_nonthinking") for row in iter_jsonl(qwen_root / relative)),
            args.overwrite,
        ))

    manifest_path = output / "manifests/artifact_build_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for entry in manifest:
        entry["path"] = str(Path(entry["path"]).relative_to(output))
    manifest_path.write_text(json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest)} compressed record artifacts to {output}")


if __name__ == "__main__":
    main()
