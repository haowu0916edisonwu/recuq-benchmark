#!/usr/bin/env python
"""Derive base-anchored self-consistency variants from existing sampled rankings (no LLM inference).

Reviewer T5QE asked whether including the issued base ranking in the self-consistency score changes the
conclusions. From the stored S=5 sampled rankings this script writes two uq_prediction_v1 files:

* self_consistency_base_anchored: mean top-k Jaccard distance between the base ranking and each sample.
* self_consistency_plus_base: mean pairwise top-k Jaccard distance over the S samples plus the base ranking.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION  # noqa: E402
from recuq_benchmark.utils.io import open_text
from recuq_benchmark.utils.logging import get_logger  # noqa: E402

VARIANTS = {
    "anchored": "self_consistency_base_anchored",
    "plus_base": "self_consistency_plus_base",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base_predictions", type=str, required=True)
    parser.add_argument("--sc_predictions", type=str, required=True, help="Self-consistency uq_predictions.jsonl")
    parser.add_argument("--output_root", type=str, required=True, help="One sub-directory per variant is created here.")
    parser.add_argument("--method_suffix", type=str, default="", help="e.g. _qwen3 to keep method names distinct.")
    parser.add_argument("--top_k", type=int, default=10)
    return parser.parse_args()


def _jaccard_distance(a: list[str], b: list[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    return 1.0 - len(sa & sb) / len(sa | sb)


def main() -> None:
    args = parse_args()
    logger = get_logger("derive_sc_base_anchored")
    base: dict[str, dict[str, Any]] = {}
    with open_text(args.base_predictions) as f:
        for line in f:
            row = json.loads(line)
            base[row["record_id"]] = row

    out_dirs = {key: Path(args.output_root) / (name + args.method_suffix) for key, name in VARIANTS.items()}
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    handles = {key: (out_dirs[key] / "uq_predictions.jsonl").open("w", encoding="utf-8") for key in VARIANTS}

    written = missing_base = skipped_samples = 0
    with open_text(args.sc_predictions) as f:
        for line in f:
            sc = json.loads(line)
            record = base.get(sc["record_id"])
            if record is None:
                missing_base += 1
                continue
            ids = record["candidate_ids"]
            issued = [ids[i] for i in record["ranking"]]
            stored_samples = sc["components"].get("sampled_rankings")
            if stored_samples is None:
                raise ValueError("Use the original self-consistency outputs containing sampled_rankings")
            samples = [sample for sample in stored_samples if sample]
            skipped_samples += len(stored_samples) - len(samples)
            anchored = [_jaccard_distance(issued, s, args.top_k) for s in samples]
            pooled = [_jaccard_distance(a, b, args.top_k) for a, b in combinations([issued, *samples], 2)]
            scores = {
                "anchored": sum(anchored) / len(anchored) if anchored else 1.0,
                "plus_base": sum(pooled) / len(pooled) if pooled else 1.0,
            }
            for key, score in scores.items():
                row = {
                    "schema_version": UQ_SCHEMA_VERSION,
                    "record_id": sc["record_id"],
                    "method": VARIANTS[key] + args.method_suffix,
                    "method_family": "sampling",
                    "uncertainty_score": score,
                    "confidence_score": 1.0 - score,
                    "source_scores": {k: None for k in SOURCE_SCORE_KEYS},
                    "components": {
                        "score_definition": (
                            "mean top-k Jaccard distance between the issued base ranking and each sampled ranking"
                            if key == "anchored"
                            else "mean pairwise top-k Jaccard distance over the sampled rankings plus the base ranking"
                        ),
                        "top_k": args.top_k,
                        "num_samples": len(samples),
                        "source_method": sc["method"],
                        "original_sc_uncertainty": sc["uncertainty_score"],
                    },
                    "metadata": sc.get("metadata", {}),
                }
                handles[key].write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    for handle in handles.values():
        handle.close()
    for key, path in out_dirs.items():
        with (path / "run_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": VARIANTS[key] + args.method_suffix,
                    "rows": written,
                    "base_predictions": args.base_predictions,
                    "sc_predictions": args.sc_predictions,
                    "top_k": args.top_k,
                    "missing_base_records": missing_base,
                    "empty_samples_skipped": skipped_samples,
                    "does_not_rerun_llm": True,
                },
                f,
                indent=2,
            )
    logger.info("Wrote %d rows per variant (missing base=%d, empty samples=%d) to %s",
                written, missing_base, skipped_samples, args.output_root)
    if missing_base:
        raise SystemExit(f"{missing_base} self-consistency rows have no base record")


if __name__ == "__main__":
    main()
