#!/usr/bin/env python3
"""Export sanitized scores and Qwen controls for the camera-ready diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_record_artifacts import clean_base, iter_jsonl, write_gzip_jsonl


ROOT = Path(__file__).resolve().parents[1]
BACKBONES = {"llama": "llama31_8b", "qwen": "qwen3_8b_nonthinking"}
SAME_PROMPT = (
    "alignment_probe_own_top10_mass",
    "same_inference_slot1_top10mass",
    "same_inference_slot1_max",
    "same_inference_seq_nll_top10",
)


def score_rows(path: Path, group: str):
    for row in iter_jsonl(path):
        yield {
            "schema_version": row["schema_version"],
            "record_id": row["record_id"],
            "method": row["method"],
            "method_family": row["method_family"],
            "uncertainty_score": row["uncertainty_score"],
            "metadata": {"diagnostic": group},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    results = args.source_root / "results"
    output = args.output_root.resolve()
    controls = results / "qwen3_8b_nonthinking/perturbation_controls/raw_predictions.jsonl"
    entries = [write_gzip_jsonl(
        output / "supplement/qwen_controlled_rankings.jsonl.gz",
        (clean_base(row, BACKBONES["qwen"]) for row in iter_jsonl(controls)),
        args.overwrite,
    )]
    for backbone, source_name in BACKBONES.items():
        uq_root = results / "uq"
        backbone_root = uq_root if backbone == "llama" else uq_root / source_name
        groups = {
            "same_prompt": {
                name: uq_root / "revision_alignment" / source_name / name for name in SAME_PROMPT
            },
            "self_consistency": {
                name: backbone_root / name
                for name in ("self_consistency_base_anchored", "self_consistency_plus_base")
            },
        }
        for group, methods in groups.items():
            for method, directory in methods.items():
                entries.append(write_gzip_jsonl(
                    output / "supplement" / group / backbone / f"{method}.jsonl.gz",
                    score_rows(directory / "uq_predictions.jsonl", group),
                    args.overwrite,
                ))
    for entry in entries:
        entry["path"] = Path(entry["path"]).relative_to(output).as_posix()
    path = output / "manifests/supplement_build_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"files": entries}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
