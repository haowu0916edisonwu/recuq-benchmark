#!/usr/bin/env python
"""Reproduce the Warm-start Error@K/TopKMass alignment diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.utils.io import open_text


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open_text(path, "r") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_predictions", type=Path, required=True)
    parser.add_argument("--labelprob_predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = {
        row["record_id"]: row
        for row in read_jsonl(args.base_predictions)
        if row.get("scenario") == "s0_warmstart"
    }
    labelprob = {row["record_id"]: row for row in read_jsonl(args.labelprob_predictions)}
    if len(base) != 4000:
        raise ValueError(f"Expected 4,000 Warm-start base records, found {len(base)}")
    if not set(base).issubset(labelprob):
        raise ValueError("LabelProb artifact is missing Warm-start record_ids")

    cutoffs = (5, 10, 15)
    labels = {cutoff: [] for cutoff in cutoffs}
    scores = {score_k: {event_k: [] for event_k in cutoffs} for score_k in cutoffs}
    for record_id, record in base.items():
        positive_indices = {idx for idx, value in enumerate(record["labels"]) if value}
        ranking = [int(value) for value in record["ranking"]]
        probabilities = labelprob[record_id]["components"]["option_probabilities"]
        for event_k in cutoffs:
            labels[event_k].append(int(not positive_indices.intersection(ranking[:event_k])))
        for score_k in cutoffs:
            top_letters = [chr(65 + idx) for idx in ranking[:score_k]]
            uncertainty = 1.0 - sum(float(probabilities[letter]) for letter in top_letters)
            for event_k in cutoffs:
                scores[score_k][event_k].append(uncertainty)

    rows = []
    for score_k in cutoffs:
        row = {"score": f"Top{score_k}Mass", "n": len(base)}
        for event_k in cutoffs:
            row[f"error_at_{event_k}_auroc"] = roc_auc_score(
                labels[event_k], scores[score_k][event_k]
            )
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
