#!/usr/bin/env python
"""Shared-temperature control for the backbone ordering reversal.

Applies the same scalar temperature T to the option-label distribution of both
backbones and writes uq_prediction_v1 files for every probe-based score
(LabelProb-Max, -Entropy, -Margin, -Top10Mass). T is a fixed constant, so no
labels are used. Evaluate the outputs with evaluate_uq_methods.py.

Default temperatures: 1 (untempered), and the mean fitted fold temperature of
each backbone (Llama 1.2275, Qwen 8.6154).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION  # noqa: E402
from recuq_benchmark.utils.io import open_text, resolve_jsonl
from recuq_benchmark.utils.logging import get_logger  # noqa: E402

LETTERS = [chr(ord("A") + i) for i in range(20)]
DEFAULT_TEMPERATURES = {"T1": 1.0, "Tllama": 1.2275, "Tqwen": 8.6154}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base_predictions", required=True)
    parser.add_argument("--labelprob", required=True, help="LabelProb uq_predictions (dir or file) with option_probabilities.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--scenario", default="s0_warmstart")
    return parser.parse_args()


def _scores(probs: list[float], top10: set[int], temperature: float) -> dict[str, float]:
    logits = [math.log(max(p, 1e-300)) / temperature for p in probs]
    m = max(logits)
    weights = [math.exp(z - m) for z in logits]
    total = sum(weights)
    q = [w / total for w in weights]
    ordered = sorted(q, reverse=True)
    return {
        "labelprob_max": 1.0 - ordered[0],
        "labelprob_entropy": -sum(x * math.log(x) for x in q if x > 0) / math.log(len(q)),
        "labelprob_margin": 1.0 - (ordered[0] - ordered[1]),
        "labelprob_top10mass": 1.0 - sum(q[i] for i in top10),
    }


def main() -> None:
    args = parse_args()
    logger = get_logger("derive_shared_temperature")
    base = {}
    with open_text(args.base_predictions) as f:
        for line in f:
            row = json.loads(line)
            if row["scenario"] == args.scenario:
                base[row["record_id"]] = row
    path = resolve_jsonl(args.labelprob)
    probe = {}
    with open_text(path) as f:
        for line in f:
            row = json.loads(line)
            if row["record_id"] in base:
                probe[row["record_id"]] = [row["components"]["option_probabilities"][a] for a in LETTERS]
    missing = set(base) - set(probe)
    if missing:
        raise SystemExit(f"{len(missing)} base records lack LabelProb rows")

    handles = {}
    for tag, temperature in DEFAULT_TEMPERATURES.items():
        for score in ("labelprob_max", "labelprob_entropy", "labelprob_margin", "labelprob_top10mass"):
            method = f"shared_{tag}_{score}"
            out = Path(args.output_root) / method
            out.mkdir(parents=True, exist_ok=True)
            handles[(tag, score)] = (method, temperature, (out / "uq_predictions.jsonl").open("w", encoding="utf-8"))
    for rid, record in base.items():
        top10 = set(record["ranking"][:10])
        for tag, temperature in DEFAULT_TEMPERATURES.items():
            values = _scores(probe[rid], top10, temperature)
            for score, value in values.items():
                method, _, handle = handles[(tag, score)]
                handle.write(json.dumps({
                    "schema_version": UQ_SCHEMA_VERSION,
                    "record_id": rid,
                    "method": method,
                    "method_family": "calibration_derived",
                    "uncertainty_score": value,
                    "confidence_score": 1.0 - value,
                    "source_scores": {k: None for k in SOURCE_SCORE_KEYS},
                    "components": {"shared_temperature": temperature, "score": score},
                    "metadata": {k: record.get(k) for k in ("dataset", "scenario", "variant_name", "instance_id")},
                }) + "\n")
    for _, _, handle in handles.values():
        handle.close()
    logger.info("Wrote %d methods x %d records to %s", len(handles), len(base), args.output_root)


if __name__ == "__main__":
    main()
