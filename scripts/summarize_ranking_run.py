#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.metrics.aggregate import aggregate_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a fixed-ranking generation directory.")
    parser.add_argument("result_dir", type=str, help="Directory containing raw_predictions.jsonl.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to result_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = result_dir / "raw_predictions.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw predictions file: {raw_path}")

    records = _read_jsonl(raw_path)
    summary = _summarize_run(records, result_dir)

    _write_json(output_dir / "run_summary.json", summary)
    _write_csv(output_dir / "parse_quality_by_dataset_scenario.csv", _parse_quality_rows(records))
    _write_csv(output_dir / "repair_distribution.csv", _repair_distribution_rows(records))
    _write_csv(output_dir / "main_metrics_by_dataset_scenario.csv", aggregate_records(records, ["dataset", "scenario"]))
    _write_csv(
        output_dir / "s3_metrics_by_history_variant.csv",
        aggregate_records(
            [record for record in records if record["scenario"] == "s3_sparsity"],
            ["dataset", "variant_name"],
        ),
    )
    _write_csv(
        output_dir / "s4_metrics_by_popularity_bucket.csv",
        aggregate_records(
            _flatten_metadata(records, "popularity_bucket", scenario="s4_longtail"),
            ["dataset", "popularity_bucket"],
        ),
    )
    _write_csv(
        output_dir / "s5_metrics_by_ablated_field.csv",
        aggregate_records(
            _flatten_metadata(records, "ablated_field", scenario="s5_robustness"),
            ["dataset", "ablated_field"],
        ),
    )

    _print_summary(summary, output_dir)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _summarize_run(records: list[dict[str, Any]], result_dir: Path) -> dict[str, Any]:
    total = len(records)
    strict_failures = [record for record in records if not record.get("parse_success")]
    repaired = [record for record in records if _parse_diag(record).get("parse_repaired")]
    fallback = [
        record
        for record in records
        if _parse_diag(record).get("repair_policy") == "fallback_original_order"
    ]
    return {
        "result_dir": str(result_dir),
        "total_records": total,
        "strict_parse_failures": len(strict_failures),
        "strict_parse_success_rate": _rate(total - len(strict_failures), total),
        "repaired": len(repaired),
        "repair_rate": _rate(len(repaired), total),
        "fallback": len(fallback),
        "fallback_rate": _rate(len(fallback), total),
        "datasets": sorted({record.get("dataset") for record in records}),
        "scenarios": sorted({record.get("scenario") for record in records}),
        "repair_distribution": _counter_to_rows(_repair_counter(records)),
        "fallback_distribution": _counter_to_rows(_fallback_counter(records)),
    }


def _parse_quality_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["scenario"])].append(record)

    rows: list[dict[str, Any]] = []
    for (dataset, scenario), group in sorted(grouped.items()):
        n = len(group)
        strict_failures = sum(1 for record in group if not record.get("parse_success"))
        repaired = sum(1 for record in group if _parse_diag(record).get("parse_repaired"))
        fallback = sum(
            1
            for record in group
            if _parse_diag(record).get("repair_policy") == "fallback_original_order"
        )
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "n": n,
                "strict_parse_failures": strict_failures,
                "strict_parse_success_rate": _rate(n - strict_failures, n),
                "repaired": repaired,
                "repair_rate": _rate(repaired, n),
                "fallback": fallback,
                "fallback_rate": _rate(fallback, n),
            }
        )
    return rows


def _repair_distribution_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, scenario, variant_name), count in sorted(_repair_counter(records).items()):
        rows.append(
            {
                "dataset": dataset,
                "scenario": scenario,
                "variant_name": variant_name,
                "repaired": count,
            }
        )
    return rows


def _repair_counter(records: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        if _parse_diag(record).get("parse_repaired"):
            counter[(record["dataset"], record["scenario"], record["variant_name"])] += 1
    return counter


def _fallback_counter(records: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        if _parse_diag(record).get("repair_policy") == "fallback_original_order":
            counter[(record["dataset"], record["scenario"], record["variant_name"])] += 1
    return counter


def _counter_to_rows(counter: Counter[tuple[str, str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": dataset,
            "scenario": scenario,
            "variant_name": variant_name,
            "count": count,
        }
        for (dataset, scenario, variant_name), count in sorted(counter.items())
    ]


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


def _parse_diag(record: dict[str, Any]) -> dict[str, Any]:
    return record.get("model_metadata", {}).get("parse_diagnostics", {})


def _rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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


def _print_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print(f"Result dir: {summary['result_dir']}")
    print(f"Total records: {summary['total_records']}")
    print(
        "Strict parse success: "
        f"{summary['strict_parse_success_rate']:.4f} "
        f"({summary['total_records'] - summary['strict_parse_failures']}/{summary['total_records']})"
    )
    print(f"Repair rate: {summary['repair_rate']:.4f} ({summary['repaired']}/{summary['total_records']})")
    print(f"Fallback rate: {summary['fallback_rate']:.4f} ({summary['fallback']}/{summary['total_records']})")
    print(f"Wrote summary files to: {output_dir}")


if __name__ == "__main__":
    main()
