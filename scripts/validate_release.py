#!/usr/bin/env python
"""Validate counts, joins, hashes, privacy fields, and reported paper values."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from build_release_manifest import release_files


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LLAMA = 45163
EXPECTED_ELIGIBLE = 42475
EXPECTED_QWEN = 4000
EXPECTED_CONDITIONS = {
    "s0_warmstart": 4000,
    "s3_sparsity": 20000,
    "s4_longtail": 1312,
    "s5_robustness": 17163,
}
FORBIDDEN_BASE_FIELDS = {
    "user_id", "ground_truth_item_ids", "raw_response", "prompt_hash",
    "history_hash", "candidate_hash", "run_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/validation_report.json")
    return parser.parse_args()


def iter_gzip_jsonl(path: Path) -> Iterator[tuple[dict[str, Any], bytes]]:
    with gzip.open(path, "rb") as handle:
        for line_number, payload in enumerate(handle, 1):
            if not payload.strip():
                continue
            row = json.loads(payload)
            if not isinstance(row, dict):
                raise AssertionError(f"{path}:{line_number}: expected JSON object")
            yield row, payload


def inspect_base(path: Path, expected: int) -> dict[str, Any]:
    ids: set[str] = set()
    scenarios: Counter[str] = Counter()
    eligible_scenarios: Counter[str] = Counter()
    forbidden_hits: Counter[str] = Counter()
    digest = hashlib.sha256()
    count = 0
    for row, payload in iter_gzip_jsonl(path):
        digest.update(payload)
        count += 1
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise AssertionError(f"Missing record_id in {path}")
        if record_id in ids:
            raise AssertionError(f"Duplicate base record_id: {record_id}")
        ids.add(record_id)
        scenario = str(row.get("scenario"))
        scenarios[scenario] += 1
        is_eligible = scenario != "s4_longtail" or row.get("popularity_bucket") == "tail"
        if is_eligible:
            eligible_scenarios[scenario] += 1
        forbidden_hits.update(FORBIDDEN_BASE_FIELDS.intersection(row))
        if len(row.get("labels", [])) != 20 or len(row.get("ranking", [])) != 20:
            raise AssertionError(f"Expected 20 labels/ranks for {record_id}")
        if sorted(int(value) for value in row["ranking"]) != list(range(20)):
            raise AssertionError(f"Ranking is not a permutation for {record_id}")
        if bool(row.get("is_error_at_10")) != bool(row.get("error_at_10")):
            raise AssertionError(f"Inconsistent Error@10 labels for {record_id}")
        serialized = payload.decode("utf-8", errors="ignore").lower()
        if "/nvme/" in serialized or "/home/" in serialized or "@gmail" in serialized:
            raise AssertionError(f"Local path or email-like identity string in {record_id}")
    if count != expected:
        raise AssertionError(f"{path}: expected {expected} rows, found {count}")
    if forbidden_hits:
        raise AssertionError(f"Forbidden base fields: {dict(forbidden_hits)}")
    return {
        "rows": count,
        "record_ids": ids,
        "scenario_counts": dict(scenarios),
        "eligible_scenario_counts": dict(eligible_scenarios),
        "eligible_rows": sum(eligible_scenarios.values()),
        "jsonl_sha256": digest.hexdigest(),
    }


def inspect_uq(path: Path, expected_ids: set[str], manifest_hash: str) -> dict[str, Any]:
    ids: set[str] = set()
    digest = hashlib.sha256()
    method: str | None = None
    family: str | None = None
    for row, payload in iter_gzip_jsonl(path):
        digest.update(payload)
        record_id = row.get("record_id")
        if record_id in ids:
            raise AssertionError(f"Duplicate UQ record_id in {path}: {record_id}")
        ids.add(record_id)
        method = method or row.get("method")
        family = family or row.get("method_family")
        if row.get("schema_version") != "uq_prediction_v1":
            raise AssertionError(f"Invalid UQ schema version in {path}")
        score = row.get("uncertainty_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise AssertionError(f"Non-finite uncertainty score in {path}:{record_id}")
    if ids != expected_ids:
        raise AssertionError(
            f"Join mismatch for {path}: missing={len(expected_ids - ids)} extra={len(ids - expected_ids)}"
        )
    if digest.hexdigest() != manifest_hash:
        raise AssertionError(f"Content hash mismatch for {path}")
    return {"rows": len(ids), "method": method, "method_family": family, "join_coverage": 1.0}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_release_files(root: Path) -> dict[str, Any]:
    manifest_path = root / "artifacts/manifests/release_files.csv"
    rows = csv_rows(manifest_path)
    listed: set[str] = set()
    for row in rows:
        relative = row["path"]
        if relative in listed:
            raise AssertionError(f"Duplicate release-manifest path: {relative}")
        listed.add(relative)
        path = root / relative
        if not path.is_file():
            raise AssertionError(f"Missing release-manifest file: {relative}")
        if path.stat().st_size != int(row["bytes"]):
            raise AssertionError(f"Size mismatch in release manifest: {relative}")
        if file_sha256(path) != row["sha256"]:
            raise AssertionError(f"SHA-256 mismatch in release manifest: {relative}")

    expected = {path.relative_to(root).as_posix() for path in release_files(root)}
    if listed != expected:
        raise AssertionError(
            f"Release-manifest coverage mismatch: missing={sorted(expected - listed)}, stale={sorted(listed - expected)}"
        )
    return {"status": "passed", "files_checked": len(rows)}


def close(actual: str | float, expected: float, tolerance: float = 5e-7) -> None:
    if abs(float(actual) - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, found {actual}")


def check_paper_values(root: Path) -> dict[str, Any]:
    tables = root / "artifacts/reference_results/paper_tables"
    t1 = csv_rows(tables / "table1_benchmark_overview.csv")
    assert {row["dataset"] for row in t1} == {"mind_20", "amazon_books", "amazon_movies", "yelp"}
    assert sum(int(row["base_instances"]) for row in t1) == EXPECTED_QWEN
    assert sum(int(row["generated_records"]) for row in t1) == EXPECTED_LLAMA
    assert sum(int(row["evaluation_eligible_records"]) for row in t1) == EXPECTED_ELIGIBLE

    t3 = csv_rows(tables / "table3_warm_start_auroc.csv")
    llama = next(row for row in t3 if row["backbone"].startswith("Llama") and row["dataset"] == "overall")
    qwen = next(row for row in t3 if row["backbone"].startswith("Qwen") and row["dataset"] == "overall")
    close(llama["labelprob_top10mass"], 0.6280276639867605)
    close(qwen["decomposed_llm_uq"], 0.558855957529661)
    close(qwen["temperature_scaled_labelprob"], 0.5382642372220433)

    for filename, expected in (
        ("table5_sparse_history_auroc.csv", 0.633594893332004),
        ("table6_tail_target_subset_auroc.csv", 0.5474440819028704),
        ("table7_item_text_robustness_auroc.csv", 0.6399001084194768),
    ):
        overall = next(row for row in csv_rows(tables / filename) if row["dataset"] == "overall")
        close(overall["labelprob_top10mass"], expected)

    paired = csv_rows(tables / "table8_paired_control_response.csv")
    perturb = next(row for row in paired if row["method"] == "Perturbation Rank Instability")
    close(perturb["directional_coverage"], 0.9873556899395272)
    close(perturb["dominant_control_accuracy"], 0.7091808686091259)
    close(perturb["macro_f1"], 0.7123876123876124)
    close(perturb["source_auroc"], 0.6748671386327973)
    close(perturb["rho"], 0.4557862176965527)

    topk = csv_rows(root / "artifacts/reference_results/diagnostics/topk_alignment.csv")
    close(next(row for row in topk if row["score"] == "Top10Mass")["error_at_10_auroc"], 0.6280279259265663)

    component = csv_rows(
        root / "artifacts/reference_results/diagnostics/component_checks/table11_error_auroc_by_method.csv"
    )
    base_top1 = next(row for row in component if row["method"] == "derived_labelprob_base_top1")
    top10 = next(row for row in component if row["method"] == "derived_labelprob_base_top10_mass")
    close(base_top1["error_auroc"], 0.5838109214359555)
    close(top10["error_auroc"], 0.6337819440548209)
    assert int(base_top1["n"]) == EXPECTED_ELIGIBLE
    rbo = next(row for row in paired if row["method"] == "RBO distance (p=0.9)")
    close(rbo["dominant_control_accuracy"], 0.746, 0.0005)
    close(perturb["signed_source_auroc"], 0.496, 0.0005)
    assert (int(perturb["clipped_n"]), int(perturb["signed_n"])) == (1819, 3617)
    return {"status": "passed", "tables_checked": [1, 3, 5, 6, 7, 8, 11, 15]}


def validate_supplement(artifacts: Path) -> dict[str, Any]:
    manifest = json.loads((artifacts / "manifests/supplement_build_manifest.json").read_text())
    for entry in manifest["files"]:
        path = artifacts / entry["path"]
        digest = hashlib.sha256()
        record_ids = set()
        count = 0
        for row, payload in iter_gzip_jsonl(path):
            digest.update(payload)
            record_ids.add(row["record_id"])
            count += 1
            if "uncertainty_score" in row and not math.isfinite(row["uncertainty_score"]):
                raise AssertionError(f"Non-finite score in {path}")
        if count != len(record_ids) or count != entry["rows"] or digest.hexdigest() != entry["jsonl_sha256"]:
            raise AssertionError(f"Supplemental file failed manifest validation: {path}")
    controls = inspect_base(artifacts / "supplement/qwen_controlled_rankings.jsonl.gz", 41_163)
    assert controls["scenario_counts"] == {"s0_warmstart": 4000, "s3_sparsity": 20000, "s5_robustness": 17163}
    return {"files": len(manifest["files"]), "qwen_controlled_rows": controls["rows"]}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    artifacts = root / "artifacts"
    manifest = json.loads((artifacts / "manifests/artifact_build_manifest.json").read_text())
    manifest_by_path = {entry["path"]: entry for entry in manifest["files"]}

    llama_rel = "base_predictions/llama_fixed_rankings.jsonl.gz"
    qwen_rel = "base_predictions/qwen_warm_start_fixed_rankings.jsonl.gz"
    supplement = validate_supplement(artifacts)
    llama = inspect_base(artifacts / llama_rel, EXPECTED_LLAMA)
    qwen = inspect_base(artifacts / qwen_rel, EXPECTED_QWEN)
    if llama["eligible_rows"] != EXPECTED_ELIGIBLE:
        raise AssertionError(f"Expected {EXPECTED_ELIGIBLE} eligible Llama rows")
    if llama["eligible_scenario_counts"] != EXPECTED_CONDITIONS:
        raise AssertionError(f"Unexpected eligible condition counts: {llama['eligible_scenario_counts']}")
    if qwen["scenario_counts"] != {"s0_warmstart": EXPECTED_QWEN}:
        raise AssertionError("Qwen release must contain only the Warm-start replication")

    for relative, info in ((llama_rel, llama), (qwen_rel, qwen)):
        expected_hash = manifest_by_path[relative]["jsonl_sha256"]
        if info["jsonl_sha256"] != expected_hash:
            raise AssertionError(f"Base content hash mismatch: {relative}")

    uq_reports = []
    for backbone, base_info in (("llama", llama), ("qwen", qwen)):
        for path in sorted((artifacts / "uq_outputs" / backbone).glob("*.jsonl.gz")):
            relative = str(path.relative_to(artifacts))
            uq_reports.append({
                "path": relative,
                **inspect_uq(path, base_info["record_ids"], manifest_by_path[relative]["jsonl_sha256"]),
            })
    report = {
        "status": "passed",
        "base_artifacts": {
            "llama": {key: value for key, value in llama.items() if key != "record_ids"},
            "qwen": {key: value for key, value in qwen.items() if key != "record_ids"},
        },
        "uq_artifacts": uq_reports,
        "supplement": supplement,
        "paper_values": check_paper_values(root),
        "release_files": check_release_files(root),
        "privacy_contract": {
            "forbidden_base_fields_absent": sorted(FORBIDDEN_BASE_FIELDS),
            "raw_prompt_or_response_text_released": False,
            "source_user_or_item_ids_released": False
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "llama_rows": llama["rows"],
        "llama_eligible_rows": llama["eligible_rows"],
        "qwen_rows": qwen["rows"],
        "uq_files": len(uq_reports),
        "report": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
