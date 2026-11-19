#!/usr/bin/env python
"""Build repository-level SHA-256 and size manifests using relative paths."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "artifacts/manifests/release_files.csv"


LOCAL_FOLDERS = {
    ".git", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "__pycache__", ".cache", "outputs", "data", "cache", "results", "logs",
    "huggingface", "wandb",
}


def release_files(root: Path):
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [name for name in subdirs if name not in LOCAL_FOLDERS]
        for name in filenames:
            path = Path(directory) / name
            if name in {".DS_Store", ".env"} or path.suffix in {".pyc", ".pyo", ".log", ".key", ".pem", ".token"}:
                continue
            if path != root / "artifacts/manifests/release_files.csv":
                yield path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def role(relative: Path) -> str:
    first = relative.parts[0]
    if first == "artifacts":
        if "base_predictions" in relative.parts:
            return "fixed_base_prediction"
        if "uq_outputs" in relative.parts:
            return "record_aligned_uq_output"
        if "reference_results" in relative.parts:
            return "reference_evaluation_output"
        if "schemas" in relative.parts:
            return "schema"
        return "artifact_metadata"
    return {
        "scripts": "executable_code",
        "recuq_benchmark": "library_code",
        "configs": "configuration",
        "docs": "documentation",
        "figures": "figure_asset",
        "tests": "test",
    }.get(first, "repository_metadata")


def main() -> None:
    rows = []
    for path in sorted(release_files(ROOT)):
        relative = path.relative_to(ROOT)
        rows.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
            "role": role(relative),
        })
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "bytes", "sha256", "role"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} entries to {CSV_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
