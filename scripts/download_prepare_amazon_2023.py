#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HF_REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

DATASETS = {"movies": "Movies_and_TV", "books": "Books"}

REVIEW_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
META_COLUMNS = [
    "parent_asin",
    "main_category",
    "title",
    "subtitle",
    "author",
    "store",
    "categories",
    "description",
    "average_rating",
    "rating_number",
    "price",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Amazon Reviews 2023 5-core interactions and metadata, "
            "then convert them to fixed-ranking UQ benchmark local parquet files."
        )
    )
    parser.add_argument("--dataset", choices=["movies", "books", "all"], required=True)
    parser.add_argument("--repo_id", type=str, default=HF_REPO_ID)
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision.")
    parser.add_argument("--output_root", type=str, default=str(REPO_ROOT / "data/raw/amazon"))
    parser.add_argument("--cache_dir", type=str, default=os.environ.get("HF_HUB_CACHE"))
    parser.add_argument("--review_chunksize", type=int, default=1_000_000)
    parser.add_argument("--meta_batch_size", type=int, default=100_000)
    parser.add_argument("--force", action="store_true", help="Overwrite existing parquet outputs.")
    parser.add_argument(
        "--copy_source",
        action="store_true",
        help="Also copy downloaded source files into the dataset raw directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = ["movies", "books"] if args.dataset == "all" else [args.dataset]
    for name in selected:
        prepare_dataset(name, args)


def prepare_dataset(name: str, args: argparse.Namespace) -> None:
    category = DATASETS[name]
    output_dir = Path(args.output_root) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = (
        (f"benchmark/5core/rating_only/{category}.csv", output_dir / "reviews.parquet",
         convert_reviews_csv, args.review_chunksize),
        (f"raw/meta_categories/meta_{category}.jsonl", output_dir / "meta.parquet",
         convert_meta_jsonl, args.meta_batch_size),
    )

    print(f"\n=== Amazon {category} ===")
    print(f"Output directory: {output_dir}")
    for source, output, convert, batch_size in sources:
        if output.is_file() and not args.force and not args.copy_source:
            print(f"Existing parquet: {output}")
            continue
        downloaded = download_hf_file(args.repo_id, source, args.cache_dir, revision=args.revision)
        if args.copy_source:
            copy_source_file(downloaded, output_dir / Path(source).name)
        convert(downloaded, output, batch_size, force=args.force)

    print(f"Ready for prepare_dataset.py:")
    for _, output, _, _ in sources:
        print(f"  {output}")


def download_hf_file(repo_id: str, filename: str, cache_dir: str | None, *, revision: str | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("huggingface_hub is required. Install it in the recuq_benchmark conda environment.") from exc

    print(f"Downloading/checking HF file: {repo_id}/{filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        cache_dir=cache_dir,
        revision=revision,
    )
    return Path(path)


def copy_source_file(source: Path, target: Path) -> None:
    import shutil

    if target.exists() and target.stat().st_size == source.stat().st_size:
        print(f"Source copy exists: {target}")
        return
    print(f"Copying source to {target}")
    shutil.copy2(source, target)


def convert_reviews_csv(source: Path, output: Path, chunksize: int, force: bool) -> None:
    if output.exists() and not force:
        print(f"Reviews parquet exists, skipping: {output}")
        return

    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pandas and pyarrow are required to convert Amazon review CSV files.") from exc

    print(f"Converting reviews CSV -> parquet: {source} -> {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    header = infer_review_header(source, pd)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        source_columns = header or REVIEW_COLUMNS
        read_kwargs = {
            "chunksize": chunksize,
            "usecols": source_columns,
            "dtype": {column: str for column in source_columns[:2]},
        }
        if header is None:
            read_kwargs.update(header=None, names=source_columns)

        for chunk in pd.read_csv(source, **read_kwargs):
            chunk = normalize_review_chunk(chunk[source_columns])
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_output, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"  reviews rows: {total_rows:,}")
    finally:
        if writer is not None:
            writer.close()

    tmp_output.replace(output)
    print(f"Wrote reviews parquet: {output} ({total_rows:,} rows)")


def infer_review_header(source: Path, pd: Any) -> list[str] | None:
    sample = pd.read_csv(source, nrows=5)
    columns = list(sample.columns)
    if set(REVIEW_COLUMNS).issubset(columns):
        return REVIEW_COLUMNS

    aliases = {
        "user_id": ["user_id", "user", "uid"],
        "parent_asin": ["parent_asin", "item_id", "asin", "item"],
        "rating": ["rating", "ratings", "overall", "score"],
        "timestamp": ["timestamp", "time", "unix_time"],
    }
    mapped: list[str] = []
    for required, candidates in aliases.items():
        match = next((column for column in columns if str(column).lower() in candidates), None)
        if match is None:
            break
        mapped.append(match)
    if len(mapped) == len(REVIEW_COLUMNS):
        return mapped

    print("Could not find named review columns; assuming headerless CSV with user_id,parent_asin,rating,timestamp.")
    return None


def normalize_review_chunk(chunk: Any) -> Any:
    rename_map = {old: new for old, new in zip(chunk.columns, REVIEW_COLUMNS) if old != new}
    chunk = chunk.rename(columns=rename_map)
    chunk = chunk[REVIEW_COLUMNS].copy()
    chunk["user_id"] = chunk["user_id"].astype(str)
    chunk["parent_asin"] = chunk["parent_asin"].astype(str)
    chunk["rating"] = chunk["rating"].astype(float)
    chunk["timestamp"] = chunk["timestamp"].astype("int64")
    return chunk


def convert_meta_jsonl(source: Path, output: Path, batch_size: int, force: bool) -> None:
    if output.exists() and not force:
        print(f"Metadata parquet exists, skipping: {output}")
        return

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to convert Amazon metadata JSONL files.") from exc

    print(f"Converting metadata JSONL -> parquet: {source} -> {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    schema = pa.schema(
        [
            pa.field("parent_asin", pa.string()),
            pa.field("main_category", pa.string()),
            pa.field("title", pa.string()),
            pa.field("subtitle", pa.string()),
            pa.field("author", pa.string()),
            pa.field("store", pa.string()),
            pa.field("categories", pa.list_(pa.string())),
            pa.field("description", pa.string()),
            pa.field("average_rating", pa.string()),
            pa.field("rating_number", pa.string()),
            pa.field("price", pa.string()),
        ]
    )
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    batch: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                batch.append(normalize_meta_row(row))
                if len(batch) >= batch_size:
                    writer, written = write_meta_batch(batch, schema, tmp_output, writer, pa, pq)
                    total_rows += written
                    print(f"  metadata rows: {total_rows:,}")
                    batch.clear()
        if batch:
            writer, written = write_meta_batch(batch, schema, tmp_output, writer, pa, pq)
            total_rows += written
            print(f"  metadata rows: {total_rows:,}")
    finally:
        if writer is not None:
            writer.close()

    tmp_output.replace(output)
    print(f"Wrote metadata parquet: {output} ({total_rows:,} rows)")


def write_meta_batch(
    batch: list[dict[str, Any]],
    schema: Any,
    output: Path,
    writer: Any,
    pa: Any,
    pq: Any,
) -> tuple[Any, int]:
    columns = {field: [row[field] for row in batch] for field in META_COLUMNS}
    table = pa.Table.from_pydict(columns, schema=schema)
    if writer is None:
        writer = pq.ParquetWriter(output, schema, compression="zstd")
    writer.write_table(table)
    return writer, len(batch)


def normalize_meta_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "parent_asin": to_text(row.get("parent_asin")),
        "main_category": to_text(row.get("main_category")),
        "title": to_text(row.get("title")),
        "subtitle": to_text(row.get("subtitle")),
        "author": to_text(row.get("author")),
        "store": to_text(row.get("store")),
        "categories": to_categories(row.get("categories"), fallback=row.get("main_category")),
        "description": to_text(row.get("description")),
        "average_rating": to_text(row.get("average_rating")),
        "rating_number": to_text(row.get("rating_number")),
        "price": to_text(row.get("price")),
    }


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(to_text(item) for item in value if to_text(item))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def to_categories(value: Any, fallback: Any = None) -> list[str]:
    categories = list(iter_categories(value))
    if not categories and fallback is not None:
        categories = list(iter_categories(fallback))
    return categories


def iter_categories(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        for part in value.split(","):
            text = part.strip()
            if text:
                yield text
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, list):
                for nested in iter_categories(item):
                    yield nested
            else:
                text = to_text(item)
                if text:
                    yield text
        return
    text = to_text(value)
    if text:
        yield text


if __name__ == "__main__":
    main()
