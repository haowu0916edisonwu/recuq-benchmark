"""Small I/O helpers shared by release scripts.

The release stores record-level JSONL files as ``.jsonl.gz`` to
keep the repository compact.  Evaluation code should otherwise behave the
same for compressed and uncompressed inputs.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import IO


def open_text(path: str | Path, mode: str = "rt", *, encoding: str = "utf-8") -> IO[str]:
    """Open plain text or gzip-compressed text based on the file suffix."""

    resolved = Path(path)
    text_mode = mode if "t" in mode else f"{mode}t"
    if resolved.suffix == ".gz":
        return gzip.open(resolved, text_mode, encoding=encoding)
    return resolved.open(text_mode.replace("t", ""), encoding=encoding)


def resolve_jsonl(path: str | Path, filename: str = "uq_predictions.jsonl") -> Path:
    """Resolve a JSONL file or a directory containing plain/gzipped JSONL."""

    resolved = Path(path)
    if not resolved.is_dir():
        return resolved
    plain = resolved / filename
    compressed = resolved / f"{filename}.gz"
    if plain.exists():
        return plain
    if compressed.exists():
        return compressed
    raise FileNotFoundError(
        f"Directory contains neither {filename} nor {filename}.gz: {resolved}"
    )
