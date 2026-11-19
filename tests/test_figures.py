from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_case_study_exports_all_documented_formats(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/make_case_study_figure.py",
            "--output_dir",
            str(tmp_path),
            "--formats",
            "pdf",
            "svg",
            "png",
        ],
        cwd=ROOT,
        check=True,
    )
    for suffix in ("pdf", "svg", "png"):
        output = tmp_path / f"fig_case_study_item_ablation.{suffix}"
        assert output.is_file()
        assert output.stat().st_size > 0
