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


def test_reliability_left_panel_matches_table_3(monkeypatch) -> None:
    import csv
    from collections import defaultdict

    import matplotlib.pyplot as plt
    import pytest

    from scripts import make_main_paper_figures as figures

    table = ROOT / "artifacts/reference_results/paper_tables/table3_warm_start_auroc.csv"
    with table.open(newline="") as handle:
        overall = {row["backbone"]: row for row in csv.DictReader(handle) if row["dataset"] == "overall"}

    columns = {
        "Top10Mass": "labelprob_top10mass",
        "Decomp-UQ": "decomposed_llm_uq",
        "Temp-Scaled": "temperature_scaled_labelprob",
        "Perturbation": "perturbation_rank_instability",
        "Self-Consistency": "self_consistency",
        "LP-Max": "labelprob_max",
        "LP-Entropy": "labelprob_entropy",
        "LP-Margin": "labelprob_margin",
    }

    def check_points(fig, output_dir, stem) -> None:
        try:
            left = fig.axes[0]
            labels = [tick.get_text() for tick in left.get_yticklabels()]
            plotted = defaultdict(list)
            for collection in left.collections:
                (x, y), = collection.get_offsets()
                plotted[labels[int(y)]].append(float(x))
            assert set(plotted) == set(columns)
            for method, column in columns.items():
                expected = [float(overall[backbone][column]) for backbone in (
                    "Llama-3.1-8B-Instruct", "Qwen3-8B non-thinking",
                )]
                assert plotted[method] == pytest.approx(expected, abs=1e-12, rel=0), method
        finally:
            plt.close(fig)

    monkeypatch.setattr(sys, "argv", ["make_main_paper_figures.py"])
    monkeypatch.setattr(figures, "save_all", check_points)
    figures.draw_reliability(figures.parse_args())
