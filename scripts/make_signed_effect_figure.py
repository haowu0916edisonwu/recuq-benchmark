#!/usr/bin/env python
"""Appendix figure: signed restoration effects before clipping.

Two side-by-side histograms (user-history, item-content) in the style of the main
paper figures (setup_style and palette from make_main_paper_figures.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_paired_control_extended as ext  # noqa: E402
import evaluate_source_response_interventions as esr  # noqa: E402
from make_main_paper_figures import COLORS, setup_style  # noqa: E402

BIN_WIDTH = 0.05
GRID = "#E8DEDA"
ZERO_LINE = "#5F646A"
NOTE_INK = "#41474E"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base_predictions",
        type=Path,
        default=ROOT / "artifacts/base_predictions/llama_fixed_rankings.jsonl.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/figures/fig_signed_restoration_effects",
        help="Output path without suffix; .pdf and .png are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_style()
    cases = ext.build_signed_cases(esr._group_base_instances(esr._read_jsonl(args.base_predictions)))
    panels = [
        ("User-history effect", np.array([c["du"] for c in cases])),
        ("Item-content effect", np.array([c["dv"] for c in cases])),
    ]
    # Bins centred on zero so the exact-zero mass sits on the reference line.
    edges = np.arange(-1.0 - BIN_WIDTH / 2, 1.0 + BIN_WIDTH, BIN_WIDTH)

    fig, axes = plt.subplots(1, 2, figsize=(6.95, 2.75), sharey=True)
    for ax, (title, values) in zip(axes, panels):
        ax.hist(values, bins=edges, color=COLORS["blue"], edgecolor="white", linewidth=0.5, zorder=2)
        ax.axvline(0.0, color=ZERO_LINE, linestyle="--", linewidth=1.1, zorder=3)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.set_title(title, loc="center", fontweight="bold", pad=5)
        ax.set_xlabel("Signed NDCG@10 difference")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        neg, zero, pos = (values < 0).mean(), (values == 0).mean(), (values > 0).mean()
        ax.text(
            0.03,
            0.96,
            f"neg  {neg:.1%}\nzero {zero:.1%}\npos  {pos:.1%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            family="DejaVu Sans Mono",
            color=NOTE_INK,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D7D2CF", "linewidth": 0.6},
            zorder=4,
        )
    axes[0].set_ylabel("Instances")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.17, wspace=0.08)

    # Keep the crop box of the submitted figure: compute the tight box with the
    # original x-label, then save with the current label inside that fixed box,
    # so that the relabeling does not change the canvas size or any coordinates.
    current = [ax.get_xlabel() for ax in axes]
    for ax in axes:
        ax.set_xlabel(r"Restoration effect ($\Delta$NDCG@10)")
    from matplotlib.backend_bases import _get_renderer
    from matplotlib.backends.backend_pdf import FigureCanvasPdf

    # Same procedure savefig(..., bbox_inches="tight", pad_inches=0.02) uses for PDF output.
    crop = fig.get_tightbbox(_get_renderer(fig, FigureCanvasPdf(fig).print_pdf)).padded(0.02)
    for ax, label in zip(axes, current):
        ax.set_xlabel(label)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches=crop)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches=crop)
    plt.close(fig)
    print(f"Wrote {args.output.with_suffix('.pdf')} and .png")


if __name__ == "__main__":
    main()
