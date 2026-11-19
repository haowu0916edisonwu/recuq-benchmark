#!/usr/bin/env python3
"""Generate the deterministic recommendation-UQ case-study figure.

The figure is read from existing artifacts only. It does not run a model or
recompute benchmark metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.utils.io import open_text

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing plotting dependencies. Install with: pip install matplotlib") from exc


COLORS = {
    "bg": "#FAFAF7",
    "mint": "#DFF3E6",
    "mint_edge": "#9BCAB0",
    "peach": "#FCE3C7",
    "peach_edge": "#E5B079",
    "lavender": "#EADDF7",
    "lavender_edge": "#B99BDC",
    "green": "#2F9461",
    "green_dark": "#267A51",
    "red": "#C94C3C",
    "red_dark": "#A83E31",
    "blue": "#8FB8DF",
    "purple": "#7357AD",
    "text": "#1F2933",
    "muted": "#5B6776",
    "line": "#6B7280",
    "chip": "#F7FAFC",
    "chip_edge": "#9AA6B2",
}

LETTERS = [chr(ord("A") + i) for i in range(20)]
DEFAULT_ORIGINAL_ID = "amazon_books::s5_robustness::original::test_4d8cff2a"
DEFAULT_ABLATED_ID = "amazon_books::s5_robustness::ablate_categories::test_4d8cff2a"
DEFAULT_METHOD = "perturbation_rank_instability"


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open_text(path, "r") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[str(row["record_id"])] = row
    return out


def target_letters(record: dict[str, Any]) -> list[str]:
    return [LETTERS[i] for i, v in enumerate(record["labels"]) if int(v) == 1]


def ranking_letters(record: dict[str, Any], k: int = 10) -> list[str]:
    return [LETTERS[int(i)] for i in record["ranking"][:k]]


def load_source_detail(path: Path, dataset: str, instance_id: str, method: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["method"] == method and row["dataset"] == dataset and row["instance_id"] == instance_id:
                return row
    raise KeyError(f"No source detail for {method} {dataset}/{instance_id} in {path}")


def rounded(ax, xy, width, height, fc, ec, lw=0.9, radius=0.018, z=1):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def draw_chip(ax, x, y, text, size=0.030, fc=None, ec=None, color=None, lw=0.8, fs=8.0, weight="semibold"):
    rounded(ax, (x, y), size, size, fc or COLORS["chip"], ec or COLORS["chip_edge"], lw=lw, radius=0.006, z=3)
    ax.text(x + size / 2, y + size / 2, text, ha="center", va="center", fontsize=fs, color=color or COLORS["text"], fontweight=weight, zorder=4)


def draw_chip_row(ax, x, y, letters, highlight=None, highlight_color=COLORS["green"], cols=10, size=0.030, gap=0.010):
    for i, letter in enumerate(letters):
        row = i // cols
        col = i % cols
        cx = x + col * (size + gap)
        cy = y - row * (size + gap)
        if letter == highlight:
            draw_chip(ax, cx, cy, letter, size=size, fc="#FFFFFF", ec=highlight_color, color=highlight_color, lw=1.8, fs=8.0, weight="heavy")
        else:
            draw_chip(ax, cx, cy, letter, size=size, fs=8.0)


def draw_ranking(ax, x, y, letters, highlight=None, highlight_color=COLORS["green"], size=0.033, gap=0.010):
    for i, letter in enumerate(letters):
        cx = x + i * (size + gap)
        if letter == highlight:
            draw_chip(ax, cx, y, letter, size=size, fc=highlight_color, ec=highlight_color, color="#FFFFFF", lw=0.8, fs=8.5, weight="heavy")
        else:
            draw_chip(ax, cx, y, letter, size=size, fs=8.4)


def draw_callout(ax, x, y, w, h, lines, fc, ec):
    rounded(ax, (x, y), w, h, fc, ec, lw=0.9, radius=0.010, z=5)
    line_h = h / (len(lines) + 0.5)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - (i + 1) * line_h, line, ha="center", va="center", fontsize=7.1, color="#FFFFFF" if fc in [COLORS["green"], COLORS["red"]] else COLORS["text"], fontweight="bold" if i == 0 else "medium", zorder=6)


def draw_bar(ax, x, y, w, h, value, max_value, label, color, label_inside=False):
    rounded(ax, (x, y), w, h, "#F7F3FB", "#CCB7E2", lw=0.7, radius=0.006, z=2)
    fill_w = max(0.012, w * min(value / max_value, 1.0))
    rounded(ax, (x, y), fill_w, h, color, color, lw=0.0, radius=0.006, z=3)
    if label_inside:
        ax.text(x + 0.008, y + h / 2, label, ha="left", va="center", fontsize=7.4, color="#FFFFFF", fontweight="semibold", zorder=4)
    else:
        ax.text(x - 0.010, y + h / 2, label, ha="right", va="center", fontsize=8.0, color=COLORS["text"], zorder=4)


def plot(args: argparse.Namespace) -> None:
    base = read_jsonl_index(args.base_predictions)
    top10 = read_jsonl_index(args.top10mass_predictions)
    perturb = read_jsonl_index(args.perturbation_predictions)

    original = base[args.original_record_id]
    ablated = base[args.ablated_record_id]
    top10_original = top10[args.original_record_id]
    top10_ablated = top10[args.ablated_record_id]
    perturb_ablated = perturb[args.ablated_record_id]
    detail = load_source_detail(args.source_details, ablated["dataset"], ablated["instance_id"], args.source_method)

    target = target_letters(ablated)[0]
    original_top10 = ranking_letters(original)
    ablated_top10 = ranking_letters(ablated)
    original_ndcg = float(original["metrics"]["ndcg@10"])
    ablated_ndcg = float(ablated["metrics"]["ndcg@10"])
    original_u = float(top10_original["uncertainty_score"])
    ablated_u = float(top10_ablated["uncertainty_score"])
    user_score = float(perturb_ablated["source_scores"]["user"])
    item_score = float(perturb_ablated["source_scores"]["item"])
    user_gain = float(detail["user_effect"])
    item_gain = float(detail["item_effect"])

    setup_style()
    fig, ax = plt.subplots(figsize=(7.25, 4.08))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "Case Study: Item-Content Ablation Causes Error@10", ha="center", va="top", fontsize=15.2, fontweight="heavy", color=COLORS["text"])
    ax.text(0.5, 0.915, "Amazon Books record; candidate labels are anonymized.", ha="center", va="top", fontsize=9.4, color=COLORS["text"])

    left = (0.045, 0.455, 0.405, 0.390)
    right = (0.550, 0.455, 0.405, 0.390)
    rounded(ax, (left[0], left[1]), left[2], left[3], COLORS["mint"], COLORS["mint_edge"], lw=0.9, radius=0.014)
    rounded(ax, (right[0], right[1]), right[2], right[3], COLORS["peach"], COLORS["peach_edge"], lw=0.9, radius=0.014)

    # Panel headers and tags.
    ax.text(left[0] + 0.018, left[1] + left[3] - 0.035, "Restored item content", ha="left", va="center", fontsize=10.8, fontweight="heavy", color=COLORS["text"])
    rounded(ax, (left[0] + left[2] - 0.150, left[1] + left[3] - 0.047), 0.132, 0.028, "#EDF8F1", COLORS["mint_edge"], lw=0.6, radius=0.006)
    ax.text(left[0] + left[2] - 0.084, left[1] + left[3] - 0.033, "s5_robustness / original", ha="center", va="center", fontsize=5.8, color=COLORS["text"])

    ax.text(right[0] + 0.018, right[1] + right[3] - 0.035, "Ablated item content", ha="left", va="center", fontsize=10.8, fontweight="heavy", color=COLORS["text"])
    rounded(ax, (right[0] + right[2] - 0.178, right[1] + right[3] - 0.047), 0.160, 0.028, "#FFF0DD", COLORS["peach_edge"], lw=0.6, radius=0.006)
    ax.text(right[0] + right[2] - 0.098, right[1] + right[3] - 0.033, "s5_robustness / ablate_categories", ha="center", va="center", fontsize=5.8, color=COLORS["text"])

    # Candidate chips.
    ax.text(left[0] + 0.018, left[1] + 0.300, "Candidate chips", ha="left", va="center", fontsize=8.6, color=COLORS["text"])
    draw_chip_row(ax, left[0] + 0.020, left[1] + 0.252, LETTERS, highlight=target, highlight_color=COLORS["green"], cols=10)
    ax.text(right[0] + 0.018, right[1] + 0.300, "Candidate chips", ha="left", va="center", fontsize=8.6, color=COLORS["text"])
    draw_chip_row(ax, right[0] + 0.020, right[1] + 0.252, LETTERS, highlight=target, highlight_color=COLORS["red"], cols=10)

    # Rankings.
    ax.text(left[0] + 0.018, left[1] + 0.125, "Frozen LLM top-10 ranking", ha="left", va="center", fontsize=8.8, color=COLORS["text"])
    draw_ranking(ax, left[0] + 0.020, left[1] + 0.065, original_top10, highlight=target, highlight_color=COLORS["green"])
    ax.text(right[0] + 0.018, right[1] + 0.125, "Frozen LLM top-10 ranking", ha="left", va="center", fontsize=8.8, color=COLORS["text"])
    draw_ranking(ax, right[0] + 0.020, right[1] + 0.065, ablated_top10, highlight=target, highlight_color=COLORS["red"])

    # Callouts avoid chip overlap by sitting below candidate rows and above ranking rows.
    draw_callout(
        ax,
        left[0] + 0.245,
        left[1] + 0.135,
        0.150,
        0.100,
        [
            f"Target {target} in top-10",
            f"NDCG@10 = {original_ndcg:.3f}",
            "is_error_at_10 = 0",
            f"Top10Mass uncertainty = {original_u:.3f}",
        ],
        COLORS["green"],
        COLORS["green_dark"],
    )
    draw_callout(
        ax,
        right[0] + 0.247,
        right[1] + 0.135,
        0.148,
        0.100,
        [
            f"Target {target} outside top-10",
            f"NDCG@10 = {ablated_ndcg:.3f}",
            "is_error_at_10 = 1",
            f"Top10Mass uncertainty = {ablated_u:.3f}",
        ],
        COLORS["red"],
        COLORS["red_dark"],
    )

    arrow = FancyArrowPatch((0.470, 0.640), (0.535, 0.640), arrowstyle="Simple,head_width=12,head_length=14,tail_width=2.0", color=COLORS["line"], linewidth=0, mutation_scale=1, zorder=2)
    ax.add_patch(arrow)
    ax.text(0.503, 0.685, "ablate item\ncategory fields", ha="center", va="center", fontsize=7.2, color=COLORS["text"])

    # Bottom diagnostic panel.
    bx, by, bw, bh = 0.045, 0.130, 0.910, 0.250
    rounded(ax, (bx, by), bw, bh, COLORS["lavender"], COLORS["lavender_edge"], lw=0.9, radius=0.014)
    ax.text(0.5, by + bh - 0.035, "Paired-control response", ha="center", va="center", fontsize=10.8, fontweight="heavy", color=COLORS["text"])

    ax.text(bx + 0.140, by + bh - 0.080, "Perturbation control scores", ha="center", va="center", fontsize=9.4, color=COLORS["text"])
    ax.text(bx + 0.590, by + bh - 0.080, "Restoration gains", ha="center", va="center", fontsize=9.4, color=COLORS["text"])

    max_source = 0.85
    draw_bar(ax, bx + 0.125, by + 0.130, 0.285, 0.032, user_score, max_source, f"User-history  score = {user_score:.3f}", COLORS["blue"], label_inside=True)
    draw_bar(ax, bx + 0.125, by + 0.075, 0.285, 0.032, item_score, max_source, f"Item-content  score = {item_score:.3f}", COLORS["green"], label_inside=True)

    max_gain = 0.56
    draw_bar(ax, bx + 0.575, by + 0.130, 0.285, 0.032, user_gain, max_gain, f"Restore user-history: ΔNDCG@10 = {user_gain:.3f}", COLORS["blue"], label_inside=True)
    draw_bar(ax, bx + 0.575, by + 0.075, 0.285, 0.032, item_gain, max_gain, f"Restore item-content: ΔNDCG@10 = {item_gain:.3f}", COLORS["purple"], label_inside=True)

    ax.text(0.045, 0.060, "Observation: scalar uncertainty flags Error@10; control scores favor item-content restoration.", ha="left", va="center", fontsize=8.3, color=COLORS["text"], fontweight="semibold")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        out = args.output_dir / f"fig_case_study_item_ablation.{fmt}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt == "png":
            kwargs["dpi"] = 450
        fig.savefig(out, **kwargs)
    plt.close(fig)
    print(f"Wrote case-study figure to {args.output_dir}")


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_predictions", type=Path, default=Path("artifacts/base_predictions/llama_fixed_rankings.jsonl.gz"))
    parser.add_argument("--top10mass_predictions", type=Path, default=Path("artifacts/uq_outputs/llama/labelprob_top10mass.jsonl.gz"))
    parser.add_argument("--perturbation_predictions", type=Path, default=Path("artifacts/uq_outputs/llama/perturbation_rank_instability.jsonl.gz"))
    parser.add_argument("--source_details", type=Path, default=Path("artifacts/reference_results/paired_control/gap_0.00/control_response_instance_details.csv"))
    parser.add_argument("--source_method", default=DEFAULT_METHOD)
    parser.add_argument("--original_record_id", default=DEFAULT_ORIGINAL_ID)
    parser.add_argument("--ablated_record_id", default=DEFAULT_ABLATED_ID)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--formats", nargs="+", default=["svg", "pdf", "png"], choices=["svg", "pdf", "png"])
    return parser.parse_args()


def main() -> None:
    plot(parse_args())


if __name__ == "__main__":
    main()
