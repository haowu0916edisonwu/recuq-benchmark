#!/usr/bin/env python3
"""Draw the data-derived result figures from released evaluation CSVs.

The script performs no model inference. It creates vector and high-resolution
versions of the backbone/condition reliability figure and the scoped
paired-control response figure. The manuscript pipeline diagram is a static
explanatory asset; an optional code-native schematic can also be generated.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = ROOT / "outputs" / "figures"

# Conventional scientific palette with a restrained neutral baseline.
COLORS = {
    "ink": "#1A1A1A",
    "gray": "#9B9FA4",
    "light_gray": "#F5F5F5",
    "red": "#B5523E",         # Top10Mass
    "blue": "#4F82B5",        # Decomposed UQ
    "green": "#2F9461",       # Perturbation
    "purple": "#7357AD",      # Temperature scaling
    "peach": "#E9C3A4",       # Sampling
    "lp_max": "#8FB8DF",      # LabelProb-Max
    "lp_entropy": "#B9D8CB",  # LabelProb-Entropy
    "lp_margin": "#D9BE8A",   # LabelProb-Margin
    "sky": "#7FA8D6",
    "sand": "#F6EBE8",
}

METHOD_ORDER = [
    "derived_labelprob_base_top10_mass",
    "kweon_decomp_llama31_8b",
    "labelprob_temp_scaled",
    "perturbation_rank_instability",
    "self_consistency_s5",
    "labelprob_llama31_8b",
    "labelprob_entropy",
    "labelprob_margin",
]

METHOD_LABELS = {
    "derived_labelprob_base_top10_mass": "Top10Mass",
    "kweon_decomp_llama31_8b": "Decomp-UQ",
    "labelprob_temp_scaled": "Temp-Scaled",
    "perturbation_rank_instability": "Perturbation",
    "self_consistency_s5": "Self-Consistency",
    "labelprob_llama31_8b": "LP-Max",
    "labelprob_entropy": "LP-Entropy",
    "labelprob_margin": "LP-Margin",
}

QWEN_METHODS = {
    "kweon_decomp_llama31_8b": "kweon_decomp_qwen3_8b_nonthinking",
    "self_consistency_s5": "self_consistency_qwen3_s5",
    "labelprob_llama31_8b": "labelprob_qwen3_8b_nonthinking",
}

REPRESENTATIVE = [
    "derived_labelprob_base_top10_mass",
    "kweon_decomp_llama31_8b",
    "perturbation_rank_instability",
]

VIEW_ORDER = ["s0_warmstart", "s3_sparsity", "s4_longtail", "s5_robustness"]
VIEW_LABELS = ["Warm-start", "Sparse-history", "Tail-target subset", "Item-text"]
VIEW_COUNTS = ["n=4,000", "n=20,000", "n=1,312", "n=17,163"]

METHOD_COLORS = {
    "derived_labelprob_base_top10_mass": COLORS["red"],
    "kweon_decomp_llama31_8b": COLORS["blue"],
    "labelprob_temp_scaled": COLORS["purple"],
    "perturbation_rank_instability": COLORS["green"],
    "self_consistency_s5": COLORS["peach"],
    "labelprob_llama31_8b": COLORS["lp_max"],
    "labelprob_entropy": COLORS["lp_entropy"],
    "labelprob_margin": COLORS["lp_margin"],
}


def method_color(method: str) -> str:
    return METHOD_COLORS.get(method, COLORS["gray"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def save_all(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.045}
        if suffix == "png":
            kwargs["dpi"] = 450
        fig.savefig(output_dir / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.7,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def add_box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    detail: str,
    color: str,
    *,
    dashed: bool = False,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.0,
        edgecolor=color,
        facecolor="#FFFFFF",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.66, title, ha="center", va="center", fontweight="bold", color=COLORS["ink"], fontsize=8.4)
    ax.text(xy[0] + width / 2, xy[1] + height * 0.34, detail, ha="center", va="center", color="#454B52", fontsize=6.7, linespacing=1.25)


def add_arrow(ax: mpl.axes.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.05, color="#4C535A"))


def draw_protocol(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.25, 2.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Context strip: it makes the Llama/Qwen and true-tail scope unambiguous.
    context = FancyBboxPatch((0.035, 0.81), 0.93, 0.14, boxstyle="round,pad=0.012,rounding_size=0.02", linewidth=0.8, edgecolor="#AEB7C2", facecolor="#F6F8FA")
    ax.add_patch(context)
    ax.text(0.055, 0.90, "Fixed-ranking records", fontweight="bold", fontsize=7.4, va="center")
    ax.text(0.245, 0.90, "Llama controlled suite:", fontsize=7.1, va="center")
    ax.text(0.405, 0.90, "Warm-start", color=COLORS["blue"], fontsize=7.1, va="center")
    ax.text(0.502, 0.90, "→ tail filter", color=COLORS["red"], fontsize=7.1, va="center")
    ax.text(0.615, 0.90, "+ Sparse-history + Item-text", fontsize=7.1, va="center")
    ax.text(0.245, 0.845, "Qwen:", fontsize=7.1, va="center")
    ax.text(0.293, 0.845, "Warm-start replication only (same 4,000 instances)", color="#454B52", fontsize=7.1, va="center")

    add_box(ax, (0.045, 0.33), 0.205, 0.32, "Base ranking artifact", "history + 20 candidates\nparsed top-20 ranking", COLORS["blue"])
    add_box(ax, (0.395, 0.33), 0.215, 0.32, "Monitor-only UQ", "probability · calibration\nvariation · perturbation · sampling", COLORS["purple"])
    add_box(ax, (0.745, 0.51), 0.20, 0.23, "Scalar reliability", "detect Error@K\nAUROC · AUPRC · risk", COLORS["red"])
    add_box(ax, (0.745, 0.18), 0.20, 0.23, "Paired-control response", "user-history vs item-text\nscoped response agreement", COLORS["green"], dashed=True)
    add_arrow(ax, (0.255, 0.49), (0.386, 0.49))
    add_arrow(ax, (0.615, 0.51), (0.735, 0.62))
    add_arrow(ax, (0.615, 0.47), (0.735, 0.29))
    ax.text(0.502, 0.265, "Reads the issued ranking; never reranks it", ha="center", va="center", fontsize=6.8, color="#454B52", style="italic")
    ax.text(0.845, 0.105, "not general causal attribution", ha="center", va="center", fontsize=6.1, color="#666D75")
    save_all(fig, output_dir, "fig_benchmark_protocol")


def scores_by_method(rows: list[dict[str, str]]) -> dict[str, float]:
    return {row["method"]: float(row["error_auroc"]) for row in rows if row.get("auroc_defined", "True") == "True"}


def bootstrap_lookup(directory: Path) -> dict[tuple[str, str], tuple[float, float]]:
    rows = read_csv(directory / "bootstrap_metrics_ci.csv")
    return {
        (row["scenario"], row["method"]): (float(row["error_auroc_ci_low"]), float(row["error_auroc_ci_high"]))
        for row in rows
        if row.get("metric", "error_auroc") == "error_auroc"
    }


def draw_reliability(args: argparse.Namespace) -> None:
    llama_dir = args.llama_eval_dir
    qwen_dir = args.qwen_eval_dir
    llama = scores_by_method(read_csv(llama_dir / "table11_error_auroc_by_method_scenario.csv"))
    qwen = scores_by_method(read_csv(qwen_dir / "table11_error_auroc_by_method.csv"))

    scenario_rows = read_csv(llama_dir / "table11_error_auroc_by_method_scenario.csv")
    by_view = {(row["scenario"], row["method"]): float(row["error_auroc"]) for row in scenario_rows}
    ci: dict[tuple[str, str], tuple[float, float]] = {}
    for scenario, directory in args.cluster_bootstrap_dir.items():
        ci.update(bootstrap_lookup(directory))

    fig, (ax_backbone, ax_view) = plt.subplots(1, 2, figsize=(7.25, 2.78), gridspec_kw={"width_ratios": [1.08, 1.30]})

    # Panel A: paired horizontal dot plot, so changing method ordering is legible.
    ordered = list(reversed(METHOD_ORDER))
    y = np.arange(len(ordered))
    ax_backbone.axhline(2.5, color="#D7E0E7", linewidth=0.75, zorder=-1)
    for i, method in enumerate(ordered):
        qwen_method = QWEN_METHODS.get(method, method)
        color = method_color(method)
        ax_backbone.plot([llama[method], qwen[qwen_method]], [i, i], color="#D6D3D1", linewidth=1.0, zorder=1)
        ax_backbone.scatter(llama[method], i, s=28, color=color, edgecolor="white", linewidth=0.45, zorder=3, marker="o")
        ax_backbone.scatter(qwen[qwen_method], i, s=28, color=color, edgecolor="white", linewidth=0.45, zorder=3, marker="D")
    ax_backbone.axvline(0.5, linestyle="--", linewidth=0.8, color="#9A9693", zorder=0)
    ax_backbone.set_xlim(0.35, 0.67)
    ax_backbone.set_yticks(y)
    ax_backbone.set_yticklabels([METHOD_LABELS[m] for m in ordered], fontsize=6.9)
    ax_backbone.set_xlabel("Warm-start Error-AUROC   (● Llama; ◆ Qwen)")
    ax_backbone.set_title("Warm-Start Method Ordering by Backbone", loc="center", fontweight="bold", pad=5)
    ax_backbone.grid(axis="x", color="#E8DEDA", linewidth=0.6)
    ax_backbone.tick_params(axis="y", length=0)

    # Panel B: three representative signals across views, with clustered CIs on controls.
    x = np.arange(len(VIEW_ORDER))
    line_styles = {
        "derived_labelprob_base_top10_mass": (COLORS["red"], "o"),
        "kweon_decomp_llama31_8b": (COLORS["blue"], "s"),
        "perturbation_rank_instability": (COLORS["green"], "^"),
    }
    for method in REPRESENTATIVE:
        color, marker = line_styles[method]
        values = [by_view[(scenario, method)] for scenario in VIEW_ORDER]
        ax_view.plot(x, values, color=color, marker=marker, markersize=4.5, linewidth=1.45, label=METHOD_LABELS[method], zorder=3)
        for j, scenario in enumerate(VIEW_ORDER[1:], start=1):
            lower, upper = ci[(scenario, method)]
            ax_view.vlines(j, lower, upper, color=color, linewidth=1.0, alpha=0.9, zorder=2)
            ax_view.hlines([lower, upper], j - 0.045, j + 0.045, color=color, linewidth=1.0, zorder=2)
    ax_view.axhline(0.5, linestyle="--", linewidth=0.8, color="#9A9693", zorder=0)
    ax_view.set_xticks(x)
    ax_view.set_xticklabels([f"{label}\n{count}" for label, count in zip(VIEW_LABELS, VIEW_COUNTS)])
    ax_view.set_ylim(0.47, 0.68)
    ax_view.set_ylabel("Overall Error-AUROC")
    ax_view.set_title("Error@10 Detection by Condition (Llama)", loc="center", fontweight="bold", pad=5)
    ax_view.grid(axis="y", color="#E8DEDA", linewidth=0.6)
    ax_view.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.50, 1.02), handlelength=1.5, columnspacing=0.9)

    fig.subplots_adjust(left=0.170, right=0.99, top=0.86, bottom=0.23, wspace=0.34)
    save_all(fig, args.output_dir, "fig_backbone_view_reliability")


def draw_paired_response(args: argparse.Namespace) -> None:
    rows = read_csv(args.source_eval_dir / "control_response_metrics.csv")
    source = {row["method"]: row for row in rows}
    ordered = list(reversed(METHOD_ORDER))
    y = np.arange(len(ordered))
    def directional_coverage(row: dict[str, str]) -> float:
        return float(row["directional_coverage"])

    coverage = np.array([directional_coverage(source[m]) for m in ordered])
    auroc = np.array([float(source[m]["source_auroc"]) for m in ordered])

    palette = [method_color(method) for method in ordered]
    fig, (ax_cov, ax_auc) = plt.subplots(1, 2, figsize=(6.95, 2.58), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax_cov.barh(y, coverage, color=palette, height=0.63, edgecolor="none")
    ax_auc.barh(y, auroc, color=palette, height=0.63, edgecolor="none")
    for axis in (ax_cov, ax_auc):
        axis.axhline(2.5, color="#D7E0E7", linewidth=0.75, zorder=-1)
    ax_auc.axvline(0.5, color="#9A9693", linestyle="--", linewidth=0.8)
    ax_cov.set_xlim(0.40, 1.07)
    ax_auc.set_xlim(0.48, 0.735)
    ax_cov.set_yticks(y)
    ax_cov.set_yticklabels([METHOD_LABELS[m] for m in ordered])
    ax_cov.set_xlabel("Directional coverage")
    ax_auc.set_xlabel("Source-AUROC")
    ax_cov.set_title("Directional Coverage", loc="center", fontweight="bold", pad=5)
    ax_auc.set_title("Paired-Control Agreement", loc="center", fontweight="bold", pad=5)
    for axis, values in ((ax_cov, coverage), (ax_auc, auroc)):
        axis.grid(axis="x", color="#E8DEDA", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        for i, value in enumerate(values):
            axis.text(value + 0.006, i, f"{value:.3f}", va="center", ha="left", fontsize=6.4, color="#41474E", clip_on=False)
    ax_auc.text(0.98, 0.04, "Dashed line: chance", transform=ax_auc.transAxes, ha="right", va="bottom", fontsize=6.2, color="#555D66")
    # Centre the scope note under the two panels, rather than under the full canvas
    # (whose extra left margin is reserved for method labels).
    fig.text(0.61, 0.012, "Scoped to the constructed user/item controls; no causal attribution", ha="center", va="bottom", fontsize=6.2, color="#555D66")
    fig.subplots_adjust(left=0.245, right=0.985, top=0.84, bottom=0.20, wspace=0.24)
    save_all(fig, args.output_dir, "fig_paired_control_response")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama_eval_dir", type=Path, default=ROOT / "artifacts/reference_results/llama/main")
    parser.add_argument("--qwen_eval_dir", type=Path, default=ROOT / "artifacts/reference_results/qwen/warm_start")
    parser.add_argument("--source_eval_dir", type=Path, default=ROOT / "artifacts/reference_results/paired_control/gap_0.00")
    parser.add_argument("--output_dir", type=Path, default=PAPER_FIGURES)
    parser.add_argument(
        "--cluster_bootstrap_dir",
        type=lambda value: Path(value),
        nargs=3,
        metavar=("SPARSE", "LONGTAIL", "ROBUST"),
        default=[
            ROOT / "artifacts/reference_results/diagnostics/cluster_bootstrap/sparse_history",
            ROOT / "artifacts/reference_results/diagnostics/cluster_bootstrap/tail_target_subset",
            ROOT / "artifacts/reference_results/diagnostics/cluster_bootstrap/item_text_robustness",
        ],
    )
    parser.add_argument("--include_protocol_schematic", action="store_true")
    args = parser.parse_args()
    args.cluster_bootstrap_dir = dict(zip(VIEW_ORDER[1:], args.cluster_bootstrap_dir))
    return args


def main() -> None:
    args = parse_args()
    setup_style()
    if args.include_protocol_schematic:
        draw_protocol(args.output_dir)
    draw_reliability(args)
    draw_paired_response(args)
    print(f"Wrote main-paper figures to {args.output_dir}")


if __name__ == "__main__":
    main()
