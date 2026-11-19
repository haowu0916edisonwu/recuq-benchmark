#!/usr/bin/env python
"""Evaluate clipped and signed paired-control agreement and ranking-distance baselines.

Extends evaluate_source_response_interventions.py (whose metric functions are reused
directly) along three axes:

* Label definitions: clipped (original), signed, and magnitude restoration effects.
* Signals: original scalar responses (clipped), signed scalar responses, Perturbation
  Rank Instability, non-UQ listwise distances (top-k Jaccard, Kendall, footrule, RBO),
  a LabelProb distribution-shift control (Jensen-Shannon), and a majority-class baseline.
* Reporting: dataset-level results, dataset-stratified bootstrap CIs, the distribution of
  signed NDCG@10 differences.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_source_response_interventions as esr  # noqa: E402  (official metric definitions)
from recuq_benchmark.utils.io import open_text
from recuq_benchmark.utils.logging import get_logger  # noqa: E402

LABEL_MODES = ("clipped", "signed", "magnitude")
# Averages of mathematically equal values can differ in the last bit; round before comparing so that such
# cases are treated as ties, as the official evaluator does for Perturbation Rank Instability.
TIE_DECIMALS = 12
DATASETS = ("mind_20", "amazon_books", "amazon_movies", "yelp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backbone", type=str, required=True, help="Label written into outputs, e.g. llama31_8b.")
    parser.add_argument("--base_predictions", type=str, required=True)
    parser.add_argument(
        "--uq_predictions",
        type=str,
        nargs="*",
        default=[],
        help="Scalar UQ files/directories covering the controlled variants (omit for ranking-only backbones).",
    )
    parser.add_argument(
        "--labelprob_predictions",
        type=str,
        default=None,
        help="LabelProb uq_predictions with option_probabilities, for the distribution-shift control.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Listwise distances between a restored ranking a and a perturbed ranking b
# ---------------------------------------------------------------------------


def _jaccard(k: int) -> Callable[[list[int], list[int]], float]:
    def dist(a: list[int], b: list[int]) -> float:
        sa, sb = set(a[:k]), set(b[:k])
        return 1.0 - len(sa & sb) / len(sa | sb)

    return dist


def _positions(ranking: list[int]) -> np.ndarray:
    pos = np.empty(len(ranking), dtype=float)
    pos[np.asarray(ranking)] = np.arange(len(ranking))
    return pos


def _kendall(a: list[int], b: list[int]) -> float:
    pa, pb = _positions(a), _positions(b)
    n = len(pa)
    discordant = 0
    for i in range(n):
        discordant += int(np.sum((pa[i] - pa[i + 1 :]) * (pb[i] - pb[i + 1 :]) < 0))
    return discordant / (n * (n - 1) / 2)


def _footrule(a: list[int], b: list[int]) -> float:
    n = len(a)
    return float(np.abs(_positions(a) - _positions(b)).sum()) / (n * n // 2)


def _rbo(a: list[int], b: list[int], p: float = 0.9) -> float:
    depth = len(a)
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    total = 0.0
    for d in range(1, depth + 1):
        seen_a.add(a[d - 1])
        seen_b.add(b[d - 1])
        total += p ** (d - 1) * len(seen_a & seen_b) / d
    return 1.0 - (1 - p) * total / (1 - p**depth)


DISTANCES: dict[str, Callable[[list[int], list[int]], float]] = {
    "Jaccard@1": _jaccard(1),
    "Jaccard@5": _jaccard(5),
    "Jaccard@10": _jaccard(10),
    "Jaccard@15": _jaccard(15),
    "Kendall distance": _kendall,
    "Spearman footrule": _footrule,
    "RBO distance (p=0.9)": _rbo,
}


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = (p + q) / 2
    kl = lambda x, y: float(np.sum(np.where(x > 0, x * np.log(np.maximum(x, 1e-12) / np.maximum(y, 1e-12)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def build_signed_cases(groups: dict[tuple[str, str], dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """All instances with both controls, with signed effects computed exactly as the official evaluator."""
    cases = []
    for (dataset, instance_id), records in sorted(groups.items()):
        full = records.get("s3_sparsity/full")
        truncs = [r for k, r in records.items() if k.startswith("s3_sparsity/truncate_")]
        orig = records.get("s5_robustness/original")
        ablates = [r for k, r in records.items() if k.startswith("s5_robustness/ablate_")]
        if not full or not truncs or not orig or not ablates:
            continue
        cases.append(
            {
                "dataset": dataset,
                "instance_id": instance_id,
                "du": round(esr._mean([esr._restoration_delta(r, full) for r in truncs]), TIE_DECIMALS),
                "dv": round(esr._mean([esr._restoration_delta(r, orig) for r in ablates]), TIE_DECIMALS),
                "s3_full": full,
                "s3_truncates": truncs,
                "s5_original": orig,
                "s5_ablates": ablates,
                "s3_full_record_id": full["record_id"],
                "s3_truncate_record_ids": [r["record_id"] for r in truncs],
                "s5_original_record_id": orig["record_id"],
                "s5_ablate_record_ids": [r["record_id"] for r in ablates],
                "s3_truncate_count": len(truncs),
                "s5_ablate_count": len(ablates),
            }
        )
    return cases


def label_cases(cases: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Attach dominant_control and label-specific effects; drop ties (same rule as the official evaluator)."""
    out = []
    for case in cases:
        du, dv = case["du"], case["dv"]
        if mode == "clipped":
            eu, ev = max(du, 0.0), max(dv, 0.0)
        elif mode == "signed":
            eu, ev = du, dv
        else:
            eu, ev = abs(du), abs(dv)
        if eu == ev:
            continue
        out.append({**case, "dominant_control": "user" if eu > ev else "item", "user_effect": eu, "item_effect": ev})
    return out


# ---------------------------------------------------------------------------
# Signals -> rows in the official row format
# ---------------------------------------------------------------------------


def _rows_from_scores(cases: list[dict[str, Any]], scores: list[tuple[float, float]], method: str) -> list[dict]:
    rows = []
    for case, (su, sv) in zip(cases, scores):
        su, sv = round(su, TIE_DECIMALS), round(sv, TIE_DECIMALS)
        rows.append(
            {
                "method": method,
                "dataset": case["dataset"],
                "instance_id": case["instance_id"],
                "dominant_control": case["dominant_control"],
                "predicted_control": esr._dominant_control({"user": su, "item": sv}),
                "user_score": su,
                "item_score": sv,
                "user_effect": case["user_effect"],
                "item_effect": case["item_effect"],
            }
        )
    return rows


def signal_rows(
    cases: list[dict[str, Any]],
    uq_by_method: dict[str, dict[str, dict[str, Any]]],
    probs: dict[str, np.ndarray],
) -> list[tuple[str, str, list[dict]]]:
    """Return (group, signal name, rows) for every signal available on this backbone."""
    out: list[tuple[str, str, list[dict]]] = []
    for method, by_id in sorted(uq_by_method.items()):
        rows, missing = esr._score_method_source_responses(cases, by_id, method)
        if missing:
            raise RuntimeError(f"{method}: {missing} paired cases lack UQ scores")
        signal = rows[0]["prediction_signal"] if rows else ""
        if signal == "source_scores":
            out.append(("source-specific UQ", method, rows))
            continue
        out.append(("scalar response (clipped, original)", method, rows))
        signed = []
        for case in cases:
            u = lambda rid: float(by_id[rid]["uncertainty_score"])
            su = esr._mean([u(r) for r in case["s3_truncate_record_ids"]]) - u(case["s3_full_record_id"])
            sv = esr._mean([u(r) for r in case["s5_ablate_record_ids"]]) - u(case["s5_original_record_id"])
            signed.append((su, sv))
        out.append(("scalar response (signed)", method, _rows_from_scores(cases, signed, method)))
    for name, dist in DISTANCES.items():
        scores = [
            (
                esr._mean([dist(c["s3_full"]["ranking"], r["ranking"]) for r in c["s3_truncates"]]),
                esr._mean([dist(c["s5_original"]["ranking"], r["ranking"]) for r in c["s5_ablates"]]),
            )
            for c in cases
        ]
        out.append(("non-UQ listwise distance", name, _rows_from_scores(cases, scores, name)))
    if probs:
        scores = [
            (
                esr._mean([_js_divergence(probs[c["s3_full_record_id"]], probs[r]) for r in c["s3_truncate_record_ids"]]),
                esr._mean([_js_divergence(probs[c["s5_original_record_id"]], probs[r]) for r in c["s5_ablate_record_ids"]]),
            )
            for c in cases
        ]
        out.append(("UQ distribution shift", "LabelProb JS shift", _rows_from_scores(cases, scores, "LabelProb JS shift")))
    majority = Counter(c["dominant_control"] for c in cases).most_common(1)[0][0]
    rows = _rows_from_scores(cases, [(0.0, 0.0)] * len(cases), "Majority class")
    for row in rows:
        row["predicted_control"] = majority
    out.append(("reference", "Majority class", rows))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def point_metrics(name: str, rows: list[dict]) -> dict[str, Any]:
    official = esr._table12_row(name, rows)
    return {
        "n": official["n"],
        "user_share": sum(r["dominant_control"] == "user" for r in rows) / len(rows),
        "dircov": sum(r["predicted_control"] != "tie" for r in rows) / len(rows),
        "d_acc": official["dominant_control_accuracy"],
        "macro_f1": official["macro_control_f1"],
        "source_auroc": official["source_auroc"],
        "rho": official["control_response_rho"],
        "confusion": official["confusion"],
    }


def bootstrap_ci(rows: list[dict], resamples: int, seed: int) -> dict[str, float]:
    y = np.array([r["dominant_control"] == "user" for r in rows], dtype=int)
    pred = np.array([r["predicted_control"] for r in rows])
    correct = (pred == np.where(y == 1, "user", "item")).astype(float)
    su = np.array([r["user_score"] for r in rows], dtype=float)
    sv = np.array([r["item_score"] for r in rows], dtype=float)
    ds = np.array([r["dataset"] for r in rows])
    strata = [np.where(ds == d)[0] for d in sorted(set(ds))]
    rng = np.random.default_rng(seed)
    accs, aucs = [], []
    for _ in range(resamples):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
        yy = y[idx]
        accs.append(correct[idx].mean())
        if 0 < yy.mean() < 1:
            aucs.append((roc_auc_score(yy, su[idx]) + roc_auc_score(1 - yy, sv[idx])) / 2)
    return {
        "d_acc_ci_low": float(np.percentile(accs, 2.5)),
        "d_acc_ci_high": float(np.percentile(accs, 97.5)),
        "source_auroc_ci_low": float(np.percentile(aucs, 2.5)),
        "source_auroc_ci_high": float(np.percentile(aucs, 97.5)),
    }


# ---------------------------------------------------------------------------
# Restoration-effect distribution
# ---------------------------------------------------------------------------


def effect_summary(cases: list[dict[str, Any]], backbone: str) -> list[dict[str, Any]]:
    out = []
    for dataset in [*DATASETS, "all"]:
        sub = [c for c in cases if dataset == "all" or c["dataset"] == dataset]
        if not sub:
            continue
        row: dict[str, Any] = {"backbone": backbone, "dataset": dataset, "instances": len(sub)}
        for key, name in (("du", "user"), ("dv", "item")):
            vals = np.array([c[key] for c in sub])
            row[f"{name}_positive"] = int((vals > 0).sum())
            row[f"{name}_negative"] = int((vals < 0).sum())
            row[f"{name}_zero"] = int((vals == 0).sum())
            row[f"{name}_mean"] = float(vals.mean())
        exact = sum(c["du"] == c["dv"] for c in sub)
        clip_ties = sum(max(c["du"], 0.0) == max(c["dv"], 0.0) for c in sub)
        row["exact_signed_ties"] = exact
        row["ties_created_by_clipping"] = clip_ties - exact
        row["clipped_label_flips_vs_signed"] = sum(
            max(c["du"], 0.0) != max(c["dv"], 0.0)
            and (max(c["du"], 0.0) > max(c["dv"], 0.0)) != (c["du"] > c["dv"])
            for c in sub
        )
        for mode in LABEL_MODES:
            labelled = label_cases(sub, mode)
            row[f"{mode}_n"] = len(labelled)
            row[f"{mode}_user"] = sum(c["dominant_control"] == "user" for c in labelled)
            row[f"{mode}_item"] = sum(c["dominant_control"] == "item" for c in labelled)
        out.append(row)
    return out


def plot_effects(cases: list[dict[str, Any]], path: Path, backbone: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    bins = np.linspace(-1, 1, 41)
    for ax, key, title in ((axes[0], "du", "User-history restoration effect"), (axes[1], "dv", "Item-content restoration effect")):
        vals = np.array([c[key] for c in cases])
        ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"$\Delta$NDCG@10 (restored $-$ perturbed)", fontsize=9)
        neg, zero, pos = (vals < 0).mean(), (vals == 0).mean(), (vals > 0).mean()
        ax.text(0.02, 0.95, f"neg {neg:.1%}\nzero {zero:.1%}\npos {pos:.1%}", transform=ax.transAxes,
                va="top", fontsize=8)
    axes[0].set_ylabel("Instances", fontsize=9)
    fig.suptitle(f"Signed restoration effects before clipping ({backbone})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    logger = get_logger("evaluate_paired_control_extended")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_records = esr._read_jsonl(Path(args.base_predictions))
    groups = esr._group_base_instances(base_records)
    cases = build_signed_cases(groups)
    logger.info("backbone=%s instances with both controls=%d", args.backbone, len(cases))

    uq_by_method = esr._read_uq_by_method(esr._resolve_uq_paths(args.uq_predictions)) if args.uq_predictions else {}
    probs: dict[str, np.ndarray] = {}
    if args.labelprob_predictions:
        path = esr._resolve_uq_paths([args.labelprob_predictions])[0]
        with open_text(path) as f:
            for line in f:
                row = json.loads(line)
                opt = row["components"]["option_probabilities"]
                probs[row["record_id"]] = np.array([opt[chr(65 + i)] for i in range(len(opt))], dtype=float)

    summary = effect_summary(cases, args.backbone)
    esr._write_csv(out_dir / "restoration_effect_distribution.csv", summary)
    plot_effects(cases, out_dir / "restoration_effect_histogram", args.backbone)

    results: list[dict[str, Any]] = []
    for mode in LABEL_MODES:
        labelled = label_cases(cases, mode)
        logger.info("label=%s cases=%d", mode, len(labelled))
        for group, name, rows in signal_rows(labelled, uq_by_method, probs):
            for dataset in ["all", *DATASETS]:
                sub = rows if dataset == "all" else [r for r in rows if r["dataset"] == dataset]
                if not sub:
                    continue
                entry = {"backbone": args.backbone, "label_mode": mode, "group": group, "signal": name,
                         "dataset": dataset, **point_metrics(name, sub)}
                if dataset == "all" and args.bootstrap_resamples > 0:
                    entry.update(bootstrap_ci(sub, args.bootstrap_resamples, args.bootstrap_seed))
                results.append(entry)
    esr._write_csv(out_dir / "paired_control_extended.csv", results)

    esr._write_json(
        out_dir / "run_metadata.json",
        {
            "backbone": args.backbone,
            "base_predictions": args.base_predictions,
            "uq_predictions": args.uq_predictions,
            "labelprob_predictions": args.labelprob_predictions,
            "instances_with_both_controls": len(cases),
            "label_modes": {
                "clipped": "argmax_s [Delta_s]_+ (original analysis; ties excluded)",
                "signed": "argmax_s Delta_s (which restoration improves NDCG@10 more)",
                "magnitude": "argmax_s |Delta_s| (which control changes NDCG@10 more)",
            },
            "metrics": "official _table12_row definitions; DirCov = share of non-tie predictions",
            "bootstrap": {"resamples": args.bootstrap_resamples, "seed": args.bootstrap_seed, "strata": "dataset"},
        },
    )
    logger.info("Wrote %s", out_dir)



if __name__ == "__main__":
    main()
