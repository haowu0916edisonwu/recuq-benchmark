#!/usr/bin/env python
"""Revision diagnostics derived from existing artifacts (no LLM inference).

Subcommands
-----------
alignment    Write uq_prediction_v1 files that separate cutoff alignment from the elicitation source
            . Evaluate them with evaluate_uq_methods.py.
calibration  Option-label probe calibration and positional bias per backbone.
positional   Positional-fallback rankings and NDCG by condition variant, Movies composition check,
             and Warm-start Error-AUROC stratified by positional fallback.
cost         Additional model calls and logged wall-clock per UQ method.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterator

import numpy as np
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION  # noqa: E402
from recuq_benchmark.utils.io import open_text, resolve_jsonl
from recuq_benchmark.utils.logging import get_logger  # noqa: E402

LETTERS = [chr(ord("A") + i) for i in range(20)]
DATASETS = ("mind_20", "amazon_books", "amazon_movies", "yelp")
SAME_INFERENCE_SCORES = (
    "slot1_top10mass",
    "slot1_max",
    "slot1_entropy",
    "slot1_margin",
    "seq_nll_top10",
    "pl_nll_top10",
    "mean_slot_entropy_top10",
    "min_token_prob_top10",
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _uq_path(path: str) -> Path:
    return resolve_jsonl(path)


def _load_uq(path: str) -> dict[str, dict[str, Any]]:
    return {row["record_id"]: row for row in _iter_jsonl(_uq_path(path))}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _positional_kind(ranking: list[int]) -> str:
    n = len(ranking)
    if ranking == list(range(n)):
        return "identity"
    if ranking == list(range(n - 1, -1, -1)):
        return "reverse"
    return "non_positional"


def _uq_row(record_id: str, method: str, family: str, score: float, components: dict, metadata: dict) -> dict:
    return {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": record_id,
        "method": method,
        "method_family": family,
        "uncertainty_score": float(score),
        "confidence_score": float(1.0 - score) if 0.0 <= score <= 1.0 else None,
        "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
        "components": components,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------


def _random_top10(record_id: str) -> list[int]:
    seed = int(hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:8], 16)
    return [int(i) for i in np.random.default_rng(seed).permutation(20)[:10]]


def cmd_alignment(args: argparse.Namespace) -> None:
    logger = get_logger("derive_revision_diagnostics.alignment")
    base = {r["record_id"]: r for r in _iter_jsonl(args.base_predictions)}
    probe = _load_uq(args.labelprob)
    temp = _load_uq(args.temp_scaled)
    decomp = _load_uq(args.decomp)
    out_root = Path(args.output_root)

    def mass_outside(probs: dict[str, float], idxs: list[int]) -> float:
        return 1.0 - sum(probs[LETTERS[i]] for i in idxs)

    variants = {
        "alignment_probe_own_top10_mass": (
            "probability_derived",
            "1 - probe mass on the probe's own 10 most probable labels (not aligned to the issued ranking)",
            lambda rid: 1.0 - sum(sorted(probe[rid]["components"]["option_probabilities"].values(), reverse=True)[:10]),
        ),
        "alignment_random_top10_mass": (
            "probability_derived",
            "1 - probe mass on a record-seeded random set of 10 labels (chance control)",
            lambda rid: mass_outside(probe[rid]["components"]["option_probabilities"], _random_top10(rid)),
        ),
        "alignment_temp_scaled_top10_mass": (
            "calibration_derived",
            "1 - temperature-scaled probe mass on the issued top-10 labels",
            lambda rid: mass_outside(temp[rid]["components"]["option_probabilities"], base[rid]["ranking"][:10]),
        ),
        "alignment_decomp_top10_mass": (
            "decomposition_derived",
            "1 - candidate-order-averaged probe mass on the issued top-10 labels",
            lambda rid: mass_outside(decomp[rid]["components"]["mean_option_probabilities"], base[rid]["ranking"][:10]),
        ),
    }
    for method, (family, definition, fn) in variants.items():
        written = 0
        path = out_root / method / "uq_predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for rid, record in base.items():
                if rid not in probe or rid not in temp or rid not in decomp:
                    raise KeyError(f"{method}: missing source UQ row for {rid}")
                meta = {k: record.get(k) for k in ("dataset", "scenario", "variant_name", "instance_id")}
                f.write(json.dumps(_uq_row(rid, method, family, fn(rid), {"score_definition": definition}, meta)) + "\n")
                written += 1
        logger.info("%s: %d rows", method, written)

    if args.same_inference:
        si = _load_uq(args.same_inference)
        for key in SAME_INFERENCE_SCORES:
            method = f"same_inference_{key}"
            path = out_root / method / "uq_predictions.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                for rid, row in si.items():
                    comps = row["components"]
                    score = comps["scores"][key]
                    if score is None:
                        raise ValueError(f"{method}: missing score for {rid}")
                    meta = dict(row.get("metadata", {}))
                    meta["teacher_forcing_exact"] = comps.get("teacher_forcing_exact")
                    f.write(json.dumps(_uq_row(rid, method, "same_inference_listwise", score,
                                               {"score_key": key, "source_method": row["method"]}, meta)) + "\n")
            logger.info("%s: %d rows", method, len(si))
    logger.info("Wrote alignment UQ files under %s", out_root)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(total)


def cmd_calibration(args: argparse.Namespace) -> None:
    logger = get_logger("derive_revision_diagnostics.calibration")
    rows = []
    for spec in args.backbone:
        name, base_path, probe_path, temp_path = spec.split(",")
        base = {r["record_id"]: r for r in _iter_jsonl(base_path) if r["scenario"] == "s0_warmstart"}
        probe, temp = _load_uq(probe_path), _load_uq(temp_path)
        ids = sorted(base)
        target = np.array([base[i]["labels"].index(1) for i in ids])
        p = np.array([[probe[i]["components"]["option_probabilities"][a] for a in LETTERS] for i in ids])
        pt = np.array([[temp[i]["components"]["option_probabilities"][a] for a in LETTERS] for i in ids])
        argmax = p.argmax(axis=1)
        correct = (argmax == target).astype(float)
        temps = sorted({round(float(temp[i]["components"]["temperature"]), 3) for i in ids})
        kinds = [_positional_kind(base[i]["ranking"]) for i in ids]
        row = {
            "backbone": name,
            "records": len(ids),
            "probe_mean_max_prob": float(p.max(axis=1).mean()),
            "probe_top1_accuracy": float(correct.mean()),
            "probe_ece_raw": _ece(p.max(axis=1), correct),
            "probe_ece_temp_scaled": _ece(pt.max(axis=1), (pt.argmax(axis=1) == target).astype(float)),
            "fold_temperature_min": temps[0],
            "fold_temperature_max": temps[-1],
            "probe_nll_positive_raw": float(-np.log(np.maximum(p[np.arange(len(ids)), target], 1e-12)).mean()),
            "probe_nll_positive_temp": float(-np.log(np.maximum(pt[np.arange(len(ids)), target], 1e-12)).mean()),
        }
        for letter in ("A", "B", "T"):
            row[f"probe_argmax_share_{letter}"] = float((argmax == LETTERS.index(letter)).mean())
        for kind in ("identity", "reverse", "non_positional"):
            row[f"issued_{kind}_share"] = kinds.count(kind) / len(kinds)
        rows.append(row)
        logger.info("%s: ECE raw=%.3f temp=%.3f argmax A=%.3f", name, row["probe_ece_raw"],
                    row["probe_ece_temp_scaled"], row["probe_argmax_share_A"])
    _write_csv(Path(args.output), rows)
    logger.info("Wrote %s", args.output)


# ---------------------------------------------------------------------------
# positional
# ---------------------------------------------------------------------------


def cmd_positional(args: argparse.Namespace) -> None:
    logger = get_logger("derive_revision_diagnostics.positional")
    out_dir = Path(args.output_dir)
    variant_rows, composition_rows, strat_rows = [], [], []
    for spec in args.backbone:
        name, base_path = spec.split(",")
        records = list(_iter_jsonl(base_path))
        groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for r in records:
            groups[(r["dataset"], r["scenario"], r["variant_name"])].append(r)
        for (ds, sc, var), rs in sorted(groups.items()):
            kinds = [_positional_kind(r["ranking"]) for r in rs]
            ndcg = np.array([r["metrics"]["ndcg@10"] for r in rs])
            nonpos = np.array([k == "non_positional" for k in kinds])
            variant_rows.append({
                "backbone": name, "dataset": ds, "scenario": sc, "variant": var, "n": len(rs),
                "identity_share": kinds.count("identity") / len(rs),
                "reverse_share": kinds.count("reverse") / len(rs),
                "positional_share": 1.0 - nonpos.mean(),
                "ndcg@10": float(ndcg.mean()),
                "hit@10": float(np.mean([r["metrics"]["hit@10"] for r in rs])),
                "non_positional_n": int(nonpos.sum()),
                "ndcg@10_non_positional": float(ndcg[nonpos].mean()) if nonpos.any() else None,
                "positional_error@10": float(np.mean([r["is_error_at_10"] for r, k in zip(rs, kinds) if k != "non_positional"]))
                if (~nonpos).any() else None,
            })
        # Composition check: S5 original NDCG on instances that do / do not have each ablation variant.
        for ds in DATASETS:
            orig = {r["instance_id"]: r for r in records
                    if r["dataset"] == ds and r["scenario"] == "s5_robustness" and r["variant_name"] == "original"}
            for var in sorted({r["variant_name"] for r in records if r["dataset"] == ds
                               and r["scenario"] == "s5_robustness" and r["variant_name"] != "original"}):
                have = {r["instance_id"] for r in records if r["dataset"] == ds and r["variant_name"] == var}
                inside = [orig[i]["metrics"]["ndcg@10"] for i in orig if i in have]
                outside = [orig[i]["metrics"]["ndcg@10"] for i in orig if i not in have]
                composition_rows.append({
                    "backbone": name, "dataset": ds, "variant": var, "instances_with_variant": len(inside),
                    "original_ndcg_with_variant": float(np.mean(inside)) if inside else None,
                    "original_ndcg_without_variant": float(np.mean(outside)) if outside else None,
                })
        logger.info("%s: %d variant groups", name, len(groups))

    for spec in args.stratify:
        name, base_path, *uq_paths = spec.split(",")
        warm = {r["record_id"]: r for r in _iter_jsonl(base_path) if r["scenario"] == "s0_warmstart"}
        ids = sorted(warm)
        y = np.array([int(warm[i]["is_error_at_10"]) for i in ids])
        kind = np.array([_positional_kind(warm[i]["ranking"]) for i in ids])
        for path in uq_paths:
            uq = _load_uq(path)
            method = next(iter(uq.values()))["method"]
            s = np.array([float(uq[i]["uncertainty_score"]) for i in ids])
            row = {"backbone": name, "method": method, "records": len(ids)}
            for label, mask in (("all", np.ones(len(ids), bool)), ("non_positional", kind == "non_positional"),
                                ("positional", kind != "non_positional")):
                row[f"{label}_n"] = int(mask.sum())
                row[f"{label}_error_rate"] = float(y[mask].mean())
                row[f"{label}_error_auroc"] = float(roc_auc_score(y[mask], s[mask])) if 0 < y[mask].mean() < 1 else None
            strat_rows.append(row)

    _write_csv(out_dir / "positional_by_variant.csv", variant_rows)
    _write_csv(out_dir / "itemtext_variant_composition.csv", composition_rows)
    if strat_rows:
        _write_csv(out_dir / "warm_error_auroc_by_positional_fallback.csv", strat_rows)
    logger.info("Wrote %s", out_dir)


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

_ELAPSED = re.compile(r"elapsed=((?:\d+h)?(?:\d+m)?\d+s)")


def _seconds(text: str) -> int:
    h = re.search(r"(\d+)h", text)
    m = re.search(r"(\d+)m", text)
    s = re.search(r"(\d+)s", text)
    return (int(h.group(1)) * 3600 if h else 0) + (int(m.group(1)) * 60 if m else 0) + (int(s.group(1)) if s else 0)


def _log_elapsed(path: str, pattern: str) -> int | None:
    """Elapsed seconds from the last line matching pattern, or the sum of matching lines when pattern ends in '+'."""
    total, last = 0, None
    summing = pattern.endswith("+")
    needle = pattern.rstrip("+")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if needle in line:
                match = _ELAPSED.search(line)
                if match:
                    last = _seconds(match.group(1))
                    total += last
    return total if summing else last


def _fmt(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def cmd_cost(args: argparse.Namespace) -> None:
    logger = get_logger("derive_revision_diagnostics.cost")
    logs = Path(args.logs_dir)
    base = list(_iter_jsonl(args.llama_base))
    per_instance: dict[tuple[str, str], int] = defaultdict(int)
    for r in base:
        if (r["scenario"] == "s3_sparsity" and r["variant_name"] != "full") or (
            r["scenario"] == "s5_robustness" and r["variant_name"] != "original"
        ):
            per_instance[(r["dataset"], r["instance_id"])] += 1
    pri_calls = np.array(list(per_instance.values()))
    by_ds = {ds: np.array([v for (d, _), v in per_instance.items() if d == ds]) for ds in DATASETS}
    pri_desc = f"{pri_calls.mean():.2f} on average (" + ", ".join(
        f"{ds} {v.min()}" + (f"-{v.max()}" if v.max() != v.min() else "") for ds, v in by_ds.items()) + ")"

    def meta_elapsed(path: str) -> int | None:
        p = Path(path)
        if not p.exists():
            return None
        return int(json.loads(p.read_text())["elapsed_seconds"])

    rows = [
        {"method": "Base listwise ranking (reference)", "extra_calls_per_record": "1 (the issued ranking)",
         "call_type": "greedy generation", "requires_labels": "no",
         "llama_seconds": _log_elapsed(logs / "llama_aligned20_base_n1000_schema.out", "Completed dataset+"),
         "llama_records": 45163,
         "qwen_seconds": _log_elapsed(logs / "qwen3_nonthinking_full.log", "Completed dataset+"), "qwen_records": 4000},
        {"method": "LabelProb-Max / Entropy / Margin / Top10Mass", "extra_calls_per_record": "1 (shared)",
         "call_type": "single forward pass", "requires_labels": "no",
         "llama_seconds": _log_elapsed(logs / "uq_labelprob_full.out", "Wrote"), "llama_records": 45163,
         "qwen_seconds": _log_elapsed(logs / "qwen3_labelprob_full.log", "Wrote"), "qwen_records": 4000},
        {"method": "Temperature-scaled LabelProb", "extra_calls_per_record": "0 (reuses LabelProb)",
         "call_type": "offline 1-D fit", "requires_labels": "yes (calibration labels)",
         "llama_seconds": None, "llama_records": None, "qwen_seconds": None, "qwen_records": None},
        {"method": "Decomposed LLM UQ", "extra_calls_per_record": "5",
         "call_type": "forward passes (candidate-order permutations)", "requires_labels": "no",
         "llama_seconds": _log_elapsed(logs / "uq_kweon_decomp_full.out", "Wrote"), "llama_records": 45163,
         "qwen_seconds": _log_elapsed(logs / "qwen3_kweon_decomp_full.log", "Wrote"), "qwen_records": 4000},
        {"method": "Self-consistency", "extra_calls_per_record": "5",
         "call_type": "sampled generations", "requires_labels": "no",
         "llama_seconds": _log_elapsed(logs / "uq_self_consistency_s5_t09_p095_full.out", "Wrote"),
         "llama_records": 45163,
         "qwen_seconds": _log_elapsed(logs / "qwen3_self_consistency_full.log", "Wrote"), "qwen_records": 4000},
        {"method": "Perturbation Rank Instability", "extra_calls_per_record": pri_desc,
         "call_type": "greedy generations under controlled inputs", "requires_labels": "no",
         "llama_seconds": None, "llama_records": None,
         "qwen_seconds": _log_elapsed(logs / "qwen3_perturbation_controls_full.log", "Completed dataset+"),
         "qwen_records": 41163},
        {"method": "Same-inference listwise scores", "extra_calls_per_record": "1",
         "call_type": "teacher-forced forward pass", "requires_labels": "no",
         "llama_seconds": meta_elapsed(args.llama_same_inference_meta), "llama_records": 4000,
         "qwen_seconds": meta_elapsed(args.qwen_same_inference_meta), "qwen_records": 4000},
    ]
    for row in rows:
        for b in ("llama", "qwen"):
            row[f"{b}_wallclock"] = _fmt(row[f"{b}_seconds"])
            row[f"{b}_seconds_per_record"] = (
                round(row[f"{b}_seconds"] / row[f"{b}_records"], 3) if row[f"{b}_seconds"] and row[f"{b}_records"] else None
            )
    _write_csv(Path(args.output), rows)
    logger.info("PRI extra generations per instance: %s", pri_desc)
    logger.info("Wrote %s", args.output)


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("alignment")
    p.add_argument("--base_predictions", required=True)
    p.add_argument("--labelprob", required=True)
    p.add_argument("--temp_scaled", required=True)
    p.add_argument("--decomp", required=True)
    p.add_argument("--same_inference", default=None)
    p.add_argument("--output_root", required=True)
    p.set_defaults(func=cmd_alignment)

    p = sub.add_parser("calibration")
    p.add_argument("--backbone", action="append", required=True, help="name,base_predictions,labelprob,temp_scaled")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_calibration)

    p = sub.add_parser("positional")
    p.add_argument("--backbone", action="append", required=True, help="name,base_predictions")
    p.add_argument("--stratify", action="append", default=[], help="name,warm_base_predictions,uq1,uq2,...")
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=cmd_positional)

    p = sub.add_parser("cost")
    p.add_argument("--logs_dir", default="logs")
    p.add_argument("--llama_base", required=True)
    p.add_argument("--llama_same_inference_meta", required=True)
    p.add_argument("--qwen_same_inference_meta", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_cost)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
