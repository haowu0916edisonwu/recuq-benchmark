from __future__ import annotations

import math


def dcg_at_k(labels_in_rank_order: list[int], k: int) -> float:
    score = 0.0
    for idx, label in enumerate(labels_in_rank_order[:k]):
        if label <= 0:
            continue
        score += (2**label - 1) / math.log2(idx + 2)
    return score


def ndcg_at_k(labels: list[int], ranking: list[int], k: int = 10) -> float:
    ranked_labels = [labels[idx] for idx in ranking if idx < len(labels)]
    ideal_labels = sorted(labels, reverse=True)
    ideal = dcg_at_k(ideal_labels, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_labels, k) / ideal


def mrr_at_k(labels: list[int], ranking: list[int], k: int = 10) -> float:
    for rank_position, candidate_idx in enumerate(ranking[:k], start=1):
        if candidate_idx < len(labels) and labels[candidate_idx] > 0:
            return 1.0 / rank_position
    return 0.0


def hit_at_k(labels: list[int], ranking: list[int], k: int = 10) -> float:
    for candidate_idx in ranking[:k]:
        if candidate_idx < len(labels) and labels[candidate_idx] > 0:
            return 1.0
    return 0.0


def top1_accuracy(labels: list[int], ranking: list[int]) -> float:
    if not ranking:
        return 0.0
    top_idx = ranking[0]
    return 1.0 if top_idx < len(labels) and labels[top_idx] > 0 else 0.0


def compute_ranking_metrics(labels: list[int], ranking: list[int], k: int = 10) -> dict[str, float]:
    hit = hit_at_k(labels, ranking, k)
    return {
        f"ndcg@{k}": ndcg_at_k(labels, ranking, k),
        f"mrr@{k}": mrr_at_k(labels, ranking, k),
        f"hit@{k}": hit,
        "top1_accuracy": top1_accuracy(labels, ranking),
        f"error@{k}": 1.0 - hit,
    }
