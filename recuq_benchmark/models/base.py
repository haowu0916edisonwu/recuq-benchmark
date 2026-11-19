from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankingOutput:
    ranking: list[int]
    scores: list[float]
    raw_response: str
    parse_success: bool
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRecommender(ABC):
    @abstractmethod
    def rank(self, history_text: list[str], candidate_texts: list[str]) -> RankingOutput:
        """Rank candidate texts for one recommendation instance."""


class DryRunRecommender(BaseRecommender):
    """Deterministic ranker used only for scaffold and dry-run checks."""

    def rank(self, history_text: list[str], candidate_texts: list[str]) -> RankingOutput:
        ranking = list(range(len(candidate_texts)))
        scores = [float(len(candidate_texts) - rank) for rank in range(len(candidate_texts))]
        return RankingOutput(
            ranking=ranking,
            scores=scores,
            raw_response=", ".join(chr(ord("A") + idx) for idx in ranking),
            parse_success=True,
            metadata={"model": "dry_run"},
        )
