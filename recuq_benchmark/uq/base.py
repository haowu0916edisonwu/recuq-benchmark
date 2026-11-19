from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from recuq_benchmark.models.base import BaseRecommender
from recuq_benchmark.scenarios.base import Variant
from recuq_benchmark.utils.cache import CacheManager


class UQEstimator(ABC):
    @abstractmethod
    def compute(
        self,
        variants: list[Variant],
        recommender: BaseRecommender,
        cache: CacheManager,
    ) -> dict[str, Any]:
        """Compute uncertainty outputs for a group of variants."""
