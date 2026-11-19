"""Recommender model interfaces."""

from recuq_benchmark.models.base import BaseRecommender, DryRunRecommender, RankingOutput
from recuq_benchmark.models.llama_instruct import CausalLMInstructRecommender, LlamaInstructRecommender

__all__ = [
    "BaseRecommender",
    "CausalLMInstructRecommender",
    "DryRunRecommender",
    "LlamaInstructRecommender",
    "RankingOutput",
]
