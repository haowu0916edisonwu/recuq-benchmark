"""Builders for the benchmark evaluation conditions."""

from recuq_benchmark.scenarios.base import ScenarioBuilder, Variant
from recuq_benchmark.scenarios.longtail import S4LongtailBuilder
from recuq_benchmark.scenarios.robustness import S5RobustnessBuilder
from recuq_benchmark.scenarios.sparsity import S3SparsityBuilder
from recuq_benchmark.scenarios.warmstart import S0WarmstartBuilder

SCENARIO_REGISTRY = {
    "s0_warmstart": S0WarmstartBuilder,
    "s3_sparsity": S3SparsityBuilder,
    "s4_longtail": S4LongtailBuilder,
    "s5_robustness": S5RobustnessBuilder,
}

__all__ = [
    "SCENARIO_REGISTRY",
    "ScenarioBuilder",
    "S0WarmstartBuilder",
    "S3SparsityBuilder",
    "S4LongtailBuilder",
    "S5RobustnessBuilder",
    "Variant",
]
