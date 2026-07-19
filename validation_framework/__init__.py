# =====================================================
# Validation & Benchmark Framework -- an evaluation-only layer on top
# of the existing SynEvac platform. Verifies AI predictions and RL
# policies against Ground Truth / simulated-outcome references, and
# evaluates Advisory System recommendation quality. Every computation
# here reads already-public APIs from ai_training, ai_explainability,
# rl_training, decision_policy, advisory_system, ground_truth,
# training_dataset, dataset_builder, campaign_analytics, and
# research_framework -- nothing in any of those packages is modified,
# and nothing here changes an existing prediction.
#
# Not to be confused with validation/ (a sibling top-level directory of
# one-off phase-completion scripts with results dumped to
# validation/results/) -- this package is the general-purpose, reusable
# library; validation/ is ad hoc and script-shaped.
# =====================================================

from validation_framework.benchmark_runner import ValidationBenchmarkResult, run_full_validation
from validation_framework.prediction_validator import (
    PredictionValidationResult,
    validate_bottleneck_model,
    validate_evacuation_time_model,
    validate_exit_usage_model,
    validate_firefighter_intelligence_consistency,
    validate_recommendation_accuracy,
    validate_smoke_prediction_model,
)
from validation_framework.recommendation_validator import RecommendationValidationResult, validate_recommendations
from validation_framework.report import generate_validation_report, write_validation_report
from validation_framework.scenario_comparator import PolicyComparisonResult, run_policy_comparison
from validation_framework.statistics import BootstrapResult, bootstrap_confidence_interval

__all__ = [
    "ValidationBenchmarkResult", "run_full_validation",
    "PredictionValidationResult",
    "validate_bottleneck_model", "validate_evacuation_time_model", "validate_exit_usage_model",
    "validate_firefighter_intelligence_consistency", "validate_recommendation_accuracy",
    "validate_smoke_prediction_model",
    "RecommendationValidationResult", "validate_recommendations",
    "generate_validation_report", "write_validation_report",
    "PolicyComparisonResult", "run_policy_comparison",
    "BootstrapResult", "bootstrap_confidence_interval",
]
