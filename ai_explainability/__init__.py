from ai_explainability.benchmark import (
    BenchmarkResult,
    benchmark_algorithms,
    best_result,
    flatten_metrics,
    to_table_rows,
)
from ai_explainability.comparison import (
    CONCERNING_EXIT_STATUSES,
    CONCERNING_STAIR_STATUSES,
    CONCERNING_ZONE_ACTIONS,
    DecisionPolicyComparisonRecord,
    compare_bottleneck_location_to_decision_policy,
    decision_policy_flagged,
    summarize_comparisons,
)
from ai_explainability.feature_importance import (
    RankedFeature,
    extract_raw_importances,
    get_feature_importances,
    rank_feature_importances,
)
from ai_explainability.permutation_importance import (
    default_score,
    permutation_importance,
    rank_permutation_importances,
)
from ai_explainability.prediction_report import PredictionReport, explain_prediction
from ai_explainability.visualization import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_model_comparison,
    plot_prediction_comparison,
    plot_residuals,
    render_benchmark_table_image,
)

__all__ = [
    "RankedFeature",
    "extract_raw_importances",
    "get_feature_importances",
    "rank_feature_importances",
    "default_score",
    "permutation_importance",
    "rank_permutation_importances",
    "PredictionReport",
    "explain_prediction",
    "BenchmarkResult",
    "benchmark_algorithms",
    "to_table_rows",
    "flatten_metrics",
    "best_result",
    "CONCERNING_EXIT_STATUSES",
    "CONCERNING_STAIR_STATUSES",
    "CONCERNING_ZONE_ACTIONS",
    "DecisionPolicyComparisonRecord",
    "decision_policy_flagged",
    "compare_bottleneck_location_to_decision_policy",
    "summarize_comparisons",
    "plot_feature_importance",
    "plot_prediction_comparison",
    "plot_model_comparison",
    "render_benchmark_table_image",
    "plot_confusion_matrix",
    "plot_residuals",
]
