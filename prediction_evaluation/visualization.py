import os
from typing import Mapping, Sequence

import matplotlib

# Agg is the non-interactive, headless raster backend -- must be
# selected before pyplot is ever imported. Same convention research_
# framework/figures.py and campaign_analytics/visualizations.py already
# establish (restated here rather than imported -- this package must
# stay import-independent of both, see its own architecture guard test).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use())

from prediction_evaluation.classification_metrics import ClassificationMetrics  # noqa: E402
from prediction_evaluation.horizon_analysis import HorizonBucketResult  # noqa: E402
from prediction_evaluation.models import MatchedEvaluation  # noqa: E402
from prediction_evaluation.pairs import bottleneck_classification_pairs, evacuation_time_regression_pairs  # noqa: E402


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 7 --
# EVALUATION-ONLY plots. Every function here takes already-computed
# evaluation data and writes a static image file -- none of them is
# imported by command_center/ or any operational UI (Phase 7's own
# explicit "do not integrate into operational UI" instruction, enforced
# by this package's own architecture guard test). No function here re-
# computes a metric that classification_metrics.py/regression_metrics.py/
# horizon_analysis.py do not already own.
# =====================================================

CHART_DPI = 100
CHART_FACECOLOR = "#3b82f6"
CHART_ACTUAL_COLOR = "#16a34a"
CHART_ERROR_COLOR = "#dc2626"


def _ensure_parent(file_path: str) -> None:

    parent = os.path.dirname(file_path)

    if parent:
        os.makedirs(parent, exist_ok=True)


def _save(fig, file_path: str) -> str:

    _ensure_parent(file_path)
    fig.tight_layout()
    fig.savefig(file_path, dpi=CHART_DPI)
    plt.close(fig)

    return file_path


# =====================================================


def plot_predicted_vs_actual_regression(evaluations: Sequence[MatchedEvaluation], file_path: str) -> str:

    y_true, y_pred = evacuation_time_regression_pairs(evaluations)

    fig, ax = plt.subplots(figsize=(6, 6))

    if y_true:

        ax.scatter(y_true, y_pred, color=CHART_FACECOLOR, alpha=0.6, s=20)
        limit = max(max(y_true), max(y_pred)) * 1.05 if y_true and y_pred else 1.0
        ax.plot([0, limit], [0, limit], color="gray", linestyle="--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)

    ax.set_xlabel("Actual evacuation time (s)")
    ax.set_ylabel("Predicted evacuation time (s)")
    ax.set_title(f"Predicted vs Actual Evacuation Time (n={len(y_true)})")

    if y_true:
        ax.legend(loc="upper left")

    return _save(fig, file_path)


def plot_error_over_time(evaluations: Sequence[MatchedEvaluation], file_path: str) -> str:

    points = []

    for evaluation in evaluations:

        evacuation_time = getattr(evaluation.prediction.payload, "evacuation_time_experimental", None)
        actual = evaluation.ground_truth.evacuation_time_seconds

        if evacuation_time is None or actual is None:
            continue

        predicted = getattr(evacuation_time, "predicted_seconds", None)
        if predicted is None:
            continue

        points.append((evaluation.prediction.timestamp, predicted - actual))

    points.sort(key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=(8, 4))

    if points:

        xs, ys = zip(*points)
        ax.plot(xs, ys, color=CHART_ERROR_COLOR, marker="o", markersize=3, linewidth=1)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Prediction timestamp (s)")
    ax.set_ylabel("Signed error: predicted - actual (s)")
    ax.set_title(f"Evacuation Time Prediction Error Over Time (n={len(points)})")

    return _save(fig, file_path)


def plot_accuracy_by_horizon(horizon_results: Mapping[float, HorizonBucketResult], file_path: str) -> str:

    horizons = sorted(horizon_results.keys())
    f1_values = [horizon_results[h].classification.f1 for h in horizons]
    mae_values = [horizon_results[h].regression.mae for h in horizons]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(
        [h for h, v in zip(horizons, f1_values) if v is not None],
        [v for v in f1_values if v is not None],
        color=CHART_FACECOLOR, marker="o",
    )
    ax1.set_xlabel("Prediction horizon (s)")
    ax1.set_ylabel("F1 score (bottleneck_occurrence)")
    ax1.set_title("Classification accuracy by horizon")
    ax1.set_ylim(0, 1)

    ax2.plot(
        [h for h, v in zip(horizons, mae_values) if v is not None],
        [v for v in mae_values if v is not None],
        color=CHART_ERROR_COLOR, marker="o",
    )
    ax2.set_xlabel("Prediction horizon (s)")
    ax2.set_ylabel("MAE (evacuation_time, s)")
    ax2.set_title("Regression error by horizon")

    return _save(fig, file_path)


def plot_confusion_matrix(metrics: ClassificationMetrics, file_path: str) -> str:

    fig, ax = plt.subplots(figsize=(4.5, 4))

    if metrics.confusion_matrix is not None:

        tn, fp, fn, tp = metrics.confusion_matrix
        matrix = [[tn, fp], [fn, tp]]

        im = ax.imshow(matrix, cmap="Blues")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center", color="black")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted: No", "Predicted: Yes"])
        ax.set_yticklabels(["Actual: No", "Actual: Yes"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Bottleneck Occurrence Confusion Matrix (n={metrics.sample_count})")

    return _save(fig, file_path)


def plot_calibration_curve(metrics: ClassificationMetrics, file_path: str) -> str:

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Perfect calibration")

    xs = [b.mean_predicted_probability for b in metrics.calibration_bins if b.sample_count > 0]
    ys = [b.observed_positive_rate for b in metrics.calibration_bins if b.sample_count > 0]

    if xs:
        ax.plot(xs, ys, color=CHART_FACECOLOR, marker="o", label="Model")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_title("Calibration Curve (bottleneck_occurrence)")
    ax.legend(loc="upper left")

    return _save(fig, file_path)


def plot_ground_truth_metric_over_time(
    evaluations: Sequence[MatchedEvaluation], metric: str, file_path: str,
) -> str:

    # A supporting, exploratory plot (Phase 7's own "occupancy/
    # congestion/queue prediction error" wording) -- honestly named:
    # this plots the GROUND-TRUTH signal itself over time (occupancy
    # count, or a 0/1 congestion-detected indicator), since neither is a
    # direct model OUTPUT to compute a numeric "error" against today
    # (see docs/architecture/prediction_evaluation.md's own "ground
    # truth definition, precisely" section) -- never fabricates a
    # prediction series that does not exist.

    if metric not in ("total_occupant_count", "congestion_detected"):
        raise ValueError(f"Unsupported metric {metric!r} -- expected 'total_occupant_count' or 'congestion_detected'.")

    points = [
        (e.ground_truth.timestamp, getattr(e.ground_truth, metric))
        for e in evaluations
        if getattr(e.ground_truth, metric) is not None
    ]
    points.sort(key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=(8, 4))

    if points:

        xs, ys = zip(*points)
        ys_numeric = [float(y) for y in ys]
        ax.plot(xs, ys_numeric, color=CHART_ACTUAL_COLOR, marker="o", markersize=3, linewidth=1)

    ax.set_xlabel("Timestamp (s)")
    ax.set_ylabel(metric)
    ax.set_title(f"Ground Truth: {metric} Over Time (n={len(points)})")

    return _save(fig, file_path)
