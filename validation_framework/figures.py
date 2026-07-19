import os

import matplotlib

# Must precede pyplot's own import -- same convention every other
# figure module in this codebase (research_framework/figures.py,
# ai_explainability/visualization.py, campaign_analytics/visualizations.py)
# already establishes.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use())

from typing import Any, Dict, Mapping, Optional, Sequence  # noqa: E402

from dataset_builder.schema import ordered_zones, zone_occupancy_columns  # noqa: E402

from research_framework.figures import (  # noqa: E402
    CHART_DPI,
    CHART_FACECOLOR,
    save_confusion_matrix,
    save_congestion_heatmap,
    save_evacuation_time_distribution,
    save_pr_curve,
    save_roc_curve,
    save_rl_reward_curve,
)
from ai_explainability.visualization import plot_prediction_comparison, plot_residuals  # noqa: E402

from validation_framework.prediction_validator import PredictionValidationResult  # noqa: E402
from validation_framework.scenario_comparator import PolicyComparisonResult  # noqa: E402


# =====================================================
# Phase 6 -- Research Figures. Confusion matrices/ROC/PR/reward-curve/
# evacuation-time-distribution/congestion-heatmap figures are already
# implemented in research_framework.figures (re-exported above) and
# residual/prediction-comparison scatter plots in ai_explainability.
# visualization (also re-exported) -- neither is reimplemented here.
# Genuinely new below: an RL learning curve across TRAINING checkpoints
# (distinct from a single policy's per-episode reward curve), an
# occupancy-over-time heatmap (vs. congestion's single-value snapshot),
# an evacuation curve, and a recommendation-change timeline -- none of
# these exist anywhere else in the platform.
# =====================================================

__all__ = [
    "CHART_DPI", "CHART_FACECOLOR",
    "save_confusion_matrix", "save_congestion_heatmap", "save_evacuation_time_distribution",
    "save_pr_curve", "save_roc_curve", "save_rl_reward_curve",
    "plot_prediction_comparison", "plot_residuals",
    "save_rl_learning_curve", "save_occupancy_heatmap", "save_evacuation_curve",
    "save_recommendation_timeline", "generate_all_figures",
]


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


def save_rl_learning_curve(checkpoint_history: Sequence[Dict[str, Any]], file_path: str) -> str:

    if not checkpoint_history:
        raise ValueError("save_rl_learning_curve() requires at least one checkpoint entry.")

    timesteps = [entry["timesteps"] for entry in checkpoint_history]
    rewards = [entry["average_reward"] for entry in checkpoint_history]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=CHART_DPI)
    ax.plot(timesteps, rewards, color=CHART_FACECOLOR, marker="o")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel("Average Reward")
    ax.set_title("RL Learning Curve")

    return _save(fig, file_path)


# =====================================================


def save_occupancy_heatmap(building: Any, timeline_rows: Sequence[Dict[str, Any]], file_path: str) -> str:

    if not timeline_rows:
        raise ValueError("save_occupancy_heatmap() requires at least one timeline row.")

    zones = ordered_zones(building)
    occupancy_columns = zone_occupancy_columns(building)

    matrix = [
        [row.get(column) or 0 for row in timeline_rows]
        for column in occupancy_columns
    ]
    times = [row.get("simulation_time") or 0.0 for row in timeline_rows]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.35 * len(zones))), dpi=CHART_DPI)
    extent = [times[0], times[-1] if len(times) > 1 else times[0] + 1.0, len(zones), 0]
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", extent=extent)
    fig.colorbar(im, ax=ax, label="Occupants")

    ax.set_yticks([index + 0.5 for index in range(len(zones))])
    ax.set_yticklabels([zone.name for zone in zones])
    ax.set_xlabel("Simulation Time (s)")
    ax.set_title("Occupancy Heatmap")

    return _save(fig, file_path)


# =====================================================


def save_evacuation_curve(timeline_rows: Sequence[Dict[str, Any]], file_path: str) -> str:

    if not timeline_rows:
        raise ValueError("save_evacuation_curve() requires at least one timeline row.")

    times = [row.get("simulation_time") or 0.0 for row in timeline_rows]
    remaining = [row.get("people_remaining") or 0 for row in timeline_rows]
    evacuated = [row.get("people_evacuated") or 0 for row in timeline_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=CHART_DPI)
    ax.plot(times, remaining, color=CHART_FACECOLOR, label="Remaining")
    ax.plot(times, evacuated, color="#22c55e", label="Evacuated")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Occupants")
    ax.set_title("Evacuation Curve")
    ax.legend()

    return _save(fig, file_path)


# =====================================================


def save_recommendation_timeline(changes: Sequence[Any], file_path: str) -> str:

    if not changes:
        raise ValueError("save_recommendation_timeline() requires at least one RecommendationChange.")

    zone_ids = sorted({change.zone_id for change in changes})
    zone_index = {zone_id: index for index, zone_id in enumerate(zone_ids)}

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.4 * len(zone_ids))), dpi=CHART_DPI)

    for change in changes:
        ax.scatter(change.timestamp, zone_index[change.zone_id], color=CHART_FACECOLOR, s=40)

    ax.set_yticks(range(len(zone_ids)))
    ax.set_yticklabels(zone_ids)
    ax.set_xlabel("Simulation Time (s)")
    ax.set_title("Recommendation Change Timeline")

    return _save(fig, file_path)


# =====================================================


def generate_all_figures(
    *,
    output_dir: str,
    prediction_results: Optional[Mapping[str, PredictionValidationResult]] = None,
    policy_comparison: Optional[PolicyComparisonResult] = None,
    rl_learning_curve_history: Sequence[Dict[str, Any]] = (),
    building: Any = None,
    sample_timeline_rows: Sequence[Dict[str, Any]] = (),
    sample_changes: Sequence[Any] = (),
) -> Dict[str, str]:

    prediction_results = prediction_results or {}
    paths: Dict[str, str] = {}

    for name, result in prediction_results.items():

        try:

            if result.y_proba is not None:

                paths[f"{name}_confusion_matrix"] = save_confusion_matrix(
                    result.y_true, result.y_pred, os.path.join(output_dir, f"{name}_confusion_matrix.png"),
                )

                classes = sorted(set(result.y_true) | set(result.y_pred))

                if len(classes) == 2:

                    positive_label = classes[-1]
                    positive_index = list(result.model.label_encoder.classes_).index(positive_label)
                    scores = [row[positive_index] for row in result.y_proba]

                    paths[f"{name}_roc_curve"] = save_roc_curve(
                        result.y_true, scores, os.path.join(output_dir, f"{name}_roc_curve.png"),
                        pos_label=positive_label,
                    )
                    paths[f"{name}_pr_curve"] = save_pr_curve(
                        result.y_true, scores, os.path.join(output_dir, f"{name}_pr_curve.png"),
                        pos_label=positive_label,
                    )

            else:

                paths[f"{name}_residuals"] = plot_residuals(
                    result.y_true, result.y_pred, os.path.join(output_dir, f"{name}_residuals.png"),
                )
                paths[f"{name}_prediction_comparison"] = plot_prediction_comparison(
                    result.y_true, result.y_pred, os.path.join(output_dir, f"{name}_prediction_comparison.png"),
                )

        except (ValueError, AttributeError, TypeError, IndexError):

            # A figure that cannot be honestly drawn from this
            # particular result (e.g. too few held-out rows, or a
            # multi-output/multiclass shape a given plot doesn't
            # support) is skipped rather than fabricated.
            continue

    if policy_comparison is not None:

        try:

            evacuation_times_by_arm = {
                label: [
                    episode.total_evacuation_time for episode in report.episodes
                    if episode.total_evacuation_time is not None
                ]
                for label, report in policy_comparison.reports.items()
            }
            paths["evacuation_time_distribution"] = save_evacuation_time_distribution(
                evacuation_times_by_arm, os.path.join(output_dir, "evacuation_time_distribution.png"),
            )

        except (ValueError, KeyError):
            pass

        if "rl_policy" in policy_comparison.reports:

            try:
                rewards = [episode.total_reward for episode in policy_comparison.reports["rl_policy"].episodes]
                paths["rl_reward_curve"] = save_rl_reward_curve(
                    rewards, os.path.join(output_dir, "rl_reward_curve.png"),
                )
            except ValueError:
                pass

    if rl_learning_curve_history:

        try:
            paths["rl_learning_curve"] = save_rl_learning_curve(
                rl_learning_curve_history, os.path.join(output_dir, "rl_learning_curve.png"),
            )
        except (ValueError, KeyError):
            pass

    if building is not None and sample_timeline_rows:

        try:
            paths["occupancy_heatmap"] = save_occupancy_heatmap(
                building, sample_timeline_rows, os.path.join(output_dir, "occupancy_heatmap.png"),
            )
        except ValueError:
            pass

        try:
            paths["evacuation_curve"] = save_evacuation_curve(
                sample_timeline_rows, os.path.join(output_dir, "evacuation_curve.png"),
            )
        except ValueError:
            pass

    if sample_changes:

        try:
            paths["recommendation_timeline"] = save_recommendation_timeline(
                sample_changes, os.path.join(output_dir, "recommendation_timeline.png"),
            )
        except ValueError:
            pass

    return paths
