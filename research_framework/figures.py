import os

import matplotlib

# Agg is the non-interactive, headless raster backend -- must be
# selected before pyplot is ever imported. Same reasoning, same
# convention as campaign_analytics/visualizations.py's own header
# (restated here rather than imported, since that module is scoped to
# single-campaign charts, not cross-arm/publication figures).
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use())
from matplotlib.patches import Rectangle  # noqa: E402

from typing import Any, Dict, Mapping, Optional, Sequence  # noqa: E402

from sklearn.metrics import (  # noqa: E402
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# =====================================================
# Phase 6 -- Publication Figures. Every function below is independently
# callable with plain, already-in-hand inputs (labels/predictions,
# fitted estimators, reward sequences, ...) -- none of them re-runs a
# simulation, re-trains a model, or re-derives a dataset. CHART_DPI/
# facecolor mirror campaign_analytics/visualizations.py's own defaults
# for visual consistency across every plot this platform produces.
# =====================================================

CHART_DPI = 100
CHART_FACECOLOR = "#3b82f6"


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


def save_confusion_matrix(
    y_true: Sequence[Any], y_pred: Sequence[Any], file_path: str, labels: Optional[Sequence[Any]] = None,
) -> str:

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    display_labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))

    fig, ax = plt.subplots(figsize=(6, 5), dpi=CHART_DPI)
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticklabels(display_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    return _save(fig, file_path)


# =====================================================


def save_roc_curve(y_true: Sequence[Any], y_score: Sequence[float], file_path: str, pos_label: Any = None) -> str:

    fpr, tpr, _thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=CHART_DPI)
    ax.plot(fpr, tpr, color=CHART_FACECOLOR, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")

    return _save(fig, file_path)


# =====================================================


def save_pr_curve(y_true: Sequence[Any], y_score: Sequence[float], file_path: str, pos_label: Any = None) -> str:

    precision, recall, _thresholds = precision_recall_curve(y_true, y_score, pos_label=pos_label)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=CHART_DPI)
    ax.plot(recall, precision, color=CHART_FACECOLOR)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")

    return _save(fig, file_path)


# =====================================================


def save_feature_importance(
    estimator: Any, feature_names: Sequence[str], file_path: str, top_n: int = 20,
) -> str:

    # Only meaningful for tree-based estimators (RandomForest/
    # GradientBoosting/xgboost -- the only algorithms ai_training.
    # models.base.build_regressor/build_classifier support) that expose
    # .feature_importances_ -- an estimator without it (unlikely, given
    # that algorithm set, but never assumed) raises a clear, honest
    # error rather than fabricating a chart of zeros.

    if not hasattr(estimator, "feature_importances_"):

        raise ValueError(
            f"{type(estimator).__name__} has no .feature_importances_ -- feature "
            f"importance is only supported for tree-based estimators.",
        )

    importances = list(estimator.feature_importances_)

    if len(importances) != len(feature_names):

        raise ValueError(
            f"estimator.feature_importances_ has {len(importances)} entries but "
            f"{len(feature_names)} feature_names were given -- these must be the "
            f"post-preprocessing names (Preprocessor.feature_names_out()), not the raw "
            f"pre-encoding column list.",
        )

    ranked = sorted(zip(feature_names, importances), key=lambda pair: pair[1], reverse=True)[:top_n]
    ranked.reverse()  # so the highest-importance feature plots at the top

    names = [name for name, _ in ranked]
    values = [value for _, value in ranked]

    fig, ax = plt.subplots(figsize=(8, max(4.0, 0.3 * len(names))), dpi=CHART_DPI)
    ax.barh(range(len(names)), values, color=CHART_FACECOLOR)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance")

    return _save(fig, file_path)


# =====================================================


def save_training_curve(
    train_scores: Sequence[float], file_path: str, val_scores: Optional[Sequence[float]] = None,
    x_label: str = "Iteration", y_label: str = "Score",
) -> str:

    # Only meaningful for an iterative/staged estimator that can report
    # a per-iteration score sequence (e.g. GradientBoostingRegressor.
    # train_score_ / staged_predict()) -- the caller supplies that
    # sequence directly; this function draws it, it does not derive it
    # (RandomForest, this platform's own default algorithm, has no
    # native staged/iterative score to plot -- this is a real,
    # documented limitation, not a gap this function papers over).

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=CHART_DPI)
    ax.plot(range(1, len(train_scores) + 1), train_scores, color=CHART_FACECOLOR, label="Train")

    if val_scores is not None:
        ax.plot(range(1, len(val_scores) + 1), val_scores, color="#f97316", label="Validation")
        ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Training Curve")

    return _save(fig, file_path)


# =====================================================


def save_rl_reward_curve(episode_rewards: Sequence[float], file_path: str) -> str:

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=CHART_DPI)
    ax.plot(range(1, len(episode_rewards) + 1), list(episode_rewards), color=CHART_FACECOLOR, marker="o")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("RL Reward Curve")

    return _save(fig, file_path)


# =====================================================


def save_evacuation_time_distribution(values_by_arm: Mapping[str, Sequence[float]], file_path: str) -> str:

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=CHART_DPI)

    for arm_name, values in values_by_arm.items():

        if values:
            ax.hist(list(values), bins=min(15, max(1, len(values))), alpha=0.5, label=arm_name)

    ax.set_xlabel("Evacuation Time (s)")
    ax.set_ylabel("Scenario Count")
    ax.set_title("Evacuation Time Distribution by Arm")
    ax.legend()

    return _save(fig, file_path)


# =====================================================


def save_fire_growth_plot(tick_results: Sequence[Any], file_path: str, zone_ids: Optional[Sequence[str]] = None) -> str:

    # One line per zone: hazard_score (TickResult.hazard_snapshot.
    # node_state(zone_id).hazard_score) over simulated time. zone_ids
    # defaults to every zone_id the *first* tick's own hazard_snapshot
    # mentions -- a zone that only ever reports the HazardNodeState()
    # default (never explicitly set) is honestly omitted rather than
    # plotted as a fabricated flat 0.0 line.

    if not tick_results:
        raise ValueError("save_fire_growth_plot() requires at least one TickResult.")

    if zone_ids is None:
        zone_ids = sorted(tick_results[0].hazard_snapshot.node_states.keys())

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=CHART_DPI)

    times = [tick.time for tick in tick_results]

    for zone_id in zone_ids:

        scores = [tick.hazard_snapshot.node_state(zone_id).hazard_score for tick in tick_results]
        ax.plot(times, scores, label=zone_id)

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Hazard Score")
    ax.set_title("Fire Growth")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize="small")

    return _save(fig, file_path)


# =====================================================


def save_congestion_heatmap(building: Any, zone_congestion: Mapping[str, float], file_path: str) -> str:

    # A spatial heatmap using each Zone's own authored rectangle
    # (x, y, width, height) -- the only geometry this platform's
    # Building model carries; no floor-plan rendering (walls, doors,
    # icons) is attempted, matching Command Center's own "id-keyed
    # values, not a floor-plan renderer" scope.

    zones = [zone for floor in building.ordered_floors() for zone in floor.zones]

    if not zones:
        raise ValueError("save_congestion_heatmap() requires a Building with at least one Zone.")

    max_value = max(zone_congestion.values(), default=0.0) or 1.0

    min_x = min(zone.x for zone in zones)
    min_y = min(zone.y for zone in zones)
    max_x = max(zone.x + zone.width for zone in zones)
    max_y = max(zone.y + zone.height for zone in zones)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=CHART_DPI)
    colormap = plt.get_cmap("YlOrRd")

    for zone in zones:

        value = zone_congestion.get(zone.id, 0.0)
        intensity = value / max_value

        ax.add_patch(
            Rectangle(
                (zone.x, zone.y), zone.width, zone.height,
                facecolor=colormap(intensity), edgecolor="black",
            )
        )
        ax.text(
            zone.x + zone.width / 2, zone.y + zone.height / 2, zone.name,
            ha="center", va="center", fontsize=8,
        )

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)
    ax.set_aspect("equal")
    ax.set_title("Congestion Heatmap")

    return _save(fig, file_path)
