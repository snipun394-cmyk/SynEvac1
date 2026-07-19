import os

from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless -- this package never opens an interactive window

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

from ai_explainability.benchmark import BenchmarkResult, flatten_metrics
from ai_explainability.feature_importance import RankedFeature


# =====================================================
# Phase 5 -- Visualizations. Every function renders one chart to a
# given file path (creating parent directories as needed) and returns
# that path -- never displays a window, never mutates the data it's
# given.
# =====================================================


def _ensure_parent_dir(path: str) -> None:

    parent = os.path.dirname(path)

    if parent:
        os.makedirs(parent, exist_ok=True)


# =====================================================


def plot_feature_importance(
    ranked_features: Sequence[RankedFeature], path: str, *, title: str = "Feature Importance", top_n: int = 15,
) -> str:

    _ensure_parent_dir(path)

    subset = list(ranked_features)[:top_n]
    names = [feature.feature for feature in subset][::-1]
    values = [feature.importance for feature in subset][::-1]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names) + 1)))
    ax.barh(names, values, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


# =====================================================


def plot_prediction_comparison(
    y_true: Sequence[float], y_pred: Sequence[float], path: str, *, title: str = "Predicted vs Actual",
) -> str:

    _ensure_parent_dir(path)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_arr, y_pred_arr, alpha=0.6, color="#4C72B0")

    lo = float(min(y_true_arr.min(), y_pred_arr.min()))
    hi = float(max(y_true_arr.max(), y_pred_arr.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


# =====================================================


def plot_model_comparison(
    benchmark_results: Sequence[BenchmarkResult], metric_key: str, path: str, *, title: Optional[str] = None,
) -> str:

    _ensure_parent_dir(path)

    labels = [result.label for result in benchmark_results]
    values = [flatten_metrics(result.metrics)[metric_key] for result in benchmark_results]

    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 1.4), 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel(metric_key)
    ax.set_title(title or f"Model Comparison -- {metric_key}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


# =====================================================


def render_benchmark_table_image(
    rows: Sequence[Dict[str, Any]], path: str, *, title: str = "Benchmark",
) -> str:

    _ensure_parent_dir(path)

    columns = list(rows[0].keys()) if rows else []
    cell_text = [[_format_cell(row[column]) for column in columns] for row in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(columns) * 1.6), 1 + 0.5 * max(1, len(rows))))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


def _format_cell(value: Any) -> str:

    if isinstance(value, float):
        return f"{value:.4f}"

    return str(value)


# =====================================================


def plot_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    path: str,
    *,
    labels: Optional[List[Any]] = None,
    title: str = "Confusion Matrix",
) -> str:

    _ensure_parent_dir(path)

    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred), key=str)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha="right")
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            ax.text(col_index, row_index, str(matrix[row_index, col_index]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path


# =====================================================


def plot_residuals(
    y_true: Sequence[float], y_pred: Sequence[float], path: str, *, title: str = "Residual Plot",
) -> str:

    _ensure_parent_dir(path)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    residuals = y_true_arr - y_pred_arr

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred_arr, residuals, alpha=0.6, color="#4C72B0")
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    return path
