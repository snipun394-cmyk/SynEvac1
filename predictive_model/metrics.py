from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 6 --
# ALL metrics this milestone reports go through this one module, so
# "how is precision computed" has exactly one answer used everywhere
# (overall, per-horizon, per-candidate-type, before/after calibration).
# =====================================================


@dataclass(frozen=True)
class ClassificationMetrics:

    n: int
    positive_rate: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    brier_score: float
    threshold: float
    confusion_matrix: Dict[str, int]
    calibration_curve: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "positive_rate": self.positive_rate,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "brier_score": self.brier_score,
            "threshold": self.threshold,
            "confusion_matrix": self.confusion_matrix,
            "calibration_curve": self.calibration_curve,
        }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5, n_calibration_bins: int = 10) -> ClassificationMetrics:

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true)
    n_pos = int(y_true.sum())

    # ROC-AUC/PR-AUC are undefined with only one class present -- can
    # happen on small slices (e.g. a rare candidate-type/horizon combo).
    if n_pos == 0 or n_pos == n:
        roc_auc = None
        pr_auc = None
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    if n_pos == 0 or n_pos == n:
        fraction_of_positives, mean_predicted_value = np.array([]), np.array([])
    else:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_calibration_bins, strategy="uniform",
        )

    return ClassificationMetrics(
        n=n,
        positive_rate=float(n_pos / n) if n else 0.0,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        threshold=threshold,
        confusion_matrix={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        calibration_curve={
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        },
    )


def metrics_by_group(
    y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray, *, threshold: float = 0.5,
) -> Dict[str, Any]:
    """compute_metrics(), sliced by an arbitrary group array (e.g.
    candidate_type). Groups with too few rows or a single class still
    get precision/recall/etc (well-defined for any confusion matrix)
    but roc_auc/pr_auc are None (undefined for one class)."""

    groups = np.asarray(groups)
    result = {}

    for group_value in sorted(set(groups.tolist())):

        mask = groups == group_value
        result[str(group_value)] = compute_metrics(y_true[mask], y_prob[mask], threshold=threshold).to_dict()

    return result
