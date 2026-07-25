from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 5 --
# class imbalance handling. Overall positive rate is 14.8% (Door 30.1%
# / Exit 5.4% / Stair 2.1%, per docs/architecture/
# predictive_dataset_campaign_v1.md Section 6) -- meaningfully
# imbalanced, worse per candidate type.
#
# Strategy used: class weighting (sample reweighting) + validation-set
# threshold tuning, ONLY. Deliberately NOT oversampling (SMOTE or
# similar): synthetic minority-class rows built by interpolating
# between real feature vectors have never been validated against this
# dataset's own leakage boundary (predictive_dataset.target_generator's
# strict "target_generator is the only module allowed to see the
# future" rule) -- an interpolated synthetic row blends two real rows'
# feature values with no guarantee the result is a physically
# consistent, non-leaking observation. Reweighting real rows changes
# their loss contribution without ever fabricating a new one, which is
# a strictly safer choice for a milestone whose own charter is
# "verify leakage boundary", not "engineer around it".
# =====================================================


def compute_class_weight_map(y: np.ndarray) -> Dict[int, float]:

    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def sample_weights_from_class_weight(y: np.ndarray, class_weight_map: Dict[int, float]) -> np.ndarray:

    return np.array([class_weight_map[int(label)] for label in y], dtype=float)


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric: str = "f1",
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[float, float]:
    """Search candidate thresholds and return (best_threshold, best_metric_value)
    maximizing `metric` ("f1" or "balanced_accuracy"). MUST be called on
    a VALIDATION split only -- never on test, and never on train (Phase
    5's "threshold tuning" is a validation-set-only step; tuning on test
    would leak test-set information into a model decision)."""

    if metric not in ("f1", "balanced_accuracy"):
        raise ValueError(f"Unsupported metric {metric!r}, expected 'f1' or 'balanced_accuracy'.")

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    metric_fn: Callable[[np.ndarray, np.ndarray], float]
    if metric == "f1":
        metric_fn = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    else:
        metric_fn = balanced_accuracy_score

    best_threshold = 0.5
    best_value = -1.0

    for threshold in thresholds:

        y_pred = (y_prob >= threshold).astype(int)
        value = metric_fn(y_true, y_pred)

        if value > best_value:
            best_value = float(value)
            best_threshold = float(threshold)

    return best_threshold, best_value
