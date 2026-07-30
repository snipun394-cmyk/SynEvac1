from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc


# =====================================================
# Prediction vs Reality Evaluation Framework milestone, Phase 4 --
# classification metrics. Pure functions over plain (y_true, y_pred,
# y_proba) lists -- never coupled to any specific model type (Phase 9's
# own "compare multiple models" requirement needs this genericity: two
# different models' predictions reduce to the SAME plain lists via
# pairs.py, then flow through the identical functions here). Reuses
# sklearn.metrics directly (already an established dependency --
# research_framework/figures.py, ai_training/, ai_registry/ all already
# depend on it) rather than reimplementing precision/recall/confusion-
# matrix arithmetic by hand.
# =====================================================


@dataclass(frozen=True)
class ClassificationMetrics:

    sample_count: int

    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    # (tn, fp, fn, tp) -- always this fixed order, regardless of class
    # balance in the sample (sklearn's own confusion_matrix with an
    # explicit labels=[False, True] argument, never inferred from
    # whichever classes happen to appear).
    confusion_matrix: Optional[Tuple[int, int, int, int]] = None

    roc_auc: Optional[float] = None
    roc_fpr: Tuple[float, ...] = field(default_factory=tuple)
    roc_tpr: Tuple[float, ...] = field(default_factory=tuple)

    # Calibration error -- mean absolute difference between predicted
    # probability and empirical outcome frequency, bucketed into 10
    # equal-width probability bins (a standard, disclosed reliability-
    # diagram bucketing, not a fabricated precision claim). None when
    # fewer than 2 buckets have any samples (not enough spread to say
    # anything honest about calibration).
    calibration_error: Optional[float] = None
    calibration_bins: Tuple["CalibrationBin", ...] = field(default_factory=tuple)

    # Mean predicted probability minus empirical positive rate -- a
    # signed bias indicator (positive = systematically over-confident
    # toward "occurs", negative = under-confident).
    confidence_bias: Optional[float] = None


@dataclass(frozen=True)
class CalibrationBin:

    bucket_low: float
    bucket_high: float
    sample_count: int
    mean_predicted_probability: Optional[float] = None
    observed_positive_rate: Optional[float] = None


def compute_classification_metrics(
    y_true: Sequence[bool], y_pred: Sequence[bool], y_proba: Sequence[float] = (),
) -> ClassificationMetrics:

    sample_count = len(y_true)

    if sample_count == 0:
        return ClassificationMetrics(sample_count=0)

    y_true_arr = np.asarray(y_true, dtype=bool)
    y_pred_arr = np.asarray(y_pred, dtype=bool)

    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0.0)
    recall = recall_score(y_true_arr, y_pred_arr, zero_division=0.0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0.0)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[False, True]).ravel()

    roc_auc = None
    roc_fpr: Tuple[float, ...] = ()
    roc_tpr: Tuple[float, ...] = ()
    calibration_error = None
    calibration_bins: Tuple[CalibrationBin, ...] = ()
    confidence_bias = None

    if y_proba and len(y_proba) == sample_count:

        y_proba_arr = np.asarray(y_proba, dtype=float)

        # ROC/AUC needs both classes present -- otherwise honestly None,
        # never a fabricated 0.5/1.0.
        if len(set(y_true_arr.tolist())) == 2:

            fpr, tpr, _thresholds = roc_curve(y_true_arr, y_proba_arr)
            roc_auc = float(auc(fpr, tpr))
            roc_fpr = tuple(float(v) for v in fpr)
            roc_tpr = tuple(float(v) for v in tpr)

        calibration_bins = _calibration_bins(y_true_arr, y_proba_arr)

        populated_bins = [b for b in calibration_bins if b.sample_count > 0]

        if len(populated_bins) >= 2:

            calibration_error = float(np.mean([
                abs(b.mean_predicted_probability - b.observed_positive_rate) for b in populated_bins
            ]))

        confidence_bias = float(np.mean(y_proba_arr) - np.mean(y_true_arr))

    return ClassificationMetrics(
        sample_count=sample_count,
        precision=float(precision), recall=float(recall), f1=float(f1),
        confusion_matrix=(int(tn), int(fp), int(fn), int(tp)),
        roc_auc=roc_auc, roc_fpr=roc_fpr, roc_tpr=roc_tpr,
        calibration_error=calibration_error, calibration_bins=calibration_bins,
        confidence_bias=confidence_bias,
    )


def _calibration_bins(y_true: np.ndarray, y_proba: np.ndarray, bucket_count: int = 10) -> Tuple[CalibrationBin, ...]:

    edges = np.linspace(0.0, 1.0, bucket_count + 1)
    bins = []

    for i in range(bucket_count):

        low, high = edges[i], edges[i + 1]

        # Inclusive on the final bucket's upper edge (probability == 1.0
        # must land somewhere), exclusive elsewhere -- standard
        # reliability-diagram convention.
        if i == bucket_count - 1:
            mask = (y_proba >= low) & (y_proba <= high)
        else:
            mask = (y_proba >= low) & (y_proba < high)

        count = int(mask.sum())

        if count == 0:
            bins.append(CalibrationBin(bucket_low=float(low), bucket_high=float(high), sample_count=0))
            continue

        bins.append(CalibrationBin(
            bucket_low=float(low), bucket_high=float(high), sample_count=count,
            mean_predicted_probability=float(y_proba[mask].mean()),
            observed_positive_rate=float(y_true[mask].mean()),
        ))

    return tuple(bins)
