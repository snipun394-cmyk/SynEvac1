from typing import Any, Dict, Optional, Sequence

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


# =====================================================
# Phase 3 -- evaluation metrics. Every function here is a pure
# function of (y_true, y_pred[, y_proba]) -- no model/dataset
# dependency, so evaluation.py and experiment.py both share this one
# implementation rather than recomputing metrics ad hoc.
# =====================================================


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> Dict[str, float]:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


# =====================================================


def multioutput_regression_metrics(
    y_true: Sequence[Sequence[float]], y_pred: Sequence[Sequence[float]], output_names: Sequence[str],
) -> Dict[str, Any]:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    per_output = {
        name: regression_metrics(y_true[:, index], y_pred[:, index])
        for index, name in enumerate(output_names)
    }

    macro_average = {
        key: float(np.nanmean([entry[key] for entry in per_output.values()]))
        for key in ("mae", "rmse", "r2")
    }

    return {"per_output": per_output, "macro_average": macro_average}


# =====================================================


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_proba: Optional[np.ndarray] = None,
    *,
    average: str = "macro",
) -> Dict[str, Any]:

    # dtype left for numpy to infer (matching BaseModel._prepare_
    # targets_for_fit()'s own choice) rather than forced to
    # dtype=object -- an object-dtype array of plain bools/strings
    # isn't recognized by sklearn's type_of_target() the same way a
    # native bool/string array is, which otherwise makes every metric
    # below raise even though y_true/y_pred describe the exact same
    # label set.
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "roc_auc": None,
    }

    if y_proba is not None:
        metrics["roc_auc"] = _safe_roc_auc(y_true, y_proba)

    return metrics


# =====================================================


def _safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> Optional[float]:

    # ROC AUC is undefined with fewer than two classes actually present
    # in y_true (the tiny campaigns ai_training's own tests run against
    # can easily produce this) -- returned as None rather than raising,
    # since Phase 3 only asks for it "where applicable".

    classes = np.unique(y_true)

    if len(classes) < 2:
        return None

    y_proba = np.asarray(y_proba)

    try:

        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))

        if y_proba.ndim == 2 and y_proba.shape[1] > 2:
            return float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=classes)
            )

        return float(roc_auc_score(y_true, y_proba))

    except ValueError:

        return None
