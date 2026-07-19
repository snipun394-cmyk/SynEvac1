from typing import Any, Dict, Sequence

from ai_training.metrics import (
    classification_metrics,
    multioutput_regression_metrics,
    regression_metrics,
)
from ai_training.models.base import BaseModel


# =====================================================
# Phase 3 -- evaluation. Each function fits a model's own predict()
# (and, for classification, predict_proba()) against held-out rows and
# reports the metrics Phase 3 asks for. No metric math lives here --
# that's entirely ai_training.metrics; this module only wires a model
# + data to it.
# =====================================================


def evaluate_regression(model: BaseModel, X_rows: Sequence[Any], y_true: Sequence[float]) -> Dict[str, float]:

    y_pred = model.predict(X_rows)

    return regression_metrics(y_true, y_pred)


# =====================================================


def evaluate_multioutput_regression(
    model: BaseModel, X_rows: Sequence[Any], y_true: Sequence[Sequence[float]], output_names: Sequence[str],
) -> Dict[str, Any]:

    predictions = model.predict(X_rows)  # List[Dict[output_name, value]], per ExitUsageModel.predict()
    y_pred = [[row[name] for name in output_names] for row in predictions]

    return multioutput_regression_metrics(y_true, y_pred, output_names)


# =====================================================


def evaluate_classification(model: BaseModel, X_rows: Sequence[Any], y_true: Sequence[Any]) -> Dict[str, Any]:

    y_pred = model.predict(X_rows)

    y_proba = None

    if model.is_classifier:

        try:
            y_proba = model.predict_proba(X_rows)
        except AttributeError:
            y_proba = None  # the underlying estimator doesn't implement predict_proba

    return classification_metrics(y_true, y_pred, y_proba)
