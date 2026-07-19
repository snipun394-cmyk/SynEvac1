from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ai_training.models.base import BaseModel

from ai_explainability.feature_importance import extract_raw_importances


# =====================================================
# Phase 2 -- Prediction Reports. Explains ONE already-computed
# prediction from an already-fitted model: what it predicted, how
# confident (classification only), which raw input features
# contributed most, and what was fed in/what target it describes.
# Built as a plain, JSON-serializable dataclass (.to_dict()) so it can
# be handed to a future Command Center dashboard without adaptation.
# =====================================================


@dataclass(frozen=True)
class PredictionReport:

    prediction: Any
    confidence: Optional[float]
    top_contributing_features: Tuple[Tuple[str, float], ...]
    input_summary: Dict[str, Any]
    target_summary: Dict[str, Any]
    scenario_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:

        return {
            "scenario_id": self.scenario_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "top_contributing_features": [
                {"feature": name, "contribution": value}
                for name, value in self.top_contributing_features
            ],
            "input_summary": dict(self.input_summary),
            "target_summary": dict(self.target_summary),
        }


# =====================================================


def explain_prediction(
    model: BaseModel,
    X_row: Dict[str, Any],
    *,
    scenario_id: Optional[str] = None,
    y_true: Optional[Any] = None,
    top_n: int = 5,
) -> PredictionReport:

    if model.preprocessor is None:
        raise RuntimeError(f"{type(model).__name__} must be fit before explaining a prediction")

    prediction = _single_prediction(model, X_row)
    confidence = _single_confidence(model, X_row)
    contributions = tuple(_local_feature_contributions(model, X_row, top_n=top_n))

    target_summary: Dict[str, Any] = {
        "target_name": model.target_name,
        "task": "classification" if model.is_classifier else "regression",
    }

    if model.output_names is not None:
        target_summary["outputs"] = list(model.output_names)

    if y_true is not None:
        target_summary["true_value"] = y_true

    return PredictionReport(
        scenario_id=scenario_id,
        prediction=prediction,
        confidence=confidence,
        top_contributing_features=contributions,
        input_summary=dict(X_row),
        target_summary=target_summary,
    )


# =====================================================


def _single_prediction(model: BaseModel, X_row: Dict[str, Any]) -> Any:

    return model.predict([X_row])[0]


# =====================================================


def _single_confidence(model: BaseModel, X_row: Dict[str, Any]) -> Optional[float]:

    if not model.is_classifier:
        return None

    try:
        probabilities = model.predict_proba([X_row])
    except (TypeError, AttributeError):
        return None

    return float(np.max(probabilities[0]))


# =====================================================


def _local_feature_contributions(
    model: BaseModel, X_row: Dict[str, Any], *, top_n: int,
) -> List[Tuple[str, float]]:

    # A lightweight, dependency-free local attribution (not a
    # substitute for SHAP/LIME, but deterministic and requires no new
    # dependency): each transformed feature's global impurity-based
    # importance, weighted by how far THIS instance's own
    # (standardized/one-hot) value sits from zero -- a feature this
    # model considers important AND that this specific row expresses
    # strongly ranks above one that's important in general but neutral
    # for this particular scenario.

    names = model.preprocessor.feature_names_out()
    importances = extract_raw_importances(model.estimator)
    transformed = model.preprocessor.transform([X_row])[0]

    contributions: Dict[str, float] = {}

    for name, importance, value in zip(names, importances, transformed):

        original_column = name.split("=", 1)[0]
        contributions[original_column] = (
            contributions.get(original_column, 0.0) + float(importance) * abs(float(value))
        )

    ordered = sorted(contributions.items(), key=lambda item: (-item[1], item[0]))

    return ordered[:top_n]
