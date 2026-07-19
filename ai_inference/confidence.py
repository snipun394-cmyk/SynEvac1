from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from ai_inference.ensemble import Ensemble
from ai_inference.loader import LoadedModel


# =====================================================
# Phase 4 -- Confidence Estimation. Every confidence score below comes
# from something a fitted model/ensemble can honestly report --
# classification uses the model's own predict_proba(); regression uses
# a Random Forest's own inter-tree prediction spread (a real,
# already-fitted ensemble-of-trees, not anything invented for this
# purpose); an Ensemble uses member agreement/dispersion. Any
# combination this module cannot honestly derive a signal from reports
# confidence=None -- never a fabricated placeholder number.
# =====================================================


@dataclass(frozen=True)
class ConfidenceReport:

    prediction: Any
    confidence: Optional[float]
    confidence_basis: str
    model_used: str
    training_dataset_version: Optional[str]
    experiment_id: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:

        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "confidence_basis": self.confidence_basis,
            "model_used": self.model_used,
            "training_dataset_version": self.training_dataset_version,
            "experiment_id": self.experiment_id,
            "model_version": self.model_version,
        }


# =====================================================
# Single-model confidence.
# =====================================================


def estimate_confidence(loaded_model: LoadedModel, X_row: Dict[str, Any]) -> Tuple[Optional[float], str]:

    model = loaded_model.model

    if model.is_classifier:

        try:
            probabilities = model.predict_proba([X_row])
        except (TypeError, AttributeError):
            return None, "unavailable_for_algorithm"

        return float(np.max(probabilities[0])), "predict_proba"

    X_transformed = model.preprocessor.transform([X_row])

    return _regression_confidence(model.estimator, X_transformed)


# =====================================================


def build_confidence_report(loaded_model: LoadedModel, X_row: Dict[str, Any]) -> ConfidenceReport:

    prediction = loaded_model.model.predict([X_row])[0]
    confidence, basis = estimate_confidence(loaded_model, X_row)

    return ConfidenceReport(
        prediction=prediction,
        confidence=confidence,
        confidence_basis=basis,
        model_used=f"{loaded_model.provenance.model_name} ({loaded_model.provenance.algorithm})",
        training_dataset_version=loaded_model.provenance.dataset_version,
        experiment_id=loaded_model.provenance.experiment_name,
        model_version=loaded_model.provenance.model_version,
    )


# =====================================================
# Regression confidence -- Random Forest inter-tree dispersion only.
# =====================================================


def _random_forest_dispersion(estimator: Any, X_row_transformed: np.ndarray) -> Optional[float]:

    # A plain RandomForestRegressor's individual trees are genuinely
    # independent predictors of the same target (bagging) -- the
    # standard deviation of their per-row predictions is a real,
    # already-available uncertainty signal. Gradient Boosting/XGBoost
    # trees are sequential *corrections*, not independent predictors,
    # so no equivalent honest signal exists for them here -- None,
    # deliberately, rather than approximating one.

    def _single_forest_std(forest: Any) -> Optional[float]:

        if not isinstance(forest, RandomForestRegressor):
            return None

        tree_predictions = [tree.predict(X_row_transformed)[0] for tree in forest.estimators_]

        return float(np.std(tree_predictions))

    if isinstance(estimator, RandomForestRegressor):
        return _single_forest_std(estimator)

    if hasattr(estimator, "estimators_"):

        # MultiOutputRegressor wrapping one RandomForestRegressor per
        # output (ExitUsageModel) -- averaged across outputs.
        stds = [_single_forest_std(sub) for sub in estimator.estimators_]

        if any(std is None for std in stds):
            return None

        return float(np.mean(stds))

    return None


# =====================================================


def _regression_confidence(estimator: Any, X_row_transformed: np.ndarray) -> Tuple[Optional[float], str]:

    std = _random_forest_dispersion(estimator, X_row_transformed)

    if std is None:
        return None, "unavailable_for_algorithm"

    prediction = np.asarray(estimator.predict(X_row_transformed))
    magnitude = max(float(np.mean(np.abs(prediction))), 1e-9)

    # Coefficient-of-variation-based score: spread relative to the
    # prediction's own magnitude, clamped to [0, 1] so it reads like a
    # confidence (low relative spread -> high confidence). An
    # interpretable derivation of a real signal, not an invented one --
    # NOT a calibrated probability the way predict_proba is.
    coefficient_of_variation = std / magnitude
    confidence = max(0.0, min(1.0, 1.0 - coefficient_of_variation))

    return confidence, "random_forest_tree_dispersion"


# =====================================================
# Ensemble confidence -- member agreement/dispersion.
# =====================================================


def build_ensemble_confidence_report(ensemble: Ensemble, X_row: Dict[str, Any]) -> ConfidenceReport:

    if ensemble.is_classifier:
        prediction, confidence = _ensemble_classification_confidence(ensemble, X_row)
        basis = "ensemble_vote_agreement"
    else:
        prediction, confidence = _ensemble_regression_confidence(ensemble, X_row)
        basis = "ensemble_prediction_dispersion"

    experiment_ids = ",".join(
        member.loaded_model.provenance.experiment_name for member in ensemble.members
    )
    model_versions = ",".join(
        member.loaded_model.provenance.model_version for member in ensemble.members
    )

    dataset_versions = {member.loaded_model.provenance.dataset_version for member in ensemble.members}
    dataset_version = (
        next(iter(dataset_versions)) if len(dataset_versions) == 1
        else ",".join(str(version) for version in dataset_versions)
    )

    return ConfidenceReport(
        prediction=prediction,
        confidence=confidence,
        confidence_basis=basis,
        model_used=f"ensemble({len(ensemble.members)} members: {ensemble.prediction_type})",
        training_dataset_version=dataset_version,
        experiment_id=experiment_ids,
        model_version=model_versions,
    )


# =====================================================


def _ensemble_classification_confidence(ensemble: Ensemble, X_row: Dict[str, Any]) -> Tuple[Any, float]:

    winner = ensemble.predict_majority_vote(X_row)

    predictions = [member.loaded_model.model.predict([X_row])[0] for member in ensemble.members]
    weights = [member.weight for member in ensemble.members]

    total_weight = sum(weights)
    winning_weight = sum(weight for prediction, weight in zip(predictions, weights) if prediction == winner)

    agreement = (winning_weight / total_weight) if total_weight else 0.0

    return winner, agreement


# =====================================================


def _ensemble_regression_confidence(ensemble: Ensemble, X_row: Dict[str, Any]) -> Tuple[Any, float]:

    prediction = ensemble.predict_average(X_row)
    raw_values = [member.loaded_model.model.predict([X_row])[0] for member in ensemble.members]

    if isinstance(raw_values[0], dict):

        coefficients_of_variation = []

        for key in raw_values[0].keys():

            values = np.array([value[key] for value in raw_values], dtype=float)
            magnitude = max(abs(float(values.mean())), 1e-9)
            coefficients_of_variation.append(float(values.std()) / magnitude)

        coefficient_of_variation = float(np.mean(coefficients_of_variation))

    else:

        values = np.array(raw_values, dtype=float)
        magnitude = max(abs(float(values.mean())), 1e-9)
        coefficient_of_variation = float(values.std()) / magnitude

    confidence = max(0.0, min(1.0, 1.0 - coefficient_of_variation))

    return prediction, confidence
