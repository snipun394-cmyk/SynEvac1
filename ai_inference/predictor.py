from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ai_inference.cache import PredictionCache
from ai_inference.loader import LoadedModel, load_model


# =====================================================
# Phase 2 -- Prediction Engine. Accepts ONE Scenario Feature dict (the
# same Dict[str, Any] row shape ai_training's own X_rows use) and
# produces predictions from already-loaded, already-fitted models --
# every call here is exactly one model.predict()/predict_proba(), no
# simulation, no training, ever.
# =====================================================


def prediction_type_key(loaded_model: LoadedModel) -> str:

    # "bottleneck" alone is ambiguous (occurrence vs. location are two
    # different trained models sharing one model_name) -- every other
    # model_name is already unambiguous as its own prediction_type.

    if loaded_model.provenance.model_name == "bottleneck":
        return f"bottleneck_{loaded_model.model.target}"

    return loaded_model.provenance.model_name


# =====================================================


@dataclass(frozen=True)
class Prediction:

    prediction_type: str
    value: Any
    probability: Optional[float]  # max predict_proba() probability, classifiers only; else None
    model_name: str
    algorithm: str
    model_version: str

    def to_dict(self) -> Dict[str, Any]:

        return {
            "prediction_type": self.prediction_type,
            "value": self.value,
            "probability": self.probability,
            "model_name": self.model_name,
            "algorithm": self.algorithm,
            "model_version": self.model_version,
        }


# =====================================================


class Predictor:

    def __init__(self, loaded_models: Sequence[LoadedModel], *, cache: Optional[PredictionCache] = None):

        self._models: Dict[str, LoadedModel] = {}

        for loaded_model in loaded_models:
            self._models[prediction_type_key(loaded_model)] = loaded_model

        self._cache = cache

    # =====================================================

    @classmethod
    def from_directories(cls, directories: Sequence[str], *, cache: Optional[PredictionCache] = None, **load_kwargs) -> "Predictor":

        loaded_models = [load_model(directory, **load_kwargs) for directory in directories]

        return cls(loaded_models, cache=cache)

    # =====================================================

    def available_predictions(self) -> List[str]:

        return sorted(self._models.keys())

    # =====================================================

    def loaded_model(self, prediction_type: str) -> LoadedModel:

        return self._get_loaded(prediction_type)

    # =====================================================

    def predict(self, prediction_type: str, X_row: Dict[str, Any]) -> Prediction:

        loaded_model = self._get_loaded(prediction_type)

        if self._cache is not None:

            value, probability = self._cache.get_or_compute(
                loaded_model.provenance.model_version, X_row, prediction_type,
                lambda: _compute_value_and_probability(loaded_model, X_row),
            )

        else:

            value, probability = _compute_value_and_probability(loaded_model, X_row)

        return Prediction(
            prediction_type=prediction_type,
            value=value,
            probability=probability,
            model_name=loaded_model.provenance.model_name,
            algorithm=loaded_model.provenance.algorithm,
            model_version=loaded_model.provenance.model_version,
        )

    # =====================================================

    def predict_all(self, X_row: Dict[str, Any]) -> Dict[str, Prediction]:

        return {
            prediction_type: self.predict(prediction_type, X_row)
            for prediction_type in self._models
        }

    # =====================================================

    def _get_loaded(self, prediction_type: str) -> LoadedModel:

        if prediction_type not in self._models:

            raise KeyError(
                f"No model loaded for prediction_type={prediction_type!r} -- "
                f"available: {self.available_predictions()}"
            )

        return self._models[prediction_type]


# =====================================================


def _compute_value_and_probability(loaded_model: LoadedModel, X_row: Dict[str, Any]):

    model = loaded_model.model
    value = model.predict([X_row])[0]

    if not model.is_classifier:
        return value, None

    try:
        probabilities = model.predict_proba([X_row])
    except (TypeError, AttributeError):
        return value, None

    return value, float(np.max(probabilities[0]))
