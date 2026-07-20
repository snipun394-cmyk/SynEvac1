from dataclasses import dataclass
from typing import Optional

from ai_inference.loader import LoadedModel, ModelProvenance

from ai_features.building_state_extractor import extract_canonical_features
from ai_features.compatibility import IncompatibleFeatureRowError, validate_feature_row
from ai_features.feature_schema import SCHEMA_VERSION

from building_state.models import BuildingState

from ai_registry.registry import ModelRegistry
from ai_registry.uncertainty import regression_ensemble_uncertainty


# =====================================================
# Phase 11's safe inference service -- BuildingState -> canonical
# extractor -> compatibility check -> registry lookup -> model ->
# structured prediction. Deliberately NOT wired into LiveOrchestrator
# anywhere (no import of live_system appears in this file or this
# package -- enforced by tests/test_ai_registry.py's own dependency-
# direction guard, matching every earlier milestone's convention).
#
# Every failure mode Phase 11 names raises InferenceUnavailableError --
# never a fabricated prediction, never a silent zero:
#   - no compatible model exists           -> registry returns None
#   - required features are unavailable    -> validate_feature_row() raises
#   - schema version mismatches            -> registry filters it out
#   - model artifact is corrupted          -> predict() exception is wrapped
#   - model is RESEARCH_ONLY               -> registry never returns one
# =====================================================


class InferenceUnavailableError(Exception):

    pass


@dataclass(frozen=True)
class EvacuationTimePrediction:

    predicted_seconds: float
    uncertainty_seconds: Optional[float]  # None means genuinely unavailable, never 0.0
    model_id: str
    model_version: str
    feature_schema_version: str
    timestamp: float


@dataclass(frozen=True)
class BottleneckOccurrencePrediction:

    probability: float
    predicted_occurrence: bool
    threshold: float
    model_id: str
    model_version: str
    feature_schema_version: str
    timestamp: float


# =====================================================


class LiveAIInferenceService:

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        bottleneck_threshold: float = 0.5,
        feature_schema_version: str = SCHEMA_VERSION,
    ):

        self._registry = registry
        self._bottleneck_threshold = bottleneck_threshold
        self._feature_schema_version = feature_schema_version

    # =====================================================

    def predict_evacuation_time(self, state: BuildingState, *, timestamp: float) -> EvacuationTimePrediction:

        row = extract_canonical_features(state)

        model, metadata = self._require_compatible_model("evacuation_time")
        self._validate_row(row, model, metadata)

        try:
            predicted = float(model.predict([row])[0])
        except Exception as exc:  # a corrupted/broken estimator must fail safely, not crash the caller
            raise InferenceUnavailableError(
                f"evacuation_time model {metadata.model_id!r} raised during predict(): {exc}",
            ) from exc

        uncertainty = regression_ensemble_uncertainty(model, [row])

        return EvacuationTimePrediction(
            predicted_seconds=predicted,
            uncertainty_seconds=uncertainty[0] if uncertainty else None,
            model_id=metadata.model_id,
            model_version=metadata.model_version,
            feature_schema_version=metadata.feature_schema_version,
            timestamp=timestamp,
        )

    # =====================================================

    def predict_bottleneck_occurrence(self, state: BuildingState, *, timestamp: float) -> BottleneckOccurrencePrediction:

        row = extract_canonical_features(state)

        model, metadata = self._require_compatible_model("bottleneck_occurrence")
        self._validate_row(row, model, metadata)

        try:

            proba = model.predict_proba([row])[0]
            classes = list(model.label_encoder.classes_)
            positive_index = classes.index(True) if True in classes else int(len(classes) > 1)
            probability = float(proba[positive_index])

        except Exception as exc:

            raise InferenceUnavailableError(
                f"bottleneck_occurrence model {metadata.model_id!r} raised during predict_proba(): {exc}",
            ) from exc

        return BottleneckOccurrencePrediction(
            probability=probability,
            predicted_occurrence=probability >= self._bottleneck_threshold,
            threshold=self._bottleneck_threshold,
            model_id=metadata.model_id,
            model_version=metadata.model_version,
            feature_schema_version=metadata.feature_schema_version,
            timestamp=timestamp,
        )

    # =====================================================

    def _require_compatible_model(self, model_type: str):

        result = self._registry.get_latest_compatible_model(
            model_type, feature_schema_version=self._feature_schema_version,
        )

        if result is None:

            raise InferenceUnavailableError(
                f"No LIVE_COMPATIBLE {model_type!r} model compatible with feature schema "
                f"version {self._feature_schema_version!r} is registered.",
            )

        return result

    # =====================================================

    def _validate_row(self, row, model, metadata) -> None:

        loaded = LoadedModel(
            model=model,
            provenance=ModelProvenance(
                experiment_name=metadata.model_id, model_name=metadata.model_type,
                algorithm="unknown", model_version=metadata.model_version,
                generated_at=metadata.training_timestamp, train_size=0, test_size=0,
                metrics={"feature_schema_version": metadata.feature_schema_version},
            ),
            directory="in-memory",
        )

        try:
            validate_feature_row(row, loaded)
        except IncompatibleFeatureRowError as exc:
            raise InferenceUnavailableError(str(exc)) from exc
