import json
import os

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

import joblib

from ai_training.models.base import BaseModel


# =====================================================
# Live-Compatible AI Model Training & Model Registry milestone -- Phase
# 2's own required metadata contract. Existing ai_training/ai_inference
# infrastructure (audited before writing this file -- see docs/
# architecture/live_ai_model_training.md §1) persists an estimator +
# Preprocessor + FeatureSchema (ai_training.models.base.ModelBundle,
# via BaseModel.save()/.load()) and a separate manifest.json (config +
# metrics + train/test sizes, ai_training/experiment.py's own
# ExperimentRunner.save_result()) -- but NEITHER carries a feature-
# column list, a schema version, or any deployability marker. This
# module is strictly additive: it saves a THIRD file, live_metadata.json,
# alongside the existing model.joblib/manifest.json a directory may
# already have, and never changes what BaseModel.save()/.load() or
# ExperimentRunner already do.
# =====================================================


class Deployability(Enum):

    # A model must explicitly declare one of these two -- "the metadata
    # must make it impossible to accidentally load a research-only model
    # into the future live inference path without an explicit
    # compatibility failure" (Phase 2's own requirement). Never inferred
    # from a filename, a directory name, or a model_type string.

    LIVE_COMPATIBLE = "LIVE_COMPATIBLE"
    RESEARCH_ONLY = "RESEARCH_ONLY"


@dataclass(frozen=True)
class ModelMetadata:

    # Every field Phase 2 names, at minimum, plus nothing else -- this
    # is a data contract, not a place for behavior.

    model_id: str
    model_type: str
    model_version: str
    training_timestamp: str
    training_dataset_identifier: str
    training_seed: int
    feature_schema_version: str
    ordered_feature_names: Tuple[str, ...]
    prediction_target: str
    model_deployability: Deployability
    training_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_metrics: Dict[str, Any] = field(default_factory=dict)
    missing_data_policy: str = ""

    # =====================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "training_timestamp": self.training_timestamp,
            "training_dataset_identifier": self.training_dataset_identifier,
            "training_seed": self.training_seed,
            "feature_schema_version": self.feature_schema_version,
            "ordered_feature_names": list(self.ordered_feature_names),
            "prediction_target": self.prediction_target,
            "model_deployability": self.model_deployability.value,
            "training_metrics": self.training_metrics,
            "validation_metrics": self.validation_metrics,
            "missing_data_policy": self.missing_data_policy,
        }

    # =====================================================

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":

        return cls(
            model_id=data["model_id"],
            model_type=data["model_type"],
            model_version=data["model_version"],
            training_timestamp=data["training_timestamp"],
            training_dataset_identifier=data["training_dataset_identifier"],
            training_seed=data["training_seed"],
            feature_schema_version=data["feature_schema_version"],
            ordered_feature_names=tuple(data["ordered_feature_names"]),
            prediction_target=data["prediction_target"],
            model_deployability=Deployability(data["model_deployability"]),
            training_metrics=dict(data.get("training_metrics", {})),
            validation_metrics=dict(data.get("validation_metrics", {})),
            missing_data_policy=data.get("missing_data_policy", ""),
        )


METADATA_FILENAME = "live_metadata.json"
MODEL_FILENAME = "model.joblib"


def save_live_model(model: BaseModel, metadata: ModelMetadata, directory: str) -> Dict[str, str]:

    os.makedirs(directory, exist_ok=True)

    model_path = os.path.join(directory, MODEL_FILENAME)
    metadata_path = os.path.join(directory, METADATA_FILENAME)

    model.save(model_path)

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata.to_dict(), handle, indent=2, sort_keys=True)

    return {"model": model_path, "metadata": metadata_path}


def load_live_metadata(directory: str) -> ModelMetadata:

    metadata_path = os.path.join(directory, METADATA_FILENAME)

    with open(metadata_path, "r", encoding="utf-8") as handle:
        return ModelMetadata.from_dict(json.load(handle))


def load_live_model_bundle(directory: str):

    # Returns the raw ModelBundle (estimator/preprocessor/feature_schema/
    # target_name/...) rather than a concrete BaseModel subclass instance
    # -- the caller (ModelRegistry) is the one place that knows which
    # concrete class (EvacuationTimeModel/BottleneckModel) a given
    # model_type maps back onto.

    return joblib.load(os.path.join(directory, MODEL_FILENAME))
