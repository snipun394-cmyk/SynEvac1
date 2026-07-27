import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import joblib


# =====================================================
# Localized Predictive Model V2 milestone, Phase 18 -- export. V1's
# model_export.py (ModelMetadata/export_model) is REUSED UNCHANGED for
# the model.joblib + model_metadata.json pair (same shape V1 produced).
# This module is ADDITIVE: it exports the extra artifacts/fields this
# milestone's own Phase 18 explicitly asks for that V1's frozen
# ModelMetadata dataclass has no field for -- training seed, split
# strategy, algorithm name/hyperparameters, calibration method, feature
# order, candidate types supported, known limitations, git commit -- and
# the calibration artifact itself (V1 never exported a calibrator; this
# milestone's own Phase 11 findings say raw probabilities should not be
# trusted, so a future consumer of this model needs the calibrator, not
# just the model). Kept as a SEPARATE file (model_metadata_v2.json)
# rather than changing model_metadata.json's shape, so nothing that
# already reads V1's export format breaks.
# =====================================================


@dataclass(frozen=True)
class ExtendedModelMetadataV2:

    model_version: str
    dataset_campaign_version: str
    dataset_schema_version: str
    dataset_feature_version: str
    dataset_target_version: str
    prediction_horizon_seconds: float
    training_seed: int
    split_strategy: str
    algorithm: str
    algorithm_library: str
    hyperparameters: Dict[str, Any]
    calibration_method: str
    calibration_artifact: str
    feature_order: Tuple[str, ...]
    candidate_types_supported: Tuple[str, ...]
    decision_threshold: float
    known_limitations: Tuple[str, ...]
    git_commit: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dataset_campaign_version": self.dataset_campaign_version,
            "dataset_schema_version": self.dataset_schema_version,
            "dataset_feature_version": self.dataset_feature_version,
            "dataset_target_version": self.dataset_target_version,
            "prediction_horizon_seconds": self.prediction_horizon_seconds,
            "training_seed": self.training_seed,
            "split_strategy": self.split_strategy,
            "algorithm": self.algorithm,
            "algorithm_library": self.algorithm_library,
            "hyperparameters": self.hyperparameters,
            "calibration_method": self.calibration_method,
            "calibration_artifact": self.calibration_artifact,
            "feature_order": list(self.feature_order),
            "candidate_types_supported": list(self.candidate_types_supported),
            "decision_threshold": self.decision_threshold,
            "known_limitations": list(self.known_limitations),
            "git_commit": self.git_commit,
            "not_wired_into_live_inference": True,
        }


def export_calibrator(calibrator: Any, output_dir: str) -> str:

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "calibrator.joblib")
    joblib.dump(calibrator, path)
    return path


def load_calibrator(path: str) -> Any:
    return joblib.load(path)


def write_metadata_v2(metadata: ExtendedModelMetadataV2, output_dir: str) -> str:

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "model_metadata_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2, default=str)
    return path


def load_metadata_v2(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
