import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib


# =====================================================
# First Localized Predictive Congestion Model milestone, Phase 12 --
# model export. THIS MODULE ONLY WRITES FILES TO data/localized_
# predictive_model_v1/ -- it is never imported by live_system/,
# building_state/, recommendation/, guidance/, dynamic_signage/,
# LiveRuntime, or any operator-facing code. Nothing in this milestone
# wires a trained model into any of those; exporting it here is purely
# so a future, EXPLICITLY SEPARATE controlled-integration milestone
# could load it, which has not happened and is not this milestone's job.
# =====================================================


@dataclass(frozen=True)
class ModelMetadata:

    model_name: str
    model_library: str
    dataset_schema_version: str
    dataset_campaign_version: str
    dataset_feature_version: str
    dataset_target_version: str
    prediction_horizon_seconds: float
    feature_names: Tuple[str, ...]
    train_scenario_count: int
    val_scenario_count: int
    test_scenario_count: int
    decision_threshold: float
    class_weight_strategy: str
    validation_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    production_readiness: str  # "READY" | "PROMISING_BUT_NEEDS_MORE_DATA" | "NOT_READY"
    production_readiness_rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_library": self.model_library,
            "dataset_schema_version": self.dataset_schema_version,
            "dataset_campaign_version": self.dataset_campaign_version,
            "dataset_feature_version": self.dataset_feature_version,
            "dataset_target_version": self.dataset_target_version,
            "prediction_horizon_seconds": self.prediction_horizon_seconds,
            "feature_names": list(self.feature_names),
            "train_scenario_count": self.train_scenario_count,
            "val_scenario_count": self.val_scenario_count,
            "test_scenario_count": self.test_scenario_count,
            "decision_threshold": self.decision_threshold,
            "class_weight_strategy": self.class_weight_strategy,
            "validation_metrics": self.validation_metrics,
            "test_metrics": self.test_metrics,
            "production_readiness": self.production_readiness,
            "production_readiness_rationale": self.production_readiness_rationale,
            "not_wired_into_live_inference": True,
        }


def export_model(model: Any, metadata: ModelMetadata, output_dir: str) -> Dict[str, str]:

    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(model, model_path)

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata.to_dict(), metadata_file, indent=2)

    return {"model_path": model_path, "metadata_path": metadata_path}


def load_model_metadata(metadata_path: str) -> Dict[str, Any]:

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def load_model(model_path: str) -> Any:
    return joblib.load(model_path)
