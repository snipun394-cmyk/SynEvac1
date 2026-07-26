from dataclasses import dataclass
from typing import Any, Dict

from predictive_dataset.campaign_config import CAMPAIGN_VERSION
from predictive_dataset.schema import SCHEMA_VERSION


# =====================================================
# Large-Scale Predictive Dataset Campaign & Validation milestone,
# Phase 11 -- explicit dataset version identifiers, so a future ML
# experiment can name exactly which schema/campaign/feature/target
# combination it trained against. FEATURE_VERSION and SCHEMA_VERSION
# are deliberately the same value today (the feature set IS the
# schema, see predictive_dataset/schema.py) -- kept as two separate
# names because they answer two different future-change questions
# ("did the FIELD LIST change" vs. "did the SCHEMA'S OWN semantics/
# classification change") that happen to coincide right now but are
# not guaranteed to stay coincident.
# =====================================================

FEATURE_VERSION = SCHEMA_VERSION

# The exact congestion/horizon semantics predictive_dataset.
# target_generator implements -- bump this if CONGESTION_THRESHOLD or
# the horizon-window definition ever changes, independently of the
# feature schema.
TARGET_VERSION = "v1-congestion-threshold-2-horizon-window"


@dataclass(frozen=True)
class DatasetVersion:

    schema_version: str
    campaign_version: str
    feature_version: str
    target_version: str
    prediction_horizon_seconds: float

    def to_dict(self) -> Dict[str, Any]:

        return {
            "schema_version": self.schema_version,
            "campaign_version": self.campaign_version,
            "feature_version": self.feature_version,
            "target_version": self.target_version,
            "prediction_horizon_seconds": self.prediction_horizon_seconds,
        }


def dataset_version(recommended_horizon_seconds: float, campaign_version: str = CAMPAIGN_VERSION) -> DatasetVersion:

    # campaign_version defaults to V1's own CAMPAIGN_VERSION so every
    # existing V1 call site (scripts/run_predictive_dataset_campaign_v1.py,
    # its own tests) is completely unaffected -- Predictive Dataset V2
    # milestone's campaign script passes its own campaign_version
    # explicitly (predictive_dataset.campaign_config_v2.CAMPAIGN_VERSION_V2)
    # instead. schema_version/feature_version/target_version are NOT
    # parameterized here: V2 reuses the exact same schema/feature/target
    # definitions as V1 (same 9 input features, same congestion-threshold-2
    # target semantics) -- only the campaign (which scenarios/topologies
    # produced the rows) differs.

    return DatasetVersion(
        schema_version=SCHEMA_VERSION,
        campaign_version=campaign_version,
        feature_version=FEATURE_VERSION,
        target_version=TARGET_VERSION,
        prediction_horizon_seconds=recommended_horizon_seconds,
    )
