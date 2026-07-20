from ai_features.building_state_extractor import extract_canonical_features
from ai_features.compatibility import (
    CompatibilityIssue,
    CompatibilityReport,
    IncompatibleFeatureRowError,
    check_model_compatibility,
    validate_feature_row,
)
from ai_features.feature_schema import (
    CANONICAL_LIVE_FEATURE_NAMES,
    CANONICAL_LIVE_SCHEMA,
    SCHEMA_VERSION,
    AIFeatureField,
    FeatureAvailability,
    field_by_name,
)
from ai_features.simulation_extractor import (
    build_building_state_at_alarm_activation,
    extract_canonical_training_row,
)

__all__ = [
    # feature_schema
    "AIFeatureField",
    "FeatureAvailability",
    "CANONICAL_LIVE_SCHEMA",
    "CANONICAL_LIVE_FEATURE_NAMES",
    "SCHEMA_VERSION",
    "field_by_name",
    # building_state_extractor
    "extract_canonical_features",
    # simulation_extractor
    "build_building_state_at_alarm_activation",
    "extract_canonical_training_row",
    # compatibility
    "CompatibilityReport",
    "CompatibilityIssue",
    "IncompatibleFeatureRowError",
    "check_model_compatibility",
    "validate_feature_row",
]
