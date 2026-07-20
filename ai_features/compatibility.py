import re

from dataclasses import dataclass, field
from typing import Tuple

from ai_features.feature_schema import CANONICAL_LIVE_SCHEMA, SCHEMA_VERSION


# =====================================================
# Phase 11's own reusable compatibility check. Today's ai_inference/
# ai_training pipeline (see docs/architecture/ai_live_feature_parity.md
# §"ai_inference schema validation" audit) performs ZERO validation
# before inference -- Predictor._compute_value_and_probability() hands
# a feature row straight to Preprocessor.transform(), which silently
# turns a missing required column into NaN (median/most-frequent
# imputed) and silently ignores any extra key. This module exists so a
# model genuinely incompatible with the canonical live-compatible
# schema fails LOUDLY, before a single prediction is ever computed --
# never a silent zero-fill or column reorder.
# =====================================================


# A reference set of legacy scenario-feature column NAMES/patterns known
# (from the exact per-model feature audit) to be simulation-ground-truth
# -only -- used only to give a more informative
# "simulation_only_dependency" verdict instead of a bare "missing
# required feature" when a legacy model's own required column is
# recognizably one of these, never used to silently accept or reject
# anything on its own.
_KNOWN_SIMULATION_ONLY_NAMES = frozenset({
    "ignition_zone", "ignition_floor", "fire_profile", "growth_time",
    "total_occupants",
    "Adult_Count", "Child_Count", "Elderly_Count", "Wheelchair_Count",
    "Visitor_Count", "Firefighter_Count",
    "Mean_Walking_Speed_Multiplier", "Mean_Reaction_Speed", "Mean_Stamina",
    "Mean_Smoke_Tolerance", "Mean_Visibility_Tolerance", "Mean_Fatigue_Resistance",
    "Mean_Mobility_Factor", "Mean_Leadership", "Mean_Risk_Aversion",
    "Mean_Route_Familiarity", "Mean_Compliance", "Mean_Helping_Likelihood",
    "Mean_Panic_Susceptibility", "Mean_Crowd_Following_Tendency",
    "Group_Count", "Grouped_Occupant_Count", "Mean_Group_Size",
})

_KNOWN_SIMULATION_ONLY_PATTERNS = (
    re.compile(r"^Zone_\d+_Occupancy$"),
    re.compile(r"^Door_\d+_State$"),
    re.compile(r"^Exit_\d+_State$"),
    re.compile(r"^Stair_\d+_State$"),
    re.compile(r"^Obstacle_\d+_State$"),
    re.compile(r"^Detector_\d+_State$"),
    re.compile(r"^Camera_\d+_State$"),
)


def _is_known_simulation_only(column: str) -> bool:

    if column in _KNOWN_SIMULATION_ONLY_NAMES:
        return True

    return any(pattern.match(column) for pattern in _KNOWN_SIMULATION_ONLY_PATTERNS)


# A reference set of known TRAINING-LABEL/outcome column names -- the
# exact-per-model audit (docs/architecture/ai_live_feature_parity.md)
# found none of the four existing models actually leak one of these into
# their own INPUT features (every one is used exclusively as `y`, never
# joined into `X_rows`). This set exists so that if a future model ever
# DID declare one of these as a required INPUT column, this checker
# gives the strongest, most specific verdict available -- "outcome_
# leakage", not merely "missing" -- rather than treating a leaked label
# as just another absent feature.
_KNOWN_OUTCOME_LEAKAGE_NAMES = frozenset({
    "total_evacuation_time", "average_evacuation_time", "last_occupant_exit_time",
    "people_evacuated", "people_trapped", "reachable_occupants", "unreachable_occupants",
    "building_cleared", "simulation_finished",
    "maximum_queue_length", "maximum_density", "maximum_congestion",
    "most_congested_exit", "most_congested_stair",
    "bottleneck_occurrence", "bottleneck_location",
    "doors_that_became_bottlenecks", "peak_congestion_location_id",
    "peak_congestion_location_type", "peak_congestion_value", "congestion_duration",
    "exit_usage_percentage", "evacuated", "exit_used",
    "next_highest_smoke_zone",
})


def _is_known_outcome_leakage(column: str) -> bool:

    return column in _KNOWN_OUTCOME_LEAKAGE_NAMES


# =====================================================


@dataclass(frozen=True)
class CompatibilityIssue:

    kind: str  # one of the five categories Phase 11 names -- see check_model_compatibility()
    detail: str


@dataclass(frozen=True)
class CompatibilityReport:

    model_name: str
    compatible: bool
    issues: Tuple[CompatibilityIssue, ...] = field(default_factory=tuple)

    def issues_of(self, kind: str) -> Tuple[CompatibilityIssue, ...]:

        return tuple(issue for issue in self.issues if issue.kind == kind)


class IncompatibleFeatureRowError(Exception):

    # Raised by validate_feature_row() -- the "fail clearly BEFORE
    # inference" half of Phase 11, for one specific X_row about to be
    # handed to Predictor.predict_all(), as opposed to
    # check_model_compatibility()'s one-time, whole-schema check.

    pass


# =====================================================


def check_model_compatibility(loaded_model, canonical_schema=CANONICAL_LIVE_SCHEMA) -> CompatibilityReport:

    # loaded_model: an ai_inference.loader.LoadedModel. Its own
    # ModelProvenance carries no feature-column list at all (see the
    # schema-validation audit) -- the only place a model's real, exact
    # training column list survives is loaded_model.model.feature_schema
    # (BaseModel.to_bundle()/from_bundle() persists it, ai_training/
    # models/base.py), so that is what this function reads.

    model_name = loaded_model.provenance.model_name
    model_columns = tuple(loaded_model.model.feature_schema.columns)
    model_column_set = set(model_columns)

    canonical_by_name = {field_.name: field_ for field_ in canonical_schema}
    canonical_names = set(canonical_by_name)

    issues = []

    # 1. Required feature missing -- every column the model actually
    # trained on that the canonical live-compatible schema cannot supply.
    for column in sorted(model_column_set - canonical_names):

        if _is_known_outcome_leakage(column):

            issues.append(CompatibilityIssue(
                "outcome_leakage",
                f"{model_name!r} requires {column!r}, a known training-label/outcome column -- "
                f"using it as an INPUT feature at inference time is invalid regardless of "
                f"live availability (the outcome cannot be known before it happens).",
            ))

        elif _is_known_simulation_only(column):

            issues.append(CompatibilityIssue(
                "simulation_only_dependency",
                f"{model_name!r} requires {column!r}, a known simulation-ground-truth-only "
                f"column with no live-observable equivalent in the canonical schema.",
            ))

        else:

            issues.append(CompatibilityIssue(
                "missing_required_feature",
                f"{model_name!r} requires {column!r}, which is not present in the canonical "
                f"live-compatible schema (version {SCHEMA_VERSION}).",
            ))

    # 2. Schema version mismatch -- only meaningful for a model this
    # milestone's own live-compatible pipeline trained (see ai_features.
    # simulation_extractor); such a model's manifest carries an explicit
    # feature_schema_version in its metadata. A legacy model (no such
    # key) is never flagged here -- its incompatibility is already fully
    # captured by every missing_required_feature/simulation_only_
    # dependency issue above.
    declared_version = loaded_model.provenance.metrics.get("feature_schema_version") if isinstance(
        loaded_model.provenance.metrics, dict,
    ) else None

    if declared_version is not None and declared_version != SCHEMA_VERSION:

        issues.append(CompatibilityIssue(
            "schema_version_mismatch",
            f"{model_name!r} was trained against canonical schema version "
            f"{declared_version!r}, but the running schema is {SCHEMA_VERSION!r}.",
        ))

    # 3. Unsupported missing-value behavior -- a shared column whose
    # numeric/categorical classification in the model's own persisted
    # FeatureSchema conflicts with this schema's declared dtype (e.g. the
    # model fit a numeric imputer/scaler against a column the canonical
    # schema documents as a string/category, or vice versa) -- silently
    # feeding it through Preprocessor.transform() would apply the wrong
    # imputation/encoding strategy without either side ever raising.
    shared_columns = model_column_set & canonical_names
    model_numeric = set(loaded_model.model.feature_schema.numeric_columns)

    for column in sorted(shared_columns):

        canonical_field = canonical_by_name[column]
        canonical_is_numeric = canonical_field.dtype in ("int", "float")
        model_is_numeric = column in model_numeric

        if canonical_is_numeric == model_is_numeric:
            continue

        if canonical_field.nullable:

            # A nullable numeric field that happened to be entirely
            # missing throughout one particular training run is a known,
            # inherent limitation of FeatureSchema.infer()'s generic
            # "categorical if no non-null value was ever observed" rule
            # (ai_training/preprocessing.py) -- not a genuine semantic
            # clash this checker should reject a model over. A NON-
            # nullable field landing in the wrong bucket has no such
            # innocent explanation and is still flagged below.
            continue

        issues.append(CompatibilityIssue(
            "unsupported_missing_value_behavior",
            f"{column!r} is trained as {'numeric' if model_is_numeric else 'categorical'} "
            f"by {model_name!r} but declared {canonical_field.dtype!r} "
            f"({'numeric' if canonical_is_numeric else 'categorical'}) by the canonical "
            f"schema -- missing-value imputation would silently use the wrong strategy.",
        ))

    return CompatibilityReport(model_name=model_name, compatible=not issues, issues=tuple(issues))


# =====================================================


def validate_feature_row(row, loaded_model) -> None:

    # The per-call guard Phase 11 also asks for: "a model that is
    # incompatible... should fail clearly BEFORE inference." Unlike
    # check_model_compatibility() (a one-time, whole-schema audit), this
    # runs immediately before every real predict_all() call a caller
    # chooses to guard this way. Raises IncompatibleFeatureRowError on
    # any missing required column or any unexpected key -- never
    # silently reorders columns or fills an unknown required feature
    # with a fabricated zero, unlike Predictor's own current behavior.

    model_columns = set(loaded_model.model.feature_schema.columns)
    row_keys = set(row.keys())

    missing = sorted(model_columns - row_keys)
    unexpected = sorted(row_keys - model_columns)

    if missing:

        raise IncompatibleFeatureRowError(
            f"{loaded_model.provenance.model_name!r} requires feature(s) {missing} which "
            f"are absent from the supplied row -- refusing to silently zero-fill them."
        )

    if unexpected:

        raise IncompatibleFeatureRowError(
            f"Feature row supplied to {loaded_model.provenance.model_name!r} contains "
            f"unexpected key(s) {unexpected} not present in the model's trained schema."
        )
