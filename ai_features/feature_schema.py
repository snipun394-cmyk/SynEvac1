from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple


# =====================================================
# Simulation-to-Live AI Feature Parity Framework -- the one explicit
# feature contract this milestone exists to establish. Every field
# here was classified against two independent audits: (1) the exact
# training feature columns every currently-trained AI model actually
# uses (dataset_builder.feature_extractor.extract_scenario_features()
# and each ai_training/models/*.py's own build_table()), and (2) what
# building_state.models.BuildingState and everything reachable from it
# can honestly represent from a real deployment, versus what is, in
# every current code path, simulation-ground-truth-only (see
# docs/architecture/ai_live_feature_parity.md for the full audit this
# schema is derived from).
#
# The central rule this module exists to enforce: the SAME feature
# names/semantics must be producible from BOTH simulation data and
# BuildingState (ai_features.simulation_extractor and ai_features.
# building_state_extractor both target exactly this schema) -- never
# two similar-looking but independently-defined "training" and "live"
# feature sets that could silently drift apart.
# =====================================================


class FeatureAvailability(Enum):

    # Phase 1's own required classification vocabulary. Assigned once,
    # per field, below -- never inferred at extraction time.

    LIVE_OBSERVABLE = auto()   # A real deployment's device/perception
                                # layer can produce this directly (e.g.
                                # CameraStatus.active, FACPSnapshot.
                                # panel_state).
    LIVE_ESTIMABLE = auto()    # Approximatable with reduced fidelity/
                                # confidence from real sensors (e.g.
                                # occupant count via multi-camera fusion).
    SIMULATION_ONLY = auto()   # Requires ground truth this codebase has
                                # no real-sensor path to today (e.g.
                                # hazard_score-derived severity, exact
                                # occupant behaviour-profile identity).
    FUTURE_INFORMATION = auto()  # Requires knowing what has not
                                  # happened yet as of the prediction
                                  # time (excluded from this schema
                                  # entirely -- see module docstring).
    OUTCOME_LEAKAGE = auto()   # A training-time-only outcome value that
                                # must never appear as an input feature
                                # (excluded from this schema entirely).
    UNKNOWN = auto()           # Not yet classified -- never assigned to
                                # a real CANONICAL_LIVE_SCHEMA field; a
                                # placeholder value for tooling only.


@dataclass(frozen=True)
class AIFeatureField:

    # One canonical feature's full metadata -- Phase 2's own required
    # shape: stable name, data type, semantic meaning, source,
    # availability classification, and (for nullable fields) the
    # missing-value policy this specific field follows. See
    # ai_features.compatibility for how this is checked against a
    # trained model's own persisted FeatureSchema before inference.

    name: str
    dtype: str  # "int" | "float" | "bool" | "str"
    semantic_meaning: str
    source: str  # dotted attribute path this field is read from
    availability: FeatureAvailability
    nullable: bool
    missing_value_note: str = ""


# =====================================================
# The canonical schema itself. Order is deterministic and part of the
# contract -- CANONICAL_LIVE_FEATURE_NAMES below is exactly this
# order, and every extractor in this package emits keys in this same
# order (Phase 13's own "feature ordering is deterministic" test).
#
# Deliberately EXCLUDED from this schema, with the reason each was
# ruled out (see docs/architecture/ai_live_feature_parity.md §3 for
# the full audit evidence):
#
#   - hazard_summary / zone_severity / HazardSeverity -- SIMULATION_ONLY
#     in every current code path (hazard_score is always simulated fire-
#     growth/smoke-propagation/tenability physics; no real-sensor-derived
#     HazardSnapshot exists anywhere in this codebase).
#   - occupant_tracks[*].classification / human_state breakdown counts --
#     currently sourced from ground-truth behaviour-profile identity
#     (SIMULATION_ONLY in current implementation), not appearance/re-ID;
#     only occupant COUNT and mean track CONFIDENCE are used below.
#   - ignition_zone / fire_profile / growth_time / occupant behaviour-
#     profile attributes / group aggregates -- pure SIMULATION_ONLY
#     ScenarioDefinition/BehaviourProfile facts, never available live.
#   - any final/total evacuation outcome (total_evacuation_time,
#     people_evacuated, exit usage, etc.) -- OUTCOME_LEAKAGE if used as
#     an input feature; these remain valid as training LABELS only
#     (see ai_features.simulation_extractor's own module docstring).
# =====================================================


CANONICAL_LIVE_SCHEMA: Tuple[AIFeatureField, ...] = (

    # ---------------------------------------------------
    # Occupancy -- derived from fused multi-camera occupant tracks
    # (BuildingState.occupant_tracks), the same signal the CCTV
    # milestone already proved end-to-end (4 raw detections -> 3
    # unique BuildingState occupants). Never the pre-simulation
    # starting-placement Zone_N_Occupancy columns the existing trained
    # models use -- those are a fundamentally different, simulation-
    # only quantity.
    # ---------------------------------------------------

    AIFeatureField(
        name="total_occupant_count",
        dtype="int",
        semantic_meaning="Current estimated total occupant count, from fused multi-camera tracks.",
        source="building_state.occupant_tracks",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=True,
        missing_value_note=(
            "None when occupancy_observed is False (no camera coverage this cycle) -- "
            "never fabricated as 0, which would silently claim a confirmed-empty building."
        ),
    ),
    AIFeatureField(
        name="occupancy_observed",
        dtype="bool",
        semantic_meaning="Whether at least one camera currently contributes occupancy evidence.",
        source="building_state.camera_observations",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="mean_occupant_track_confidence",
        dtype="float",
        semantic_meaning="Mean FusedTrack.confidence across all current occupant tracks.",
        source="building_state.occupant_tracks[*].confidence",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=True,
        missing_value_note="None when there are no current occupant tracks to average.",
    ),

    # ---------------------------------------------------
    # Camera asset health -- genuine device-management facts.
    # ---------------------------------------------------

    AIFeatureField(
        name="camera_total_count",
        dtype="int",
        semantic_meaning="Total number of registered Camera Assets, regardless of state.",
        source="building_state.camera_observations",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="camera_active_count",
        dtype="int",
        semantic_meaning="Number of Camera Assets currently active.",
        source="building_state.active_assets.active_camera_ids",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="camera_offline_count",
        dtype="int",
        semantic_meaning="Number of Camera Assets currently inactive/offline.",
        source="building_state.active_assets.offline_camera_ids",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),

    # ---------------------------------------------------
    # Sensor (Smoke/Heat Detector) asset health.
    # ---------------------------------------------------

    AIFeatureField(
        name="sensor_total_count",
        dtype="int",
        semantic_meaning="Total number of registered smoke/heat detector assets.",
        source="building_state.smoke_detector_states + building_state.heat_detector_states",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="sensor_active_count",
        dtype="int",
        semantic_meaning="Number of smoke/heat detector assets currently active.",
        source="building_state.active_assets.active_sensor_ids",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="sensor_offline_count",
        dtype="int",
        semantic_meaning="Number of smoke/heat detector assets currently inactive/offline.",
        source="building_state.active_assets.offline_sensor_ids",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),

    # ---------------------------------------------------
    # Detector condition (alarm/fault), split by kind, always paired
    # with a coverage count -- so "0 alarms" is never confused with
    # "no detectors of this kind exist at all" (Phase 10's own explicit
    # "no smoke detector coverage in a zone" scenario).
    # ---------------------------------------------------

    AIFeatureField(
        name="smoke_detector_coverage_count",
        dtype="int",
        semantic_meaning="Number of smoke detectors currently reporting a state at all.",
        source="building_state.smoke_detector_states",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="smoke_detector_alarm_count",
        dtype="int",
        semantic_meaning="Number of smoke detectors currently in ALARM.",
        source="building_state.smoke_detector_states[*].reading.alarm_active",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="smoke_detector_fault_count",
        dtype="int",
        semantic_meaning="Number of smoke detectors currently in FAULT (health_status != OK).",
        source="building_state.smoke_detector_states[*].status.health_status",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="heat_detector_coverage_count",
        dtype="int",
        semantic_meaning="Number of heat detectors currently reporting a state at all.",
        source="building_state.heat_detector_states",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="heat_detector_alarm_count",
        dtype="int",
        semantic_meaning="Number of heat detectors currently in ALARM.",
        source="building_state.heat_detector_states[*].reading.alarm_active",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="heat_detector_fault_count",
        dtype="int",
        semantic_meaning="Number of heat detectors currently in FAULT (health_status != OK).",
        source="building_state.heat_detector_states[*].status.health_status",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="building_alarm_status",
        dtype="str",
        semantic_meaning=(
            "Aggregated ALARM > FAULT > NORMAL status across every detector "
            "BuildingStateEstimator was handed this cycle."
        ),
        source="building_state.building_alarm_status",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
        missing_value_note=(
            "Always present (BuildingState's own field defaults to NORMAL when no "
            "detector is configured at all) -- pair with sensor_total_count to tell "
            "'confirmed clear' apart from 'no detector coverage exists.'"
        ),
    ),

    # ---------------------------------------------------
    # Fire Alarm Control Panel -- every FACPSnapshot field matches a
    # real panel integration's outputs (see the observability audit).
    # facp_available gates every other facp_* field explicitly, rather
    # than defaulting panel_state to "NORMAL" when no FACP exists.
    # ---------------------------------------------------

    AIFeatureField(
        name="facp_available",
        dtype="bool",
        semantic_meaning="Whether a Fire Alarm Control Panel snapshot is present this cycle.",
        source="building_state.facp_status",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="facp_panel_state",
        dtype="str",
        semantic_meaning="The FACP's own panel_state (NORMAL/FAULT/ALARM/ALARM_ACKNOWLEDGED/ALARM_SILENCED).",
        source="building_state.facp_status.panel_state",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None (never 'NORMAL') when facp_available is False.",
    ),
    AIFeatureField(
        name="facp_active_alarm_source_count",
        dtype="int",
        semantic_meaning="Number of sources currently in FACPSnapshot.active_alarm_source_ids.",
        source="building_state.facp_status.active_alarm_source_ids",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when facp_available is False.",
    ),
    AIFeatureField(
        name="facp_active_fault_source_count",
        dtype="int",
        semantic_meaning="Number of sources currently in FACPSnapshot.active_fault_source_ids.",
        source="building_state.facp_status.active_fault_source_ids",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when facp_available is False.",
    ),
    AIFeatureField(
        name="facp_acknowledged",
        dtype="bool",
        semantic_meaning="Whether the panel's current alarm has been operator-acknowledged.",
        source="building_state.facp_status.acknowledged",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when facp_available is False.",
    ),
    AIFeatureField(
        name="facp_silenced",
        dtype="bool",
        semantic_meaning="Whether the panel's current alarm has been operator-silenced.",
        source="building_state.facp_status.silenced",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when facp_available is False.",
    ),

    # ---------------------------------------------------
    # Building Control -- only ever CONFIRMED provider state (never a
    # pending recommendation), per ControlStateSnapshot's own contract.
    # ---------------------------------------------------

    AIFeatureField(
        name="control_status_available",
        dtype="bool",
        semantic_meaning="Whether a Building Control snapshot is present this cycle.",
        source="building_state.control_status",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="control_pending_request_count",
        dtype="int",
        semantic_meaning="Number of Building Control requests currently awaiting approval.",
        source="building_state.control_status.pending_count",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when control_status_available is False.",
    ),
    AIFeatureField(
        name="control_confirmed_entry_count",
        dtype="int",
        semantic_meaning="Number of CONFIRMED Building Control actions currently in effect.",
        source="building_state.control_status.entries",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when control_status_available is False.",
    ),
)


CANONICAL_LIVE_FEATURE_NAMES: Tuple[str, ...] = tuple(field.name for field in CANONICAL_LIVE_SCHEMA)

SCHEMA_VERSION = "1.0"


def field_by_name(name: str) -> AIFeatureField:

    for field in CANONICAL_LIVE_SCHEMA:

        if field.name == name:
            return field

    raise KeyError(f"{name!r} is not a field of CANONICAL_LIVE_SCHEMA.")
