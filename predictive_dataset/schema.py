from typing import Tuple

from ai_features.feature_schema import AIFeatureField, FeatureAvailability


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone, Phase 6/7 --
# the per-candidate feature schema, plus the live-parity classification
# every field must carry. Reuses ai_features.feature_schema's own
# AIFeatureField shape and FeatureAvailability vocabulary rather than
# inventing a second, parallel one (that module's own module docstring:
# "the SAME feature names/semantics must be producible from BOTH
# simulation data and [a live source]... never two similar-looking but
# independently-defined... feature sets").
#
# Every field below was kept OUT unless it has a credible LIVE source
# today (Phase 6/7's own "no simulation-exclusive omniscience" rule).
# The one family investigated and explicitly EXCLUDED is per-candidate
# hazard exposure (e.g. "hazard score of the zone(s) this candidate
# touches"): ai_features.feature_schema's own, already-audited
# exclusion list rules this SIMULATION_ONLY ("hazard_score is always
# simulated fire-growth/smoke-propagation/tenability physics; no real-
# sensor-derived HazardSnapshot exists anywhere in this codebase") --
# reused here rather than re-litigated, per this module's own "do not
# invent a second incompatible definition" discipline. It belongs in a
# future schema version only once real hazard sensing exists.
#
# ROW IDENTITY columns (scenario_id/observation_time/candidate_id/
# candidate_type) are NOT part of this schema -- they are bookkeeping,
# not trainable input features, exactly like dataset_builder.schema's
# own SCENARIO_METADATA_COLUMNS are kept separate from feature columns.
# =====================================================


GLOBAL_CONTEXT_SCHEMA: Tuple[AIFeatureField, ...] = (

    AIFeatureField(
        name="total_active_occupant_count",
        dtype="int",
        semantic_meaning=(
            "Current whole-building occupant count still evacuating (not yet arrived/exited), "
            "at observation time -- global context, not localized to this candidate."
        ),
        source="SIM: MultiAgentSimulationResult.occupants (not yet arrived). "
               "LIVE: LiveOccupantManager.canonical_occupancy(time).total_observed_count.",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=False,
    ),
)


CANDIDATE_CONTEXT_SCHEMA: Tuple[AIFeatureField, ...] = (

    AIFeatureField(
        name="candidate_type",
        dtype="str",
        semantic_meaning="DOOR | EXIT | STAIR -- matches navigation.edge.Edge.EDGE_TYPES.",
        source="SIM/LIVE: navigation.edge.Edge.edge_type (structural, shared Navigation Graph type).",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="candidate_capacity",
        dtype="int",
        semantic_meaning=(
            "Simulation-style engineering capacity estimate for this candidate "
            "(crowd_intelligence.capacity / simulator.capacity's own DefaultCapacityModel/"
            "StairCapacityModel) -- a structural design estimate, never a measured live flow rate."
        ),
        source="SIM/LIVE: IDENTICAL call -- crowd_intelligence.capacity.{door,exit,stair}_capacity(), "
               "a pure function of the Door/Exit/Staircase object, no simulation/live state involved.",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when the underlying capacity model cannot derive one (e.g. no width).",
    ),
    AIFeatureField(
        name="candidate_walking_distance",
        dtype="float",
        semantic_meaning="Physical walking distance (meters) this candidate's edge represents.",
        source="SIM/LIVE: IDENTICAL -- navigation.edge.Edge.walking_distance, from the shared "
               "Navigation Graph (never simulation- or live-only geometry).",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=True,
        missing_value_note="None when an endpoint has no derivable geometry (Edge.walking_distance's own contract).",
    ),
    AIFeatureField(
        name="candidate_traversable",
        dtype="bool",
        semantic_meaning=(
            "Whether this candidate can currently be walked at all (locked/blocked/obstacle-checked)."
        ),
        source="SIM/LIVE: navigation.edge.Edge.traversable, evaluated against the structural "
               "Door.locked/active, Exit.is_blocked, and blocking-obstacle state known at "
               "observation time. Does not yet incorporate mid-scenario ScenarioEvent door/exit/"
               "stair overrides -- a disclosed v1 simplification, see module docstring below.",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="candidate_adjacent_zone_occupancy",
        dtype="int",
        semantic_meaning="Occupant count in this candidate's own approach-side zone right now.",
        source="SIM: OccupancySnapshot.observation_at(edge.from_node).occupant_count (from the "
               "same TickResult dataset_builder.timeline already reads for Zone_N_Occupancy). "
               "LIVE: LiveOccupantManager.canonical_occupancy(time).occupant_ids_by_zone[from_node].",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=True,
        missing_value_note="None when the approach-side node has no zone (should not occur for a "
                            "well-formed candidate) -- never fabricated as 0.",
    ),
    AIFeatureField(
        name="candidate_queue_length",
        dtype="int",
        semantic_meaning="Count of occupants currently queued/accumulating at this candidate.",
        source="SIM: exact discrete-event queue-admission bookkeeping (OccupantTimelineStep."
               "queue_wait_time interval, the SAME technique dataset_builder.timeline._current_"
               "queue_length already uses, filtered to this candidate's edge id only). "
               "LIVE: crowd_intelligence.models.AssetApproachMetrics.queue_candidate_count "
               "(occupants near this asset classified STATIONARY) -- a geometric/behavioral proxy, "
               "not the same underlying mechanism, but the same semantic question.",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=True,
        missing_value_note="LIVE: None when AssetApproachMetrics.position_available is False "
                            "(no occupant near this asset has a usable world_position). "
                            "SIM: never None -- exact ground truth is always computable.",
    ),
    AIFeatureField(
        name="candidate_approaching_count",
        dtype="int",
        semantic_meaning="Count of occupants currently committed to / approaching this candidate.",
        source="SIM: occupants whose fixed Route's final edge is this candidate and who have "
               "departed but not yet arrived by observation time (whole-route commitment, a "
               "long-range demand signal known from route assignment at scenario start -- see "
               "target_generator.py's leakage note, this is NOT a future-outcome read). "
               "LIVE: crowd_intelligence.models.AssetApproachMetrics.approaching_count "
               "(short-range, ~3m geometric proximity + heading). Both answer 'how much demand "
               "is headed for this candidate', at different look-ahead ranges -- disclosed, not "
               "papered over.",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=True,
        missing_value_note="LIVE: None when AssetApproachMetrics.position_available is False.",
    ),
    AIFeatureField(
        name="candidate_congestion_level",
        dtype="str",
        semantic_meaning=(
            "IntensityLevel name (LOW/MODERATE/HIGH/VERY_HIGH/CRITICAL) classifying demand "
            "(queue + approaching) against candidate_capacity."
        ),
        source="SIM/LIVE: IDENTICAL call -- crowd_intelligence.congestion.compute_congestion_level("
               "QueueMetrics(candidate_queue_length, candidate_approaching_count), "
               "candidate_capacity, thresholds). Only the queue/approaching COUNTS feeding this "
               "call differ in methodology (see above); the classification step itself is shared code.",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=True,
        missing_value_note="None when candidate_capacity is None/<=0, or neither queue nor "
                            "approaching evidence is available.",
    ),
)


CANDIDATE_FEATURE_SCHEMA: Tuple[AIFeatureField, ...] = GLOBAL_CONTEXT_SCHEMA + CANDIDATE_CONTEXT_SCHEMA

CANDIDATE_FEATURE_NAMES: Tuple[str, ...] = tuple(field.name for field in CANDIDATE_FEATURE_SCHEMA)

# Bookkeeping/identity columns every dataset row carries in addition to
# the trainable features above. scenario_id/observation_time/candidate_id
# are never fed to a model as an input. candidate_type is deliberately
# listed both here (row identity, used for grouping/inspection) AND in
# CANDIDATE_CONTEXT_SCHEMA above (it is also a legitimate categorical
# input feature) -- the two uses are not in conflict, the same way a
# join key can also carry information.
ROW_IDENTITY_COLUMNS: Tuple[str, ...] = (
    "scenario_id",
    "observation_time",
    "candidate_id",
    "candidate_type",
)

SCHEMA_VERSION = "1.0"


def field_by_name(name: str) -> AIFeatureField:

    for field in CANDIDATE_FEATURE_SCHEMA:

        if field.name == name:
            return field

    raise KeyError(f"{name!r} is not a field of CANDIDATE_FEATURE_SCHEMA.")
