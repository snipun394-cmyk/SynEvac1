from typing import Tuple

from ai_features.feature_schema import AIFeatureField, FeatureAvailability
from predictive_dataset.schema import CANDIDATE_FEATURE_SCHEMA, ROW_IDENTITY_COLUMNS


# =====================================================
# Predictive Dataset V4 milestone, Phase 1/2 -- the FIRST milestone to
# actually promote experimental fields into a versioned canonical
# schema (every prior tier -- V2.1's 3 experimental fields, V3's
# structural-diversity work, the Cross-Topology Generalization
# Investigation's normalized/graph-context fields -- stayed additive-
# external, never touching predictive_dataset/schema.py's
# SCHEMA_VERSION or CANDIDATE_FEATURE_SCHEMA; see that investigation's
# own Phase 3 research finding for the full history).
#
# `predictive_dataset/schema.py` (SCHEMA_VERSION "1.0",
# CANDIDATE_FEATURE_SCHEMA's original 9 fields) is IMPORTED, never
# mutated -- Dataset V1/V2/V3 remain reproducible against their own
# original schema exactly as before. This module defines a NEW,
# separately versioned schema (SCHEMA_VERSION_V4 = "4.0") that EXTENDS
# the frozen 9 with 6 promoted fields:
#
#   - 3 fields V2.1 already validated operationally (candidate_recent_
#     flow_rate, candidate_congestion_trend, candidate_alternative_
#     route_count) -- used in every campaign since Dataset V2.1/V2.2/V3,
#     but never previously given a formal AIFeatureField schema entry.
#     Given one here for the first time, using the exact semantics
#     already established by predictive_dataset/simulation_extractor_
#     v2_1.py and predictive_dataset/live_extractor_v2_1.py (both
#     UNCHANGED by this milestone).
#
#   - 3 NEW fields validated by the Cross-Topology Generalization
#     Investigation (commit e8d728a) as the only representation change
#     that improved EVERY family holdout (average +3.3% PR-AUC):
#     candidate_betweenness_centrality, candidate_is_bridge,
#     candidate_upstream_catchment_count. Computed via
#     predictive_dataset.graph_context_v4.compute_graph_context_for_
#     building -- see that module's own docstring for the full
#     mathematical/physical/complexity/leakage documentation each field
#     inherits verbatim.
#
# Any dataset/model built against SCHEMA_VERSION "1.0" remains fully
# loadable/reproducible -- this module is purely additive, imported
# from, never merged into, schema.py.
# =====================================================


V4_PROMOTED_EXPERIMENTAL_FIELDS: Tuple[AIFeatureField, ...] = (

    AIFeatureField(
        name="candidate_recent_flow_rate",
        dtype="int",
        semantic_meaning=(
            "Count of occupants who COMPLETED crossing this candidate's edge during "
            "the trailing 60-second window ending at observation time -- a throughput/"
            "rate signal, distinct from candidate_queue_length (current backlog) and "
            "candidate_approaching_count (future demand)."
        ),
        source="SIM: predictive_dataset.simulation_extractor_v2_1._recent_flow_rate (completed-crossing "
               "count over movement_result.occupants). LIVE: predictive_dataset.live_extractor_v2_1."
               "_live_recent_flow_rate -- Exit reuses evacuation_progress.models.ExitFlow."
               "recent_flow_per_minute directly (same 60s window); Stair reuses stair_flow.models."
               "StairFlowMetrics.exits (a genuine, camera/tracking-derived completed-crossing count, "
               "proved semantically equivalent to the simulation definition -- see docs/architecture/"
               "stair_predictive_feature_live_parity.md) when a caller supplies `stair_flow_snapshot`, "
               "else falls back to the original Door/Stair zone_transitions proxy; Door always uses "
               "that proxy (queries live_occupants.history.OccupantHistory.zone_transitions for the "
               "matching zone pair -- no per-asset transition evidence analogous to Stair exists for Door).",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=True,
        missing_value_note="LIVE: None when the underlying evacuation_snapshot/occupants/stair_flow_"
                            "snapshot source is unavailable this cycle. SIM: never None -- exact ground "
                            "truth is always computable.",
    ),
    AIFeatureField(
        name="candidate_congestion_trend",
        dtype="str",
        semantic_meaning=(
            "RISING | STABLE | FALLING | UNKNOWN -- direction of change in a demand proxy "
            "(queue_length + approaching_count) over the trailing 30-second window."
        ),
        source="SIM: predictive_dataset.simulation_extractor_v2_1._congestion_trend (demand-proxy delta "
               "vs. time-30s). LIVE: predictive_dataset.live_extractor_v2_1._live_congestion_trend -- "
               "crowd_intelligence.models.AssetApproachMetrics.trend, already computed for every "
               "Door/Exit/Stair asset by crowd_intelligence.trends.TrendTracker.",
        availability=FeatureAvailability.LIVE_ESTIMABLE,
        nullable=False,
        missing_value_note="UNKNOWN is itself a real, honest value (not enough history yet) -- "
                            "never coerced to None or a fabricated STABLE.",
    ),
    AIFeatureField(
        name="candidate_alternative_route_count",
        dtype="int",
        semantic_meaning="Count of OTHER candidates (any Door/Exit/Stair) sharing at least one zone with this one.",
        source="SIM/LIVE: IDENTICAL -- predictive_dataset.simulation_extractor_v2_1.build_alternative_route_counts, "
               "purely structural (CandidateIdentity.zone_ids), zero occupancy/time dependence, computed "
               "once per Building and reused by both extractors.",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
)


V4_PROMOTED_GRAPH_CONTEXT_FIELDS: Tuple[AIFeatureField, ...] = (

    AIFeatureField(
        name="candidate_betweenness_centrality",
        dtype="float",
        semantic_meaning=(
            "Edge betweenness centrality (Brandes' algorithm, walking-distance-weighted, "
            "normalized to [0, 1]): the fraction of all-pairs shortest paths in the building "
            "that pass through this candidate's edge. How structurally CENTRAL this candidate "
            "is to the whole evacuation graph, not just its own local queue."
        ),
        source="SIM/LIVE: IDENTICAL -- predictive_dataset.graph_context_v4.compute_graph_context_for_building, "
               "which calls navigation.graph_builder.NavigationGraphGenerator().build(building), the EXACT "
               "same class live_runtime/factory.py's own navigation_graph construction already uses, "
               "against the SAME models.building.Building object (Designer canvas building IS LiveRuntime's "
               "building -- no separate live graph representation exists). Computed once per building "
               "shape, never per row/tick.",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
        missing_value_note="Always a real computed value (0.0 for an edge on zero shortest paths) -- never None.",
    ),
    AIFeatureField(
        name="candidate_is_bridge",
        dtype="bool",
        semantic_meaning=(
            "True iff removing this candidate's edge increases the graph's connected-component "
            "count (a cut-edge). Whether this candidate is a structural SINGLE POINT OF FAILURE "
            "for some zone's evacuation, by design -- not a momentary blockage "
            "(candidate_traversable's job)."
        ),
        source="SIM/LIVE: IDENTICAL -- same function/class chain as candidate_betweenness_centrality above.",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
    ),
    AIFeatureField(
        name="candidate_upstream_catchment_count",
        dtype="int",
        semantic_meaning=(
            "Count of zones whose walking-distance-weighted shortest path to OUTSIDE passes "
            "through this candidate's edge. A STATIC structural proxy for how much of the "
            "building's total occupant-carrying demand depends on this candidate under normal "
            "shortest-path routing -- never how many currently do (that remains "
            "candidate_queue_length/candidate_approaching_count's job)."
        ),
        source="SIM/LIVE: IDENTICAL -- same function/class chain as candidate_betweenness_centrality above.",
        availability=FeatureAvailability.LIVE_OBSERVABLE,
        nullable=False,
        missing_value_note="Always a real computed value (0 for an edge on zero shortest paths) -- never None.",
    ),
)


CANDIDATE_FEATURE_SCHEMA_V4: Tuple[AIFeatureField, ...] = (
    CANDIDATE_FEATURE_SCHEMA + V4_PROMOTED_EXPERIMENTAL_FIELDS + V4_PROMOTED_GRAPH_CONTEXT_FIELDS
)

CANDIDATE_FEATURE_NAMES_V4: Tuple[str, ...] = tuple(field.name for field in CANDIDATE_FEATURE_SCHEMA_V4)

SCHEMA_VERSION_V4 = "4.0"


def field_by_name_v4(name: str) -> AIFeatureField:

    for field in CANDIDATE_FEATURE_SCHEMA_V4:

        if field.name == name:
            return field

    raise KeyError(f"{name!r} is not a field of CANDIDATE_FEATURE_SCHEMA_V4.")


# `topology_family`/`structural_variant_id` are dataset/campaign
# bookkeeping (which generator built a scenario) -- per the Cross-
# Topology Generalization Investigation's own Phase 14 audit, they
# NEVER belong in a production feature schema regardless of any
# diagnostic result, and are deliberately absent here, exactly like
# ROW_IDENTITY_COLUMNS (imported, re-exported for convenience, unchanged).
__all__ = [
    "CANDIDATE_FEATURE_SCHEMA_V4",
    "CANDIDATE_FEATURE_NAMES_V4",
    "SCHEMA_VERSION_V4",
    "field_by_name_v4",
    "ROW_IDENTITY_COLUMNS",
]
