from typing import Any, Dict

from predictive_dataset.candidate import CandidateIdentity
from predictive_dataset.graph_context_v4 import CandidateGraphContext
from predictive_dataset.schema_v4 import CANDIDATE_FEATURE_NAMES_V4
from predictive_dataset.simulation_extractor_v2_1 import extract_experimental_candidate_features


# =====================================================
# Predictive Dataset V4 milestone, Phase 1/2 -- THE SIMULATION
# EXTRACTOR for the 3 promoted graph-context fields, on top of V2.1's
# already-established 3 experimental fields (reused verbatim via
# extract_experimental_candidate_features -- V1/V2/V2.1's own modules
# are UNCHANGED by this milestone). `graph_context` is precomputed ONCE
# per Building via predictive_dataset.graph_context_v4.
# compute_graph_context_for_building() and threaded in by the campaign
# runner -- never recomputed per row/tick (same discipline
# alternative_route_counts already established in V2.1).
# =====================================================


def extract_v4_candidate_features(
    candidate: CandidateIdentity,
    edge,
    time: float,
    *,
    building,
    movement_result,
    occupancy_snapshot,
    alternative_route_counts: Dict[str, int],
    graph_context: Dict[str, CandidateGraphContext],
) -> Dict[str, Any]:
    """The frozen V1 9-field + V2.1's 3-field experimental dict, PLUS the
    3 promoted graph-context fields. Output keys are exactly
    CANDIDATE_FEATURE_NAMES_V4 (mechanically checked by
    tests/test_predictive_dataset_schema_v4.py)."""

    base = extract_experimental_candidate_features(
        candidate, edge, time,
        building=building, movement_result=movement_result, occupancy_snapshot=occupancy_snapshot,
        alternative_route_counts=alternative_route_counts,
    )

    context = graph_context.get(candidate.candidate_id)

    # graph_context is keyed by every candidate_id enumerate_candidates()
    # returns for this SAME building (Phase 1's "always a real computed
    # value" guarantee) -- a missing key here would indicate a genuine
    # building/graph_context mismatch bug, not honest missingness, so
    # this is intentionally NOT defensively defaulted to 0.0/False/0.
    if context is None:
        raise KeyError(
            f"No graph_context entry for candidate {candidate.candidate_id!r} -- "
            f"graph_context must be computed from the SAME building this candidate came from."
        )

    base["candidate_betweenness_centrality"] = context.betweenness_centrality
    base["candidate_is_bridge"] = context.is_bridge
    base["candidate_upstream_catchment_count"] = context.upstream_catchment_count

    assert set(base.keys()) == set(CANDIDATE_FEATURE_NAMES_V4), (
        f"extract_v4_candidate_features output keys diverged from CANDIDATE_FEATURE_NAMES_V4: "
        f"missing={set(CANDIDATE_FEATURE_NAMES_V4) - set(base.keys())}, "
        f"extra={set(base.keys()) - set(CANDIDATE_FEATURE_NAMES_V4)}"
    )

    return base
