from typing import Any, Dict, Optional, Sequence

from predictive_dataset.candidate import CandidateIdentity
from predictive_dataset.graph_context_v4 import CandidateGraphContext
from predictive_dataset.live_extractor_v2_1 import extract_live_experimental_candidate_features

from stair_flow.models import StairFlowSnapshot


# =====================================================
# Predictive Dataset V4 milestone, Phase 3 -- THE LIVE EXTRACTOR for
# the 3 promoted graph-context fields. NOT wired into LiveRuntime this
# milestone (per the charter) -- this module exists solely to PROVE the
# live system can produce the same semantics a future model would
# consume, mirroring predictive_dataset/live_extractor_v2_1.py's own
# "deliberately thin, reuses whatever the live system already computes"
# discipline.
#
# `graph_context` is precomputed ONCE (at Designer-edit time, not per
# cycle) via predictive_dataset.graph_context_v4.
# compute_graph_context_for_building(building) against LiveRuntime's own
# `building` (== the Designer canvas's live models.building.Building,
# per live_runtime/factory.py/live_runtime_launcher/session.py's
# construction chain) -- the EXACT SAME function
# predictive_dataset/simulation_extractor_v4.py calls, proving sim/live
# parity by construction rather than by a second, independently-written
# implementation that could silently drift.
#
# Stair Predictive-Feature Live Parity milestone -- `stair_flow_snapshot`
# is likewise OPTIONAL and simply threaded through to
# extract_live_experimental_candidate_features() unchanged (see that
# module's own docstring and predictive_dataset.live_extractor_v2_1.
# build_stair_flow_snapshot_for_prediction() for how to build one). Still
# NOT wired into LiveRuntime -- this parameter only makes a genuinely
# parity-proven Stair candidate_recent_flow_rate available to a caller
# that supplies it, never enables model inference on its own.
# =====================================================


def extract_live_v4_candidate_features(
    candidate: CandidateIdentity,
    edge,
    time: float,
    *,
    building,
    crowd_snapshot,
    graph_context: Dict[str, CandidateGraphContext],
    occupancy_facts=None,
    alternative_route_counts: Dict[str, int],
    evacuation_snapshot=None,
    occupants: Optional[Sequence] = None,
    stair_flow_snapshot: Optional[StairFlowSnapshot] = None,
) -> Dict[str, Any]:

    base = extract_live_experimental_candidate_features(
        candidate, edge, time,
        building=building, crowd_snapshot=crowd_snapshot, occupancy_facts=occupancy_facts,
        alternative_route_counts=alternative_route_counts, evacuation_snapshot=evacuation_snapshot,
        occupants=occupants, stair_flow_snapshot=stair_flow_snapshot,
    )

    context = graph_context.get(candidate.candidate_id)

    if context is None:
        raise KeyError(
            f"No graph_context entry for candidate {candidate.candidate_id!r} -- "
            f"graph_context must be computed from the SAME building this candidate came from."
        )

    base["candidate_betweenness_centrality"] = context.betweenness_centrality
    base["candidate_is_bridge"] = context.is_bridge
    base["candidate_upstream_catchment_count"] = context.upstream_catchment_count

    return base
