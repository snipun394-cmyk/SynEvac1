from typing import Any, Dict, Optional

from navigation.edge import Edge
from navigation.node import Node

from predictive_dataset.candidate import CandidateIdentity
from predictive_dataset.congestion import candidate_capacity as sim_style_capacity


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone, Phase 12 --
# THE LIVE EXTRACTOR. Deliberately thin: crowd_intelligence.engine.
# CrowdIntelligenceEngine already computes genuine per-candidate
# (door_metrics/exit_metrics/stair_metrics, keyed by the SAME asset id
# this package uses as candidate_id) queue/approach/congestion metrics
# every LiveRuntime cycle -- this module reuses that CrowdIntelligenceSnapshot
# directly rather than building a second, competing live intelligence
# engine (explicitly out of scope for this milestone). The one thing
# this module adds is remapping AssetApproachMetrics' shape onto
# predictive_dataset.schema.CANDIDATE_FEATURE_SCHEMA's field names, so
# a caller gets the SAME schema simulation_extractor.py produces.
#
# `occupancy_facts` is whatever live_occupants.manager.LiveOccupantManager.
# canonical_occupancy(time) already returns (the SAME canonical grouping
# crowd_intelligence/evacuation_progress/emergency_response all already
# read, per docs/architecture/canonical_live_occupancy.md) -- passed in
# by the caller rather than fetched here, since this module has no
# LiveOccupantManager reference of its own and must not construct one.
# =====================================================


def extract_live_candidate_features(
    candidate: CandidateIdentity,
    edge,
    *,
    building,
    crowd_snapshot,
    occupancy_facts=None,
) -> Dict[str, Any]:

    metrics = _metrics_for(candidate, crowd_snapshot)

    capacity = (
        metrics.simulation_style_capacity if metrics is not None
        else sim_style_capacity(candidate.candidate_type, edge, building)
    )

    if metrics is not None and metrics.position_available:
        queue_length: Optional[int] = metrics.queue_candidate_count
        approaching_count: Optional[int] = metrics.approaching_count
        congestion_level = metrics.congestion_level.name if metrics.congestion_level is not None else None
    else:
        queue_length = None
        approaching_count = None
        congestion_level = None

    return {
        "total_active_occupant_count": (
            occupancy_facts.total_observed_count if occupancy_facts is not None else None
        ),
        "candidate_type": candidate.candidate_type,
        "candidate_capacity": capacity,
        "candidate_walking_distance": edge.walking_distance,
        "candidate_traversable": edge.traversable,
        "candidate_adjacent_zone_occupancy": _adjacent_zone_occupancy(edge, occupancy_facts),
        "candidate_queue_length": queue_length,
        "candidate_approaching_count": approaching_count,
        "candidate_congestion_level": congestion_level,
    }


# =====================================================


def _metrics_for(candidate: CandidateIdentity, crowd_snapshot):

    if crowd_snapshot is None:
        return None

    if candidate.candidate_type == Edge.DOOR:
        return crowd_snapshot.door(candidate.candidate_id)

    if candidate.candidate_type == Edge.EXIT:
        return crowd_snapshot.exit(candidate.candidate_id)

    if candidate.candidate_type == Edge.STAIR:
        return crowd_snapshot.stair(candidate.candidate_id)

    return None


# =====================================================


def _adjacent_zone_occupancy(edge, occupancy_facts) -> Optional[int]:

    if occupancy_facts is None:
        return None

    if edge.from_node == Node.OUTSIDE_NODE_ID:
        return None

    occupant_ids = occupancy_facts.occupant_ids_by_zone.get(edge.from_node)

    return len(occupant_ids) if occupant_ids is not None else 0
