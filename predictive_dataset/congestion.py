from typing import Optional

from navigation.edge import Edge

from crowd_intelligence.capacity import door_capacity, exit_capacity, stair_capacity
from crowd_intelligence.congestion import CongestionThresholds, compute_congestion_level
from crowd_intelligence.queue import QueueMetrics


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone -- the ONE
# shared implementation of "candidate capacity" and "candidate
# congestion classification", called identically by both
# simulation_extractor.py and live_extractor.py (schema.py's own
# CANDIDATE_CONTEXT_SCHEMA documents this as literally the same code
# path on both sides, not just the same semantics). Neither extractor
# re-derives capacity or re-implements the demand/capacity ratio
# classification -- both call these two functions.
# =====================================================

_DEFAULT_THRESHOLDS = CongestionThresholds()


def candidate_capacity(candidate_type: str, edge: Edge, building) -> Optional[int]:

    reference = edge.reference

    if candidate_type == Edge.DOOR:
        return door_capacity(reference)

    if candidate_type == Edge.EXIT:
        return exit_capacity(reference)

    if candidate_type == Edge.STAIR:
        return stair_capacity(reference, building)

    return None


def candidate_congestion_level(
    queue_length: Optional[int],
    approaching_count: Optional[int],
    capacity: Optional[int],
    thresholds: CongestionThresholds = _DEFAULT_THRESHOLDS,
) -> Optional[str]:

    if queue_length is None or approaching_count is None:
        return None

    metrics = QueueMetrics(queue_candidate_count=queue_length, approaching_count=approaching_count)

    level = compute_congestion_level(metrics, capacity, thresholds)

    return level.name if level is not None else None
