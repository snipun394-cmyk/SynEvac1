from dataclasses import dataclass
from typing import Optional, Sequence

from behavior_recognition.observation import RecognizedBehavior

from live_occupants.occupant import LiveOccupant

from crowd_intelligence.flow import AssetSide, evaluate_approach, occupants_near_asset


DEFAULT_APPROACH_REGION_DEPTH = 3.0  # meters -- a documented project assumption (Phase 7), not a validated queueing-theory value


@dataclass(frozen=True)
class QueueMetrics:

    approaching_count: int = 0
    queue_candidate_count: int = 0
    estimated_queue_length: Optional[int] = None
    mean_approach_speed: Optional[float] = None


def compute_queue_metrics(
    occupants: Sequence[LiveOccupant], sides: Sequence[AssetSide], approach_region_depth: float = DEFAULT_APPROACH_REGION_DEPTH,
) -> QueueMetrics:

    # Phase 7's own pipeline: define a configurable approach region ->
    # find occupants inside it (crowd_intelligence.flow.occupants_near_
    # asset, itself already honest about missing world_position) ->
    # evaluate movement toward the asset -> detect slow/stationary
    # accumulation. NEVER fabricates a queue from zone occupancy alone --
    # every occupant counted here has an individually-verified world
    # position near this specific asset.

    nearby = occupants_near_asset(occupants, sides, approach_region_depth)

    if not nearby:
        # A genuine, position-confirmed zero -- the caller (crowd_
        # intelligence.engine.CrowdIntelligenceEngine) only ever reaches
        # this function once position_available has already been
        # established, so "nobody nearby" here is a real observed zero,
        # never the "position unavailable" case (that stays None, but
        # one layer up, on AssetApproachMetrics itself).
        return QueueMetrics(estimated_queue_length=0)

    approaching_count = 0
    queue_candidate_count = 0
    approach_speeds = []

    for occupant, _distance in nearby:

        evidence = evaluate_approach(occupant.world_position, occupant.current_floor_id, occupant.history.position_samples, sides)

        if evidence is not None and evidence.is_approaching:
            approaching_count += 1

        if evidence is not None and evidence.approach_speed is not None:
            approach_speeds.append(evidence.approach_speed)

        # "Slow/stationary accumulation" (Phase 7) -- reuses the SAME
        # behavior classification already computed by behavior_recognition
        # (RuleBasedBehaviorRecognizer, upstream of LiveOccupant.behavior)
        # rather than inventing a second, competing speed threshold here.
        if occupant.behavior == RecognizedBehavior.STATIONARY:
            queue_candidate_count += 1

    mean_approach_speed = (sum(approach_speeds) / len(approach_speeds)) if approach_speeds else None

    return QueueMetrics(
        approaching_count=approaching_count,
        queue_candidate_count=queue_candidate_count,
        estimated_queue_length=queue_candidate_count,
        mean_approach_speed=mean_approach_speed,
    )
