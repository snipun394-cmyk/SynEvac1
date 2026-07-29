from typing import FrozenSet, Mapping, Sequence, Tuple

from stair_perception.models import StairObservation, StairObservationStatus, StairOccupancySnapshot


def compute_stair_occupancy_snapshot(
    stair_ids: Sequence[str],
    occupant_ids_by_stair: Mapping[str, Tuple[str, ...]],
    covered_stair_ids: FrozenSet[str],
    timestamp: float,
) -> StairOccupancySnapshot:

    # Observable Stair Perception milestone -- a PURE grouping function,
    # exactly mirroring live_occupants.occupancy.compute_occupancy_facts()'s
    # own "does NOT decide who counts, does NOT re-scan raw occupants"
    # convention: `occupant_ids_by_stair` is taken directly from an
    # already-computed live_occupants.occupancy.OccupancyFacts (Phase
    # 10's own "do not reintroduce duplicate occupancy computations"),
    # and `covered_stair_ids` is taken directly from camera_calibration.
    # stair_lookup.covered_stair_ids() (Phase 19's own geometrically-
    # derived coverage answer) -- this function only combines the two
    # into the OBSERVED-vs-UNKNOWN distinction Phase 18 requires.
    #
    # `stair_ids`: every Staircase id that exists in the Building this
    # cycle (so a genuinely-covered stair with zero current occupants
    # still gets an explicit OBSERVED/empty entry, never silently
    # absent -- StairOccupancySnapshot.observation_for()'s own default
    # already handles "entirely absent" as UNKNOWN, but being explicit
    # here is what lets a caller enumerate "every stair we know about,"
    # not just the ones that happened to have someone on them).

    observations = {}

    for stair_id in stair_ids:

        occupant_ids = occupant_ids_by_stair.get(stair_id, ())

        if stair_id in covered_stair_ids:

            observations[stair_id] = StairObservation(
                stair_id=stair_id,
                status=StairObservationStatus.OBSERVED,
                occupant_ids=tuple(occupant_ids),
            )

        else:

            # UNKNOWN -- no honest basis to report an occupancy count,
            # regardless of whether occupant_ids_by_stair happens to
            # (impossibly, absent a bug elsewhere) carry an entry for
            # this id. occupant_ids stays deliberately empty -- see
            # StairObservation.occupant_ids' own docstring.
            observations[stair_id] = StairObservation(
                stair_id=stair_id,
                status=StairObservationStatus.UNKNOWN,
            )

    return StairOccupancySnapshot(timestamp=timestamp, observations=observations)
