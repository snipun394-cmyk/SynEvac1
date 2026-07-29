import math
from typing import Optional, Tuple

from trajectory_intelligence.models import MovementStatus, TrajectoryConfig


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 3 -- purely geometric, per-occupant
# movement facts derived from LiveOccupant.world_position/world_velocity
# and LiveOccupant.history.position_samples. No Navigation Graph, no
# BuildingState, no route/safety reasoning here at all (that is route_
# progress.py's own, separate concern) -- this module only ever answers
# "how has this occupant physically moved," never "is that movement
# useful evacuation progress."
# =====================================================


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:

    return math.hypot(a[0] - b[0], a[1] - b[1])


def _confirmed_floor_change(a_floor_id: Optional[str], b_floor_id: Optional[str]) -> bool:

    # Observable Stair Perception milestone -- the smallest fix that
    # keeps this module's own documented boundary intact ("no Navigation
    # Graph, no BuildingState here at all"): PositionSample now carries
    # an OPTIONAL per-sample floor_id (live_occupants.history.
    # PositionSample), itself already-available positional/geometric
    # metadata, not a graph query. A CONFIRMED floor change (both known,
    # and different) means the raw Euclidean distance between these two
    # samples is being computed across two floors' own, UNRELATED local
    # coordinate spaces (see models.staircase.Staircase's own docstring
    # on why from_position/to_position never share one coordinate
    # system) -- physically meaningless, the exact artifact the prior
    # audit's Trajectory Intelligence section found (compute_movement_
    # facts() was floor-blind while route_progress.py's own
    # _floor_transition_uncertain()/_has_direct_stair_edge() already
    # correctly recognizes a legitimate Zone-A -> Stair -> Zone-B
    # crossing at the ROUTE layer). Returns False (i.e. "treat as safe
    # to compute") whenever floor context is not confirmed different --
    # either genuinely the same floor, or floor_id simply unavailable
    # (a pre-milestone caller, or a test object built without it) --
    # never suppressing legitimate same-floor movement data just because
    # floor context happens to be missing.

    if a_floor_id is None or b_floor_id is None:
        return False

    return a_floor_id != b_floor_id


def compute_movement_facts(occupant, config: TrajectoryConfig):

    # Returns a dict of the plain MovementFacts fields TrajectoryResult
    # wants (rather than a separate dataclass this module alone uses --
    # engine.py assembles the final TrajectoryResult from this plus
    # route_progress.py/anomaly.py's own contributions).

    samples = occupant.history.position_samples

    position_available = occupant.world_position is not None
    position_sample_count = len(samples)

    current_position = occupant.world_position
    previous_position = samples[-2].world_position if len(samples) >= 2 else None

    distance_travelled = None
    net_displacement = None
    trajectory_duration = None
    movement_direction = None

    if len(samples) >= 2:

        # Observable Stair Perception milestone -- a pair straddling a
        # CONFIRMED floor change (see _confirmed_floor_change() above)
        # contributes 0 to distance_travelled rather than a physically
        # meaningless cross-floor Euclidean jump; every other pair is
        # summed exactly as before.
        distance_travelled = sum(
            _distance(samples[i - 1].world_position, samples[i].world_position)
            for i in range(1, len(samples))
            if not _confirmed_floor_change(samples[i - 1].floor_id, samples[i].floor_id)
        )
        trajectory_duration = samples[-1].timestamp - samples[0].timestamp

        # net_displacement compares the WINDOW's own endpoints (first vs.
        # last sample), so it must be guarded by THEIR floor identity,
        # not the last consecutive pair's -- a genuinely different check
        # from the one below, even though both use the same helper.
        if not _confirmed_floor_change(samples[0].floor_id, samples[-1].floor_id):
            net_displacement = _distance(samples[0].world_position, samples[-1].world_position)

        last = samples[-1].world_position
        prior = samples[-2].world_position

        if last != prior and not _confirmed_floor_change(samples[-2].floor_id, samples[-1].floor_id):
            movement_direction = math.atan2(last[1] - prior[1], last[0] - prior[0])

    current_speed = occupant.world_velocity

    if current_speed is None and len(samples) >= 2 and not _confirmed_floor_change(samples[-2].floor_id, samples[-1].floor_id):

        dt = samples[-1].timestamp - samples[-2].timestamp

        if dt > 0:
            current_speed = _distance(samples[-2].world_position, samples[-1].world_position) / dt

    movement_status = MovementStatus.UNKNOWN

    if current_speed is not None:
        movement_status = (
            MovementStatus.STATIONARY if current_speed <= config.stationary_speed_threshold_m_s
            else MovementStatus.MOVING
        )

    zone_transition_history = tuple(
        record.to_zone_id for record in occupant.history.zone_transitions if record.to_zone_id is not None
    )

    return {
        "current_position": current_position,
        "previous_position": previous_position,
        "distance_travelled": distance_travelled,
        "net_displacement": net_displacement,
        "movement_direction": movement_direction,
        "current_speed": current_speed,
        "trajectory_duration": trajectory_duration,
        "zone_transition_history": zone_transition_history,
        "movement_status": movement_status,
        "position_available": position_available,
        "position_sample_count": position_sample_count,
    }


def is_stale(occupant, now: float, config: TrajectoryConfig) -> bool:

    # Phase 15/17 -- camera-offline / long cross-camera blind interval
    # honesty check. Deliberately reuses LiveOccupant.last_seen (the
    # SAME field live_occupants.lifecycle.is_expired() already uses for
    # occupant expiry) rather than inventing a second staleness clock --
    # this engine's own staleness_seconds is independently configurable
    # (and typically shorter) so trajectory evidence can honestly go
    # stale well before LiveOccupantManager's own expire_after_seconds
    # would drop the occupant entirely.

    return (now - occupant.last_seen) > config.staleness_seconds
