from typing import Optional, Sequence, Tuple

from live_occupants.state import OccupantStatus


DEFAULT_EXIT_PROXIMITY_THRESHOLD = 2.0  # meters
DEFAULT_EXPIRE_AFTER_SECONDS = 30.0

# Configure this to match (or exceed) the SAME timeout_seconds a
# caller's cross_camera_identity.transition_model.TransitionModel
# already uses -- "respect CrossCameraIdentity timeout policy" (Phase
# 6) is satisfied as a CONFIGURATION convention here, not a live query
# into that package's own registry, keeping live_occupants' own
# dependency surface minimal (it never imports cross_camera_identity
# at all). An occupant should never be reported EXPIRED here before its
# underlying global identity itself expires there.


def next_status_on_update(current_status: OccupantStatus) -> OccupantStatus:

    # Live Occupant Digital Twin milestone, Phase 6 -- any update() call
    # (first-ever, or a recovery from TEMPORARILY_LOST/EXITED) reports
    # ACTIVE. NEW is reserved exclusively for LiveOccupantManager's own
    # very-first construction of a LiveOccupant -- this function is
    # never consulted for that case (there is no "current_status" yet).
    # A recovery from EXITED simply means the earlier "likely left
    # through this exit" guess was wrong -- honestly corrected the
    # moment real evidence (a fresh sighting) contradicts it, never
    # left stuck on a stale guess.

    return OccupantStatus.ACTIVE


def next_status_on_missing(current_status: OccupantStatus, near_exit: bool) -> OccupantStatus:

    # Called once per occupant per cycle they were NOT observed at all
    # (live_occupants.manager.LiveOccupantManager.sweep_missing()).
    # EXPIRED is terminal and idempotent (defensive -- an EXPIRED
    # occupant is normally already removed from the manager's own
    # store, so this branch should not normally be reached).

    if current_status == OccupantStatus.EXPIRED:
        return OccupantStatus.EXPIRED

    if near_exit:
        return OccupantStatus.EXITED

    return OccupantStatus.TEMPORARILY_LOST


def is_expired(last_seen: float, now: float, expire_after_seconds: float) -> bool:

    return (now - last_seen) > expire_after_seconds


# =====================================================
# Exit proximity (a pure geometry helper -- Phase 13's own "Manager
# depends only on ... Geometry" allowance; models.exit.Exit itself is
# never imported here, only its already-established start_point/
# end_point shape, so this function is testable with any plain object
# exposing that shape.)
# =====================================================


def _distance_point_to_segment(point: Tuple[float, float], seg_start: Tuple[float, float], seg_end: Tuple[float, float]) -> float:

    px, py = point
    ax, ay = seg_start
    bx, by = seg_end

    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy

    if length_squared == 0.0:
        # Degenerate (zero-length) segment -- distance to the single point.
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5

    t = ((px - ax) * dx + (py - ay) * dy) / length_squared
    t = max(0.0, min(1.0, t))

    closest_x = ax + t * dx
    closest_y = ay + t * dy

    return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5


def is_near_exit(
    world_position: Optional[Tuple[float, float]],
    exits: Sequence[object],
    threshold: float,
) -> bool:

    # An honest, computed geometric proximity check -- never a guess
    # dressed up as certainty (see live_occupants.state.OccupantStatus.
    # EXITED's own docstring on why it stays recoverable). False
    # whenever there is genuinely nothing to check against (no world
    # position, or no exits supplied) -- never a fabricated default.

    if world_position is None or not exits:
        return False

    for exit_obj in exits:

        distance = _distance_point_to_segment(world_position, tuple(exit_obj.start_point), tuple(exit_obj.end_point))

        if distance <= threshold:
            return True

    return False
