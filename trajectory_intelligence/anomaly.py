import math
from typing import Optional, Tuple

from trajectory_intelligence.history import OccupantTrajectoryState
from trajectory_intelligence.models import AnomalyFlag, RouteProgressStatus, TrajectoryConfig, hazard_severity_at_least


# =====================================================
# Live Occupant Trajectory, Movement Anomaly & Route-Deviation
# Intelligence milestone, Phase 8/9/10/11 -- single-occupant anomaly
# detection. Every function returns a plain bool/flag (never mutates
# anything) -- engine.py is the only place flags are assembled onto a
# TrajectoryResult. Every detector requires PERSISTENCE (a configured
# minimum sample count and/or duration) before flagging -- Phase 8/9's
# own explicit "do not trigger from one noisy frame" requirement.
# =====================================================


def moving_away_from_safe_exit(history_state: OccupantTrajectoryState, now: float, config: TrajectoryConfig) -> bool:

    samples = history_state.route_distance_samples

    if len(samples) < 2:
        return False

    streak = 0

    for i in range(len(samples) - 1, 0, -1):

        if samples[i].distance - samples[i - 1].distance >= config.route_distance_tolerance_m:
            streak += 1
        else:
            break

    if streak < config.moving_away_min_samples - 1:
        return False

    duration = samples[-1].timestamp - samples[-(streak + 1)].timestamp

    return duration >= config.moving_away_min_duration_seconds


# =====================================================


def movement_stalled(history_state: OccupantTrajectoryState, config: TrajectoryConfig) -> bool:

    # Phase 10 -- insufficient ROUTE progress for a configurable period,
    # entirely independent of physical movement/RecognizedBehavior.
    # STATIONARY (Phase 27 test 12's own "stationary alone must not
    # equal stall" requirement) -- this only ever reads route_distance_
    # samples, never occupant.behavior/human_state.

    samples = history_state.route_distance_samples

    if len(samples) < 2:
        return False

    running_min = samples[0].distance
    last_improvement_time = samples[0].timestamp

    for sample in samples[1:]:

        if sample.distance < running_min - config.route_distance_tolerance_m:
            running_min = sample.distance
            last_improvement_time = sample.timestamp

    return (samples[-1].timestamp - last_improvement_time) >= config.movement_stall_duration_seconds


# =====================================================


def repeated_route_reversal(occupant, config: TrajectoryConfig) -> bool:

    zone_transitions = occupant.history.zone_transitions

    zone_sequence = tuple(
        record.to_zone_id for record in zone_transitions[-config.reversal_window_transitions:]
        if record.to_zone_id is not None
    )

    if len(zone_sequence) >= 3:
        return _count_ab_reversals(zone_sequence) >= config.reversal_min_events

    # World-space fallback (Phase 9) -- only engaged when zone identity
    # is unavailable/too sparse to judge a zone-level reversal.
    position_samples = occupant.history.position_samples[-config.reversal_window_transitions:]

    if len(position_samples) < 3:
        return False

    directions = []

    for i in range(1, len(position_samples)):

        a = position_samples[i - 1].world_position
        b = position_samples[i].world_position

        if a == b:
            continue

        directions.append(math.atan2(b[1] - a[1], b[0] - a[0]))

    return _count_direction_reversals(directions, config.reversal_min_angle_degrees) >= config.reversal_min_events


def _count_ab_reversals(sequence: Tuple[str, ...]) -> int:

    count = 0

    for i in range(2, len(sequence)):

        if sequence[i] == sequence[i - 2] and sequence[i] != sequence[i - 1]:
            count += 1

    return count


def _count_direction_reversals(directions, min_angle_degrees: float) -> int:

    count = 0
    min_angle_radians = math.radians(min_angle_degrees)

    for i in range(1, len(directions)):

        delta = abs(directions[i] - directions[i - 1])
        delta = min(delta, 2 * math.pi - delta)

        if delta >= min_angle_radians:
            count += 1

    return count


# =====================================================


def currently_in_hazardous_zone(occupant, building_state, config: TrajectoryConfig) -> bool:

    # Phase 11 -- zone-level ONLY (no live edge-level hazard gradient
    # exists downstream of BuildingStateEstimator, see route_progress.
    # py's own module docstring) -- MOVES_TOWARD_HAZARD is deliberately
    # not computed anywhere in this package for the same reason.
    #
    # ENTERED_HAZARDOUS_ZONE vs REMAINS_IN_HAZARDOUS_ZONE is a cross-
    # cycle distinction engine.py makes by comparing this cycle's result
    # against OccupantTrajectoryState.was_in_hazardous_zone -- this
    # function only ever answers the single-cycle question.

    if occupant.current_zone_id is None or building_state is None or building_state.hazard_summary is None:
        return False

    severity = building_state.hazard_summary.zone_severities.get(occupant.current_zone_id)

    return hazard_severity_at_least(severity, config.hazard_zone_entry_severity_floor)
