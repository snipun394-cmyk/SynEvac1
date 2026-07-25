from dataclasses import dataclass
from typing import Optional


# =====================================================
# Per-Candidate Predictive AI Data Foundation milestone, Phase 9/10 --
# THE TARGET GENERATOR. This is the ONLY module in predictive_dataset/
# allowed to inspect what happens strictly after `time` -- everything
# in simulation_extractor.py must stay ignorant of it (Phase 8/9's own
# "keep this boundary extremely obvious in code" instruction). Never
# imported by simulation_extractor.py, live_extractor.py, or anything
# that ships live; only by predictive_dataset.dataset_builder (offline
# dataset assembly) and predictive_dataset's own tests.
#
# CONGESTION DEFINITION -- reused, not reinvented (Phase 5's own "do
# not invent a second incompatible definition"): ground_truth.
# bottleneck._congestion_duration_for_edge()'s own already-audited,
# disclosed policy threshold ("'Congested' is defined here as 2+
# occupants concurrently on the same edge") is the SAME threshold used
# below, now evaluated PER CANDIDATE PER TIMESTEP instead of as a
# whole-run duration. This is target-only, simulation-ground-truth-only
# machinery -- it reads exact edge-occupancy intervals a live deployment
# could never honestly reconstruct after the fact, which is exactly why
# it belongs here and not in a live-parity-constrained feature.
#
# CURRENTLY_CONGESTED vs. FUTURE_CONGESTION_TARGET (Phase 5/10) -- a
# candidate already congested AT `time` gets currently_congested=True
# and target=None (NOT APPLICABLE, not a fabricated trivial positive):
# "will this become congested in the next T seconds" is not a
# meaningful question to ask about something already congested right
# now. Phase 15's class-balance analysis reports these rows separately
# rather than silently dropping or silently counting them.
# =====================================================


CONGESTION_THRESHOLD = 2  # occupants concurrently on the candidate's edge -- ground_truth.bottleneck's own value


@dataclass(frozen=True)
class CandidateLabel:

    candidate_id: str
    observation_time: float
    horizon: float

    currently_congested: bool

    # None only when currently_congested is True (Phase 10's "not
    # applicable" policy). Otherwise True/False, always a real answer
    # -- there is no additional "unknown" state for a completed
    # simulation run (unlike the live-only "position_available" concept
    # elsewhere in this codebase, target generation always has full
    # ground truth for its own already-completed run).
    target: Optional[bool]

    # Phase 10 case E diagnostic -- whether ANY occupant was recorded on
    # this candidate's edge at all during (time, time + horizon]. False
    # alongside target=False distinguishes "definitively unused/
    # unavailable during the window" from "used, but never reached the
    # congestion threshold" -- both are legitimate negatives, but this
    # flag keeps the distinction visible for analysis rather than
    # collapsing them.
    had_any_activity_in_window: bool


def generate_candidate_label(
    candidate_id: str,
    movement_result,
    time: float,
    horizon: float,
    *,
    congestion_threshold: int = CONGESTION_THRESHOLD,
) -> CandidateLabel:

    currently_congested = _edge_occupant_count(movement_result, candidate_id, time) >= congestion_threshold

    window_start = time
    window_end = time + horizon

    if currently_congested:
        target = None
    else:
        target = _congested_within_window(
            movement_result, candidate_id, window_start, window_end, congestion_threshold,
        )

    had_activity = _had_any_activity_in_window(movement_result, candidate_id, window_start, window_end)

    return CandidateLabel(
        candidate_id=candidate_id,
        observation_time=time,
        horizon=horizon,
        currently_congested=currently_congested,
        target=target,
        had_any_activity_in_window=had_activity,
    )


# =====================================================


def _edge_occupant_count(movement_result, candidate_id: str, time: float) -> int:

    count = 0

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:

            if step.edge.id != candidate_id:
                continue

            if step.start_time <= time <= step.end_time:
                count += 1

    return count


# =====================================================


def _congested_within_window(
    movement_result, candidate_id: str, window_start: float, window_end: float, threshold: int,
) -> bool:

    count = _edge_occupant_count(movement_result, candidate_id, window_start)

    if count >= threshold:
        return True

    events = []

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:

            if step.edge.id != candidate_id:
                continue

            if window_start < step.start_time <= window_end:
                events.append((step.start_time, 1))

            if window_start < step.end_time <= window_end:
                events.append((step.end_time, -1))

    # Arrivals (+1) processed before departures (-1) at the same
    # instant -- a simultaneous arrival/departure should still register
    # the momentary peak, not silently skip past the threshold.
    events.sort(key=lambda event: (event[0], -event[1]))

    for _, delta in events:

        count += delta

        if count >= threshold:
            return True

    return False


# =====================================================


def _had_any_activity_in_window(movement_result, candidate_id: str, window_start: float, window_end: float) -> bool:

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:

            if step.edge.id != candidate_id:
                continue

            if step.start_time <= window_end and step.end_time >= window_start:
                return True

    return False
