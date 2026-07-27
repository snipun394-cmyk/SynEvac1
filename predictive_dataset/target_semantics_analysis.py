from typing import Any, Dict, List, Tuple

from predictive_dataset.target_generator import CONGESTION_THRESHOLD

# =====================================================
# Localized Predictive Model V2.2 milestone, Phase 7/8 -- ANALYSIS-ONLY
# machinery quantifying how much of the current congestion target
# (target_generator.CONGESTION_THRESHOLD concurrent occupants, any
# duration) is a zero-duration timestamp-boundary artifact vs. genuine
# sustained overlap, per candidate type. Pulled out of scripts/
# run_target_semantics_audit_v2_2.py into its own module specifically so
# this logic is unit-testable (this milestone's own "add focused tests
# for... target-semantics analysis" instruction) -- the script itself
# stays a thin orchestration loop over real simulation runs.
#
# NEVER imported by simulation_extractor.py/live_extractor.py or any of
# their v2_1 counterparts (this is POST-HOC analysis of a completed
# run's full timeline, the same "movement_result already holds the
# whole run" discipline target_generator.py itself already established
# -- reused, not a new leakage channel). Does NOT modify target_
# generator.py or predictive_dataset/schema.py; the production target
# is untouched.
# =====================================================


def episode_durations_and_gaps(movement_result, candidate_id: str) -> Tuple[List[float], List[float]]:
    """Returns (episode_durations, adjacent_gaps) for one candidate's
    edge, from its real occupant-step event timeline.

    episode_durations: every maximal interval where concurrent occupant
    count >= CONGESTION_THRESHOLD (the same event-sweep target_
    generator._congested_within_window uses internally, generalized to
    the whole run instead of one prediction window).

    adjacent_gaps: for every pair of temporally-adjacent occupant steps
    on this edge (sorted by start_time), the gap between one step's
    end_time and the next's start_time -- 0.0 means an exact FIFO
    handoff (no real-world idle time between the two occupants)."""

    steps = []
    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.edge.id == candidate_id:
                steps.append((step.start_time, step.end_time))

    if not steps:
        return [], []

    steps.sort()
    gaps = [steps[i + 1][0] - steps[i][1] for i in range(len(steps) - 1)]

    events = []
    for start, end in steps:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda e: (e[0], -e[1]))

    durations = []
    count = 0
    episode_start = None
    for t, delta in events:
        prev = count
        count += delta
        if prev < CONGESTION_THRESHOLD <= count:
            episode_start = t
        if prev >= CONGESTION_THRESHOLD > count and episode_start is not None:
            durations.append(t - episode_start)
            episode_start = None

    return durations, gaps


def counterfactual_positive(
    movement_result, candidate_id: str, time: float, horizon: float, min_duration: float,
) -> bool:
    """A candidate is counterfactually positive at `time` if, within
    (time, time+horizon], there exists an episode of >=CONGESTION_
    THRESHOLD concurrent occupants sustained for at least min_duration
    seconds. min_duration=0.0 reproduces target_generator.
    generate_candidate_label()'s real target exactly (any registered
    threshold-crossing, including a zero-duration boundary touch) --
    used by the audit script as its own internal consistency check."""

    steps = []
    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.edge.id == candidate_id:
                steps.append((step.start_time, step.end_time))

    if not steps:
        return False

    events = []
    for start, end in steps:
        if time < start <= time + horizon:
            events.append((start, 1))
        if time < end <= time + horizon:
            events.append((end, -1))

    count = sum(1 for start, end in steps if start <= time <= end)
    if count >= CONGESTION_THRESHOLD and min_duration <= 0.0:
        return True

    events.sort(key=lambda e: (e[0], -e[1]))
    episode_start = time if count >= CONGESTION_THRESHOLD else None

    for t, delta in events:
        prev = count
        count += delta
        if prev < CONGESTION_THRESHOLD <= count:
            episode_start = t
        if prev >= CONGESTION_THRESHOLD > count and episode_start is not None:
            if (t - episode_start) >= min_duration:
                return True
            episode_start = None

    if episode_start is not None and (time + horizon - episode_start) >= min_duration:
        return True

    return False


def summarize_durations(durations: List[float]) -> Dict[str, Any]:
    """Pure-Python summary (no numpy dependency) -- kept simple/testable;
    the campaign script itself uses numpy for the full-scale run, this
    is only used by unit tests to cross-check small fixtures."""

    if not durations:
        return {"n_episodes": 0}

    sorted_durations = sorted(durations)
    n = len(sorted_durations)

    def _percentile(p: float) -> float:
        index = min(n - 1, int(round(p / 100.0 * (n - 1))))
        return sorted_durations[index]

    zero_count = sum(1 for d in durations if d <= 1e-9)

    return {
        "n_episodes": n,
        "mean": sum(durations) / n,
        "median": _percentile(50.0),
        "p95": _percentile(95.0),
        "zero_duration_count": zero_count,
        "zero_duration_fraction": zero_count / n,
    }
