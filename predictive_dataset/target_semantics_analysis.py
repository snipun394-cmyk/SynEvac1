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


def _interval_episodes(intervals: List[Tuple[float, float]], threshold: int) -> List[Tuple[float, float, float]]:
    """Shared event-sweep primitive: given a list of (start, end) real
    intervals (any semantics -- edge-occupancy OR queue-membership),
    return (episode_start, episode_end, duration) for every maximal
    sub-interval where the number of concurrently-open intervals is
    >= threshold. `end` is treated as exclusive here (matches
    predictive_dataset.simulation_extractor._current_queue_length's own
    `join_time <= time < start_time` half-open convention) -- callers
    using inclusive occupancy intervals pass (start_time, end_time) and
    accept that a same-instant handoff still registers a momentary
    (possibly zero-duration) crossing, exactly like target_generator.py's
    own existing behavior."""

    if not intervals:
        return []

    events = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda e: (e[0], -e[1]))

    episodes = []
    count = 0
    episode_start = None
    for t, delta in events:
        prev = count
        count += delta
        if prev < threshold <= count:
            episode_start = t
        if prev >= threshold > count and episode_start is not None:
            episodes.append((episode_start, t, t - episode_start))
            episode_start = None

    return episodes


def _interval_episode_durations(intervals: List[Tuple[float, float]], threshold: int) -> List[float]:
    """Durations only -- thin wrapper over _interval_episodes()."""

    return [duration for _, _, duration in _interval_episodes(intervals, threshold)]


def queue_intervals(movement_result, candidate_id: str) -> List[Tuple[float, float]]:
    """Every occupant's REAL waiting interval [join_time, start_time) on
    this candidate's edge -- join_time = step.start_time - step.
    queue_wait_time, the SAME quantity predictive_dataset.simulation_
    extractor._current_queue_length already reads. This represents
    "waiting to be admitted onto the edge" (a BEFORE-the-candidate
    phenomenon), distinct from queue_episode_durations' occupancy
    sibling (which represents "physically present ON the candidate's
    edge")."""

    intervals = []
    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.edge.id != candidate_id:
                continue
            join_time = step.start_time - step.queue_wait_time
            if step.start_time > join_time:  # zero-length waits contribute no interval
                intervals.append((join_time, step.start_time))
    return intervals


def occupancy_intervals(movement_result, candidate_id: str) -> List[Tuple[float, float]]:
    """Every occupant's [start_time, end_time] PRESENCE interval on this
    candidate's edge -- "physically on the edge", as opposed to
    queue_intervals()'s "waiting to be admitted"."""

    intervals = []
    for timeline in movement_result.occupants.values():
        for step in timeline.steps:
            if step.edge.id == candidate_id:
                intervals.append((step.start_time, step.end_time))
    return intervals


def queue_episodes(movement_result, candidate_id: str, threshold: int = 1) -> List[Tuple[float, float, float]]:
    """(episode_start, episode_end, duration) for every maximal interval
    where >=`threshold` occupants are simultaneously WAITING (queued,
    not yet admitted) for this candidate's edge. Unlike occupancy
    episodes, a queue episode's boundaries come from continuous
    queue_wait_time accumulation, not a single admission instant -- it
    cannot degenerate to zero duration the way a FIFO occupancy handoff
    does, UNLESS every wait on this edge happens to be genuinely
    instantaneous (queue_wait_time == 0 for everyone)."""

    return _interval_episodes(queue_intervals(movement_result, candidate_id), threshold)


def occupancy_episodes(movement_result, candidate_id: str, threshold: int = 2) -> List[Tuple[float, float, float]]:
    """(episode_start, episode_end, duration) for every maximal interval
    where >=`threshold` occupants are simultaneously PRESENT on this
    candidate's edge (inclusive [start_time, end_time] intervals --
    matches target_generator.py's own _edge_occupant_count semantics
    exactly, so threshold=2 here reproduces the production Target V1
    definition)."""

    return _interval_episodes(occupancy_intervals(movement_result, candidate_id), threshold)


def queue_episode_durations(movement_result, candidate_id: str, threshold: int = 1) -> List[float]:
    """Durations only -- thin wrapper over queue_episodes()."""

    return [d for _, _, d in queue_episodes(movement_result, candidate_id, threshold)]


def occupancy_episode_durations(movement_result, candidate_id: str, threshold: int = 2) -> List[float]:
    """Durations only -- thin wrapper over occupancy_episodes()."""

    return [d for _, _, d in occupancy_episodes(movement_result, candidate_id, threshold)]


# =====================================================
# ONSET / PERSISTENCE semantics -- shared by the threshold-sensitivity
# sweep (this milestone's Phase 4) and predictive_dataset.
# target_generator_v2 (the frozen definition, Phase 8). A "qualifying"
# episode is one whose FULL (retrospectively-known) duration meets a
# minimum-persistence bar; "currently congested" and "onset" are BOTH
# evaluated using only the ELAPSED portion of that episode as of the
# time in question (t - episode_start >= min_duration), never its
# eventual future duration -- this is what keeps the definition
# leak-safe: target-generation code may inspect (t, t+horizon] (the
# established, existing rule -- predictive_dataset.target_generator.py's
# own module docstring), but "was t itself already congested" must only
# ever depend on information at or before t.
# =====================================================


def qualifying_onset_times(episodes: List[Tuple[float, float, float]], min_duration: float) -> List[Tuple[float, float]]:
    """For every episode whose FULL duration >= min_duration, returns
    (onset_time, episode_end) where onset_time = episode_start +
    min_duration -- the first instant at which that episode's ELAPSED
    duration reaches the persistence bar (never its start, unless
    min_duration is 0)."""

    return [
        (start + min_duration, end)
        for start, end, duration in episodes
        if duration >= min_duration
    ]


def is_persistently_congested(onset_times: List[Tuple[float, float]], time: float) -> bool:
    """True iff `time` falls within [onset_time, episode_end) for some
    qualifying episode -- i.e. a persistence-qualifying episode is
    ALREADY (as of `time`) underway. Only ever compares `time` against
    onset_time <= time, itself derived from an episode_start <= time --
    never inspects anything after `time`."""

    return any(onset <= time < end for onset, end in onset_times)


def persistent_onset_within_window(
    onset_times: List[Tuple[float, float]], time: float, horizon: float,
) -> bool:
    """True iff some qualifying episode's onset_time falls strictly
    within (time, time + horizon] -- "a new, meaningfully-persistent
    congestion condition begins within the prediction window". Callers
    must check `not is_persistently_congested(onset_times, time)` first
    (the "already congested -> not applicable" exclusion) -- this
    function does not repeat that check itself."""

    return any(time < onset <= time + horizon for onset, _ in onset_times)


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
