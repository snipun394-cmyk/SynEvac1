from dataclasses import dataclass
from typing import List, Optional, Tuple

from predictive_dataset.target_semantics_analysis import (
    is_persistently_congested,
    occupancy_episodes,
    persistent_onset_within_window,
    qualifying_onset_times,
    queue_episodes,
)

# =====================================================
# Predictive Congestion Target V2 milestone -- THE FROZEN TARGET
# DEFINITION. This is a NEW, SEPARATE module from predictive_dataset/
# target_generator.py (Target V1), which is left COMPLETELY UNTOUCHED
# and remains byte-for-byte reproducible -- this milestone's own Hard
# Rule #3 ("do not silently redefine congestion") means V1 stays a real,
# selectable target version forever, not deleted or edited in place.
#
# WHY TARGET V1 WAS INVALID FOR DOOR/STAIR (docs/architecture/
# predictive_congestion_target_v2.md has the full investigation):
# V1 defines "congested" as >=2 occupants CONCURRENTLY ON a candidate's
# edge, at ANY duration. For Door/Stair, whose capacity is structurally
# 1 (Door's derived-from-width default floors to 1; Stair's derived-
# from-width-and-vertical-travel default also floors to 1 -- see
# docs/architecture/localized_predictive_model_v2_2.md Section 2), the
# discrete-event simulator's admission-queue mechanism (simulator/
# coordinator.py) NEVER lets 2 occupants genuinely overlap on the edge
# -- the only way V1's ">=2 concurrent" threshold ever fires for Door/
# Stair is the exact INSTANT one occupant's admission ends and the next
# occupant (already queued) is admitted, a duration-ZERO artifact of
# target_generator.py's own inclusive `start_time <= time <= end_time`
# bounds. Confirmed exhaustively at full V2.2 scale: 100% of Door's
# 54,832 and 100% of Stair's 31,419 V1 "congestion" episodes have
# EXACTLY zero duration. Exit (capacity 50, effectively unconstrained
# for these scenarios) never hits this admission ceiling, so its V1
# episodes are genuine sustained overlap (mean 20.4s) -- V1 was never
# invalid for Exit.
#
# TARGET V2 FORMULA -- "v2-persistent-demand-service-imbalance":
# ONE universal definition, applied identically to every candidate type
# (no type-conditional branching in this module) -- built from TWO
# raw signals every candidate already has real intervals for:
#
#   QUEUE persistence: >=1 occupant continuously WAITING (already
#     admission-queued, not yet on the edge -- see queue_intervals())
#     for the SAME candidate, sustained for >= MIN_PERSISTENCE_SECONDS.
#
#   OCCUPANCY persistence: >=2 occupants continuously PRESENT on the
#     candidate's edge (V1's own threshold, reused not reinvented),
#     sustained for >= MIN_PERSISTENCE_SECONDS.
#
# A candidate is "meaningfully congested" if EITHER mechanism's episode
# has persisted for >= MIN_PERSISTENCE_SECONDS. Because Door/Stair's
# capacity-1 admission model means they STRUCTURALLY never produce a
# nonzero-duration occupancy episode (proven above), and Exit's
# capacity-50 admission model means it STRUCTURALLY never produces ANY
# queue episode at all (confirmed empirically: 0 queue episodes across
# every V2.2-topology scenario checked), this single OR'd formula
# automatically reduces to "queue-based" for Door/Stair and
# "occupancy-based" for Exit WITHOUT any explicit per-type rule --
# satisfying this milestone's own Phase 6 preference for one universal
# formula over arbitrary type-specific labels.
#
# THRESHOLDS -- evidence-based, not invented (docs/architecture/
# predictive_congestion_target_v2.md Section 5's full sensitivity
# sweep): at MIN_PERSISTENCE_SECONDS=3.0s, Door retains 98.3% and Stair
# retains 90.5% of their real queue episodes (a persistence floor that
# only discards genuinely momentary blips), and Exit retains 79.2% of
# its real occupancy episodes. The SAME 3.0s value is used for both
# mechanisms deliberately -- a per-mechanism-tuned pair of different
# constants would risk exactly the "arbitrary type-specific tuning to
# improve class balance" this milestone's own Phase 6 prohibits; one
# shared persistence floor, applied to whichever raw signal is
# structurally meaningful for that candidate, is not arbitrary.
#
# ONSET SEMANTICS (Phase 21 of this milestone): target=1 iff the
# candidate is NOT already meaningfully congested at `time` (an
# already-congested candidate returns target=None, "not applicable",
# exactly like V1's own currently_congested policy) AND a qualifying
# episode's persistence bar is FIRST CROSSED within (time, time+horizon]
# -- "will congestion BEGIN", not "does congestion exist at some point
# during the window". Both currently_congested(time) and the onset
# check are evaluated using only the ELAPSED portion of each episode as
# of the time being checked (predictive_dataset.target_semantics_
# analysis.qualifying_onset_times/is_persistently_congested/
# persistent_onset_within_window's own leak-safety guarantee) -- never
# an episode's full, only-known-in-retrospect duration.
#
# LEAKAGE BOUNDARY: same discipline as target_generator.py itself --
# this module is the ONLY place (besides target_generator.py) allowed
# to inspect (time, time+horizon]; simulation_extractor.py/
# live_extractor.py and their v2_1 counterparts must never import it
# (tests/test_predictive_dataset_target_v2_architecture_guards.py).
# =====================================================


TARGET_VERSION_V2 = "v2-persistent-demand-service-imbalance"

QUEUE_THRESHOLD = 1  # occupants; Door/Stair's own structural capacity is 1, so ANY wait is demand > capacity
OCCUPANCY_THRESHOLD = 2  # V1's own CONGESTION_THRESHOLD, reused verbatim -- see predictive_dataset.target_generator
MIN_PERSISTENCE_SECONDS = 3.0  # evidence-based floor, see module docstring


@dataclass(frozen=True)
class CandidateLabelV2:

    candidate_id: str
    observation_time: float
    horizon: float

    currently_congested: bool

    # None only when currently_congested is True -- same "not
    # applicable" policy as Target V1's CandidateLabel.
    target: Optional[bool]

    had_any_activity_in_window: bool

    target_version: str = TARGET_VERSION_V2


def compute_qualifying_onsets(
    movement_result,
    candidate_id: str,
    *,
    queue_threshold: int = QUEUE_THRESHOLD,
    occupancy_threshold: int = OCCUPANCY_THRESHOLD,
    min_persistence_seconds: float = MIN_PERSISTENCE_SECONDS,
) -> List[Tuple[float, float]]:
    """The (onset_time, episode_end) list for ONE candidate, computed
    ONCE per (scenario, candidate) -- NOT per tick. Callers generating
    labels across many observation times for the same candidate MUST
    precompute this once and pass it to generate_candidate_label_v2()
    via `onsets=`, rather than letting it recompute the full episode
    sweep on every call -- episodes depend only on candidate_id (via
    movement_result), never on `time`, so recomputing per-tick would be
    pure wasted work at campaign scale (this milestone's own Phase 24
    "avoid unnecessary full-row duplication / measure throughput"
    instruction)."""

    queue_onsets = qualifying_onset_times(
        queue_episodes(movement_result, candidate_id, threshold=queue_threshold), min_persistence_seconds,
    )
    occupancy_onsets = qualifying_onset_times(
        occupancy_episodes(movement_result, candidate_id, threshold=occupancy_threshold), min_persistence_seconds,
    )
    return queue_onsets + occupancy_onsets


def generate_candidate_label_v2(
    candidate_id: str,
    movement_result,
    time: float,
    horizon: float,
    *,
    onsets: Optional[List[Tuple[float, float]]] = None,
    queue_threshold: int = QUEUE_THRESHOLD,
    occupancy_threshold: int = OCCUPANCY_THRESHOLD,
    min_persistence_seconds: float = MIN_PERSISTENCE_SECONDS,
) -> CandidateLabelV2:
    """Pass a precomputed `onsets` (from compute_qualifying_onsets(),
    called once per candidate) when labeling many ticks for the same
    candidate -- if omitted, this recomputes it from scratch (correct,
    just slower; convenient for tests/one-off calls)."""

    if onsets is None:
        onsets = compute_qualifying_onsets(
            movement_result, candidate_id,
            queue_threshold=queue_threshold, occupancy_threshold=occupancy_threshold,
            min_persistence_seconds=min_persistence_seconds,
        )

    currently_congested = is_persistently_congested(onsets, time)

    if currently_congested:
        target = None
    else:
        target = persistent_onset_within_window(onsets, time, horizon)

    had_activity = _had_any_activity_in_window(movement_result, candidate_id, time, time + horizon)

    return CandidateLabelV2(
        candidate_id=candidate_id,
        observation_time=time,
        horizon=horizon,
        currently_congested=currently_congested,
        target=target,
        had_any_activity_in_window=had_activity,
    )


# =====================================================


def _had_any_activity_in_window(movement_result, candidate_id: str, window_start: float, window_end: float) -> bool:
    """Identical semantics to predictive_dataset.target_generator's own
    private helper of the same name -- reimplemented here (not
    imported) so target_generator.py stays completely untouched."""

    for timeline in movement_result.occupants.values():
        for step in timeline.steps:

            if step.edge.id != candidate_id:
                continue

            if step.start_time <= window_end and step.end_time >= window_start:
                return True

    return False
