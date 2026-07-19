import hashlib
import random

from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from simulator.occupant import OccupantState


class DynamicHumanState(Enum):

    # Human Population & Assisted Evacuation Modeling Phase 2 --
    # ground-truth-owned, deliberately a *separate* type from
    # perception.models.human_observation.HumanState even though
    # several labels line up -- the same "deliberately a separate
    # type... even though the labels line up" reasoning
    # perception.models.building_observation.PerceptionSeverity's own
    # docstring already applies to HazardSeverity/PerceptionSeverity,
    # one layer over: this one is the provable, simulated *fact* of
    # what an occupant did; a future vision-model-backed
    # HumanObservation is, at best, a partial, noisy *observation* of
    # it. ground_truth/ must never import perception/ (perception's own
    # Ground Truth adapters already depend on simulator/ground_truth-
    # shaped data, never the other way -- see
    # simulation_runtime.human_observation_bridge's own docstring for
    # exactly this direction).
    #
    # NORMAL is this type's own neutral default (no equivalent value
    # exists on perception.HumanState, which is Optional instead --
    # ground truth always has a provable answer for what an occupant's
    # own record shows, so there is no "unknown" case to leave as
    # None).

    NORMAL = auto()
    WALKING = auto()
    RUNNING = auto()
    WAITING = auto()
    HELPING_ANOTHER_OCCUPANT = auto()
    BEING_ASSISTED = auto()
    FALLEN = auto()
    POSSIBLE_INJURY = auto()


@dataclass(frozen=True)
class OccupantBehaviorRecord:

    # One occupant's full dynamic-behavior summary for the run --
    # dynamic_state is the single most notable state they reached
    # (see _PRIORITY below); the four ever_* flags are independent of
    # it and of each other, since an occupant can, for instance, both
    # have been assisted *and* later fallen -- dataset_builder/
    # ground_truth's own count columns (Phase 6/7) need the
    # independent flags, not just the one summarized state.

    occupant_id: str
    dynamic_state: DynamicHumanState
    ever_helping: bool
    ever_assisted: bool
    ever_fallen: bool
    ever_possible_injury: bool


# =====================================================
# Fall/injury derivation -- Phase 2's own hard requirement: "Possible
# Injury must arise naturally from simulation events rather than being
# assigned as a static occupant profile." Modeled as a deterministic,
# seeded probability conditioned on two signals the simulation already
# produced for every step: how long this specific crossing made the
# occupant queue (a crowd-crush proxy -- the same queue_wait_time
# dataset_builder.timeline._current_queue_length() already reads) and
# how hazardous the zone they were departing was at that instant (the
# same tick.hazard_snapshot.node_state(zone_id).hazard_score every
# other ground_truth/*.py module already reads). Neither threshold nor
# probability below is a validated life-safety figure -- the same
# honest "illustrative, not validated" disclosure
# DEFAULT_PROFILE_REGISTRY's own walking speeds/compliance levels and
# HazardSeverity's own score cutoffs already carry throughout this
# codebase. A fall can only ever occur while crossing an edge under
# both a real crowd-crush and a real hazard signal -- never assigned
# ahead of time, and never possible for an occupant whose run never
# produced either signal.
# =====================================================

FALL_QUEUE_WAIT_THRESHOLD_SECONDS = 20.0
FALL_HAZARD_SCORE_THRESHOLD = 0.5
FALL_PROBABILITY_PER_QUALIFYING_STEP = 0.08
POSSIBLE_INJURY_PROBABILITY_GIVEN_FALLEN = 0.4

# A step's own hazard exposure at or above this (but below the fall
# threshold's crowding requirement) is still worth fleeing -- RUNNING
# instead of WALKING, no fall risk attached.
RUNNING_HAZARD_SCORE_THRESHOLD = 0.35

_PRIORITY = (
    DynamicHumanState.POSSIBLE_INJURY,
    DynamicHumanState.FALLEN,
    DynamicHumanState.BEING_ASSISTED,
    DynamicHumanState.HELPING_ANOTHER_OCCUPANT,
    DynamicHumanState.RUNNING,
    DynamicHumanState.WALKING,
    DynamicHumanState.WAITING,
    DynamicHumanState.NORMAL,
)


def _derive_behavior_seed(scenario_seed: int, occupant_id: str) -> int:

    # Same "sha256 of a stable key, independently restated per package"
    # convention behaviour_profile_resolver.registrar._derive_occupant_seed()
    # and simulation_interactive.replanning.derive_replan_seed() already
    # each establish -- deterministic given (scenario_seed, occupant_id),
    # so the identical Scenario always produces the identical fall/
    # injury outcomes on repeated analysis (see
    # tests.test_human_population.DeterminismTests).

    key = f"{scenario_seed}|ground_truth.human_behavior|{occupant_id}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()

    return int.from_bytes(digest[:8], "big")


class _HazardTimeline:

    # A per-zone, time-sorted hazard_score lookup built once from
    # tick_results -- the same "last known reading holds until the
    # next one" convention command_center.incident_data.IncidentData.
    # frame_at() already establishes for exactly this shape of query,
    # restated locally rather than imported (ground_truth/ must not
    # depend on command_center/).

    def __init__(self, tick_results):

        self._times = [tick.time for tick in tick_results]
        self._ticks = list(tick_results)

    # =====================================================

    def hazard_score_at(self, zone_id: str, time: float) -> float:

        if not self._times:
            return 0.0

        index = bisect_right(self._times, time) - 1
        index = max(0, min(index, len(self._ticks) - 1))

        return self._ticks[index].hazard_snapshot.node_state(zone_id).hazard_score


# =====================================================


def compute_occupant_behavior(scenario, movement_result, tick_results) -> Dict[str, OccupantBehaviorRecord]:

    helper_ids, assisted_ids = _assistance_ids(scenario)
    hazard_timeline = _HazardTimeline(tick_results)

    records = {}

    for occupant_id, timeline in movement_result.occupants.items():

        states_reached = set()

        ever_fallen = False
        ever_possible_injury = False

        rng = random.Random(_derive_behavior_seed(scenario.metadata.seed, occupant_id))

        if timeline.state == OccupantState.STATIONARY:
            states_reached.add(DynamicHumanState.WAITING)

        for step in timeline.steps:

            queue_wait = step.queue_wait_time
            hazard_score = hazard_timeline.hazard_score_at(step.from_node.id, step.start_time)

            if queue_wait > 0:
                states_reached.add(DynamicHumanState.WAITING)

            if (
                queue_wait >= FALL_QUEUE_WAIT_THRESHOLD_SECONDS
                and hazard_score >= FALL_HAZARD_SCORE_THRESHOLD
                and rng.random() < FALL_PROBABILITY_PER_QUALIFYING_STEP
            ):

                ever_fallen = True
                states_reached.add(DynamicHumanState.FALLEN)

                if rng.random() < POSSIBLE_INJURY_PROBABILITY_GIVEN_FALLEN:
                    ever_possible_injury = True
                    states_reached.add(DynamicHumanState.POSSIBLE_INJURY)

                # A fall is this step's own notable event; it does not
                # also need to separately register as ordinary
                # WALKING/RUNNING traversal.
                continue

            if hazard_score >= RUNNING_HAZARD_SCORE_THRESHOLD:
                states_reached.add(DynamicHumanState.RUNNING)
            else:
                states_reached.add(DynamicHumanState.WALKING)

        ever_helping = occupant_id in helper_ids
        ever_assisted = occupant_id in assisted_ids

        if ever_helping:
            states_reached.add(DynamicHumanState.HELPING_ANOTHER_OCCUPANT)

        if ever_assisted:
            states_reached.add(DynamicHumanState.BEING_ASSISTED)

        dynamic_state = _summarize(states_reached)

        records[occupant_id] = OccupantBehaviorRecord(
            occupant_id=occupant_id,
            dynamic_state=dynamic_state,
            ever_helping=ever_helping,
            ever_assisted=ever_assisted,
            ever_fallen=ever_fallen,
            ever_possible_injury=ever_possible_injury,
        )

    return records


# =====================================================


def _assistance_ids(scenario) -> Tuple[set, set]:

    helper_ids = set()
    assisted_ids = set()

    for occupant in scenario.occupants:

        if occupant.assisting_occupant_id:
            helper_ids.add(occupant.occupant_id)
            assisted_ids.add(occupant.assisting_occupant_id)

    return helper_ids, assisted_ids


# =====================================================


def _summarize(states_reached) -> DynamicHumanState:

    for state in _PRIORITY:
        if state in states_reached:
            return state

    return DynamicHumanState.NORMAL


# =====================================================
# Scenario-level aggregate counts -- Phase 6/7's own required shape
# (Dataset Builder's Helping_Group_Count/Assisted_Occupant_Count/
# Fallen_Count/Possible_Injury_Count, Ground Truth's "Assisted
# Occupants"). One pass over compute_occupant_behavior()'s own output,
# never recomputed independently.
# =====================================================


def summarize_behavior_counts(records: Dict[str, OccupantBehaviorRecord]) -> Dict[str, int]:

    return {
        "helping_group_count": sum(1 for record in records.values() if record.ever_helping),
        "assisted_occupant_count": sum(1 for record in records.values() if record.ever_assisted),
        "fallen_count": sum(1 for record in records.values() if record.ever_fallen),
        "possible_injury_count": sum(1 for record in records.values() if record.ever_possible_injury),
    }
