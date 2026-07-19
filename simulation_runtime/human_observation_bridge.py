from typing import Mapping, Optional, Tuple

from perception.models.human_observation import (
    BehaviorEvent,
    HumanClassification,
    HumanObservation,
    HumanState,
)
from perception.providers.human_observation_provider import HumanObservationProvider

from simulator.multi_agent_result import MultiAgentSimulationResult, OccupantTimeline


# Production Pipeline Completion -- Human Observation Bridge.
# simulation_runtime/ is architecturally forbidden from importing
# behaviour_profile_resolver (docs/architecture/simulation_runtime.md
# §19, enforced by tests.test_simulation_runtime.
# SimulationRuntimePackageDependencyDirectionTests), so this module
# cannot classify a behaviour_profile_id itself -- classification_by_id
# is instead supplied already-resolved by a caller that *is* allowed to
# import behaviour_profile_resolver (e.g. designer.campaign.
# campaign_worker, which already does, via
# behaviour_profile_resolver.category.occupant_category()). Optional,
# defaulting to empty: a caller that supplies nothing reproduces this
# class's original, honest "no signal at all" behavior
# (HumanClassification.UNKNOWN for everyone) exactly.
#
# ground_truth.human_behavior.DynamicHumanState's own dynamic_state
# names, translated to this package's own HumanState -- restated as
# literals (not imported: simulation_runtime is allowed to depend on
# ground_truth, but this keeps the two enums' own membership
# independent, the same "same vocabulary, independently restated"
# convention ground_truth.decision_events._COLUMN_BY_EVENT_TYPE already
# uses for the identical reason). POSSIBLE_INJURY has no dedicated
# HumanState of its own (perception.models.human_observation.HumanState's
# own docstring: that hedge belongs entirely to Phase 5's InferenceFlag)
# -- it always implies a fall already happened, so it maps to FALLEN,
# never a fabricated new state.
_STATE_BY_DYNAMIC_STATE_NAME = {
    "FALLEN": HumanState.FALLEN,
    "POSSIBLE_INJURY": HumanState.FALLEN,
    "HELPING_ANOTHER_OCCUPANT": HumanState.HELPING_ANOTHER_OCCUPANT,
    "BEING_ASSISTED": HumanState.BEING_ASSISTED,
}


def _apply_behavior_record(state, record) -> Tuple[Optional[HumanState], frozenset]:

    # record is a ground_truth.human_behavior.OccupantBehaviorRecord --
    # duck-typed here (attribute access only, no import) so this module
    # never needs a hard dependency on ground_truth/ to accept one.
    # Only overrides `state` when the record shows something more
    # notable than this class's own already-precise per-instant
    # walking/waiting/standing resolution (see _resolve_with_steps) --
    # dynamic_state is a whole-run summary, not time-indexed, so it is
    # only ever used to add information _resolve() could not have
    # honestly produced on its own (a fall, an assist relationship),
    # never to overwrite an ordinary WALKING/STANDING/WAITING result.

    behavior_events = set()

    if getattr(record, "ever_helping", False):
        behavior_events.add(BehaviorEvent.HELPING_ANOTHER_PERSON)

    if getattr(record, "ever_assisted", False):
        behavior_events.add(BehaviorEvent.GROUPED_MOVEMENT)

    dynamic_state = getattr(record, "dynamic_state", None)
    dynamic_state_name = getattr(dynamic_state, "name", None)

    overridden_state = _STATE_BY_DYNAMIC_STATE_NAME.get(dynamic_state_name)

    if overridden_state is not None:
        state = overridden_state

    return state, frozenset(behavior_events)


class GroundTruthHumanObservationProvider(HumanObservationProvider):

    # Simulation Runtime -- the Ground Truth adapter for HumanObservation,
    # living here rather than in perception/providers/ for exactly the
    # reason simulation_runtime.occupancy_bridge.MovementTimelineOccupancyProvider
    # already lives here rather than there: perception/providers/'s own
    # existing Ground Truth adapters (GroundTruthCameraProvider/
    # GroundTruthSmokeDetectorProvider/GroundTruthHeatDetectorProvider)
    # are architecturally forbidden from importing simulator directly
    # (see tests.test_perception_ground_truth_providers.
    # GroundTruthAdapterDependencyDirectionTests) -- they only ever reach
    # Ground Truth through an intermediary Provider interface (Occupancy
    # Provider, in their case). No such per-occupant-position
    # intermediary exists (GroundTruthCameraProvider's own "V1
    # IMPLEMENTATION LIMITATION" comment already names this precise gap:
    # "the simulator does not yet expose individual occupant positions
    # through any public interface"), and inventing one only for this
    # class to import would be architecture this phase was not asked to
    # build. simulation_runtime is instead where MultiAgentSimulationResult
    # and Perception-shaped interfaces are already both legitimately
    # in scope (see occupancy_bridge.py's own identical reasoning) -- so
    # this class reads MultiAgentSimulationResult.occupants directly,
    # which already *is* exactly the per-occupant feed a HumanObservation
    # needs (one OccupantTimeline per occupant_id), unlike
    # OccupancyProvider/OccupancySnapshot's zone-aggregated shape.
    #
    # This is a simulation-backed stand-in, not a vision model -- it
    # exists so the rest of the platform (live_system, Command Center,
    # tests) can be exercised against real, honestly-derived
    # HumanObservation data before any real perception model exists,
    # this integration phase's own stated purpose.
    #
    # classification_by_id/behavior_records are optional and additive
    # (Production Pipeline Completion): both default to empty,
    # reproducing this class's original, honest "no signal at all"
    # behavior (HumanClassification.UNKNOWN, no behavior_events) for any
    # existing caller that constructs this class the old way. When a
    # caller *does* supply them, both are genuinely known, already-
    # computed facts about this run -- classification_by_id is the
    # caller's own behaviour_profile_resolver.occupant_category()
    # lookup (this module cannot build that itself, see this class's
    # own module-level docstring above), and behavior_records is
    # ground_truth.human_behavior.compute_occupant_behavior()'s own
    # already-completed, whole-run summary for this exact
    # movement_result -- never a guess made up by this class. Still
    # never fabricated: an occupant/firefighter with no
    # classification_by_id entry, or no behavior_records entry, still
    # reports UNKNOWN / no notable state, exactly as before.
    #
    # confidence is a fixed 1.0, the same documented convention
    # GroundTruthCameraProvider's own frame_observation_at() already
    # uses: this adapter reports Ground Truth position/state with
    # certainty, and makes no claim about a real vision model's future
    # accuracy.

    CONFIDENCE = 1.0

    def __init__(
        self,
        movement_result: MultiAgentSimulationResult,
        classification_by_id: Optional[Mapping[str, HumanClassification]] = None,
        behavior_records: Optional[Mapping[str, object]] = None,
    ):

        self._movement_result = movement_result
        self._classification_by_id = classification_by_id or {}
        self._behavior_records = behavior_records or {}

    # =====================================================

    def observations_at(self, time: float) -> Mapping[str, HumanObservation]:

        observations = {}

        for occupant_id, timeline in self._movement_result.occupants.items():

            resolved = self._resolve(timeline, time)

            if resolved is None:
                # Evacuated (arrived and departed the monitored area) --
                # no longer observed, the same "absence means not
                # currently observed" convention every other Perception
                # mapping already follows.
                continue

            zone_id, floor_id, state = resolved

            classification = self._classification_by_id.get(occupant_id, HumanClassification.UNKNOWN)
            behavior_events: frozenset = frozenset()

            record = self._behavior_records.get(occupant_id)

            if record is not None:
                state, behavior_events = _apply_behavior_record(state, record)

            observations[occupant_id] = HumanObservation(
                person_id=occupant_id,
                zone_id=zone_id,
                floor_id=floor_id,
                confidence=self.CONFIDENCE,
                classification=classification,
                state=state,
                behavior_events=behavior_events,
                last_observed_time=time,
            )

        return observations

    # =====================================================

    def _resolve(
        self, timeline: OccupantTimeline, time: float,
    ) -> Optional[Tuple[Optional[str], Optional[str], HumanState]]:

        if not timeline.steps:
            return self._resolve_no_steps(timeline)

        return self._resolve_with_steps(timeline, time)

    # =====================================================

    def _resolve_no_steps(
        self, timeline: OccupantTimeline,
    ) -> Tuple[Optional[str], Optional[str], HumanState]:

        # No OccupantTimelineStep exists for this occupant at all --
        # either a trivial (zero-edge) route (they started at, and
        # never left, their own goal), or route is None entirely
        # (STATIONARY -- chose not to move -- or UNREACHABLE -- no
        # route could be planned). Both are fixed for the occupant's
        # entire life in the run, so the same answer applies at any
        # requested time.

        if timeline.route is not None and timeline.route.nodes:

            node = timeline.route.nodes[0]
            return node.id, node.floor_id, HumanState.WAITING

        # route is None -- OccupantTimeline itself carries no start_id
        # in this case (only a Route would have), so there is no
        # honest zone/floor to report -- left None rather than guessed.
        # Still reported as observed (this person is known to exist and
        # not moving), just with unknown location.
        return None, None, HumanState.WAITING

    # =====================================================

    def _resolve_with_steps(
        self, timeline: OccupantTimeline, time: float,
    ) -> Optional[Tuple[Optional[str], Optional[str], HumanState]]:

        if time < timeline.depart_time:

            start_node = timeline.route.nodes[0]
            return start_node.id, start_node.floor_id, HumanState.NEVER_MOVING_YET

        for step in timeline.steps:

            # An occupant is queued for [join_time, start_time) -- the
            # same interval dataset_builder.timeline._current_queue_length()
            # already establishes (join_time = start_time -
            # queue_wait_time, computed once, in
            # simulator/coordinator.py's _admit_onto_edge()).
            join_time = step.start_time - step.queue_wait_time

            if time < join_time:
                # Departed (or arrived from a previous hop) but not yet
                # queued for this one -- holding at from_node.
                return step.from_node.id, step.from_node.floor_id, HumanState.STANDING

            if time < step.start_time:
                return step.from_node.id, step.from_node.floor_id, HumanState.WAITING

            if time < step.end_time:
                # Mid-traversal -- attributed to the zone they departed
                # from (still their last confirmed position; see this
                # module's own class docstring on why this differs from
                # MovementTimelineOccupancyProvider's "neither zone"
                # choice for aggregate occupancy counting).
                return step.from_node.id, step.from_node.floor_id, HumanState.WALKING

        # time is at or past the final step's end_time.
        if timeline.arrival_time is not None and time >= timeline.arrival_time:
            return None  # evacuated -- no longer observed

        # Reached the end of the step list without a recorded
        # arrival_time at or before `time` -- not expected in a
        # completed MultiAgentSimulationResult (arrival_time is always
        # set to the final step's end_time for an ARRIVED occupant),
        # but handled conservatively rather than raising: report them
        # holding at their last known node.
        last_step = timeline.steps[-1]
        return last_step.to_node.id, last_step.to_node.floor_id, HumanState.STANDING
