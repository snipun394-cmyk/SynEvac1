from bisect import bisect_right
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

from models.building import Building

from behaviour_profile_resolver.category import OccupantCategory, occupant_category

from perception.models.human_observation import BehaviorEvent, HumanClassification, HumanObservation, HumanState

from scenario.engineering_state import DeviceAvailability, DoorState, StairAvailability
from scenario.scenario import Scenario

from decision_policy import announcement_policy, exit_policy, stair_policy, zone_policy
from decision_policy.policy import DecisionPolicy

from advisory_system import AdvisoryInputs, AdvisoryOrchestrator, AdvisoryReport, RecommendationHistory

from speaker_manager.manager import SpeakerManager

from voice_evacuation.adapter import civilian_announcements_to_voice_messages
from voice_evacuation.broadcast_log import BroadcastLog
from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.models import VoiceMessage
from voice_evacuation.provider import SimulationVoiceOutputProvider

from building_control.advisory_adapter import translate_report
from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider

from dataset_builder.schema import (
    camera_state_columns,
    detector_state_columns,
    door_state_columns,
    exit_state_columns,
    ordered_cameras,
    ordered_detectors,
    ordered_doors,
    ordered_exits,
    ordered_stairs,
    ordered_zones,
    stair_state_columns,
    zone_occupancy_columns,
)
from dataset_builder.timeline_schema import (
    zone_smoke_columns,
    zone_temperature_columns,
    zone_visibility_columns,
)


# =====================================================
# IncidentData / IncidentFrame -- the one read-only view model every
# Command Center panel renders from. Bundles together artifacts a
# completed run already produced (Building, Scenario, GroundTruth,
# DecisionPolicy, and per-tick Timeline Dataset rows from
# dataset_builder.timeline.extract_timeline_rows) and resolves them
# into one IncidentFrame per timestep -- id-keyed, never
# column-name-keyed, so a panel reads frame.zone_occupancy["zone-a"]
# rather than row["Zone_1_Occupancy"]. The column <-> id mapping is
# rebuilt on each load using dataset_builder.schema's own ordering
# helpers (the exact functions the rows were produced with), never
# re-derived independently.
#
# Nothing in this module simulates, generates, or recomputes an
# incident -- every field is read directly off an already-produced
# artifact, or falls back to a "no data yet" default. The Command
# Center is a visualization layer only.
# =====================================================


@dataclass(frozen=True)
class IncidentFrame:

    time: float = 0.0

    ignition_zone_id: Optional[str] = None
    current_fire_zone_ids: FrozenSet[str] = frozenset()
    current_hazard_score: Optional[float] = None

    zone_occupancy: Mapping[str, Optional[float]] = field(default_factory=dict)
    zone_smoke: Mapping[str, Optional[float]] = field(default_factory=dict)
    zone_temperature: Mapping[str, Optional[float]] = field(default_factory=dict)
    zone_visibility: Mapping[str, Optional[float]] = field(default_factory=dict)

    door_states: Mapping[str, str] = field(default_factory=dict)
    exit_states: Mapping[str, str] = field(default_factory=dict)
    stair_states: Mapping[str, str] = field(default_factory=dict)
    detector_states: Mapping[str, str] = field(default_factory=dict)
    camera_states: Mapping[str, str] = field(default_factory=dict)

    people_evacuated: Optional[int] = None
    people_remaining: Optional[int] = None
    people_trapped: Optional[int] = None

    current_congestion: Optional[float] = None
    maximum_congestion_so_far: Optional[float] = None
    current_evacuation_rate: Optional[float] = None
    current_queue_length: Optional[int] = None
    current_route_utilization: Optional[float] = None
    current_bottleneck: Optional[str] = None

    # Human Perception Integration -- additive. Keyed by
    # HumanObservation.person_id (a perception tracking identity, not
    # a Building Model id). dataset_builder's own Timeline CSV schema
    # has no per-person columns (so _frame_from_row()/_baseline_frame()
    # still never derive this from a timeline row), but Production
    # Pipeline Completion adds a second, genuinely available source:
    # IncidentData.__post_init__ builds one classification+dynamic-state
    # snapshot from self.scenario (occupant/firefighter
    # behaviour_profile_id, via behaviour_profile_resolver.
    # occupant_category()) and self.ground_truth.occupant_behavior (if a
    # GroundTruth was loaded), and attaches it to every frame -- the
    # same "known for the whole run, not per-instant" honesty
    # ground_truth.human_behavior.OccupantBehaviorRecord.dynamic_state
    # itself already carries (see _build_human_observations() below).
    # Empty only when no Scenario occupants/firefighters exist. A live
    # source (live_system.integration.DashboardCommandCenterGateway)
    # remains free to build its own, richer, per-instant IncidentFrame
    # directly from a LiveBuildingSnapshot instead.
    human_observations: Mapping[str, HumanObservation] = field(default_factory=dict)

    # Human Decision Engine Phase 7 -- additive. Keyed by the same
    # occupant/firefighter_id human_observations already uses where
    # both exist (a live source populates both), but populated
    # independently -- a dynamic-registration run has no
    # HumanObservation at all (no perception layer is involved), only
    # this. Each value is a plain dict (never a nested dataclass --
    # same convention every other id-keyed field on this frame already
    # follows) with whichever of "decision"/"goal"/"group"/
    # "assistance_status"/"rescue_priority"/"firefighter_task" keys
    # this person actually has -- a civilian never has "firefighter_
    # task", an ordinary unassisted occupant has no "group"/
    # "assistance_status" -- HumanPanel reads each with .get(...) and
    # shows "-" for whichever this frame never supplied, never
    # fabricating a value a completed-simulation-replay artifact (which
    # never populates this field at all -- see this module's own
    # _frame_from_row()/_baseline_frame()) was never given.
    human_decisions: Mapping[str, Dict[str, Any]] = field(default_factory=dict)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "current_fire_zone_ids", frozenset(self.current_fire_zone_ids))
        object.__setattr__(self, "zone_occupancy", MappingProxyType(dict(self.zone_occupancy)))
        object.__setattr__(self, "zone_smoke", MappingProxyType(dict(self.zone_smoke)))
        object.__setattr__(self, "zone_temperature", MappingProxyType(dict(self.zone_temperature)))
        object.__setattr__(self, "zone_visibility", MappingProxyType(dict(self.zone_visibility)))
        object.__setattr__(self, "door_states", MappingProxyType(dict(self.door_states)))
        object.__setattr__(self, "exit_states", MappingProxyType(dict(self.exit_states)))
        object.__setattr__(self, "stair_states", MappingProxyType(dict(self.stair_states)))
        object.__setattr__(self, "detector_states", MappingProxyType(dict(self.detector_states)))
        object.__setattr__(self, "camera_states", MappingProxyType(dict(self.camera_states)))
        object.__setattr__(self, "human_observations", MappingProxyType(dict(self.human_observations)))
        object.__setattr__(self, "human_decisions", MappingProxyType(dict(self.human_decisions)))


# =====================================================


@dataclass(frozen=True)
class IncidentData:

    # ground_truth/decision_policy are typed Any to keep this module
    # free of a hard import dependency on either package's internals
    # beyond the one attribute each field-mapper below reads -- same
    # "typed Any, not imported for its own sake" convention
    # decision_policy.policy.DecisionInputs.ground_truth already uses.

    building: Building
    scenario: Scenario
    ground_truth: Any = None
    decision_policy: Any = None
    timeline_rows: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    # =====================================================

    def __post_init__(self):

        object.__setattr__(self, "timeline_rows", tuple(self.timeline_rows))

        if self.timeline_rows:

            frames = tuple(
                _frame_from_row(self.building, row) for row in self.timeline_rows
            )

        else:

            frames = (_baseline_frame(self.building, self.scenario),)

        human_observations = _build_human_observations(self.scenario, self.ground_truth)

        if human_observations:
            frames = tuple(replace(frame, human_observations=human_observations) for frame in frames)

        object.__setattr__(self, "_frames", frames)
        object.__setattr__(self, "_frame_times", tuple(frame.time for frame in frames))
        object.__setattr__(self, "_names", _object_name_lookup(self.building))

        advisory_reports, history = _build_advisory_reports(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
            static_decision_policy=self.decision_policy, human_observations=human_observations,
            frames=frames,
        )
        object.__setattr__(self, "_advisory_reports", advisory_reports)
        object.__setattr__(self, "_recommendation_history", history)

        active_voice_messages_by_frame, broadcast_log = _build_voice_broadcasts(
            building=self.building, scenario=self.scenario, ground_truth=self.ground_truth,
            frames=frames, advisory_reports=advisory_reports,
        )
        object.__setattr__(self, "_active_voice_messages_by_frame", active_voice_messages_by_frame)
        object.__setattr__(self, "_broadcast_log", broadcast_log)

        # Building Control & Safety Systems Integration -- one shared
        # BuildingControlController for this incident's whole viewing
        # session (never reconstructed per frame, same "one shared
        # instance" convention _build_voice_broadcasts() above already
        # establishes for VoiceEvacuationController), so pending/
        # approved/confirmed state persists exactly as it would during
        # a real operator session as frames are navigated. Unlike voice
        # broadcasts, this is deliberately NOT precomputed for every
        # frame up front: approval is a human-in-the-loop action
        # (Phase 5), so requests are ingested lazily as each frame is
        # actually viewed (see ingest_control_recommendations() below,
        # called from BuildingControlsPanel.show_frame()) rather than
        # auto-approved on a schedule.
        #
        # action_executor=None here is deliberate and honest: this is a
        # static replay of an already-completed run, not a live
        # SimulationContext -- there is nothing for a Door/Exit control
        # instruction to actually mutate, so SimulationControlProvider
        # correctly reports those as provider-FAILED rather than
        # fabricating a confirmation (see that class's own docstring).
        # Stair Pressurization/Smoke Exhaust/Deluge have no backing
        # physics anywhere in this codebase regardless of context, so
        # they confirm honestly here the same way they would live.
        object.__setattr__(self, "_control_provider", SimulationControlProvider(self.building))
        object.__setattr__(
            self, "_control_controller", BuildingControlController(self.building, self._control_provider),
        )

    # =====================================================

    @property
    def frames(self) -> Tuple[IncidentFrame, ...]:

        return self._frames

    # =====================================================

    def name_for(self, object_id: Optional[str]) -> str:

        # One id -> name lookup shared by every panel (Zone, Door,
        # Exit, Staircase, Detector, Camera, Floor all share one id
        # namespace project-wide) -- falls back to the id itself so a
        # panel never has to special-case a missing/unknown id.

        if not object_id:
            return "-"

        return self._names.get(object_id, object_id)

    # =====================================================

    @property
    def frame_count(self) -> int:

        return len(self._frames)

    # =====================================================

    @property
    def has_timeline(self) -> bool:

        return bool(self.timeline_rows)

    # =====================================================

    @property
    def duration(self) -> float:

        return self._frame_times[-1] if self._frame_times else 0.0

    # =====================================================

    def frame_at(self, time: float) -> IncidentFrame:

        # Nearest preceding frame -- the same "last known state holds
        # until the next reading" convention every snapshot_at(time)
        # accessor in this codebase already uses.

        times = self._frame_times

        index = bisect_right(times, time) - 1
        index = max(0, min(index, len(self._frames) - 1))

        return self._frames[index]

    # =====================================================

    def frame_at_index(self, index: int) -> IncidentFrame:

        index = max(0, min(index, len(self._frames) - 1))

        return self._frames[index]

    # =====================================================

    def advisory_report_at_index(self, index: int) -> Optional[AdvisoryReport]:

        # One advisory_system.AdvisoryReport per IncidentFrame -- see
        # _build_advisory_reports() below for exactly what is (and is
        # not) recomputed per frame. None only when no GroundTruth was
        # ever loaded for this incident (nothing to advise from).

        if not self._advisory_reports:
            return None

        index = max(0, min(index, len(self._advisory_reports) - 1))

        return self._advisory_reports[index]

    # =====================================================

    def active_voice_messages_at_index(self, index: int) -> Mapping[str, VoiceMessage]:

        # Zoned Voice Evacuation & Speaker Network Framework -- one
        # zone_id -> VoiceMessage snapshot per IncidentFrame, deterministically
        # replayed once in __post_init__ (see _build_voice_broadcasts()
        # below) by routing each frame's already-computed AdvisoryReport.
        # civilian_announcements through the same
        # voice_evacuation.adapter/VoiceEvacuationController this
        # milestone's Simulation path already uses -- never a live
        # simulation, purely a deterministic reconstruction from
        # artifacts this IncidentData already holds. Empty when no
        # GroundTruth was ever loaded (nothing to advise from, so
        # nothing to broadcast either).

        if not self._active_voice_messages_by_frame:
            return {}

        index = max(0, min(index, len(self._active_voice_messages_by_frame) - 1))

        return self._active_voice_messages_by_frame[index]

    # =====================================================

    @property
    def control_controller(self) -> BuildingControlController:

        # The one BuildingControlController shared across this
        # incident's whole viewing session -- see __post_init__ above.
        return self._control_controller

    # =====================================================

    def ingest_control_recommendations(self, report: Optional[AdvisoryReport]) -> None:

        # Translates report.building_recommendations into ControlRequests
        # (building_control.advisory_adapter.translate_report) and
        # submits each into the shared controller. Submission is
        # idempotent (BuildingControlController.submit() dedups any
        # request already live for the same system/target/action), so
        # calling this repeatedly for the same or an earlier frame is
        # always safe -- BuildingControlsPanel calls this once per
        # show_frame().

        if report is None:
            return

        for request in translate_report(report):
            self._control_controller.submit(request)

    # =====================================================

    def active_voice_messages_at_time(self, time: float) -> Mapping[str, VoiceMessage]:

        # Same "nearest preceding frame" resolution as frame_at(time)
        # above -- for a caller (VoiceEvacuationPanel) that is handed an
        # IncidentFrame rather than a bare index, since frame.time is
        # exactly what advisory_report_at_index()'s own caller
        # (command_center.dashboard.Dashboard.show_frame()) already uses
        # to resolve its own AdvisoryReport for the same frame.

        if not self._active_voice_messages_by_frame:
            return {}

        times = self._frame_times

        index = bisect_right(times, time) - 1
        index = max(0, min(index, len(self._active_voice_messages_by_frame) - 1))

        return self._active_voice_messages_by_frame[index]

    # =====================================================

    @property
    def voice_broadcast_history(self) -> tuple:

        # The full, accumulated BroadcastInstruction history across
        # every frame this incident's own playback produced -- the same
        # "display every recommendation/event ever issued" role
        # recommendation_history already plays for civilian announcements.

        return self._broadcast_log.all_instructions()

    # =====================================================

    @property
    def recommendation_history(self) -> tuple:

        # The full, accumulated RecommendationHistory across every
        # frame this incident's own playback produced -- Phase 4's own
        # "display every recommendation ever issued," not merely the
        # currently-showing frame's.

        return self._recommendation_history.changes

    # =====================================================

    def occupant_count_by_zone(self) -> Dict[str, int]:

        counts: Dict[str, int] = {}

        for occupant in self.scenario.occupants:
            counts[occupant.zone_id] = counts.get(occupant.zone_id, 0) + 1

        return counts


# =====================================================
# One id -> name lookup spanning every engineering object type. ids
# are unique project-wide regardless of type (the same convention
# models/connectable_space.py already relies on), so a single flat
# dict is always unambiguous.
# =====================================================


def _object_name_lookup(building: Building) -> Dict[str, str]:

    names: Dict[str, str] = {}

    for floor in building.ordered_floors():

        names[floor.id] = floor.name

        for zone in floor.zones:
            names[zone.id] = zone.name
        for door in floor.doors:
            names[door.id] = door.name or "Door"
        for exit_obj in floor.exits:
            names[exit_obj.id] = exit_obj.name or "Exit"
        for stair in floor.stairs:
            names[stair.id] = stair.name or "Staircase"
        for detector in floor.detectors:
            names[detector.id] = detector.name or "Detector"
        for camera in floor.cameras:
            names[camera.id] = camera.name or "Camera"

    return names


# =====================================================
# Scenario/GroundTruth -> HumanObservation. Production Pipeline
# Completion -- Command Center's HumanPanel already reads
# frame.human_observations honestly (see command_center.human_panel's
# own docstring); the only thing missing was ever populating it for the
# real, file-based load_incident() path (as opposed to a hypothetical
# live SensorFusionPerceptionGateway, which no concrete deployment
# configures yet). Every value here is read directly off Scenario
# (already loaded) or GroundTruth.occupant_behavior (already loaded,
# already computed by ground_truth.human_behavior.
# compute_occupant_behavior() for this exact run) -- nothing is
# inferred or guessed. Categories with no genuinely equivalent
# HumanClassification member (STAFF, VISITOR, UNKNOWN) are left
# unmapped -- UNKNOWN, never guessed.
# =====================================================

_CLASSIFICATION_BY_CATEGORY = {
    OccupantCategory.ADULT: HumanClassification.ADULT,
    OccupantCategory.CHILD: HumanClassification.CHILD,
    OccupantCategory.ELDERLY: HumanClassification.ELDERLY,
    OccupantCategory.WHEELCHAIR_USER: HumanClassification.WHEELCHAIR_USER,
    OccupantCategory.FIREFIGHTER: HumanClassification.FIREFIGHTER,
    OccupantCategory.FIRE_WARDEN: HumanClassification.FIRE_WARDEN,
}

# ground_truth.human_behavior.DynamicHumanState's own dynamic_state
# names (restated as literals, not imported -- command_center must not
# depend on ground_truth's internals beyond the plain dict shape
# GroundTruth.to_dict()/occupant_behavior already exposes, the same
# "typed Any, not imported for its own sake" convention this module's
# own IncidentData.ground_truth field already documents). POSSIBLE_
# INJURY always implies a fall already happened, so it maps to FALLEN,
# never a fabricated new state -- see
# simulation_runtime.human_observation_bridge's own identical table for
# the same reasoning.
_STATE_BY_DYNAMIC_STATE_NAME = {
    "FALLEN": HumanState.FALLEN,
    "POSSIBLE_INJURY": HumanState.FALLEN,
    "HELPING_ANOTHER_OCCUPANT": HumanState.HELPING_ANOTHER_OCCUPANT,
    "BEING_ASSISTED": HumanState.BEING_ASSISTED,
}


def _build_human_observations(scenario: Scenario, ground_truth: Any) -> Dict[str, HumanObservation]:

    behavior_by_id: Dict[str, Dict[str, Any]] = {}

    for record in getattr(ground_truth, "occupant_behavior", None) or ():
        occupant_id = record.get("occupant_id")
        if occupant_id:
            behavior_by_id[occupant_id] = record

    observations: Dict[str, HumanObservation] = {}

    for occupant in scenario.occupants:

        classification = _CLASSIFICATION_BY_CATEGORY.get(
            occupant_category(occupant.behaviour_profile_id), HumanClassification.UNKNOWN,
        )
        observations[occupant.occupant_id] = _observation_for(
            occupant.occupant_id, occupant.zone_id, classification,
            behavior_by_id.get(occupant.occupant_id),
        )

    for firefighter in scenario.firefighters:

        classification = _CLASSIFICATION_BY_CATEGORY.get(
            occupant_category(firefighter.behaviour_profile_id), HumanClassification.FIREFIGHTER,
        )
        observations[firefighter.firefighter_id] = _observation_for(
            firefighter.firefighter_id, firefighter.entry_zone_id, classification,
            behavior_by_id.get(firefighter.firefighter_id),
        )

    return observations


def _observation_for(person_id, zone_id, classification, behavior_record) -> HumanObservation:

    state = None
    behavior_events = set()

    if behavior_record is not None:

        if behavior_record.get("ever_helping"):
            behavior_events.add(BehaviorEvent.HELPING_ANOTHER_PERSON)

        if behavior_record.get("ever_assisted"):
            behavior_events.add(BehaviorEvent.GROUPED_MOVEMENT)

        state = _STATE_BY_DYNAMIC_STATE_NAME.get(behavior_record.get("dynamic_state"))

    return HumanObservation(
        person_id=person_id,
        zone_id=zone_id,
        confidence=1.0,
        classification=classification,
        state=state,
        behavior_events=frozenset(behavior_events),
    )


# =====================================================
# Frame -> DecisionPolicy -> AdvisoryReport. Command Center's own
# integration layer, not a backend redesign: every function called
# below (decision_policy.zone_policy.compute_zone_decisions(),
# exit_policy.compute_exit_decisions(), stair_policy.
# compute_stair_decisions(), announcement_policy.generate_announcements(),
# advisory_system.AdvisoryOrchestrator.generate_report()) is an
# already-public, already-tested function from an already-existing
# package, called here in exactly the same order decision_policy.
# policy.generate_policy() itself already uses -- the only thing new is
# calling them once per already-computed IncidentFrame instead of once
# per whole scenario, using that frame's own current_fire_zone_ids
# (already a field on IncidentFrame) instead of the campaign export's
# single, final one. rescue_priorities/rescue_order/firefighter_
# deployment_priority/human_rescue_priorities are NOT recomputed per
# frame -- rescue_policy.compute_rescue_priorities() itself has no
# current_fire_zone_ids parameter at all (it depends only on Scenario's
# own occupant placement and GroundTruth's own risk scores, neither of
# which varies by frame), so the whole-run DecisionPolicy's own values
# are reused unchanged for those fields, never re-derived.
#
# This is what makes Phase 4/7's "recommendation evolution" genuine
# rather than staged: as playback advances through frames whose
# current_fire_zone_ids differ, zone/exit/stair/announcement decisions
# for the affected zones can genuinely change, and
# RecommendationHistory (one shared instance, walked frame-by-frame in
# order below) records exactly those changes -- never a fabricated
# "10:03 -> 10:05" sequence.
# =====================================================


def _build_advisory_reports(
    *, building, scenario, ground_truth, static_decision_policy, human_observations, frames,
) -> Tuple[Tuple[Any, ...], "RecommendationHistory"]:

    history = RecommendationHistory()

    if ground_truth is None:
        return (), history

    zones = ordered_zones(building)
    exits = ordered_exits(building)
    stairs = ordered_stairs(building)

    orchestrator = AdvisoryOrchestrator(history=history)
    reports = []

    for frame in frames:

        zone_decisions = zone_policy.compute_zone_decisions(
            scenario=scenario, ground_truth=ground_truth, zones=zones,
            current_fire_zone_ids=frame.current_fire_zone_ids,
        )
        exit_decisions = exit_policy.compute_exit_decisions(
            scenario=scenario, ground_truth=ground_truth, exits=exits, zone_decisions=zone_decisions,
        )
        stair_decisions = stair_policy.compute_stair_decisions(
            ground_truth=ground_truth, stairs=stairs, current_fire_zone_ids=frame.current_fire_zone_ids,
        )
        announcements = announcement_policy.generate_announcements(zone_decisions)

        frame_policy = DecisionPolicy(
            scenario_id=ground_truth.scenario_id,
            zone_decisions=zone_decisions,
            exit_decisions=exit_decisions,
            stair_decisions=stair_decisions,
            announcements=announcements,
            rescue_priorities=(
                static_decision_policy.rescue_priorities if static_decision_policy is not None else ()
            ),
            rescue_order=(
                static_decision_policy.rescue_order if static_decision_policy is not None else ()
            ),
            firefighter_deployment_priority=(
                static_decision_policy.firefighter_deployment_priority
                if static_decision_policy is not None else ()
            ),
            human_rescue_priorities=(
                static_decision_policy.human_rescue_priorities
                if static_decision_policy is not None else ()
            ),
        )

        inputs = AdvisoryInputs(
            building=building, scenario=scenario, ground_truth=ground_truth,
            decision_policy=frame_policy, human_observations=human_observations,
            simulation_time=frame.time,
            # Building System State -- this frame's own replayed door/
            # exit/stair/detector/camera readings (dataset_builder.
            # timeline's own per-tick engineering-event replay, already
            # computed, never re-derived here). Distinct from Ground
            # Truth's own whole-run hazard fields above: a door/exit
            # scheduled to lock or fail partway through the incident is
            # exactly the kind of genuinely time-varying fact this
            # optional AdvisoryInputs field exists for (see that
            # dataclass's own docstring).
            building_system_state={
                **{k: {"state": v} for k, v in frame.door_states.items()},
                **{k: {"state": v} for k, v in frame.exit_states.items()},
                **{k: {"state": v} for k, v in frame.stair_states.items()},
                **{k: {"state": v} for k, v in frame.detector_states.items()},
                **{k: {"state": v} for k, v in frame.camera_states.items()},
            },
        )

        reports.append(orchestrator.generate_report(inputs))

    return tuple(reports), history


# =====================================================
# AdvisoryReport.civilian_announcements -> VoiceMessage -> Speaker
# routing. Command Center's own integration layer, same discipline as
# _build_advisory_reports() above: every function called below
# (speaker_manager.manager.SpeakerManager.discover_speakers(),
# voice_evacuation.adapter.civilian_announcements_to_voice_messages(),
# voice_evacuation.controller.VoiceEvacuationController.broadcast())
# is an already-existing, already-tested public function, called here
# once per already-computed IncidentFrame/AdvisoryReport pair. One
# VoiceEvacuationController is shared across every frame (never
# reconstructed per frame) so priority/supersession accumulates exactly
# as it would during a real playback -- the same "one shared instance,
# walked frame-by-frame in order" convention _build_advisory_reports()
# already establishes for RecommendationHistory above.
#
# zone_actions is recomputed per frame via zone_policy.compute_zone_
# decisions() -- the exact same call _build_advisory_reports() already
# makes for its own frame_policy, independently re-run here rather than
# threaded through as an extra return value, so this function stays a
# clean, additive, read-only pass over already-produced artifacts
# without changing _build_advisory_reports()'s own signature or return
# shape.
# =====================================================


def _build_voice_broadcasts(
    *, building, scenario, ground_truth, frames, advisory_reports,
) -> Tuple[Tuple[Mapping[str, VoiceMessage], ...], "BroadcastLog"]:

    broadcast_log = BroadcastLog()

    if ground_truth is None or not advisory_reports:
        return (), broadcast_log

    zones = ordered_zones(building)

    speaker_manager = SpeakerManager()
    speaker_manager.discover_speakers(building)

    controller = VoiceEvacuationController(speaker_manager, SimulationVoiceOutputProvider())

    active_by_frame = []

    for frame, report in zip(frames, advisory_reports):

        zone_decisions = zone_policy.compute_zone_decisions(
            scenario=scenario, ground_truth=ground_truth, zones=zones,
            current_fire_zone_ids=frame.current_fire_zone_ids,
        )
        zone_actions = {decision["zone_id"]: decision["action"] for decision in zone_decisions}

        voice_messages = civilian_announcements_to_voice_messages(
            report.civilian_announcements, frame.time, zone_actions=zone_actions,
        )

        for message in voice_messages:
            controller.broadcast(message, frame.time)

        active_by_frame.append(controller.active_messages())

    # broadcast_log accumulates on the controller itself; reuse the
    # exact same log object rather than copying its contents.
    return tuple(active_by_frame), controller.broadcast_log


# =====================================================
# Timeline-row -> IncidentFrame. Rebuilds the exact column <-> id
# zip dataset_builder.timeline.extract_timeline_rows() used to produce
# each row, then reads every scalar field straight off the row by key.
# =====================================================


def _frame_from_row(building: Building, row: Dict[str, Any]) -> IncidentFrame:

    zones = ordered_zones(building)

    zone_occupancy = {
        zone.id: row.get(column)
        for column, zone in zip(zone_occupancy_columns(building), zones)
    }
    zone_smoke = {
        zone.id: row.get(column)
        for column, zone in zip(zone_smoke_columns(building), zones)
    }
    zone_temperature = {
        zone.id: row.get(column)
        for column, zone in zip(zone_temperature_columns(building), zones)
    }
    zone_visibility = {
        zone.id: row.get(column)
        for column, zone in zip(zone_visibility_columns(building), zones)
    }

    door_states = {
        door.id: row.get(column)
        for column, door in zip(door_state_columns(building), ordered_doors(building))
    }
    exit_states = {
        exit_obj.id: row.get(column)
        for column, exit_obj in zip(exit_state_columns(building), ordered_exits(building))
    }
    stair_states = {
        stair.id: row.get(column)
        for column, stair in zip(stair_state_columns(building), ordered_stairs(building))
    }
    detector_states = {
        detector.id: row.get(column)
        for column, detector in zip(detector_state_columns(building), ordered_detectors(building))
    }
    camera_states = {
        camera.id: row.get(column)
        for column, camera in zip(camera_state_columns(building), ordered_cameras(building))
    }

    raw_fire_zones = row.get("current_fire_zones") or ""
    fire_zone_ids = frozenset(zone_id for zone_id in raw_fire_zones.split(";") if zone_id)

    return IncidentFrame(
        time=row.get("simulation_time", 0.0),
        ignition_zone_id=row.get("ignition_zone"),
        current_fire_zone_ids=fire_zone_ids,
        current_hazard_score=row.get("current_hazard_score"),
        zone_occupancy=zone_occupancy,
        zone_smoke=zone_smoke,
        zone_temperature=zone_temperature,
        zone_visibility=zone_visibility,
        door_states=door_states,
        exit_states=exit_states,
        stair_states=stair_states,
        detector_states=detector_states,
        camera_states=camera_states,
        people_evacuated=row.get("people_evacuated"),
        people_remaining=row.get("people_remaining"),
        people_trapped=row.get("people_trapped"),
        current_congestion=row.get("current_congestion"),
        maximum_congestion_so_far=row.get("maximum_congestion_so_far"),
        current_evacuation_rate=row.get("current_evacuation_rate"),
        current_queue_length=row.get("current_queue_length"),
        current_route_utilization=row.get("current_route_utilization"),
        current_bottleneck=row.get("current_bottleneck"),
    )


# =====================================================
# Baseline frame -- used only when no Timeline Dataset rows are
# available at all (a static, pre-ignition incident view). Resolved
# entirely from Scenario overrides falling back to Building baseline
# fields, the same "Scenario override, else Building baseline"
# convention dataset_builder.timeline._initial_engineering_state()
# already establishes -- reimplemented locally (rather than imported,
# since that helper is private) using only the same public
# Building/Scenario fields and scenario.engineering_state enums it
# reads.
# =====================================================


def _baseline_frame(building: Building, scenario: Scenario) -> IncidentFrame:

    zone_occupancy: Dict[str, int] = {}

    for occupant in scenario.occupants:
        zone_occupancy[occupant.zone_id] = zone_occupancy.get(occupant.zone_id, 0) + 1

    door_states: Dict[str, str] = {}
    for door in ordered_doors(building):

        if door.locked:
            door_states[door.id] = DoorState.LOCKED.name
        elif not door.active:
            door_states[door.id] = DoorState.CLOSED.name
        else:
            door_states[door.id] = DoorState.OPEN.name

    for state in scenario.door_states:
        door_states[state.door_id] = state.state.name

    exit_states: Dict[str, str] = {
        exit_obj.id: ("CLOSED" if exit_obj.is_blocked else "OPEN")
        for exit_obj in ordered_exits(building)
    }
    for state in scenario.exit_states:
        exit_states[state.exit_id] = "OPEN" if state.is_open else "CLOSED"

    stair_states: Dict[str, str] = {
        stair.id: StairAvailability.AVAILABLE.name for stair in ordered_stairs(building)
    }
    for state in scenario.stair_states:
        stair_states[state.stair_id] = state.availability.name

    detector_states: Dict[str, str] = {
        detector.id: (
            DeviceAvailability.AVAILABLE.name if detector.active else DeviceAvailability.FAILED.name
        )
        for detector in ordered_detectors(building)
    }
    for state in scenario.detector_states:
        detector_states[state.detector_id] = state.availability.name

    camera_states: Dict[str, str] = {
        camera.id: (
            DeviceAvailability.AVAILABLE.name if camera.active else DeviceAvailability.FAILED.name
        )
        for camera in ordered_cameras(building)
    }
    for state in scenario.camera_states:
        camera_states[state.camera_id] = state.availability.name

    total_occupants = len(scenario.occupants)

    return IncidentFrame(
        time=0.0,
        ignition_zone_id=scenario.fire.ignition_zone_id if scenario.fire is not None else None,
        current_fire_zone_ids=frozenset(),
        current_hazard_score=None,
        zone_occupancy=zone_occupancy,
        door_states=door_states,
        exit_states=exit_states,
        stair_states=stair_states,
        detector_states=detector_states,
        camera_states=camera_states,
        people_evacuated=0,
        people_remaining=total_occupants,
        people_trapped=0,
    )


# =====================================================
# Optional convenience loader for on-disk artifacts. Purely additive:
# IncidentData itself never requires this -- a caller may just as well
# construct one directly from in-memory objects it already has. This
# only reads already-written files; it never generates, simulates, or
# validates anything.
# =====================================================


def load_incident(
    project_path: str,
    scenario_id: str,
    scenario_storage_root: str,
    ground_truth_path: Optional[str] = None,
    decision_policy_path: Optional[str] = None,
    timeline_rows_path: Optional[str] = None,
) -> IncidentData:

    import json

    from serialization.serializer import Serializer
    from scenario_storage.storage import load_scenario_by_id

    from ground_truth.labels import GroundTruth
    from decision_policy import DecisionPolicy

    project = Serializer.load(project_path)

    scenario = load_scenario_by_id(scenario_id, scenario_storage_root)

    ground_truth = None
    if ground_truth_path:
        with open(ground_truth_path, "r", encoding="utf-8") as handle:
            ground_truth = GroundTruth.from_dict(json.load(handle))

    decision_policy = None
    if decision_policy_path:
        with open(decision_policy_path, "r", encoding="utf-8") as handle:
            decision_policy = DecisionPolicy.from_dict(json.load(handle))

    timeline_rows: Tuple[Dict[str, Any], ...] = ()
    if timeline_rows_path:
        with open(timeline_rows_path, "r", encoding="utf-8") as handle:
            timeline_rows = tuple(json.load(handle))

    return IncidentData(
        building=project.building,
        scenario=scenario,
        ground_truth=ground_truth,
        decision_policy=decision_policy,
        timeline_rows=timeline_rows,
    )
