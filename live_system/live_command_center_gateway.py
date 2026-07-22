from typing import Dict, Optional, Tuple

from models.building import Building

from scenario.engineering_state import DeviceAvailability

from hazard.severity import HazardSeverity

from building_control.types import ControlSystemType

from facp.models import DetectorConditionReport

from building_state.models import BuildingState

from command_center.incident_data import IncidentFrame
from command_center.data_source import CommandCenterMode, CommandCenterSnapshot, SnapshotConsistency

from live_system.event_bus import EventBus
from live_system.state_manager import StateManager


# =====================================================
# Live Command Center Integration milestone -- the LIVE
# CommandCenterDataSource implementation. Lives in `live_system`, not
# `command_center` (see command_center/data_source.py's own module
# docstring for why), mirroring live_ai_gateway.py/live_advisory_
# gateway.py's own established "thin Protocol implementation, real
# adapter composes an already-public API, never recomputes anything
# itself" shape exactly.
#
# THE ONE RULE THIS MODULE HOLDS TO: every field on the
# CommandCenterSnapshot this class produces is read straight off
# StateManager's own already-computed LiveBuildingSnapshot -- this
# class never instantiates BuildingStateEstimator, LiveAIInferenceService,
# or AdvisoryOrchestrator, never calls CameraManager/SensorManager/
# MultiCameraFusionEngine/SimulatedFACP directly, and never imports any
# of camera decoding, HumanDetector, IdentityResolver, or hardware
# providers (mechanically enforced -- see
# tests/test_live_command_center.py::CommandCenterLiveIntegrationGuardTests).
# Command Center is a VIEW over whatever LiveOrchestrator already
# produced; this module is the seam that makes that honest.
# =====================================================


def frame_from_building_state(building_state: Optional[BuildingState]) -> Optional[IncidentFrame]:

    # The one place a canonical BuildingState is adapted into the
    # per-tick shape every existing Command Center panel already knows
    # how to render -- deliberately best-effort and honest: every
    # IncidentFrame field BuildingState has no live equivalent for is
    # left at its own honest empty/None default (never fabricated), and
    # every field it DOES have a genuine source for is read straight off
    # BuildingState, never guessed. This is the "richer, per-instant
    # IncidentFrame directly from a LiveBuildingSnapshot" IncidentFrame's
    # own docstring already anticipated a live source would build.
    #
    # zone_smoke/zone_temperature/zone_visibility/stair_states/
    # ignition_zone_id/current_hazard_score/people_evacuated/
    # people_trapped/human_observations/human_decisions all stay at
    # their empty/None defaults -- BuildingState carries no live source
    # for any of them today (confirmed by this milestone's own Phase 1
    # investigation): no per-zone smoke/temperature/visibility reading,
    # no per-door/exit "stair" engineering state, no concept of an
    # "ignition zone" or a single scalar hazard score, no Timeline-
    # Dataset-derived evacuated/trapped headcount, and no per-person
    # classification on FusedTrack (Phase 7's own explicit instruction:
    # show an aggregate "Estimated Occupants: N" per zone, never
    # invented per-occupant coordinates or identities).

    if building_state is None:
        return None

    zone_occupancy: Dict[str, Optional[float]] = {
        node_id: observation.occupant_count
        for node_id, observation in building_state.zone_occupancy.observations.items()
    }

    current_fire_zone_ids = frozenset(
        zone_id for zone_id, severity in building_state.hazard_summary.zone_severities.items()
        if severity in (HazardSeverity.HIGH, HazardSeverity.CRITICAL)
    )

    door_states: Dict[str, str] = {}
    exit_states: Dict[str, str] = {}

    if building_state.control_status is not None:

        for entry in building_state.control_status.entries:

            if entry.system_type == ControlSystemType.DOOR and entry.target_id:
                door_states[entry.target_id] = entry.action.name
            elif entry.system_type == ControlSystemType.EXIT and entry.target_id:
                exit_states[entry.target_id] = entry.action.name

    # detector_states/camera_states carry AVAILABILITY (is the device
    # physically able to function), the exact same DeviceAvailability
    # vocabulary _baseline_frame()/_frame_from_row() already write into
    # these fields for Replay -- NOT alarm/fault condition (that
    # richer, live-only concept surfaces separately, in the new Live
    # Status panel, straight off BuildingState.smoke_detector_states/
    # heat_detector_states/facp_status -- see command_center/
    # live_status_panel.py).
    detector_states: Dict[str, str] = {}
    for sensor_id, asset in {**building_state.smoke_detector_states, **building_state.heat_detector_states}.items():
        detector_states[sensor_id] = (
            DeviceAvailability.AVAILABLE.name if asset.status.active else DeviceAvailability.FAILED.name
        )

    camera_states: Dict[str, str] = {
        camera_id: (DeviceAvailability.AVAILABLE.name if asset.status.active else DeviceAvailability.FAILED.name)
        for camera_id, asset in building_state.camera_observations.items()
    }

    known_occupancy = [count for count in zone_occupancy.values() if count is not None]
    people_remaining = int(sum(known_occupancy)) if known_occupancy else None

    return IncidentFrame(
        time=building_state.timestamp,
        current_fire_zone_ids=current_fire_zone_ids,
        zone_occupancy=zone_occupancy,
        door_states=door_states,
        exit_states=exit_states,
        detector_states=detector_states,
        camera_states=camera_states,
        people_remaining=people_remaining,
    )


def detector_condition(asset) -> str:

    # The richer, live-only ALARM/FAULT/NORMAL condition string the
    # Live Status panel shows (as opposed to frame_from_building_state()'s
    # coarser AVAILABLE/FAILED). Reuses facp.models.DetectorConditionReport.
    # from_status_and_reading()'s own already-established "health fault
    # outranks alarm, alarm outranks clear" priority rule verbatim,
    # rather than re-deriving it -- `asset` is a building_state.models.
    # DetectorAssetState (status + reading).

    return DetectorConditionReport.from_status_and_reading(asset.status, asset.reading).state.name


# =====================================================


_EVENT_TYPE_LABELS = {
    "SENSOR_UPDATE": "Sensor readings updated",
    "ALARM_ACTIVATED": "Alarm activated",
    "HAZARD_UPDATED": "Hazard state updated",
    "OCCUPANCY_UPDATED": "Occupancy updated",
    "BUILDING_STATE_UPDATED": "BuildingState updated",
    "AI_PREDICTION_UPDATED": "AI prediction updated",
    "ADVISORY_REPORT_UPDATED": "Advisory report updated",
    "RECOMMENDATION_UPDATED": "Recommendation updated",
    "OPERATOR_ACKNOWLEDGEMENT": "Operator acknowledgement",
}


def _describe_event(event) -> str:

    # Formats only fields the EventBus/payload types already guarantee
    # -- never a fabricated narrative (e.g. a specific detector id this
    # function cannot honestly know without inspecting a payload shape
    # this module deliberately does not import anything new to parse).

    label = _EVENT_TYPE_LABELS.get(event.event_type.name, event.event_type.name.replace("_", " ").title())

    if event.event_type.name == "AI_PREDICTION_UPDATED" and getattr(event.payload, "bottleneck", None) is not None:
        label = f"Bottleneck probability updated to {event.payload.bottleneck.probability:.2f}"

    elif event.event_type.name == "ADVISORY_REPORT_UPDATED":
        count = len(getattr(event.payload, "civilian_announcements", ()) or ())
        label = f"Advisory report updated ({count} civilian announcement(s))"

    return f"{event.timestamp:.1f}s -- {label}"


# =====================================================


class LiveCommandCenterDataSource:

    # The real, Live CommandCenterDataSource. Composes an already-
    # constructed StateManager (owned by a caller-run LiveOrchestrator)
    # and an already-known Building (deployment configuration -- the
    # one static fact BuildingState itself never carries, same as
    # EstimatorBuildingStateGateway's own callers already supply their
    # own Building/Scenario elsewhere). An optional EventBus enables
    # Phase 15's bounded recent-event list; without one, recent_events
    # stays honestly empty rather than fabricated.

    def __init__(
        self,
        state_manager: StateManager,
        *,
        building: Building,
        event_bus: Optional[EventBus] = None,
        recent_events_limit: int = 20,
    ):

        self.mode = CommandCenterMode.LIVE

        self._state_manager = state_manager
        self._building = building
        self._event_bus = event_bus
        self._recent_events_limit = recent_events_limit

        self._running = False

    # =====================================================

    def start(self) -> None:

        self._running = True

    # =====================================================

    def stop(self) -> None:

        self._running = False

    # =====================================================

    @property
    def is_running(self) -> bool:

        return self._running

    # =====================================================

    def current_snapshot(self) -> CommandCenterSnapshot:

        building_name = self._building.name if self._building is not None else None

        if not self._running:

            return CommandCenterSnapshot(
                mode=CommandCenterMode.LIVE, building=self._building, building_name=building_name,
                consistency=SnapshotConsistency.UNAVAILABLE,
            )

        snapshot = self._state_manager.current()
        building_state = snapshot.building_state

        if building_state is None:

            return CommandCenterSnapshot(
                mode=CommandCenterMode.LIVE, building=self._building, building_name=building_name,
                consistency=SnapshotConsistency.UNAVAILABLE, recent_events=self._recent_events(),
            )

        timestamps = snapshot.component_timestamps

        return CommandCenterSnapshot(
            mode=CommandCenterMode.LIVE,
            timestamp=building_state.timestamp,
            building_name=building_name,
            building=self._building,
            frame=frame_from_building_state(building_state),
            advisory_report=snapshot.advisory_report,
            decision_policy=None,
            building_state=building_state,
            ai_prediction_snapshot=snapshot.ai_prediction_snapshot,
            evacuation_progress=snapshot.evacuation_progress,
            emergency_response=snapshot.emergency_response,
            trajectory_intelligence=snapshot.trajectory_intelligence,
            evacuation_recommendation=snapshot.evacuation_recommendation,
            building_state_timestamp=timestamps.get("building_state"),
            ai_prediction_timestamp=timestamps.get("ai_prediction_snapshot"),
            advisory_timestamp=timestamps.get("advisory_report"),
            evacuation_progress_timestamp=timestamps.get("evacuation_progress"),
            emergency_response_timestamp=timestamps.get("emergency_response"),
            trajectory_intelligence_timestamp=timestamps.get("trajectory_intelligence"),
            evacuation_recommendation_timestamp=timestamps.get("evacuation_recommendation"),
            consistency=self._resolve_consistency(timestamps),
            recent_events=self._recent_events(),
        )

    # =====================================================

    def _resolve_consistency(self, timestamps) -> SnapshotConsistency:

        # Phase 14's own smallest honest mechanism. building_state's own
        # timestamp is the reference cycle -- if AI/Advisory have never
        # been populated at all this run, that is PARTIAL (a valid,
        # honestly-labeled configuration, e.g. no live_ai_gateway wired
        # up yet), never silently presented as CURRENT; if they HAVE
        # been populated but at a different cycle than building_state's
        # own latest timestamp, that is the mismatched-cycle case the
        # brief's own example names explicitly -- STALE, not CURRENT.

        building_state_time = timestamps.get("building_state")
        ai_time = timestamps.get("ai_prediction_snapshot")
        advisory_time = timestamps.get("advisory_report")

        if building_state_time is None:
            return SnapshotConsistency.UNAVAILABLE

        if ai_time is None or advisory_time is None:
            return SnapshotConsistency.PARTIAL

        if building_state_time == ai_time == advisory_time:
            return SnapshotConsistency.CURRENT

        return SnapshotConsistency.STALE

    # =====================================================

    def _recent_events(self) -> Tuple[str, ...]:

        if self._event_bus is None:
            return ()

        events = self._event_bus.history[-self._recent_events_limit:]

        return tuple(_describe_event(event) for event in events)
