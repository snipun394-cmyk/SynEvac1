from dataclasses import dataclass, field
from typing import Mapping, Tuple

from hazard.node_state import HazardNodeState
from hazard.snapshot import HazardSnapshot

from occupancy.observation import OccupancyObservation
from occupancy.snapshot import OccupancySnapshot

from camera_manager.manager import CameraManager
from sensor_manager.manager import SensorManager

from multi_camera_fusion.track import FusedTrack, FusionResult, TrackHistory

from perception.models.human_observation import HumanClassification, HumanState

from sandbox.occupant import SandboxOccupantState

from facp.engine import SimulatedFACP
from facp.models import DetectorConditionReport, PanelEvent

from building_state.consistency import ConsistencyWarning, check_consistency
from building_state.estimator import BuildingStateEstimator
from building_state.models import BuildingState

from designer.perception_debug_runner import PerceptionDebugRunner


# A ground-truth-only mapping (never Ground Truth's own DynamicHumanState,
# never Perception's own HumanState-derivation logic) purely so this
# debug view's occupant tracks show *something* more useful than "no
# state known" -- see BuildingStateDebugRunner's own docstring on why
# this is an allowed debug-only shortcut, not a second behavior model.
_SANDBOX_STATE_TO_HUMAN_STATE = {
    SandboxOccupantState.MOVING: HumanState.WALKING,
    SandboxOccupantState.IDLE: HumanState.STANDING,
    SandboxOccupantState.ROUTED: HumanState.STANDING,
    SandboxOccupantState.ARRIVED: HumanState.STANDING,
    SandboxOccupantState.UNREACHABLE: HumanState.WAITING,
}


@dataclass(frozen=True)
class BuildingStateDebugSnapshot:

    # One run()'s complete output for the Building State Debug Panel --
    # the BuildingState itself, the ConsistencyWarnings computed from it
    # (never stored on BuildingState itself, per that type's own "no
    # AI/decision field" contract), and zone_names for display, the same
    # convenience field PerceptionDebugSnapshot already carries.

    state: BuildingState
    warnings: Tuple[ConsistencyWarning, ...]
    zone_names: Mapping[str, str] = field(default_factory=dict)


class BuildingStateDebugRunner:

    # Designer/debug-only glue -- composes the Building State Estimator
    # (building_state/estimator.py) with a PerceptionDebugRunner instance
    # it owns privately, the same "glue module, not a new algorithm" role
    # perception_debug_runner.PerceptionDebugRunner already establishes
    # for the Perception Debug Panel. Implements no estimation logic of
    # its own beyond:
    #
    #   1. Reconstructing a HazardSnapshot/OccupancySnapshot from the
    #      GroundTruthZoneReading rows PerceptionDebugRunner.run()
    #      already returns (its own `ground_truth` mapping) -- a pure
    #      re-shaping of already-computed values, not a second Ground
    #      Truth computation.
    #   2. Ground-truth-backed occupant tracks, built directly from the
    #      running Sandbox's own SandboxOccupant positions -- the same
    #      allowance PerceptionDebugRunner._ground_truth_occupancy()
    #      already exercises for zone occupancy counts, one level more
    #      detailed (per-occupant instead of per-zone) because, unlike a
    #      real camera network, the Sandbox already knows each
    #      occupant's identity with certainty. This is NOT a substitute
    #      for multi_camera_fusion.engine.MultiCameraFusionEngine (real
    #      per-camera re-identification stays untouched and unused
    #      here) -- it is this debug panel's own ground-truth
    #      comparison view, exactly like PerceptionDebugRunner's own
    #      "Ground Truth" tab is for Perception.
    #   3. CameraManager/SensorManager asset discovery, purely to read
    #      each asset's own online/offline CameraStatus/SensorStatus --
    #      neither manager is modified, subclassed, or reimplemented.
    #
    # Owns its own hazard-override map (independent of the Perception
    # Debug Panel's own injector, via its own private PerceptionDebugRunner
    # instance) for the same reason PerceptionDebugRunner owns one: no
    # fire/smoke simulation is wired into the Designer UI yet. The two
    # panels' overrides are intentionally independent so neither debug
    # tool depends on the other being open.
    #
    # Also owns one SimulatedFACP (facp/engine.py) -- the building-wide
    # Fire Alarm Control Panel coordinating whatever smoke/heat detector
    # states this runner already computed for BuildingState. Never
    # recomputes a smoke_level/temperature reading itself; only wraps
    # each already-computed SensorStatus/reading pair into a
    # DetectorConditionReport (see that type's own from_status_and_
    # reading() classmethod) and hands the whole set to SimulatedFACP.
    # evaluate() each run() cycle, then folds the resulting FACPSnapshot
    # into BuildingState additively via BuildingStateEstimator's own
    # facp_snapshot parameter.

    def __init__(self):

        self._perception_runner = PerceptionDebugRunner()

        self._camera_manager = CameraManager()
        self._sensor_manager = SensorManager()

        self._estimator = BuildingStateEstimator()
        self._facp = SimulatedFACP(panel_id="FACP-1")

    # =====================================================

    def set_zone_hazard(self, zone_id: str, smoke_level, temperature):

        self._perception_runner.set_zone_hazard(zone_id, smoke_level, temperature)

    # =====================================================

    def clear_zone_hazard(self, zone_id: str):

        self._perception_runner.clear_zone_hazard(zone_id)

    # =====================================================

    def clear_all_hazard_overrides(self):

        self._perception_runner.clear_all_hazard_overrides()

    # =====================================================
    # FACP operator controls -- these operate ONLY on this runner's own
    # SimulatedFACP (Phase 7's own explicit "never pretend to control
    # real hardware" requirement). Re-raises facp.models.
    # InvalidPanelOperation as-is for the panel widget to catch and
    # display; this runner adds no policy of its own about when these
    # are legal to call.
    # =====================================================

    def acknowledge_facp(self, time: float, operator_id=None) -> PanelEvent:

        return self._facp.acknowledge(time, operator_id=operator_id)

    # =====================================================

    def silence_facp(self, time: float, operator_id=None) -> PanelEvent:

        return self._facp.silence(time, operator_id=operator_id)

    # =====================================================

    def reset_facp(self, time: float, operator_id=None) -> PanelEvent:

        return self._facp.reset(time, operator_id=operator_id)

    # =====================================================

    def run(self, building, sandbox_manager, time: float) -> BuildingStateDebugSnapshot:

        perception_snapshot = self._perception_runner.run(building, sandbox_manager, time)

        hazard_snapshot = self._reconstruct_hazard_snapshot(perception_snapshot)
        occupancy_snapshot = self._reconstruct_occupancy_snapshot(perception_snapshot)

        self._camera_manager.discover_cameras(building)
        self._sensor_manager.discover_sensors(building)

        camera_statuses = self._camera_manager.all_statuses()
        smoke_statuses, heat_statuses = self._sensor_statuses_by_type()

        fusion_result = self._ground_truth_fusion_result(sandbox_manager, time)

        detector_conditions = self._build_detector_condition_reports(
            smoke_statuses, perception_snapshot.smoke_detector_readings,
            heat_statuses, perception_snapshot.heat_detector_readings,
        )
        self._facp.evaluate(detector_conditions, time)
        facp_snapshot = self._facp.current_snapshot(time)

        state = self._estimator.estimate(
            time,
            hazard_snapshot=hazard_snapshot,
            occupancy_snapshot=occupancy_snapshot,
            camera_statuses=camera_statuses,
            camera_observations=perception_snapshot.camera_observations,
            smoke_detector_statuses=smoke_statuses,
            smoke_detector_readings=perception_snapshot.smoke_detector_readings,
            heat_detector_statuses=heat_statuses,
            heat_detector_readings=perception_snapshot.heat_detector_readings,
            fusion_result=fusion_result,
            facp_snapshot=facp_snapshot,
        )

        warnings = check_consistency(state)

        return BuildingStateDebugSnapshot(
            state=state,
            warnings=warnings,
            zone_names=perception_snapshot.zone_names,
        )

    # =====================================================

    def _reconstruct_hazard_snapshot(self, perception_snapshot) -> HazardSnapshot:

        return HazardSnapshot(
            timestamp=perception_snapshot.timestamp,
            node_states={
                zone_id: HazardNodeState(
                    smoke_level=reading.smoke_level,
                    temperature=reading.temperature,
                    hazard_score=reading.hazard_score,
                )
                for zone_id, reading in perception_snapshot.ground_truth.items()
            },
        )

    # =====================================================

    def _reconstruct_occupancy_snapshot(self, perception_snapshot) -> OccupancySnapshot:

        return OccupancySnapshot(
            timestamp=perception_snapshot.timestamp,
            observations={
                zone_id: OccupancyObservation(occupant_count=reading.occupant_count)
                for zone_id, reading in perception_snapshot.ground_truth.items()
            },
        )

    # =====================================================

    def _sensor_statuses_by_type(self):

        smoke_statuses = []
        heat_statuses = []

        for status in self._sensor_manager.all_statuses():

            if status.sensor_type == "SmokeDetector":
                smoke_statuses.append(status)
            elif status.sensor_type == "HeatDetector":
                heat_statuses.append(status)

        return tuple(smoke_statuses), tuple(heat_statuses)

    # =====================================================

    def _build_detector_condition_reports(
        self, smoke_statuses, smoke_readings, heat_statuses, heat_readings,
    ):

        smoke_reading_by_id = {reading.detector_id: reading for reading in smoke_readings}
        heat_reading_by_id = {reading.detector_id: reading for reading in heat_readings}

        reports = {
            status.sensor_id: DetectorConditionReport.from_status_and_reading(
                status, smoke_reading_by_id.get(status.sensor_id),
            )
            for status in smoke_statuses
        }
        reports.update({
            status.sensor_id: DetectorConditionReport.from_status_and_reading(
                status, heat_reading_by_id.get(status.sensor_id),
            )
            for status in heat_statuses
        })

        return reports

    # =====================================================

    def _ground_truth_fusion_result(self, sandbox_manager, time: float) -> FusionResult:

        occupants = getattr(sandbox_manager, "occupants", []) if sandbox_manager is not None else []

        tracks = tuple(
            FusedTrack(
                track_id=occupant.occupant_id,
                source_camera_ids=(),
                floor_id=occupant.floor_id,
                zone_id=occupant.zone_id,
                classification=HumanClassification.UNKNOWN,
                human_state=_SANDBOX_STATE_TO_HUMAN_STATE.get(occupant.state),
                confidence=1.0,
                timestamp=time,
                history=TrackHistory(
                    track_id=occupant.occupant_id,
                    previous_camera_id=None,
                    current_camera_id=None,
                    last_observation_time=time,
                    current_zone_id=occupant.zone_id,
                    current_floor_id=occupant.floor_id,
                ),
            )
            for occupant in occupants
        )

        return FusionResult(timestamp=time, tracks=tracks)
