from typing import Iterable, Optional

from hazard.severity import HazardSeverity
from hazard.snapshot import HazardSnapshot

from models.sensor_asset import DetectorState, HealthStatus

from occupancy.snapshot import OccupancySnapshot

from camera_manager.status import CameraStatus
from sensor_manager.status import SensorStatus

from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading

from multi_camera_fusion.track import FusionResult

from facp.models import FACPSnapshot

from building_control.snapshot import ControlStateSnapshot

from building_state.models import (
    ActiveAssetsSummary,
    BuildingState,
    CameraAssetState,
    DetectorAssetState,
    HazardSummary,
)


# Ordinal ranking used only to find the single worst zone/overall
# severity below -- kept as an explicit tuple rather than comparing
# HazardSeverity.value directly, so this file never depends on the
# incidental integer values auto() happens to assign in hazard/severity.py.
_SEVERITY_RANK = (
    HazardSeverity.NONE,
    HazardSeverity.LOW,
    HazardSeverity.MODERATE,
    HazardSeverity.HIGH,
    HazardSeverity.CRITICAL,
)


class BuildingStateEstimator:

    # Fuses already-collected observations -- a HazardSnapshot, an
    # OccupancySnapshot, CameraStatus/SensorStatus asset status, typed
    # camera/detector readings, an optional multi-camera FusionResult,
    # an optional Fire Alarm Control Panel FACPSnapshot (see
    # facp/engine.py::SimulatedFACP, which computes it -- this
    # estimator only ever passes it through unchanged onto
    # BuildingState.facp_status), and an optional Building Control
    # ControlStateSnapshot (see building_control.controller.
    # BuildingControlController.snapshot(), same pass-through-only
    # discipline, onto BuildingState.control_status) -- into one
    # immutable BuildingState.
    # Performs no AI reasoning, no
    # reinforcement learning, and no decision-making of any kind: every
    # value on the resulting BuildingState is either passed through
    # unchanged or classified through an already-centralized rule
    # (HazardSeverity.from_score(), SensorManager's own ALARM > FAULT >
    # NORMAL alarm priority). It never mutates any argument it is given.
    #
    # Deliberately takes pre-collected data rather than a Building/
    # CameraManager/SensorManager/HazardProvider/OccupancyProvider
    # itself: this estimator must not know, and must never be made to
    # know, whether its inputs came from a running Simulation, a
    # Replay, or a Live deployment (Phase 6's "future compatibility"
    # requirement) -- collecting from Camera Manager/Sensor Manager/a
    # hazard or occupancy provider/a fusion engine is each caller's own
    # job (see designer/building_state_debug_runner.py for the
    # Designer/Sandbox composition, simulation_runtime.runtime.
    # SimulationRuntime.tick() for where the same ingredients are
    # already collected once per tick in the batch runtime).

    def estimate(
        self,
        timestamp: float,
        *,
        hazard_snapshot: HazardSnapshot,
        occupancy_snapshot: OccupancySnapshot,
        camera_statuses: Iterable[CameraStatus] = (),
        camera_observations: Iterable[CameraFrameObservation] = (),
        smoke_detector_statuses: Iterable[SensorStatus] = (),
        smoke_detector_readings: Iterable[SmokeDetectorReading] = (),
        heat_detector_statuses: Iterable[SensorStatus] = (),
        heat_detector_readings: Iterable[HeatDetectorReading] = (),
        fusion_result: Optional[FusionResult] = None,
        facp_snapshot: Optional[FACPSnapshot] = None,
        control_snapshot: Optional[ControlStateSnapshot] = None,
    ) -> BuildingState:

        # Materialized up front -- every argument is iterated more than
        # once below (once to build the per-asset state, again to build
        # ActiveAssetsSummary), so a caller passing a one-shot generator
        # would otherwise silently see an empty second pass.
        camera_statuses = tuple(camera_statuses)
        smoke_detector_statuses = tuple(smoke_detector_statuses)
        heat_detector_statuses = tuple(heat_detector_statuses)

        camera_states = self._build_camera_states(camera_statuses, camera_observations)
        smoke_states = self._build_detector_states(smoke_detector_statuses, smoke_detector_readings)
        heat_states = self._build_detector_states(heat_detector_statuses, heat_detector_readings)

        occupant_tracks = (
            {track.track_id: track for track in fusion_result.tracks}
            if fusion_result is not None
            else {}
        )

        return BuildingState(
            timestamp=timestamp,
            occupant_tracks=occupant_tracks,
            zone_occupancy=occupancy_snapshot,
            camera_observations=camera_states,
            smoke_detector_states=smoke_states,
            heat_detector_states=heat_states,
            hazard_summary=self._summarize_hazard(hazard_snapshot),
            building_alarm_status=self._aggregate_alarm_status(smoke_states, heat_states),
            active_assets=self._summarize_active_assets(
                camera_statuses, smoke_detector_statuses, heat_detector_statuses,
            ),
            facp_status=facp_snapshot,
            control_status=control_snapshot,
        )

    # =====================================================

    def _build_camera_states(self, camera_statuses, camera_observations):

        observation_by_camera_id = {
            observation.camera_id: observation for observation in camera_observations
        }

        return {
            status.camera_id: CameraAssetState(
                status=status,
                frame_observation=observation_by_camera_id.get(status.camera_id),
            )
            for status in camera_statuses
        }

    # =====================================================

    def _build_detector_states(self, detector_statuses, detector_readings):

        reading_by_sensor_id = {
            reading.detector_id: reading for reading in detector_readings
        }

        return {
            status.sensor_id: DetectorAssetState(
                status=status,
                reading=reading_by_sensor_id.get(status.sensor_id),
            )
            for status in detector_statuses
        }

    # =====================================================

    def _summarize_hazard(self, hazard_snapshot: HazardSnapshot) -> HazardSummary:

        zone_severities = {
            zone_id: HazardSeverity.from_score(node_state.hazard_score)
            for zone_id, node_state in hazard_snapshot.node_states.items()
        }

        overall_severity = HazardSeverity.NONE
        worst_zone_id = None

        for zone_id, severity in zone_severities.items():

            if _SEVERITY_RANK.index(severity) > _SEVERITY_RANK.index(overall_severity):

                overall_severity = severity
                worst_zone_id = zone_id

        return HazardSummary(
            overall_severity=overall_severity,
            worst_zone_id=worst_zone_id,
            zone_severities=zone_severities,
        )

    # =====================================================

    def _aggregate_alarm_status(self, smoke_states, heat_states) -> DetectorState:

        # Same ALARM > FAULT > NORMAL priority SensorManager.
        # aggregate_alarm_state() already establishes, restated here
        # (never imported -- see that method's own docstring on why
        # SensorManager never computes a hazard-driven state itself)
        # over BuildingState's own asset states instead of a caller-
        # supplied DetectorState mapping.

        all_states = tuple(smoke_states.values()) + tuple(heat_states.values())

        if any(asset.reading is not None and asset.reading.alarm_active for asset in all_states):
            return DetectorState.ALARM

        if any(asset.status.health_status != HealthStatus.OK for asset in all_states):
            return DetectorState.FAULT

        return DetectorState.NORMAL

    # =====================================================

    def _summarize_active_assets(
        self, camera_statuses, smoke_detector_statuses, heat_detector_statuses,
    ) -> ActiveAssetsSummary:

        sensor_statuses = smoke_detector_statuses + heat_detector_statuses

        return ActiveAssetsSummary(
            active_camera_ids=tuple(sorted(s.camera_id for s in camera_statuses if s.active)),
            offline_camera_ids=tuple(sorted(s.camera_id for s in camera_statuses if not s.active)),
            active_sensor_ids=tuple(sorted(s.sensor_id for s in sensor_statuses if s.active)),
            offline_sensor_ids=tuple(sorted(s.sensor_id for s in sensor_statuses if not s.active)),
        )
