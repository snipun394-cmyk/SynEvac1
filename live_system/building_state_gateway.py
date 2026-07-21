from typing import Callable, Iterable, Optional, Protocol

from hazard.snapshot import HazardSnapshot

from occupancy.snapshot import OccupancySnapshot

from camera_manager.status import CameraStatus
from sensor_manager.status import SensorStatus

from perception.models.camera_observation import CameraFrameObservation
from perception.models.heat_detector_observation import HeatDetectorReading
from perception.models.smoke_detector_observation import SmokeDetectorReading

from multi_camera_fusion.track import FusionResult

from facp.models import FACPSnapshot

from building_control.snapshot import ControlStateSnapshot

from building_state.estimator import BuildingStateEstimator
from building_state.models import BuildingState


# =====================================================
# Canonical Live BuildingState Runtime Assembly -- the seam this
# milestone adds. BuildingStateEstimator.estimate() already accepts
# every one of these ingredients as an independently optional keyword
# argument (see building_state/estimator.py's own signature); this
# module's entire job is promoting each one from "a value this cycle"
# to "a value-at-time provider a caller injects", so LiveOrchestrator
# can obtain a fresh BuildingState every cycle without importing
# CameraManager, SensorManager, MultiCameraFusionEngine, SimulatedFACP,
# BuildingControlController, ReplayFrameSource, or any future
# RTSPFrameSource itself (enforced by
# LiveSystemPackageDependencyDirectionTests in tests/test_live_system.py).
#
# No aggregation logic is duplicated here -- every provider below
# supplies one already-computed value; BuildingStateEstimator.estimate()
# remains the one and only place those values are fused into a
# BuildingState. A provider left as None (the default for all eleven)
# means exactly what it already means to BuildingStateEstimator itself:
# "nothing available for this input yet", never a fabricated reading --
# so a gateway with every provider left unset still produces a valid,
# empty BuildingState (no cameras/sensors/FACP/control configured is a
# normal, non-error runtime state, not a crash).
# =====================================================


HazardSnapshotProvider = Callable[[float], HazardSnapshot]
OccupancySnapshotProvider = Callable[[float], OccupancySnapshot]

CameraStatusProvider = Callable[[float], Iterable[CameraStatus]]
CameraObservationProvider = Callable[[float], Iterable[CameraFrameObservation]]
FusionResultProvider = Callable[[float], Optional[FusionResult]]

SmokeDetectorStatusProvider = Callable[[float], Iterable[SensorStatus]]
SmokeDetectorReadingProvider = Callable[[float], Iterable[SmokeDetectorReading]]
HeatDetectorStatusProvider = Callable[[float], Iterable[SensorStatus]]
HeatDetectorReadingProvider = Callable[[float], Iterable[HeatDetectorReading]]

FACPSnapshotProvider = Callable[[float], Optional[FACPSnapshot]]
ControlSnapshotProvider = Callable[[float], Optional[ControlStateSnapshot]]


class BuildingStateGateway(Protocol):

    # The one method live_system.orchestrator.LiveOrchestrator actually
    # calls -- same "thin Protocol, real adapter composes an already-
    # public API" shape as PerceptionGateway/AIInferenceGateway/
    # DecisionPolicyGateway/CommandCenterGateway in live_system.integration.
    # A test that wants a deterministic double simply implements this
    # Protocol; it never has to subclass EstimatorBuildingStateGateway.

    def collect(self, time: float) -> BuildingState: ...


class EstimatorBuildingStateGateway:

    # The real adapter -- composes BuildingStateEstimator with whatever
    # subset of the eleven providers above a caller wires up. Camera/
    # fusion inputs are deliberately received as already-produced values
    # (a CameraStatusProvider, a FusionResult already fused by whoever's
    # MultiCameraFusionEngine is running this cycle -- Simulation, Replay,
    # or a future Live adapter) rather than this class owning a
    # CameraManager/MultiCameraFusionEngine itself: composing those
    # managers for a given deployment is the caller's job (see
    # designer/building_state_debug_runner.py for the Designer/Sandbox
    # composition this class generalizes the same pattern from), exactly
    # mirroring why BuildingStateEstimator itself never imports
    # CameraManager/SensorManager either.

    def __init__(
        self,
        estimator: Optional[BuildingStateEstimator] = None,
        hazard_snapshot_provider: Optional[HazardSnapshotProvider] = None,
        occupancy_snapshot_provider: Optional[OccupancySnapshotProvider] = None,
        camera_status_provider: Optional[CameraStatusProvider] = None,
        camera_observation_provider: Optional[CameraObservationProvider] = None,
        fusion_result_provider: Optional[FusionResultProvider] = None,
        smoke_detector_status_provider: Optional[SmokeDetectorStatusProvider] = None,
        smoke_detector_reading_provider: Optional[SmokeDetectorReadingProvider] = None,
        heat_detector_status_provider: Optional[HeatDetectorStatusProvider] = None,
        heat_detector_reading_provider: Optional[HeatDetectorReadingProvider] = None,
        facp_snapshot_provider: Optional[FACPSnapshotProvider] = None,
        control_snapshot_provider: Optional[ControlSnapshotProvider] = None,
    ):

        self._estimator = estimator if estimator is not None else BuildingStateEstimator()

        self._hazard_snapshot_provider = hazard_snapshot_provider
        self._occupancy_snapshot_provider = occupancy_snapshot_provider

        self._camera_status_provider = camera_status_provider
        self._camera_observation_provider = camera_observation_provider
        self._fusion_result_provider = fusion_result_provider

        self._smoke_detector_status_provider = smoke_detector_status_provider
        self._smoke_detector_reading_provider = smoke_detector_reading_provider
        self._heat_detector_status_provider = heat_detector_status_provider
        self._heat_detector_reading_provider = heat_detector_reading_provider

        self._facp_snapshot_provider = facp_snapshot_provider
        self._control_snapshot_provider = control_snapshot_provider

    # =====================================================

    def collect(self, time: float) -> BuildingState:

        # Live Perception -> BuildingState Integration Bridge milestone
        # -- fusion_result is resolved FIRST, deliberately, before every
        # other provider below. Python evaluates keyword arguments in
        # the order they are WRITTEN, and live_runtime.factory.
        # build_live_runtime()'s own fusion_result_provider is the one
        # provider with a real side effect: it is what actually calls
        # live_camera_pipeline.pipeline.LiveCameraPipeline.run_cycle()
        # this cycle, which is also what populates
        # live_occupants.manager.LiveOccupantManager (consumed by that
        # same factory's hazard_snapshot_provider/occupancy_snapshot_
        # provider, backed by sensor_fusion.engine.SensorFusionEngine).
        # Every provider here was originally designed to be independent
        # or a caller could freely omit any subset (still true) -- but
        # once a caller's OWN providers share a hidden dependency like
        # this one, resolving fusion_result first is what makes
        # hazard_snapshot/occupancy_snapshot see this cycle's fresh
        # camera-pipeline output rather than last cycle's (or, on the
        # very first cycle, nothing at all). Reordering this is the
        # only change in this file -- no provider's own contract,
        # BuildingStateEstimator, or BuildingState changed at all.
        fusion_result = self._resolve_optional(self._fusion_result_provider, time)

        return self._estimator.estimate(
            time,
            hazard_snapshot=self._resolve(
                self._hazard_snapshot_provider, time, default_factory=HazardSnapshot,
            ),
            occupancy_snapshot=self._resolve(
                self._occupancy_snapshot_provider, time, default_factory=OccupancySnapshot,
            ),
            camera_statuses=self._resolve_iterable(self._camera_status_provider, time),
            camera_observations=self._resolve_iterable(self._camera_observation_provider, time),
            smoke_detector_statuses=self._resolve_iterable(self._smoke_detector_status_provider, time),
            smoke_detector_readings=self._resolve_iterable(self._smoke_detector_reading_provider, time),
            heat_detector_statuses=self._resolve_iterable(self._heat_detector_status_provider, time),
            heat_detector_readings=self._resolve_iterable(self._heat_detector_reading_provider, time),
            fusion_result=fusion_result,
            facp_snapshot=self._resolve_optional(self._facp_snapshot_provider, time),
            control_snapshot=self._resolve_optional(self._control_snapshot_provider, time),
        )

    # =====================================================
    # Internals -- three small resolution shapes, one per "what an
    # absent provider honestly defaults to" (an empty snapshot, an
    # empty tuple, or None), never a fabricated non-empty reading.
    # =====================================================

    def _resolve(self, provider, time, *, default_factory):

        if provider is None:
            return default_factory()

        return provider(time)

    def _resolve_iterable(self, provider, time):

        if provider is None:
            return ()

        return tuple(provider(time))

    def _resolve_optional(self, provider, time):

        if provider is None:
            return None

        return provider(time)
