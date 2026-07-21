from typing import Callable, Mapping, Optional

from models.engineering_asset import DeviceMode

from camera_manager.manager import CameraManager

from sensor_manager.manager import SensorManager

from multi_camera_fusion.engine import MultiCameraFusionEngine

from speaker_manager.manager import SpeakerManager

from live_camera_pipeline.detection_provider import LiveCameraPipelineDetectionProvider
from live_camera_pipeline.pipeline import LiveCameraPipeline

from voice_evacuation.controller import VoiceEvacuationController
from voice_evacuation.provider import SimulationVoiceOutputProvider

from building_control.controller import BuildingControlController
from building_control.providers import SimulationControlProvider

from live_system.building_state_gateway import EstimatorBuildingStateGateway
from live_system.event_bus import EventBus
from live_system.live_command_center_gateway import LiveCommandCenterDataSource
from live_system.orchestrator import LiveOrchestrator

from command_center.live_operator_action_gateway import LiveOperatorActionGateway

from live_occupants.manager import LiveOccupantManager

from sensor_fusion.engine import SensorFusionEngine

from live_perception.coordinator import LivePerceptionFusionCoordinator
from live_perception.providers import (
    LiveFACPObservationProvider, LiveHeatObservationProvider,
    LiveOccupantObservationProvider, LiveSmokeObservationProvider,
)

from crowd_intelligence.engine import CrowdIntelligenceEngine

from live_system.crowd_intelligence_gateway import EngineCrowdIntelligenceGateway

from live_runtime.runtime import LiveRuntime


# =====================================================
# Production Live Runtime Composition Root milestone -- the one factory
# responsible for assembling a coherent LiveRuntime out of already-
# existing, already-tested components. Every import above is a package
# this factory COMPOSES; none of their logic is duplicated here.
#
# Nothing in this module performs network I/O, opens a camera
# connection, or otherwise "goes live" on its own -- constructing a
# LiveRuntime (or calling this function) never calls start() on
# anything. Connection/start remains an explicit, separate call the
# caller makes on the returned LiveRuntime (Phase 3's own hard
# requirement, mirroring RTSPFrameSource's own "constructor performs
# zero network I/O" discipline one layer up).
# =====================================================


def build_live_runtime(
    building,
    *,
    frame_sources: Optional[Mapping[str, object]] = None,
    human_detector: Optional[object] = None,
    identity_resolver: Optional[object] = None,
    tracker: Optional[object] = None,
    behavior_recognizer: Optional[object] = None,
    cross_camera_identity_resolver: Optional[object] = None,
    world_projector: Optional[object] = None,
    live_occupant_manager: Optional[LiveOccupantManager] = None,
    sensor_fusion_engine: Optional[SensorFusionEngine] = None,
    crowd_intelligence_engine: Optional[CrowdIntelligenceEngine] = None,
    smoke_detector_reading_provider: Optional[Callable[[float], object]] = None,
    heat_detector_reading_provider: Optional[Callable[[float], object]] = None,
    camera_manager: Optional[CameraManager] = None,
    sensor_manager: Optional[SensorManager] = None,
    fusion_engine: Optional[MultiCameraFusionEngine] = None,
    facp: Optional[object] = None,
    speaker_manager: Optional[SpeakerManager] = None,
    occupancy_snapshot_provider: Optional[Callable[[float], object]] = None,
    voice_output_provider: Optional[object] = None,
    building_control_provider: Optional[object] = None,
    live_ai_gateway: Optional[object] = None,
    live_advisory_gateway: Optional[object] = None,
    event_bus: Optional[EventBus] = None,
    recent_events_limit: int = 20,
    interval_seconds: float = 1.0,
) -> LiveRuntime:

    # =====================================================
    # Live Perception -> BuildingState Integration Bridge milestone --
    # closes the gap Phase 1's own investigation found: this factory
    # previously wired human_detector/identity_resolver into
    # LiveCameraPipeline but NEVER tracker/behavior_recognizer/
    # cross_camera_identity_resolver/world_projector/live_occupant_manager
    # (every one of those milestones' own work existed only as a
    # standalone, separately-tested package, never actually reachable
    # from build_live_runtime() at all) -- and BuildingState's
    # hazard_snapshot was ALWAYS the gateway's own empty default in
    # production (no hazard_snapshot_provider was ever wired), while
    # smoke_detector_reading_provider/heat_detector_reading_provider
    # were never wired either (only detector STATUS, never an actual
    # reading). All five new tracker-chain parameters above are
    # OPTIONAL and threaded into the SAME single LiveCameraPipeline
    # instance already constructed below -- there is exactly one
    # LiveCameraPipeline for every camera in frame_sources, so
    # live_occupant_manager is automatically the ONE shared instance
    # every camera contributes to (Phase 5's own "not one per camera"
    # requirement), never duplicated.
    # =====================================================

    live_occupant_manager = live_occupant_manager if live_occupant_manager is not None else LiveOccupantManager()
    sensor_fusion_engine = sensor_fusion_engine if sensor_fusion_engine is not None else SensorFusionEngine()

    # Live Occupancy, Crowd Density & Congestion Intelligence milestone --
    # exactly ONE CrowdIntelligenceEngine for this live session (Phase
    # 10), reading the SAME shared live_occupant_manager every other
    # stage above reads, plus this session's own Building (read-only
    # geometry -- Zone/Door/Exit/Staircase). Always constructed (a
    # caller may still supply their own, e.g. with custom thresholds),
    # never conditionally, mirroring live_occupant_manager/
    # sensor_fusion_engine's own "always exists, even with zero
    # occupants yet" default above.
    crowd_intelligence_engine = (
        crowd_intelligence_engine if crowd_intelligence_engine is not None
        else CrowdIntelligenceEngine(building, live_occupant_manager)
    )

    # =====================================================
    # Digital Twin / asset-management layer (Phase 1/4) -- exactly one
    # CameraManager/SensorManager/MultiCameraFusionEngine/SpeakerManager
    # for this live session, whether freshly constructed here or
    # supplied by a caller that already owns one (e.g. a future Designer
    # integration reusing its own CameraManagerPanel.manager instead of
    # a second, duplicate instance -- Phase 1 item 9's own named risk).
    # =====================================================

    camera_manager = camera_manager if camera_manager is not None else CameraManager()
    camera_manager.discover_cameras(building)

    sensor_manager = sensor_manager if sensor_manager is not None else SensorManager()
    sensor_manager.discover_sensors(building)

    fusion_engine = fusion_engine if fusion_engine is not None else MultiCameraFusionEngine()

    speaker_manager = speaker_manager if speaker_manager is not None else SpeakerManager()
    speaker_manager.discover_speakers(building)

    # =====================================================
    # Camera ingestion (Phase 3) -- wired through CameraManager's own
    # production routing (register_detection_provider/set_camera_mode/
    # all_detections), the same path docs/architecture/
    # cctv_integration_readiness.md documents, never the CameraManager-
    # bypassing shortcut a test harness might take. Only ever built when
    # a caller actually supplied frame_sources/human_detector/
    # identity_resolver -- "no cameras" (Phase 8 item 1) or "camera
    # configured but no frame source" (item 2) both simply leave
    # camera_pipeline/fusion_result_provider unset, never an error.
    # =====================================================

    frame_sources = dict(frame_sources or {})
    camera_pipeline = None
    fusion_result_provider = None

    if frame_sources and human_detector is not None and identity_resolver is not None:

        detection_provider = LiveCameraPipelineDetectionProvider()

        camera_pipeline = LiveCameraPipeline(
            frame_sources=frame_sources, human_detector=human_detector,
            identity_resolver=identity_resolver, detection_provider=detection_provider,
            tracker=tracker, behavior_recognizer=behavior_recognizer,
            cross_camera_identity_resolver=cross_camera_identity_resolver,
            world_projector=world_projector, live_occupant_manager=live_occupant_manager,
        )

        camera_manager.register_detection_provider(DeviceMode.LIVE, detection_provider)

        for camera_id in frame_sources:
            camera_manager.set_camera_mode(camera_id, DeviceMode.LIVE)

        def fusion_result_provider(time: float):

            camera_pipeline.run_cycle(time)
            detections = camera_manager.all_detections(time)

            return fusion_engine.fuse(detections, time)

    # =====================================================
    # FACP (Phase 1 item 5/8) -- read-only passthrough only. This
    # factory never calls facp.evaluate()/acknowledge()/silence()/
    # reset() itself -- exactly as building_state_gateway.py's own
    # facp_snapshot_provider contract requires ("the estimator/gateway
    # never call acknowledge()/silence()/reset()/evaluate() themselves");
    # feeding it real detector condition reports each cycle is a
    # separate, already-existing SensorManager+FACP integration concern
    # this milestone does not reinvent, left to the caller (or a future
    # milestone) to wire, the same way Designer's own
    # BuildingStateDebugRunner does it today.
    # =====================================================

    facp_snapshot_provider = (lambda time: facp.current_snapshot(time)) if facp is not None else None

    # =====================================================
    # Sensor status (bookkeeping only -- SensorManager.all_statuses()
    # needs no external reading, unlike a detector's actual hazard
    # reading, which this factory does not fabricate a source for).
    # =====================================================

    def smoke_detector_status_provider(time: float):
        return tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "SmokeDetector")

    def heat_detector_status_provider(time: float):
        return tuple(s for s in sensor_manager.all_statuses() if s.sensor_type == "HeatDetector")

    # =====================================================
    # Voice Evacuation / Building Control (Phase 4/6) -- exactly one
    # VoiceEvacuationController/BuildingControlController per live
    # session, both built here and threaded into the SAME
    # LiveOperatorActionGateway a caller later hands to Command Center
    # (command_center.main_window.MainWindow.enable_live_mode) -- never
    # a second, independently-constructed controller anywhere else.
    # None whenever no provider was supplied (NO_PROVIDER capability,
    # per docs/architecture/live_operator_action_routing.md §4) --
    # never fabricated.
    # =====================================================

    voice_evacuation_controller = (
        VoiceEvacuationController(speaker_manager, voice_output_provider)
        if voice_output_provider is not None else None
    )

    building_control_controller = (
        BuildingControlController(building, building_control_provider)
        if building_control_provider is not None else None
    )

    operator_action_gateway = LiveOperatorActionGateway(
        voice_controller=voice_evacuation_controller, control_controller=building_control_controller,
    )

    # =====================================================
    # Live Perception -> BuildingState Integration Bridge milestone,
    # Phase 2/3/6 -- exactly ONE SensorFusionEngine for this live
    # session (sensor_fusion_engine, defaulted above), fed by whichever
    # production observation providers this deployment actually has
    # evidence for. LiveOccupantObservationProvider is always present
    # (live_occupant_manager itself always exists, even with zero
    # occupants yet -- an honestly empty provider, never an error);
    # the detector-reading and FACP providers are added ONLY when a
    # caller actually supplied a real reading source/FACP (Phase 4's
    # own "never fabricate a source" discipline extended one layer
    # up -- no reading provider configured means no SMOKE/HEAT
    # Observations at all, not a fabricated non-alarming one).
    # =====================================================

    perception_providers = [LiveOccupantObservationProvider(live_occupant_manager)]

    if smoke_detector_reading_provider is not None:
        perception_providers.append(
            LiveSmokeObservationProvider(sensor_manager, reading_provider=smoke_detector_reading_provider)
        )

    if heat_detector_reading_provider is not None:
        perception_providers.append(
            LiveHeatObservationProvider(sensor_manager, reading_provider=heat_detector_reading_provider)
        )

    if facp is not None:
        perception_providers.append(
            LiveFACPObservationProvider(sensor_manager, snapshot_provider=facp_snapshot_provider)
        )

    perception_fusion_coordinator = LivePerceptionFusionCoordinator(
        providers=perception_providers, engine=sensor_fusion_engine,
    )

    # =====================================================
    # BuildingState assembly -- reuses EstimatorBuildingStateGateway
    # verbatim (Canonical Live BuildingState Runtime Assembly milestone),
    # never reimplemented. Every provider left unconfigured resolves to
    # that gateway's own already-established honest empty default.
    #
    # hazard_snapshot_provider is wired for the FIRST time this
    # milestone -- production BuildingState.hazard_summary was always
    # empty before (see this function's own Phase 1 investigation
    # comment above). occupancy_snapshot_provider prefers a caller-
    # supplied value (backward compatible with every existing caller
    # that already wires its own, e.g. a Ground-Truth-backed one) and
    # falls back to the SensorFusionEngine-derived one otherwise --
    # never silently overriding an explicit caller choice.
    # =====================================================

    building_state_gateway = EstimatorBuildingStateGateway(
        camera_status_provider=lambda time: camera_manager.all_statuses(),
        fusion_result_provider=fusion_result_provider,
        facp_snapshot_provider=facp_snapshot_provider,
        hazard_snapshot_provider=perception_fusion_coordinator.hazard_snapshot_provider,
        occupancy_snapshot_provider=(
            occupancy_snapshot_provider if occupancy_snapshot_provider is not None
            else perception_fusion_coordinator.occupancy_snapshot_provider
        ),
        smoke_detector_status_provider=smoke_detector_status_provider,
        smoke_detector_reading_provider=smoke_detector_reading_provider,
        heat_detector_status_provider=heat_detector_status_provider,
        heat_detector_reading_provider=heat_detector_reading_provider,
        control_snapshot_provider=(
            (lambda time: building_control_controller.snapshot())
            if building_control_controller is not None else None
        ),
    )

    event_bus = event_bus if event_bus is not None else EventBus()

    orchestrator = LiveOrchestrator(
        event_bus=event_bus,
        building_state_gateway=building_state_gateway,
        crowd_intelligence_gateway=EngineCrowdIntelligenceGateway(crowd_intelligence_engine),
        live_ai_gateway=live_ai_gateway,
        live_advisory_gateway=live_advisory_gateway,
        interval_seconds=interval_seconds,
    )

    # =====================================================
    # Command Center connection (Phase 6) -- reads StateManager.current()
    # only; never constructs BuildingStateEstimator/LiveAIInferenceService/
    # AdvisoryOrchestrator/CameraManager/SensorManager/FACP itself
    # (mechanically enforced -- tests/test_live_command_center.py::
    # CommandCenterLiveIntegrationGuardTests, unmodified by this
    # milestone).
    # =====================================================

    command_center_data_source = LiveCommandCenterDataSource(
        orchestrator.state_manager, building=building, event_bus=event_bus,
        recent_events_limit=recent_events_limit,
    )

    return LiveRuntime(
        building=building,
        camera_manager=camera_manager,
        sensor_manager=sensor_manager,
        fusion_engine=fusion_engine,
        facp=facp,
        speaker_manager=speaker_manager,
        frame_sources=frame_sources,
        camera_pipeline=camera_pipeline,
        orchestrator=orchestrator,
        command_center_data_source=command_center_data_source,
        operator_action_gateway=operator_action_gateway,
        voice_evacuation_controller=voice_evacuation_controller,
        building_control_controller=building_control_controller,
        live_occupant_manager=live_occupant_manager,
        sensor_fusion_engine=sensor_fusion_engine,
        perception_fusion_coordinator=perception_fusion_coordinator,
        crowd_intelligence_engine=crowd_intelligence_engine,
    )


# =====================================================


def build_offline_demo_runtime(
    building,
    *,
    frame_sources: Optional[Mapping[str, object]] = None,
    human_detector: Optional[object] = None,
    identity_resolver: Optional[object] = None,
    action_executor: Optional[object] = None,
    **kwargs,
) -> LiveRuntime:

    # Phase 3's named "OFFLINE LIVE DEMO MODE" entry point -- the exact
    # same build_live_runtime() above, with sensible offline-only
    # defaults for the two output providers so a caller demonstrating
    # the full chain does not need to import voice_evacuation.provider/
    # building_control.providers itself. Requires zero physical CCTV,
    # zero network access, zero hardware -- frame_sources is expected to
    # be ReplayFrameSource instances (or left empty for a no-camera
    # demo), and both providers are the existing Simulation-only ones.
    # `action_executor` is passed straight through to
    # SimulationControlProvider exactly as building_control.providers
    # already documents (None is the honest "no running simulation to
    # mutate" default -- Door/Exit control confirms=False, never a
    # fabricated CONFIRMED).

    kwargs.setdefault("voice_output_provider", SimulationVoiceOutputProvider())
    kwargs.setdefault("building_control_provider", SimulationControlProvider(building, action_executor))

    return build_live_runtime(
        building, frame_sources=frame_sources, human_detector=human_detector,
        identity_resolver=identity_resolver, **kwargs,
    )
