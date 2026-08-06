from typing import Dict, Mapping, Optional


# =====================================================
# Production Live Runtime Composition Root milestone.
#
# Phase 1's own investigation (docs/architecture/live_runtime_
# composition.md) confirmed there is no production entry point
# anywhere in this repository that constructs and owns a coherent live
# SynEvac runtime -- `main.py` -> `core.app.SynEvacApp` ->
# `designer.windows.main_window.MainWindow` only ever launches the
# Designer, and every existing `CameraManager`/`SensorManager`/
# `MultiCameraFusionEngine`/`SimulatedFACP`/`LiveOrchestrator`/
# `LiveOperatorActionGateway` assembly found anywhere in the codebase
# lives inside a test module (`tests/test_live_command_center.py::
# _LiveChain` is the closest existing precedent this class formalizes).
#
# LiveRuntime is deliberately NOT a new subsystem. It owns REFERENCES
# to already-existing, already-tested components (composition and
# lifecycle management only, per this milestone's own Phase 2
# instruction) -- it contains no camera-management, sensor-management,
# fusion, FACP, AI, advisory, voice, or building-control LOGIC of its
# own. Every attribute below is exactly the object a caller (this
# package's own factory.py, or a future test/deployment composing one
# by hand) already constructed; LiveRuntime's only original code is
# start()/stop()/is_running/run_cycle().
# =====================================================


class LiveRuntime:

    def __init__(
        self,
        *,
        building,
        camera_manager,
        sensor_manager,
        fusion_engine,
        facp,
        speaker_manager,
        frame_sources: Mapping[str, object],
        camera_pipeline,
        orchestrator,
        command_center_data_source,
        operator_action_gateway,
        camera_view_gateway=None,
        voice_evacuation_controller,
        building_control_controller,
        sign_manager=None,
        dynamic_signage_controller=None,
        emergency_light_manager=None,
        fire_safety_asset_manager=None,
        fire_water_infrastructure_manager=None,
        live_occupant_manager=None,
        sensor_fusion_engine=None,
        perception_fusion_coordinator=None,
        crowd_intelligence_engine=None,
        evacuation_progress_engine=None,
        trajectory_intelligence_engine=None,
        emergency_response_engine=None,
        evacuation_recommendation_engine=None,
        evacuation_guidance_engine=None,
        recommendation_layer=None,
        warden_notification_controller=None,
        execution_layer=None,
        dynamic_signage_planner=None,
    ):

        # Digital Twin / asset-management layer -- Phase 4's shared-
        # instance requirement: exactly one of each, threaded into
        # every closure/gateway below that needs it. Never reconstructed
        # or duplicated anywhere else in this class.
        self.building = building
        self.camera_manager = camera_manager
        self.sensor_manager = sensor_manager
        self.fusion_engine = fusion_engine
        self.facp = facp
        self.speaker_manager = speaker_manager

        # Live Dynamic Evacuation Signage milestone -- exactly one
        # SignManager/DynamicSignageController for this live session,
        # mirroring speaker_manager/voice_evacuation_controller's own
        # "shared instance, never duplicated" role. Both None whenever
        # this deployment/test does not wire signage at all -- never
        # fabricated.
        self.sign_manager = sign_manager
        self.dynamic_signage_controller = dynamic_signage_controller

        # Manual Call Points & Emergency Lighting milestone -- exactly
        # one EmergencyLightManager for this live session, same shared-
        # instance discipline as every other manager above. Manual Call
        # Points need no new manager of their own -- see sensor_manager.
        # manager.SensorManager.discover_sensors(), which now discovers
        # them alongside Smoke/Heat Detector.
        self.emergency_light_manager = emergency_light_manager

        # Fire Suppression & Water-Based Safety Asset Digital Twin
        # milestone -- exactly one FireSafetyAssetManager for this live
        # session, same shared-instance discipline as every other
        # manager above.
        self.fire_safety_asset_manager = fire_safety_asset_manager

        # Fire Water Supply & Suppression Infrastructure milestone --
        # exactly one FireWaterInfrastructureManager for this live
        # session, same shared-instance discipline as every other
        # manager above.
        self.fire_water_infrastructure_manager = fire_water_infrastructure_manager

        # Camera ingestion -- camera_id -> CameraFrameSource (Replay
        # today; RTSPFrameSource, unchanged, once physical CCTV access
        # exists -- see docs/architecture/live_runtime_composition.md
        # §C). None of these are started merely by being present here;
        # see start() below. camera_pipeline is None whenever no frame
        # sources were configured at all (Phase 8 item 1, "no cameras").
        self.frame_sources: Dict[str, object] = dict(frame_sources)
        self.camera_pipeline = camera_pipeline

        # The live cycle itself -- LiveRuntime never reimplements
        # sequencing; run_cycle() below is a pure forward.
        self.orchestrator = orchestrator

        # Presentation / operator-action seams -- the exact same
        # instances a caller hands to command_center.main_window.
        # MainWindow.enable_live_mode(data_source, operator_action_gateway).
        self.command_center_data_source = command_center_data_source
        self.operator_action_gateway = operator_action_gateway

        # Live CCTV Dashboard milestone -- the same "exact same instance
        # a caller hands to command_center.main_window.MainWindow.
        # enable_live_mode()" discipline as operator_action_gateway
        # immediately above. None whenever no CameraManager was
        # supplied to build_live_runtime() (Phase 1's own "None whenever
        # no provider was supplied" convention).
        self.camera_view_gateway = camera_view_gateway
        self.voice_evacuation_controller = voice_evacuation_controller
        self.building_control_controller = building_control_controller

        # Live Perception -> BuildingState Integration Bridge milestone
        # -- exactly one LiveOccupantManager/SensorFusionEngine for this
        # live session (Phase 5/6's own "not one per camera, not a
        # hidden second engine" requirement), plus the coordinator that
        # composes them with this session's own production observation
        # providers. Deliberately untyped, like every other attribute on
        # this class (Phase 2's "composition and lifecycle management
        # only" -- this file imports no concrete collaborator class at
        # all, mechanically enforced by tests/
        # test_live_runtime_architecture_guards.py::
        # GatewayIsTheOnlyExecutionSeamTests::
        # test_live_runtime_container_holds_no_concrete_class_imports).
        self.live_occupant_manager = live_occupant_manager
        self.sensor_fusion_engine = sensor_fusion_engine
        self.perception_fusion_coordinator = perception_fusion_coordinator

        # Live Occupancy, Crowd Density & Congestion Intelligence
        # milestone -- exactly ONE CrowdIntelligenceEngine for this live
        # session (Phase 10's own "wire exactly ONE shared
        # CrowdIntelligenceEngine" requirement), same untyped-attribute
        # discipline as every other collaborator on this class.
        self.crowd_intelligence_engine = crowd_intelligence_engine

        # Live Evacuation Progress, Flow & Clearance Intelligence
        # milestone -- exactly ONE EvacuationProgressEngine for this
        # live session (Phase 11's own "wire exactly ONE shared"
        # requirement), same untyped-attribute discipline as every
        # other collaborator on this class.
        self.evacuation_progress_engine = evacuation_progress_engine

        # Live Occupant Trajectory, Movement Anomaly & Route-Deviation
        # Intelligence milestone -- exactly ONE TrajectoryIntelligenceEngine
        # for this live session, same untyped-attribute discipline as
        # every other collaborator on this class.
        self.trajectory_intelligence_engine = trajectory_intelligence_engine

        # Live Emergency Response & Rescue Priority Intelligence
        # milestone -- exactly ONE EmergencyResponseIntelligenceEngine
        # for this live session, same untyped-attribute discipline as
        # every other collaborator on this class.
        self.emergency_response_engine = emergency_response_engine

        # Live Dynamic Evacuation Recommendation Engine milestone --
        # exactly ONE EvacuationRecommendationEngine for this live
        # session, same untyped-attribute discipline as every other
        # collaborator on this class.
        self.evacuation_recommendation_engine = evacuation_recommendation_engine

        # Live Evacuation Guidance & Zoned Message Planning milestone --
        # exactly ONE EvacuationGuidanceEngine for this live session,
        # same untyped-attribute discipline as every other collaborator
        # on this class.
        self.evacuation_guidance_engine = evacuation_guidance_engine

        # The Recommendation Layer milestone -- exactly ONE
        # RecommendationLayer for this live session, same untyped-
        # attribute discipline as every other collaborator on this
        # class.
        self.recommendation_layer = recommendation_layer

        # The Execution Layer V1 milestone -- exactly ONE
        # WardenNotificationController (the one genuinely new execution
        # controller this milestone adds, same NO_PROVIDER-under-LIVE
        # discipline as voice_evacuation_controller/building_control_
        # controller above) and exactly ONE ExecutionLayer (an
        # orchestration/coordinating layer over all four controllers,
        # never a replacement execution authority -- see execution_
        # layer/layer.py's own docstring), same untyped-attribute
        # discipline as every other collaborator on this class.
        self.warden_notification_controller = warden_notification_controller
        self.execution_layer = execution_layer

        # Live Dynamic Evacuation Signage milestone -- exactly ONE
        # DynamicSignagePlanner for this live session, same untyped-
        # attribute discipline as every other collaborator on this class.
        self.dynamic_signage_planner = dynamic_signage_planner

        self._running = False

    # =====================================================

    @property
    def is_running(self) -> bool:

        return self._running

    # =====================================================

    def start(self) -> None:

        # Idempotent -- calling start() on an already-running runtime is
        # a safe no-op (Phase 5's own explicit requirement), even though
        # the underlying LiveOrchestrator.start() itself raises
        # LiveSystemAlreadyRunningError on a double call. LiveRuntime's
        # own _running flag is the one place that double-start guard
        # lives; LiveOrchestrator itself is never modified.

        if self._running:
            return

        started_camera_ids = []

        try:

            for camera_id, source in self.frame_sources.items():

                # One failed camera must never crash the entire runtime
                # (Phase 5) -- RTSPFrameSource/ReplayFrameSource already
                # convert every connection failure into an honest status
                # internally and never raise from start(); this is
                # defense in depth for any caller-supplied
                # CameraFrameSource that does not hold to that contract.
                try:
                    source.start()
                    started_camera_ids.append(camera_id)
                except Exception:
                    pass

            # Core infrastructure -- if either of these two raises
            # unexpectedly, roll back whatever camera sources were just
            # started rather than leaving a half-started runtime (Phase
            # 5: "partial startup failure does not leave half the
            # system running").
            self.orchestrator.start()

            if self.command_center_data_source is not None:
                self.command_center_data_source.start()

        except Exception:

            for camera_id in started_camera_ids:
                try:
                    self.frame_sources[camera_id].stop()
                except Exception:
                    pass

            if self.orchestrator.is_running:
                self.orchestrator.stop()

            raise

        self._running = True

    # =====================================================

    def stop(self) -> None:

        # Always safe to call, including repeatedly and including on a
        # runtime that never fully started (Phase 5) -- every
        # sub-component's own stop() is already independently safe to
        # call more than once (LiveOrchestrator.stop(), CameraFrameSource.
        # stop(), LiveCommandCenterDataSource.stop() all simply set a
        # flag), and each is additionally wrapped here so one component
        # failing to stop cleanly never prevents the others from
        # stopping (Phase 5/8: "component fails during shutdown").

        for source in self.frame_sources.values():

            try:
                source.stop()
            except Exception:
                pass

        try:
            self.orchestrator.stop()
        except Exception:
            pass

        if self.command_center_data_source is not None:

            try:
                self.command_center_data_source.stop()
            except Exception:
                pass

        self._running = False

    # =====================================================

    def run_cycle(self, time: float):

        # A pure forward -- every stage's own sequencing, gating, and
        # failure handling (a missing/failed AI model never blocking
        # BuildingState assembly, a missing advisory gateway never
        # blocking AI inference, etc.) is already LiveOrchestrator's own
        # existing, tested behavior; LiveRuntime adds no logic of its
        # own here.

        return self.orchestrator.run_cycle(time)

    # =====================================================

    def tick_execution_layer(self, time: float):

        # The Execution Layer V1 milestone -- deliberately a SEPARATE
        # method from run_cycle() above, never folded into it.
        # LiveOrchestrator is mechanically forbidden from importing
        # voice_evacuation/building_control.controller/dynamic_signage.
        # controller (tests/test_live_runtime_architecture_guards.py::
        # LiveOrchestratorCannotDirectlyCallControllersTests) -- those
        # controllers live here, on LiveRuntime, one layer above the
        # orchestrator, so ExecutionLayer (which reads them) cannot be
        # ticked from inside run_cycle() without violating that guard.
        # Returns None (never raises) if no execution_layer was
        # constructed for this session.

        if self.execution_layer is None:
            return None

        return self.execution_layer.compute(time)
