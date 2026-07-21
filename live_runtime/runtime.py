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
        voice_evacuation_controller,
        building_control_controller,
        live_occupant_manager=None,
        sensor_fusion_engine=None,
        perception_fusion_coordinator=None,
        crowd_intelligence_engine=None,
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
