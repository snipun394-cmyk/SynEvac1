from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager

from command_center.main_window import MainWindow as CommandCenterMainWindow

from credential_store.local_file_store import LocalFileCredentialStore

from live_runtime.factory import build_live_runtime, build_offline_demo_runtime

from live_runtime_launcher.modes import ApplicationMode, RuntimeLifecycleState
from live_runtime_launcher.rtsp_camera_sources import build_rtsp_frame_sources
from live_runtime_launcher.human_detector_wiring import build_yolo_human_detector

from tracking.simple_tracker import SimpleSingleCameraTracker


# =====================================================
# Application Live Runtime Launcher milestone -- the ONE place, above
# live_runtime/factory.py itself, that decides WHEN a LiveRuntime gets
# constructed/started/stopped and WHEN a Command Center window gets
# opened against it. Owns exactly one LiveRuntime and, once opened,
# exactly one Command Center MainWindow for its own lifetime -- never a
# second of either (Phase 6's own "no duplicate runtime" requirement).
#
# Composition and lifecycle only, the same discipline live_runtime/
# runtime.py itself already follows one layer down: construct() never
# calls start() (building a session, or constructing its runtime, opens
# no camera/network connection -- CameraManager.discover_cameras() only
# ever reads Building geometry); start()/stop() never reconstruct the
# underlying LiveRuntime, so Command Center always keeps looking at the
# SAME orchestrator/StateManager across a stop()/start() cycle.
# =====================================================


class LiveRuntimeSession:

    def __init__(
        self, mode: ApplicationMode, credential_store=None,
        human_detector_weights_path=None, human_detector_device: str = "cpu",
    ):

        if mode is ApplicationMode.DESIGNER:
            raise ValueError("LiveRuntimeSession is only for ApplicationMode.LIVE/OFFLINE_DEMO.")

        self.mode = mode
        self.runtime = None
        self.state = RuntimeLifecycleState.STOPPED
        self.last_error = None

        # CP PLUS NVR -> SynEvac1 Live Runtime Integration milestone --
        # the SAME CredentialStore a caller (MainWindow.__init__'s own
        # self._credential_store) already resolves saved camera
        # passwords through, reused here rather than duplicated -- see
        # designer/windows/main_window.py. Defaults to a fresh
        # LocalFileCredentialStore() (the same default ~/.synevac/
        # credentials.json path) so every existing caller/test that
        # never passes one stays unaffected.
        self.credential_store = (
            credential_store if credential_store is not None else LocalFileCredentialStore()
        )

        # Camera 1 Live Human-Detection Integration milestone -- an
        # explicit, OPTIONAL opt-in (default None) rather than a
        # hardcoded model path: no caller/test that never supplies one
        # is affected, matching this launcher's own "never fabricate a
        # provider" discipline (ai_status()/provider_capabilities()
        # above). None means exactly what it already means everywhere
        # else in this class -- no real human detector configured, so
        # construct() below never builds one, and camera_pipeline stays
        # None exactly as before this milestone.
        self.human_detector_weights_path = human_detector_weights_path
        self.human_detector_device = human_detector_device

        self._command_center_window = None

    # =====================================================

    @property
    def is_running(self) -> bool:

        return self.runtime is not None and self.runtime.is_running

    # =====================================================

    def construct(self, building) -> None:

        # Failure Isolation milestone requirement (Phase 10) -- a
        # missing/invalid project must degrade the SESSION honestly,
        # never crash the application. An empty Building (zero floors)
        # is not an error at all: build_live_runtime() already handles
        # it (every manager's own discover_*() simply finds nothing),
        # so it is deliberately NOT special-cased here.

        if building is None:
            self.runtime = None
            self.state = RuntimeLifecycleState.FAILED
            self.last_error = "No Building is loaded -- open or create a project first."
            return

        factory = build_live_runtime if self.mode is ApplicationMode.LIVE else build_offline_demo_runtime

        # CP PLUS NVR -> SynEvac1 Live Runtime Integration milestone --
        # closes the composition gap named (but deliberately left open)
        # by the Application Live Runtime Launcher milestone's own
        # "no RTSP-from-Camera-asset builder exists yet" note (see
        # live_runtime_launcher/modes.py). Only ApplicationMode.LIVE
        # ever gets real RTSPFrameSource instances -- OFFLINE_DEMO stays
        # exactly as before (Simulation* providers only, zero network
        # I/O), matching this launcher's own established "LIVE means
        # real hardware, OFFLINE_DEMO means Simulation" split.
        #
        # The CameraManager is built HERE (not left for build_live_
        # runtime() to construct its own) specifically so the SAME
        # instance backs both frame_sources' status_callback bridge
        # (build_rtsp_frame_sources) and self._any_configured_camera_
        # offline()'s later reads -- build_live_runtime()'s own
        # `camera_manager` parameter exists exactly to let a caller
        # reuse an already-built manager instead of a second, duplicate
        # one (factory.py's own Phase 1 item 9 comment).
        factory_kwargs = {}

        if self.mode is ApplicationMode.LIVE:

            camera_manager = CameraManager()
            camera_manager.discover_cameras(building)

            factory_kwargs["camera_manager"] = camera_manager
            factory_kwargs["frame_sources"] = build_rtsp_frame_sources(
                building, camera_manager, self.credential_store,
            )

        try:

            # Camera 1 Live Human-Detection Integration milestone --
            # closes the SECOND composition gap: build_live_runtime()'s
            # own camera_pipeline gate needs frame_sources AND
            # human_detector AND identity_resolver all non-None (see
            # live_runtime/factory.py). Only attempted when there is at
            # least one real configured camera to feed it -- a
            # human_detector_weights_path supplied against a building
            # with no Live-mode camera correctly builds nothing here,
            # same "no cameras" honesty build_live_runtime() itself
            # already establishes. Constructed INSIDE this try block
            # (not before it) so a bad weights path (UltralyticsYOLO
            # Backend's own ModelWeightsNotFoundError) degrades this
            # session to an honest FAILED state -- exactly this
            # method's own Phase 10 "never crash the application"
            # requirement above -- rather than raising uncaught.
            if (
                self.mode is ApplicationMode.LIVE
                and factory_kwargs.get("frame_sources")
                and self.human_detector_weights_path is not None
            ):

                human_detector, identity_resolver = build_yolo_human_detector(
                    self.human_detector_weights_path, device=self.human_detector_device,
                )

                factory_kwargs["human_detector"] = human_detector
                factory_kwargs["identity_resolver"] = identity_resolver

                # Camera 1 Live Detection -> Tracking/Building-State
                # Integration milestone -- closes the THIRD composition
                # gap. Without a tracker, LiveCameraPipeline.run_cycle()
                # feeds identity_resolver the detector's own raw,
                # per-frame-only local_track_id (YOLOHumanDetector's own
                # docstring: "NOT stable across frames"), and its
                # pending_occupant_updates stays all-None (see
                # pipeline.py's own run_cycle(), `else: camera_pending_
                # updates = [None] * len(raw)`) -- meaning live_occupants.
                # manager.LiveOccupantManager.update() is NEVER called at
                # all, even though build_live_runtime() always constructs
                # one. SimpleSingleCameraTracker (tracking/simple_tracker.py)
                # is the EXISTING, already-tested, non-ML production
                # tracker docs/architecture/human_detection.md's own
                # real-world validation combination already paired with
                # this exact detector/identity_resolver pairing -- reused
                # verbatim here, never a second/new tracker. Needs no
                # weights/credentials/machine-specific config of its own,
                # so it is always paired with a real human_detector
                # (never a separate opt-in flag).
                factory_kwargs["tracker"] = SimpleSingleCameraTracker()

            self.runtime = factory(building, **factory_kwargs)

        except Exception as exc:

            self.runtime = None
            self.state = RuntimeLifecycleState.FAILED
            self.last_error = str(exc)
            return

        self.state = RuntimeLifecycleState.STOPPED
        self.last_error = None

    # =====================================================

    def start(self) -> None:

        if self.runtime is None:

            self.state = RuntimeLifecycleState.FAILED
            self.last_error = "No runtime constructed -- call construct(building) first."
            return

        if self.runtime.is_running:
            return  # idempotent, mirrors LiveRuntime.start() itself

        self.state = RuntimeLifecycleState.STARTING

        try:
            self.runtime.start()
        except Exception as exc:

            self.state = RuntimeLifecycleState.FAILED
            self.last_error = str(exc)
            return

        self.state = (
            RuntimeLifecycleState.DEGRADED if self._any_configured_camera_offline()
            else RuntimeLifecycleState.RUNNING
        )
        self.last_error = None

    # =====================================================

    def stop(self) -> None:

        # Always safe to call, including before any start() and
        # including repeatedly (Phase 11) -- LiveRuntime.stop() itself
        # already guarantees this; this method adds only the session's
        # own state bookkeeping on top.

        if self.runtime is not None:
            self.runtime.stop()

        self.state = RuntimeLifecycleState.STOPPED
        self.last_error = None

    # =====================================================

    def open_command_center(self) -> CommandCenterMainWindow:

        # Phase 6's own core requirement, mechanically satisfied by
        # construction, not merely asserted: this method never builds a
        # second LiveRuntime/StateManager -- it hands Command Center
        # THIS session's own runtime.command_center_data_source/
        # operator_action_gateway, the exact same objects
        # runtime.orchestrator.state_manager and every operator-action
        # controller live behind.

        if self.runtime is None:
            raise RuntimeError("No LiveRuntime constructed -- start the Live Runtime first.")

        if self._command_center_window is None:
            self._command_center_window = CommandCenterMainWindow()

        self._command_center_window.enable_live_mode(
            self.runtime.command_center_data_source, self.runtime.operator_action_gateway,
        )

        return self._command_center_window

    # =====================================================

    def ai_status(self) -> str:

        # AI Configuration Honesty milestone requirement (Phase 8) --
        # this launcher never constructs a live_ai_gateway itself (no
        # default-construction helper exists anywhere in this
        # repository -- confirmed by docs/architecture/
        # synevac_end_to_end_architecture_review.md §1/§14), so this is
        # always "NOT CONFIGURED" today; the check itself is honest and
        # forward-compatible with a future caller that does supply one.

        if self.runtime is not None and self.runtime.orchestrator.live_ai_gateway is not None:
            return "CONFIGURED"

        return "NOT CONFIGURED"

    # =====================================================

    def provider_capabilities(self) -> dict:

        # Provider Capability Honesty milestone requirement (Phase 9) --
        # distinguishes NO_PROVIDER from SIMULATION; never labels
        # anything LIVE merely because a controller object exists. No
        # camera/human_detector/identity_resolver is ever supplied by
        # this launcher (building one from a Designer Camera asset would
        # require implementing a physical hardware protocol, explicitly
        # out of scope this milestone), so "camera" is always
        # NO_PROVIDER today; Voice/Dynamic Signage/Building Control are
        # SIMULATION under OFFLINE_DEMO (build_offline_demo_runtime()
        # defaults Simulation* providers in) and NO_PROVIDER under LIVE
        # (no physical provider implementation exists yet, per Phase 9's
        # own "keep that explicit" instruction).

        if self.runtime is None:
            return {
                "camera": "NO_PROVIDER", "voice": "NO_PROVIDER",
                "dynamic_signage": "NO_PROVIDER", "building_control": "NO_PROVIDER",
            }

        return {
            "camera": "LIVE" if self.runtime.frame_sources else "NO_PROVIDER",
            "voice": self._provider_label(self.runtime.voice_evacuation_controller),
            "dynamic_signage": self._provider_label(self.runtime.dynamic_signage_controller),
            "building_control": self._provider_label(self.runtime.building_control_controller),
        }

    # =====================================================

    def shutdown(self) -> None:

        # Application Startup/Shutdown Lifecycle milestone requirement
        # (Phase 11) -- called once, from MainWindow.closeEvent(), so no
        # frame source/orchestrator/Command Center refresh timer survives
        # the Designer window closing (no zombie runtime).

        self.stop()

        if self._command_center_window is not None:
            self._command_center_window.close()
            self._command_center_window = None

    # =====================================================
    # Internals
    # =====================================================

    def _any_configured_camera_offline(self) -> bool:

        if self.runtime is None or not self.runtime.frame_sources:
            return False

        for camera_id in self.runtime.frame_sources:

            if self.runtime.camera_manager.connection_status(camera_id) != CameraConnectionState.ONLINE:
                return True

        return False

    def _provider_label(self, controller) -> str:

        # This launcher only ever supplies a provider under OFFLINE_DEMO
        # (always a Simulation* one, via build_offline_demo_runtime()'s
        # own defaults) -- a controller being non-None here can
        # therefore never mean anything but SIMULATION today. A future
        # caller wiring a real hardware provider into LIVE mode would
        # extend this, not silently mislabel it LIVE now.

        return "NO_PROVIDER" if controller is None else "SIMULATION"
