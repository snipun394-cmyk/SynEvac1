from camera_manager.connection_status import CameraConnectionState

from command_center.main_window import MainWindow as CommandCenterMainWindow

from live_runtime.factory import build_live_runtime, build_offline_demo_runtime

from live_runtime_launcher.modes import ApplicationMode, RuntimeLifecycleState


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

    def __init__(self, mode: ApplicationMode):

        if mode is ApplicationMode.DESIGNER:
            raise ValueError("LiveRuntimeSession is only for ApplicationMode.LIVE/OFFLINE_DEMO.")

        self.mode = mode
        self.runtime = None
        self.state = RuntimeLifecycleState.STOPPED
        self.last_error = None

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

        try:
            self.runtime = factory(building)
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
