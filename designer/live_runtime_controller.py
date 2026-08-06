import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox

from live_runtime_launcher.session import LiveRuntimeSession


# Live CCTV Dashboard Refresh Rate milestone -- this timer now drives TWO
# derived cadences from one tick, never two timers: acquire_frames() (frame
# caching only, no AI) every tick, ~10Hz; the full run_cycle() (YOLO/
# tracking/the whole LiveOrchestrator, unchanged cost) every FULL_CYCLE_
# EVERY_N_TICKS'th tick, still ~1Hz overall -- matching LiveOrchestrator's
# own default interval_seconds=1.0 (live_system/orchestrator.py) and
# command_center.main_window.LIVE_REFRESH_INTERVAL_MS exactly as before,
# just phased differently (see _on_tick()). ACQUISITION_INTERVAL_MS is the
# new base cadence this class was previously called LIVE_CYCLE_INTERVAL_MS
# under, before acquisition and the full cycle were decomposed.
ACQUISITION_INTERVAL_MS = 100
FULL_CYCLE_EVERY_N_TICKS = 10


class LiveRuntimeController:

    # Application Live Runtime Launcher milestone -- the mediator
    # between LiveRuntimePanel (raw widgets, no backend knowledge of its
    # own) and LiveRuntimeSession (the one place build_live_runtime()/
    # build_offline_demo_runtime() actually get called), same "controller
    # mediates, window/panel stays dumb" convention designer.campaign.
    # campaign_controller.CampaignController already established for
    # Campaign Studio. Every button the panel exposes is wired here to
    # exactly one already-existing LiveRuntimeSession capability -- this
    # class calls no live_runtime/live_system/command_center API
    # directly, only LiveRuntimeSession's own public methods.
    #
    # Expose Real Live Camera Occupant State In SynEvac UI milestone --
    # this class is ALSO now the one place a running LiveRuntime's own
    # run_cycle() gets driven automatically. Phase 1's own investigation
    # found a genuine, previously-undiscovered gap here: Command Center's
    # own live_refresh_timer (command_center/main_window.py) ONLY ever
    # re-renders whatever LiveCommandCenterDataSource.current_snapshot()
    # already holds -- by its own explicit design ("never runs perception/
    # fusion/AI inference... on the GUI thread") -- and nothing else in the
    # real application ever called run_cycle() at all (only tests/scripts
    # did). Without this, BuildingState/live_occupants/every other Live
    # panel would stay frozen forever in a real Designer session, no
    # matter how well-wired the UI panels themselves are. This tick timer
    # is a THIN driver only -- it calls the existing, unmodified
    # LiveRuntime.run_cycle() exactly once per interval; it performs no
    # tracking/fusion/state logic itself.

    def __init__(self, panel, get_building, credential_store=None, human_detector_weights_path=None):

        self.panel = panel
        self.get_building = get_building
        self.credential_store = credential_store

        # Live CCTV Dashboard milestone -- root-cause fix for "camera
        # shows Online but every frame consumer (Designer's own
        # LiveCameraViewPanel, and now Command Center's Live CCTV tab)
        # shows nothing": on_start() below constructs LiveRuntimeSession
        # WITHOUT this, ever, prior to this fix -- human_detector_
        # weights_path had no path from this class's own __init__ to
        # LiveRuntimeSession at all, so live_runtime.factory.
        # build_live_runtime()'s own gate ("frame_sources and
        # human_detector is not None and identity_resolver is not
        # None") was permanently unsatisfiable through the real
        # application, and camera_pipeline -- the ONLY thing that ever
        # populates latest_frame()/latest_detections() -- was silently
        # None for every real Designer-driven Live session that ever
        # ran, regardless of a real, connected, Online RTSP camera.
        # None (the default) reproduces every existing caller's/test's
        # prior behavior exactly -- only a caller that explicitly opts
        # in (designer/windows/main_window.py, when a real local
        # weights file exists) gets a real camera_pipeline.
        self.human_detector_weights_path = human_detector_weights_path

        self.session = None

        # The Recommendation Layer milestone -- notified once per real
        # run_cycle() tick, so MainWindow can refresh its Recommendation
        # panel from freshly-computed live state. None (the default) is
        # a valid, guarded state, same convention as CameraManagerPanel.
        # on_camera_changed.
        self.on_cycle_callback = None

        # Live CCTV Dashboard Refresh Rate milestone -- counts ticks so
        # _on_tick() below can derive the slower (~1Hz) full-cycle
        # cadence from this single fast (~10Hz) timer, phased so the
        # FIRST tick after Start already runs a full cycle (matching
        # this class's own prior single-cadence behavior exactly,
        # rather than waiting FULL_CYCLE_EVERY_N_TICKS ticks for it).
        self._tick_count = 0

        self._tick_timer = QTimer()
        self._tick_timer.setInterval(ACQUISITION_INTERVAL_MS)
        self._tick_timer.timeout.connect(self._on_tick)

        self.panel.start_button.clicked.connect(self.on_start)
        self.panel.stop_button.clicked.connect(self.on_stop)
        self.panel.open_command_center_button.clicked.connect(self.on_open_command_center)
        self.panel.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self._refresh_panel()

    # =====================================================

    def _on_tick(self) -> None:

        if self.session is None or not self.session.is_running:
            return

        runtime = self.session.runtime
        self._tick_count += 1

        # Live CCTV Dashboard Refresh Rate milestone -- exactly one of
        # these two branches runs per tick, never both: run_cycle()
        # already composes acquire_frames() as its own first step
        # (live_camera_pipeline/pipeline.py), so calling both here
        # would be a genuine second read within the same ~100ms tick,
        # not a shared acquisition. YOLO/tracking/the full orchestrator
        # still runs at the same ~1Hz cadence as before (phased so the
        # very first tick already runs one, matching prior behavior);
        # every other tick only refreshes the frame cache, ~10Hz, at
        # essentially zero cost (no detection, no tracking, no
        # orchestrator stages).
        if self._tick_count % FULL_CYCLE_EVERY_N_TICKS == 1:

            runtime.run_cycle(time.time())

            if self.on_cycle_callback is not None:
                self.on_cycle_callback()

        elif runtime.camera_pipeline is not None:

            runtime.camera_pipeline.acquire_frames(time.time())

    # =====================================================

    def on_mode_changed(self, _index):

        # Switching mode while a STOPPED session from the previous mode
        # is still constructed forces a fresh construct() on the next
        # Start -- never silently reusing an OFFLINE_DEMO session's
        # Simulation providers under a LIVE-mode label, or vice versa.
        # A RUNNING session is left alone; the operator must Stop first
        # (Safe Startup Semantics milestone requirement -- switching a
        # combo box must never itself stop a running runtime).

        if self.session is not None and not self.session.is_running:
            self.session = None

        self._refresh_panel()

    # =====================================================

    def on_start(self):

        building = self.get_building()

        if self.session is None:

            self.session = LiveRuntimeSession(
                self.panel.selected_mode(), credential_store=self.credential_store,
                human_detector_weights_path=self.human_detector_weights_path,
            )
            self.session.construct(building)

        if self.session.runtime is not None:
            self.session.start()

        if self.session.is_running:
            self._tick_count = 0
            self._tick_timer.start()

        self._refresh_panel()

        if self.session.last_error:
            QMessageBox.warning(self.panel, "Live Runtime", self.session.last_error)

    # =====================================================

    def on_stop(self):

        self._tick_timer.stop()

        if self.session is not None:
            self.session.stop()

        self._refresh_panel()

    # =====================================================

    def on_open_command_center(self):

        if self.session is None or self.session.runtime is None:

            QMessageBox.information(
                self.panel, "Live Runtime", "Start the Live Runtime before opening Command Center.",
            )
            return

        window = self.session.open_command_center()

        window.show()
        window.raise_()
        window.activateWindow()

    # =====================================================

    def stop_and_reset(self):

        # Called by MainWindow whenever the loaded project is about to
        # change (new_project()/open_project()) -- a running session
        # points at the Building instance about to be replaced/
        # discarded, the same "stop the loop before the project
        # disappears" discipline MainWindow.stop_simulation() already
        # applies to the Manual Simulation Sandbox.

        self._tick_timer.stop()

        if self.session is not None:
            self.session.shutdown()

        self.session = None
        self._refresh_panel()

    # =====================================================

    def shutdown(self):

        # Called once from MainWindow.closeEvent() -- see stop_and_reset()
        # above for why this also tears down any open Command Center
        # window, not merely the LiveRuntime itself.

        self.stop_and_reset()

    # =====================================================

    def _refresh_panel(self):

        if self.session is None:

            self.panel.set_state(
                "STOPPED", ai_status="NOT CONFIGURED",
                capabilities={
                    "camera": "NO_PROVIDER", "voice": "NO_PROVIDER",
                    "dynamic_signage": "NO_PROVIDER", "building_control": "NO_PROVIDER",
                },
                error=None,
            )
            return

        self.panel.set_state(
            self.session.state.value, ai_status=self.session.ai_status(),
            capabilities=self.session.provider_capabilities(), error=self.session.last_error,
        )
