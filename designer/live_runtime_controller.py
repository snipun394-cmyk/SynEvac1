import time

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox

from live_runtime_launcher.session import LiveRuntimeSession


# Matches LiveOrchestrator's own default interval_seconds=1.0 (live_system/
# orchestrator.py) and command_center.main_window.LIVE_REFRESH_INTERVAL_MS --
# the existing "~1Hz live cycle" convention this milestone's own tick timer
# reuses rather than inventing a second cadence.
LIVE_CYCLE_INTERVAL_MS = 1000


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

    def __init__(self, panel, get_building, credential_store=None):

        self.panel = panel
        self.get_building = get_building
        self.credential_store = credential_store

        self.session = None

        # The Recommendation Layer milestone -- notified once per real
        # run_cycle() tick, so MainWindow can refresh its Recommendation
        # panel from freshly-computed live state. None (the default) is
        # a valid, guarded state, same convention as CameraManagerPanel.
        # on_camera_changed.
        self.on_cycle_callback = None

        self._tick_timer = QTimer()
        self._tick_timer.setInterval(LIVE_CYCLE_INTERVAL_MS)
        self._tick_timer.timeout.connect(self._on_tick)

        self.panel.start_button.clicked.connect(self.on_start)
        self.panel.stop_button.clicked.connect(self.on_stop)
        self.panel.open_command_center_button.clicked.connect(self.on_open_command_center)
        self.panel.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self._refresh_panel()

    # =====================================================

    def _on_tick(self) -> None:

        if self.session is not None and self.session.is_running:

            self.session.runtime.run_cycle(time.time())

            if self.on_cycle_callback is not None:
                self.on_cycle_callback()

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

            self.session = LiveRuntimeSession(self.panel.selected_mode(), credential_store=self.credential_store)
            self.session.construct(building)

        if self.session.runtime is not None:
            self.session.start()

        if self.session.is_running:
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
