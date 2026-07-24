from PyQt6.QtWidgets import QMessageBox

from live_runtime_launcher.session import LiveRuntimeSession


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

    def __init__(self, panel, get_building):

        self.panel = panel
        self.get_building = get_building

        self.session = None

        self.panel.start_button.clicked.connect(self.on_start)
        self.panel.stop_button.clicked.connect(self.on_stop)
        self.panel.open_command_center_button.clicked.connect(self.on_open_command_center)
        self.panel.mode_combo.currentIndexChanged.connect(self.on_mode_changed)

        self._refresh_panel()

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

            self.session = LiveRuntimeSession(self.panel.selected_mode())
            self.session.construct(building)

        if self.session.runtime is not None:
            self.session.start()

        self._refresh_panel()

        if self.session.last_error:
            QMessageBox.warning(self.panel, "Live Runtime", self.session.last_error)

    # =====================================================

    def on_stop(self):

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
