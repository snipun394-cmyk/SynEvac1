from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMainWindow, QMessageBox

from command_center.building_view import OVERLAY_MODES
from command_center.dashboard import Dashboard
from command_center.data_source import CommandCenterMode
from command_center.incident_data import load_incident
from command_center.theme import COMMAND_CENTER_STYLESHEET

LIVE_REFRESH_INTERVAL_MS = 1000

# Live CCTV Dashboard Refresh Rate milestone -- a SECOND, independent,
# dedicated timer, faster than live_refresh_timer above, but scoped to
# ONLY the camera grid's video display. Deliberately not a change to
# LIVE_REFRESH_INTERVAL_MS itself -- every other Live panel (BuildingState,
# AI, evacuation progress, etc.) still only has genuinely new data once
# per ~1Hz run_cycle(), so re-rendering them at 10Hz would just repaint
# identical state ~9 times out of 10 for no benefit. Camera video is the
# one panel with materially fresher data available in between full
# orchestrator cycles now that LiveCameraPipeline.acquire_frames()
# (live_camera_pipeline package) is polled by Designer's own tick
# timer at the same ~10Hz cadence.
CAMERA_REFRESH_INTERVAL_MS = 100


# =====================================================
# MainWindow -- the Command Center's top-level window. Owns the one
# playback clock (a QTimer), same convention designer/windows/
# main_window.py's Manual Simulation Sandbox already establishes:
# Dashboard/TimelinePanel/BuildingView stay unaware of wall-clock time
# entirely, they only ever receive a frame index to display.
#
# This is a visualization layer only -- it never runs a simulation,
# never generates a scenario, and never recomputes GroundTruth/
# DecisionPolicy. Loading an incident only ever reads already-written
# files (a saved .syn Building project, a stored Scenario, and
# optional GroundTruth/DecisionPolicy/Timeline Dataset JSON a prior
# pipeline run already produced).
# =====================================================


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SynEvac Command Center")
        self.resize(1600, 900)
        self.setStyleSheet(COMMAND_CENTER_STYLESHEET)

        self.dashboard = Dashboard()
        self.setCentralWidget(self.dashboard)

        self._frame_accumulator = 0.0

        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(200)
        self.playback_timer.timeout.connect(self._on_playback_tick)

        # Live Command Center Integration milestone -- a SECOND,
        # independent QTimer, mutually exclusive with playback_timer
        # (Phase 13's own explicit "approximately 1 Hz" target,
        # matching LiveOrchestrator's own default cycle interval --
        # see live_system.orchestrator.LiveOrchestrator's own
        # interval_seconds=1.0 default). Only ever reads an already-
        # computed CommandCenterSnapshot (self.live_data_source.
        # current_snapshot()) -- never runs perception/fusion/AI
        # inference/advisory generation on the GUI thread.
        self.live_data_source = None
        self.live_refresh_timer = QTimer(self)
        self.live_refresh_timer.setInterval(LIVE_REFRESH_INTERVAL_MS)
        self.live_refresh_timer.timeout.connect(self._on_live_refresh_tick)

        # Live CCTV Dashboard Refresh Rate milestone -- see
        # CAMERA_REFRESH_INTERVAL_MS above. Reads ONLY the already-
        # public camera_gateway reference Dashboard already holds
        # (set_camera_gateway()) -- never touches live_data_source,
        # never runs perception/fusion/AI inference, same "display-only"
        # discipline live_refresh_timer itself already follows, just at
        # a faster cadence for this one panel.
        self.camera_refresh_timer = QTimer(self)
        self.camera_refresh_timer.setInterval(CAMERA_REFRESH_INTERVAL_MS)
        self.camera_refresh_timer.timeout.connect(self._on_camera_refresh_tick)

        self._create_actions()
        self._create_menu()
        self._connect_signals()

    # =====================================================

    def _create_actions(self):

        self.load_incident_action = QAction("Load Incident...", self)
        self.load_incident_action.triggered.connect(self.load_incident_dialog)

        # Simulation Replay Studio V1 -- additive, sits alongside
        # load_incident_dialog() rather than replacing it (see
        # open_scenario_dialog()'s own docstring).
        self.open_scenario_action = QAction("Open Scenario...", self)
        self.open_scenario_action.triggered.connect(self.open_scenario_dialog)

        self.overlay_action_group = QActionGroup(self)
        self.overlay_action_group.setExclusive(True)

        self.overlay_actions = {}
        for mode in OVERLAY_MODES:

            action = QAction(mode, self)
            action.setCheckable(True)
            action.triggered.connect(lambda _checked, m=mode: self.dashboard.set_overlay_mode(m))
            self.overlay_action_group.addAction(action)
            self.overlay_actions[mode] = action

        self.overlay_actions[OVERLAY_MODES[0]].setChecked(True)

        # Live Command Center Integration milestone -- Phase 5's mode
        # toggle, same QActionGroup/checkable-action convention the
        # Overlay submenu above already establishes. Switching mode
        # never constructs a second application/window -- both actions
        # target this same MainWindow/Dashboard instance.
        self.mode_action_group = QActionGroup(self)
        self.mode_action_group.setExclusive(True)

        self.replay_mode_action = QAction("Replay", self)
        self.replay_mode_action.setCheckable(True)
        self.replay_mode_action.setChecked(True)
        self.replay_mode_action.triggered.connect(self.switch_to_replay_mode)
        self.mode_action_group.addAction(self.replay_mode_action)

        self.live_mode_action = QAction("Live", self)
        self.live_mode_action.setCheckable(True)
        self.live_mode_action.setEnabled(False)
        self.live_mode_action.triggered.connect(self._on_live_mode_action_triggered)
        self.mode_action_group.addAction(self.live_mode_action)

    # =====================================================

    def _create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.open_scenario_action)
        file_menu.addAction(self.load_incident_action)

        view_menu = menubar.addMenu("View")
        overlay_menu = view_menu.addMenu("Overlay")
        for action in self.overlay_actions.values():
            overlay_menu.addAction(action)

        mode_menu = view_menu.addMenu("Mode")
        mode_menu.addAction(self.replay_mode_action)
        mode_menu.addAction(self.live_mode_action)

    # =====================================================

    def _connect_signals(self):

        self.dashboard.timeline_panel.play_button.clicked.connect(self.play_playback)
        self.dashboard.timeline_panel.pause_button.clicked.connect(self.pause_playback)
        self.dashboard.timeline_panel.reset_button.clicked.connect(self.reset_playback)

    # =====================================================
    # Loading an incident
    # =====================================================

    def load_incident_dialog(self):

        project_path, _ = QFileDialog.getOpenFileName(
            self, "Load Incident -- Building Project", "", "SynEvac Project (*.syn)",
        )
        if not project_path:
            return

        scenario_storage_root = QFileDialog.getExistingDirectory(self, "Scenario Storage Root")
        if not scenario_storage_root:
            return

        scenario_id, ok = QInputDialog.getText(self, "Scenario", "Scenario ID:")
        if not ok or not scenario_id:
            return

        ground_truth_path, _ = QFileDialog.getOpenFileName(
            self, "Ground Truth JSON (optional -- Cancel to skip)", "", "JSON (*.json)",
        )
        decision_policy_path, _ = QFileDialog.getOpenFileName(
            self, "Decision Policy JSON (optional -- Cancel to skip)", "", "JSON (*.json)",
        )
        timeline_rows_path, _ = QFileDialog.getOpenFileName(
            self, "Timeline Dataset Rows JSON (optional -- Cancel to skip)", "", "JSON (*.json)",
        )

        try:

            incident_data = load_incident(
                project_path=project_path,
                scenario_id=scenario_id,
                scenario_storage_root=scenario_storage_root,
                ground_truth_path=ground_truth_path or None,
                decision_policy_path=decision_policy_path or None,
                timeline_rows_path=timeline_rows_path or None,
            )

        except Exception as error:

            QMessageBox.critical(self, "Load Incident", f"Failed to load incident:\n{error}")
            return

        self.load_incident_data(incident_data)

    # =====================================================
    # Simulation Replay Studio V1 -- "open a generated scenario by id"
    # against a campaign output directory (designer/campaign/
    # campaign_worker.py / research_framework/runner.py's own shared
    # layout), rather than six separate raw file pickers. Deferred
    # import: replay_studio already depends on command_center (it
    # reuses this MainWindow/Dashboard as a library), so importing it
    # back at module load time here would be circular -- this method-
    # local import is the same "avoid a real circular import" pattern
    # already used elsewhere in this codebase (e.g.
    # human_decision_engine.groups's own local simulator.occupant import).
    # =====================================================

    def open_scenario_dialog(self):

        from replay_studio.open_scenario_dialog import OpenScenarioDialog

        dialog = OpenScenarioDialog(self)

        if dialog.exec() != OpenScenarioDialog.DialogCode.Accepted:
            return

        artifacts = dialog.resolved_artifacts()
        if artifacts is None:
            return

        try:

            incident_data = load_incident(**artifacts)

        except Exception as error:

            QMessageBox.critical(self, "Open Scenario", f"Failed to load scenario:\n{error}")
            return

        self.load_incident_data(incident_data)

    # =====================================================

    def load_incident_data(self, incident_data):

        # The one non-dialog entry point -- a caller that already has
        # an in-memory IncidentData (e.g. an embedding script that just
        # finished running the pipeline) can hand it straight to the
        # Command Center without going through the file dialogs above.

        self.pause_playback()

        self._frame_accumulator = 0.0

        self.dashboard.set_incident(incident_data)

    # =====================================================
    # Playback -- MainWindow's one clock. Dashboard/TimelinePanel/
    # BuildingView never see wall-clock time, only the frame index
    # this loop advances them to.
    # =====================================================

    def play_playback(self):

        self.playback_timer.start()

    # =====================================================

    def pause_playback(self):

        self.playback_timer.stop()

    # =====================================================

    def reset_playback(self):

        self.pause_playback()

        self._frame_accumulator = 0.0

        self.dashboard.set_frame_index(0)

    # =====================================================

    def _on_playback_tick(self):

        if self.dashboard.frame_count == 0:
            self.pause_playback()
            return

        self._frame_accumulator += self.dashboard.timeline_panel.speed_multiplier()

        while self._frame_accumulator >= 1.0:

            self._frame_accumulator -= 1.0

            next_index = self.dashboard.frame_index + 1

            if next_index >= self.dashboard.frame_count:
                self.pause_playback()
                return

            self.dashboard.set_frame_index(next_index)

    # =====================================================
    # Live Command Center Integration milestone -- Phase 5/13. Mutually
    # exclusive with playback: entering Live mode pauses (never
    # destroys) the Replay QTimer/state, so switching back to Replay
    # (Phase 17's "Replay mode after Live mode" / "Live mode after
    # Replay mode" scenarios) always resumes exactly where the operator
    # left off, never re-loading or losing the loaded IncidentData.
    # =====================================================

    def enable_live_mode(self, data_source, operator_action_gateway=None, camera_gateway=None) -> None:

        # The one call a caller that has already constructed and
        # started a LiveOrchestrator (with its own StateManager) makes
        # to hand this MainWindow a live_system.live_command_center_
        # gateway.LiveCommandCenterDataSource. MainWindow never
        # constructs one itself -- it has no knowledge of
        # LiveOrchestrator/StateManager/BuildingStateGateway at all
        # (mechanically enforced -- see
        # tests/test_live_command_center.py::CommandCenterLiveIntegrationGuardTests).
        #
        # operator_action_gateway (Live Operator Action Routing milestone)
        # is the same kind of opaque, caller-constructed object --
        # MainWindow never constructs or imports a
        # command_center.live_operator_action_gateway.LiveOperatorActionGateway
        # itself, only forwards whatever it was handed straight into
        # Dashboard.set_operator_action_gateway(). None (the default)
        # keeps every Live panel's honest NO_PROVIDER fallback, the same
        # as never calling this at all.
        #
        # camera_gateway (Live CCTV Dashboard milestone) is the same
        # kind of opaque, caller-constructed object one collaborator
        # over -- MainWindow never constructs or imports a
        # command_center.live_camera_view_gateway.LiveCameraViewGateway
        # itself either, only forwards it into Dashboard.
        # set_camera_gateway(). None (the default) keeps the Live CCTV
        # tab's own honest "no live camera session active" empty state.

        self.live_data_source = data_source
        self.live_mode_action.setEnabled(True)

        self.dashboard.set_operator_action_gateway(operator_action_gateway)
        self.dashboard.set_camera_gateway(camera_gateway)

        data_source.start()

        self.live_mode_action.setChecked(True)
        self._activate_live_mode()

    # =====================================================

    def _on_live_mode_action_triggered(self) -> None:

        if self.live_data_source is None:

            # No live data source has been supplied yet (enable_live_mode()
            # was never called) -- honestly refuse rather than switch to
            # a mode with nothing to show, and keep the action group's
            # own checked state consistent with what actually happened.
            self.live_mode_action.setChecked(False)
            self.replay_mode_action.setChecked(True)
            QMessageBox.information(
                self, "Live Mode", "No live data source is configured for this Command Center session.",
            )
            return

        self._activate_live_mode()

    # =====================================================

    def _activate_live_mode(self) -> None:

        self.pause_playback()
        self.dashboard.set_mode(CommandCenterMode.LIVE)

        self._on_live_refresh_tick()
        self.live_refresh_timer.start()

        self.dashboard.refresh_camera_grid()
        self.camera_refresh_timer.start()

    # =====================================================

    def _on_camera_refresh_tick(self) -> None:

        self.dashboard.refresh_camera_grid()

    # =====================================================

    def switch_to_replay_mode(self) -> None:

        self.live_refresh_timer.stop()
        self.camera_refresh_timer.stop()

        if self.live_data_source is not None:
            self.live_data_source.stop()

        self.dashboard.set_mode(CommandCenterMode.REPLAY)
        self.replay_mode_action.setChecked(True)

    # =====================================================

    def _on_live_refresh_tick(self) -> None:

        if self.live_data_source is None:
            return

        snapshot = self.live_data_source.current_snapshot()
        self.dashboard.apply_snapshot(snapshot)

    # =====================================================

    def closeEvent(self, event):

        self.pause_playback()
        self.live_refresh_timer.stop()
        self.camera_refresh_timer.stop()

        if self.live_data_source is not None:
            self.live_data_source.stop()

        event.accept()
