from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QActionGroup
from PyQt6.QtWidgets import QFileDialog, QInputDialog, QMainWindow, QMessageBox

from command_center.building_view import OVERLAY_MODES
from command_center.dashboard import Dashboard
from command_center.incident_data import load_incident
from command_center.theme import COMMAND_CENTER_STYLESHEET


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

        self._create_actions()
        self._create_menu()
        self._connect_signals()

    # =====================================================

    def _create_actions(self):

        self.load_incident_action = QAction("Load Incident...", self)
        self.load_incident_action.triggered.connect(self.load_incident_dialog)

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

    # =====================================================

    def _create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        file_menu.addAction(self.load_incident_action)

        view_menu = menubar.addMenu("View")
        overlay_menu = view_menu.addMenu("Overlay")
        for action in self.overlay_actions.values():
            overlay_menu.addAction(action)

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

    def closeEvent(self, event):

        self.pause_playback()
        event.accept()
