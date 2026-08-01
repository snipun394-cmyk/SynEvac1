from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from replay_studio.session import discover_scenario_ids, resolve_scenario_artifacts


# =====================================================
# OpenScenarioDialog -- Simulation Replay Studio V1's "User selects:
# Scenario #4832" requirement. Replaces MainWindow.load_incident_dialog()'s
# own six-separate-file-pickers flow with a two-step one (pick a campaign
# output directory, pick a scenario id already discovered inside it) for
# exactly this milestone's own on-disk layout -- load_incident_dialog()
# itself is untouched, still available for a caller pointing at files
# that don't follow this layout at all.
# =====================================================


class OpenScenarioDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Open Scenario")
        self.resize(480, 420)

        self._output_dir = None
        self.selected_scenario_id = None

        layout = QVBoxLayout(self)

        dir_row = QHBoxLayout()
        self.dir_label = QLabel("No campaign output directory selected.")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._on_browse)
        dir_row.addWidget(self.dir_label, 1)
        dir_row.addWidget(self.browse_button)
        layout.addLayout(dir_row)

        layout.addWidget(QLabel("Scenarios found:"))
        self.scenario_list = QListWidget()
        self.scenario_list.itemDoubleClicked.connect(self._on_accept)
        layout.addWidget(self.scenario_list, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self._on_accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.open_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

    # =====================================================

    def _on_browse(self):

        directory = QFileDialog.getExistingDirectory(self, "Campaign Output Directory")
        if not directory:
            return

        self._output_dir = directory
        self.dir_label.setText(directory)

        self.scenario_list.clear()
        scenario_ids = discover_scenario_ids(directory)

        if not scenario_ids:
            QMessageBox.information(self, "Open Scenario", "No scenarios found in this directory.")

        self.scenario_list.addItems(scenario_ids)

    # =====================================================

    def _on_accept(self):

        item = self.scenario_list.currentItem()

        if self._output_dir is None or item is None:
            QMessageBox.warning(
                self, "Open Scenario", "Select a campaign output directory and a scenario.",
            )
            return

        self.selected_scenario_id = item.text()
        self.accept()

    # =====================================================

    def resolved_artifacts(self):

        if self._output_dir is None or self.selected_scenario_id is None:
            return None

        return resolve_scenario_artifacts(self._output_dir, self.selected_scenario_id)
