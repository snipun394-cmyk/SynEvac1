from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from speaker_manager.manager import SpeakerManager


class SpeakerManagerPanel(QWidget):

    # Building-wide Speaker Asset administration -- the Designer-facing
    # view onto SpeakerManager (speaker_manager/manager.py), mirroring
    # designer.widgets.camera_manager_panel.CameraManagerPanel's own
    # "dumb widget, MainWindow pushes updates in" convention exactly.
    # The one public entry point is refresh(building), called by
    # MainWindow wherever it already calls self.camera_manager_panel.
    # refresh(...). Makes clear, per row, which zone(s) each speaker
    # covers -- Phase 10's own "visually clear which zones are covered
    # by which speakers" requirement -- via the same Zone(s) column
    # CameraManagerPanel already uses for the identical purpose.
    #
    # Owns its own SpeakerManager instance, rebuilt (via discover_
    # speakers()) on every refresh() -- same "own the manager, never
    # take one injected" convention CameraManagerPanel already
    # establishes. Every interactive edit (enable/disable) mutates the
    # real Speaker object through SpeakerManager and then re-runs
    # refresh() in full.

    def __init__(self):

        super().__init__()

        self.manager = SpeakerManager()

        self._last_building = None

        layout = QVBoxLayout()
        self.setLayout(layout)

        filter_row = QHBoxLayout()

        filter_row.addWidget(QLabel("Floor:"))
        self.floor_filter = QComboBox()
        self.floor_filter.currentIndexChanged.connect(self._on_floor_filter_changed)
        filter_row.addWidget(self.floor_filter)

        filter_row.addWidget(QLabel("Zone:"))
        self.zone_filter = QComboBox()
        self.zone_filter.currentIndexChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self.zone_filter)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._rerun_last)
        filter_row.addWidget(refresh_button)

        layout.addLayout(filter_row)

        self.speaker_table = QTableWidget()
        self.speaker_table.setColumnCount(7)
        self.speaker_table.setHorizontalHeaderLabels(
            ["Name", "ID", "Floor", "Zone(s)", "Active", "Speaker Type", "Health"],
        )
        self.speaker_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.speaker_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.speaker_table.setRowCount(0)
        layout.addWidget(self.speaker_table)

    # =====================================================
    # Public entry point
    # =====================================================

    def refresh(self, building):

        self._last_building = building

        self.manager.discover_speakers(building)

        floors = building.ordered_floors() if building is not None else []

        self._floor_names = {floor.id: floor.name for floor in floors}
        self._zone_names = {
            zone.id: zone.name or zone.id for floor in floors for zone in floor.zones
        }
        self._zones_by_floor = {
            floor.id: [zone.id for zone in floor.zones] for floor in floors
        }

        self._refresh_floor_filter(floors)
        self._refresh_zone_filter()
        self._refresh_table()

    # =====================================================
    # Filters
    # =====================================================

    def _refresh_floor_filter(self, floors):

        current_floor_id = self.floor_filter.currentData()

        self.floor_filter.blockSignals(True)
        self.floor_filter.clear()

        self.floor_filter.addItem("All Floors", None)

        for floor in floors:
            self.floor_filter.addItem(floor.name, floor.id)

        index = self.floor_filter.findData(current_floor_id)
        self.floor_filter.setCurrentIndex(index if index != -1 else 0)

        self.floor_filter.blockSignals(False)

    # =====================================================

    def _refresh_zone_filter(self):

        current_zone_id = self.zone_filter.currentData()

        self.zone_filter.blockSignals(True)
        self.zone_filter.clear()

        self.zone_filter.addItem("All Zones", None)

        selected_floor_id = self.floor_filter.currentData()

        zone_ids = (
            self._zones_by_floor.get(selected_floor_id, [])
            if selected_floor_id is not None
            else list(self._zone_names.keys())
        )

        for zone_id in zone_ids:
            self.zone_filter.addItem(self._zone_names.get(zone_id, zone_id), zone_id)

        index = self.zone_filter.findData(current_zone_id)
        self.zone_filter.setCurrentIndex(index if index != -1 else 0)

        self.zone_filter.blockSignals(False)

    # =====================================================

    def _on_floor_filter_changed(self, _index):

        self._refresh_zone_filter()
        self._refresh_table()

    # =====================================================

    def _on_filter_changed(self, _index):

        self._refresh_table()

    # =====================================================
    # Table
    # =====================================================

    def _refresh_table(self):

        floor_filter_id = self.floor_filter.currentData()
        zone_filter_id = self.zone_filter.currentData()

        speakers = self._filtered_speakers(floor_filter_id, zone_filter_id)

        self.speaker_table.setRowCount(len(speakers))

        for row, speaker in enumerate(speakers):
            self._populate_row(row, speaker)

    # =====================================================

    def _filtered_speakers(self, floor_filter_id, zone_filter_id):

        speakers = self.manager.all_speakers()

        if floor_filter_id is not None:
            speakers = tuple(s for s in speakers if s.floor_id == floor_filter_id)

        if zone_filter_id is not None:
            speakers = tuple(s for s in speakers if zone_filter_id in s.zone_ids)

        return speakers

    # =====================================================

    def _populate_row(self, row, speaker):

        self.speaker_table.setItem(row, 0, QTableWidgetItem(speaker.name))
        self.speaker_table.setItem(row, 1, QTableWidgetItem(speaker.id))
        self.speaker_table.setItem(
            row, 2, QTableWidgetItem(self._floor_names.get(speaker.floor_id, speaker.floor_id)),
        )

        zone_names = ", ".join(
            self._zone_names.get(zone_id, zone_id) for zone_id in speaker.zone_ids
        ) or "(none)"
        self.speaker_table.setItem(row, 3, QTableWidgetItem(zone_names))

        active_checkbox = QCheckBox()
        active_checkbox.setChecked(speaker.active)
        active_checkbox.toggled.connect(
            lambda checked, speaker_id=speaker.id: self._on_active_toggled(speaker_id, checked)
        )
        self.speaker_table.setCellWidget(row, 4, active_checkbox)

        self.speaker_table.setItem(row, 5, QTableWidgetItem(speaker.speaker_type))

        status = self.manager.speaker_status(speaker.id)
        self.speaker_table.setItem(row, 6, QTableWidgetItem(status.health_status))

    # =====================================================
    # Handlers
    # =====================================================

    def _on_active_toggled(self, speaker_id, checked):

        if checked:
            self.manager.enable_speaker(speaker_id)
        else:
            self.manager.disable_speaker(speaker_id)

        self._rerun_last()

    # =====================================================

    def _rerun_last(self):

        if self._last_building is None:
            return

        self.refresh(self._last_building)
