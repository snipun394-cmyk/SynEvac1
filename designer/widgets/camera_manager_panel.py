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

from models.engineering_asset import DeviceMode

from camera_manager.connection_status import CameraConnectionState
from camera_manager.manager import CameraManager


class CameraManagerPanel(QWidget):

    # Building-wide Camera Asset administration -- the Designer-facing
    # view onto CameraManager (camera_manager/manager.py), the single
    # place cameras are managed. Same "dumb widget, MainWindow pushes
    # updates in" convention as PropertyPanel/PerceptionDebugPanel: the
    # one public entry point is refresh(building), called by MainWindow
    # wherever it already calls self.property_panel.refresh()/
    # self.perception_debug_panel.refresh(...).
    #
    # Unlike the Property Panel (which only ever shows the one
    # currently-selected item on the currently-displayed floor), this
    # panel lists every camera in the whole Building regardless of
    # which floor the canvas currently shows -- CameraManager.
    # discover_cameras() is building-wide by design (see that method's
    # own docstring), and that is exactly this panel's point: an
    # operator managing cameras across an entire building, not just
    # the one floor happens to be on screen.
    #
    # Owns its own CameraManager instance, rebuilt (via
    # discover_cameras()) on every refresh() -- same "own the debug
    # runner, never take one injected" convention PerceptionDebugPanel
    # already establishes for its own PerceptionDebugRunner. Every
    # interactive edit (enable/disable, mode change) mutates the real
    # Camera object through CameraManager and then re-runs refresh()
    # in full, the same "edit, then _rerun_last()" pattern the
    # Perception Debug Panel's own Hazard Injector already uses --
    # simpler and safer than partially patching table rows in place.

    def __init__(self):

        super().__init__()

        self.manager = CameraManager()

        self._last_building = None

        # Notified after any edit made through this panel (enable/
        # disable, mode change) that could change what other views
        # show for the same Camera object -- the Property Panel (if
        # that camera happens to be selected) and the Camera Coverage
        # overlay both read the exact same Camera instance. None (the
        # default) is a valid, guarded state, same convention as
        # PropertyPanel.on_visual_change.
        self.on_camera_changed = None

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

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(7)
        self.camera_table.setHorizontalHeaderLabels(
            ["Name", "ID", "Floor", "Zone(s)", "Active", "Mode", "Status"],
        )
        self.camera_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.camera_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.camera_table.setRowCount(0)
        layout.addWidget(self.camera_table)

    # =====================================================
    # Public entry point
    # =====================================================

    def refresh(self, building):

        self._last_building = building

        self.manager.discover_cameras(building)

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

        # Changing which floor is selected changes which zones are
        # even valid choices -- rebuild the zone filter to match
        # before re-filtering the table.
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

        cameras = self._filtered_cameras(floor_filter_id, zone_filter_id)

        self.camera_table.setRowCount(len(cameras))

        for row, camera in enumerate(cameras):
            self._populate_row(row, camera)

    # =====================================================

    def _filtered_cameras(self, floor_filter_id, zone_filter_id):

        cameras = self.manager.all_cameras()

        if floor_filter_id is not None:
            cameras = tuple(c for c in cameras if c.floor_id == floor_filter_id)

        if zone_filter_id is not None:
            cameras = tuple(c for c in cameras if zone_filter_id in c.zone_ids)

        return cameras

    # =====================================================

    def _populate_row(self, row, camera):

        self.camera_table.setItem(row, 0, QTableWidgetItem(camera.name))
        self.camera_table.setItem(row, 1, QTableWidgetItem(camera.id))
        self.camera_table.setItem(
            row, 2, QTableWidgetItem(self._floor_names.get(camera.floor_id, camera.floor_id)),
        )

        zone_names = ", ".join(
            self._zone_names.get(zone_id, zone_id) for zone_id in camera.zone_ids
        ) or "(none)"
        self.camera_table.setItem(row, 3, QTableWidgetItem(zone_names))

        active_checkbox = QCheckBox()
        active_checkbox.setChecked(camera.active)
        active_checkbox.toggled.connect(
            lambda checked, camera_id=camera.id: self._on_active_toggled(camera_id, checked)
        )
        self.camera_table.setCellWidget(row, 4, active_checkbox)

        mode_combo = QComboBox()

        for mode in DeviceMode.ALL:
            mode_combo.addItem(mode)

        mode_index = mode_combo.findText(camera.mode)

        if mode_index != -1:
            mode_combo.setCurrentIndex(mode_index)

        mode_combo.currentIndexChanged.connect(
            lambda index, camera_id=camera.id, combo=mode_combo: self._on_mode_changed(
                camera_id, combo.itemText(index),
            )
        )
        self.camera_table.setCellWidget(row, 5, mode_combo)

        status = self.manager.camera_status(camera.id)
        self.camera_table.setItem(row, 6, QTableWidgetItem(self._status_text(camera, status)))

    # =====================================================

    def _status_text(self, camera, status) -> str:

        active_text = "Active" if status.active else "Disabled"

        provider_text = (
            "provider registered" if status.has_detection_provider
            else "no provider (placeholder)"
        )

        detail_text = self._mode_detail_text(camera, status)

        return f"{active_text} | {status.mode} | {provider_text} | {detail_text}"

    # =====================================================
    # CCTV Pipeline End-to-End Offline Validation milestone, Phase 8 --
    # a truthful, mode-aware detail appended to the existing status
    # column. Every input here already exists (CameraStatus.mode/
    # has_detection_provider, CameraManager's own runtime-only
    # connection_status(), Camera.connection's own credential_ref/
    # password fields) -- nothing here is fabricated, and "Live" never
    # reports anything but "Not Connected"/"Online" strictly from
    # whatever connection_status() was actually told (see
    # camera_manager/connection_status.py -- nothing sets it to ONLINE
    # today, so a real camera never shows connected until a real Live
    # adapter exists and reports it).
    # =====================================================

    def _mode_detail_text(self, camera, status) -> str:

        connection_state = self.manager.connection_status(camera.id)

        if status.mode == DeviceMode.SIMULATION:

            return "Ready" if status.has_detection_provider else "Not Configured"

        if status.mode == DeviceMode.REPLAY:

            if not status.has_detection_provider:
                return "No Source"

            if connection_state == CameraConnectionState.STREAM_UNAVAILABLE:
                return "Source Missing"

            return "Source Loaded"

        if status.mode == DeviceMode.LIVE:

            if not camera.connection.credential_ref and not camera.connection.password:
                return "Credentials Missing"

            if connection_state == CameraConnectionState.ONLINE:
                return "Online"

            return "Not Connected"

        return ""

    # =====================================================
    # Handlers
    # =====================================================

    def _on_active_toggled(self, camera_id, checked):

        if checked:
            self.manager.enable_camera(camera_id)
        else:
            self.manager.disable_camera(camera_id)

        self._notify_camera_changed()
        self._rerun_last()

    # =====================================================

    def _on_mode_changed(self, camera_id, mode):

        self.manager.set_camera_mode(camera_id, mode)

        self._notify_camera_changed()
        self._rerun_last()

    # =====================================================

    def _notify_camera_changed(self):

        if self.on_camera_changed:
            self.on_camera_changed()

    # =====================================================

    def _rerun_last(self):

        if self._last_building is None:
            return

        self.refresh(self._last_building)
