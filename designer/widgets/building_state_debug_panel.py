from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from models.sensor_asset import DetectorState

from facp.models import InvalidPanelOperation, PanelState

from designer.building_state_debug_runner import BuildingStateDebugRunner


_STATUS_OK = QColor("#2e7d32")
_STATUS_WARN = QColor("#f9a825")
_STATUS_BAD = QColor("#c62828")

_ALARM_COLOR = {
    DetectorState.NORMAL: _STATUS_OK,
    DetectorState.ALARM: _STATUS_BAD,
    DetectorState.FAULT: _STATUS_WARN,
}

_PANEL_STATE_COLOR = {
    PanelState.NORMAL: _STATUS_OK,
    PanelState.ALARM: _STATUS_BAD,
    PanelState.FAULT: _STATUS_WARN,
    PanelState.ALARM_ACKNOWLEDGED: _STATUS_WARN,
    PanelState.ALARM_SILENCED: _STATUS_WARN,
}


class BuildingStateDebugPanel(QWidget):

    # Verification/visualization-only -- this widget never calls into
    # the Building State Estimator or the Consistency checks directly.
    # It owns one BuildingStateDebugRunner (Designer glue, see
    # designer/building_state_debug_runner.py) and only ever reads what
    # that runner already computed by driving the unmodified
    # BuildingStateEstimator/check_consistency functions. Follows this
    # codebase's established "dumb widget, MainWindow pushes updates in"
    # convention (see PerceptionDebugPanel/BottomInfoBar/PropertyPanel):
    # the only public entry point is refresh(building, sandbox_manager,
    # time), called by MainWindow exactly where it already calls
    # self.perception_debug_panel.refresh(...).

    def __init__(self):

        super().__init__()

        self._runner = BuildingStateDebugRunner()

        self._last_building = None
        self._last_sandbox_manager = None
        self._last_time = 0.0

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self._build_occupants_tab()
        self._build_zone_occupancy_tab()
        self._build_camera_tab()
        self._build_smoke_tab()
        self._build_heat_tab()
        self._build_hazard_alarm_tab()
        self._build_facp_tab()
        self._build_diagnostics_tab()

    # =====================================================
    # Tab construction
    # =====================================================

    def _build_occupants_tab(self):

        self.occupants_table = self._make_table(
            ["Track ID", "Floor", "Zone", "Classification", "Human State", "Confidence"],
        )
        self.tabs.addTab(self._wrap(self.occupants_table), "1. Occupant Tracks")

    # =====================================================

    def _build_zone_occupancy_tab(self):

        self.zone_occupancy_table = self._make_table(
            ["Zone", "Occupant Count", "Confidence"],
        )
        self.tabs.addTab(self._wrap(self.zone_occupancy_table), "2. Zone Occupancy")

    # =====================================================

    def _build_camera_tab(self):

        self.camera_table = self._make_table(
            ["Camera ID", "Active", "Mode", "Zone(s)", "Estimated Occupants", "Confidence"],
        )
        self.tabs.addTab(self._wrap(self.camera_table), "3. Cameras")

    # =====================================================

    def _build_smoke_tab(self):

        self.smoke_table = self._make_table(
            ["Sensor ID", "Active", "Health", "Zone(s)", "Alarm", "Confidence"],
        )
        self.tabs.addTab(self._wrap(self.smoke_table), "4. Smoke Detectors")

    # =====================================================

    def _build_heat_tab(self):

        self.heat_table = self._make_table(
            ["Sensor ID", "Active", "Health", "Zone(s)", "Alarm", "Confidence"],
        )
        self.tabs.addTab(self._wrap(self.heat_table), "5. Heat Detectors")

    # =====================================================

    def _build_hazard_alarm_tab(self):

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        self.alarm_status_label = QLabel("Building Alarm Status: -")
        container_layout.addWidget(self.alarm_status_label)

        self.overall_severity_label = QLabel("Overall Hazard Severity: -")
        container_layout.addWidget(self.overall_severity_label)

        self.hazard_summary_table = self._make_table(["Zone", "Severity"])
        container_layout.addWidget(self.hazard_summary_table)

        self.tabs.addTab(container, "6. Hazard & Alarm Summary")

    # =====================================================

    def _build_facp_tab(self):

        # Fire Alarm Control Panel (Phase 7) -- displays whatever
        # facp.engine.SimulatedFACP already computed for this cycle
        # (via BuildingState.facp_status, see building_state_debug_
        # runner.py) and offers Acknowledge/Silence/Reset buttons that
        # operate ONLY on this runner's own SimulatedFACP -- never real
        # hardware, exactly as Phase 7 requires.

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        self.facp_panel_state_label = QLabel("Panel State: -")
        container_layout.addWidget(self.facp_panel_state_label)

        button_row = QHBoxLayout()

        self.facp_acknowledge_button = QPushButton("Acknowledge")
        self.facp_acknowledge_button.clicked.connect(self._on_facp_acknowledge)
        button_row.addWidget(self.facp_acknowledge_button)

        self.facp_silence_button = QPushButton("Silence")
        self.facp_silence_button.clicked.connect(self._on_facp_silence)
        button_row.addWidget(self.facp_silence_button)

        self.facp_reset_button = QPushButton("Reset")
        self.facp_reset_button.clicked.connect(self._on_facp_reset)
        button_row.addWidget(self.facp_reset_button)

        container_layout.addLayout(button_row)

        container_layout.addWidget(QLabel("Active Alarm Sources"))
        self.facp_alarm_table = self._make_table(["Source Asset ID"])
        container_layout.addWidget(self.facp_alarm_table)

        container_layout.addWidget(QLabel("Active Fault Sources"))
        self.facp_fault_table = self._make_table(["Source Asset ID"])
        container_layout.addWidget(self.facp_fault_table)

        container_layout.addWidget(QLabel("Recent Panel Events"))
        self.facp_event_table = self._make_table(
            ["Time", "Event Type", "Source Asset", "Floor", "Zone(s)", "Panel State After"],
        )
        container_layout.addWidget(self.facp_event_table)

        self.tabs.addTab(container, "7. Fire Alarm Control Panel")

    # =====================================================

    def _build_diagnostics_tab(self):

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)

        note = QLabel(
            "Diagnostic warnings only -- generated from the current BuildingState by "
            "building_state.consistency.check_consistency(), never fed back into the "
            "state itself and never a decision/recommendation of any kind."
        )
        note.setWordWrap(True)
        container_layout.addWidget(note)

        self.diagnostics_table = self._make_table(["Type", "Message", "Zone", "Asset"])
        container_layout.addWidget(self.diagnostics_table)

        self.tabs.addTab(container, "8. Diagnostics")

    # =====================================================
    # Helpers
    # =====================================================

    def _make_table(self, headers) -> QTableWidget:

        table = QTableWidget()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setRowCount(0)

        return table

    # =====================================================

    def _wrap(self, table) -> QWidget:

        container = QWidget()
        container_layout = QVBoxLayout()
        container.setLayout(container_layout)
        container_layout.addWidget(table)

        return container

    # =====================================================

    def _set_row(self, table, row, values, color=None):

        for column, value in enumerate(values):

            item = QTableWidgetItem(str(value))

            if color is not None:
                item.setForeground(color)

            table.setItem(row, column, item)

    # =====================================================
    # Public entry point -- called by MainWindow after every
    # simulation tick/step/reset, exactly where it already calls
    # self.perception_debug_panel.refresh(...).
    # =====================================================

    def refresh(self, building, sandbox_manager, time: float):

        self._last_building = building
        self._last_sandbox_manager = sandbox_manager
        self._last_time = time

        if building is None:
            return

        snapshot = self._runner.run(building, sandbox_manager, time)

        self._refresh_occupants_table(snapshot)
        self._refresh_zone_occupancy_table(snapshot)
        self._refresh_camera_table(snapshot)
        self._refresh_detector_table(self.smoke_table, snapshot.state.smoke_detector_states, snapshot)
        self._refresh_detector_table(self.heat_table, snapshot.state.heat_detector_states, snapshot)
        self._refresh_hazard_alarm_tab(snapshot)
        self._refresh_facp_tab(snapshot)
        self._refresh_diagnostics_table(snapshot)

    # =====================================================

    def _zone_name(self, snapshot, zone_id):

        return snapshot.zone_names.get(zone_id, zone_id) if zone_id else "-"

    # =====================================================

    def _refresh_occupants_table(self, snapshot):

        state = snapshot.state
        tracks = sorted(state.occupant_tracks.values(), key=lambda track: track.track_id)

        self.occupants_table.setRowCount(len(tracks))

        for row, track in enumerate(tracks):

            self._set_row(
                self.occupants_table, row,
                [
                    track.track_id,
                    track.floor_id or "-",
                    self._zone_name(snapshot, track.zone_id),
                    track.classification.name,
                    track.human_state.name if track.human_state else "-",
                    f"{track.confidence:.2f}",
                ],
            )

    # =====================================================

    def _refresh_zone_occupancy_table(self, snapshot):

        state = snapshot.state
        zone_ids = sorted(snapshot.zone_names.keys(), key=lambda zone_id: snapshot.zone_names[zone_id])

        self.zone_occupancy_table.setRowCount(len(zone_ids))

        for row, zone_id in enumerate(zone_ids):

            observation = state.zone_occupancy.observation_at(zone_id)

            self._set_row(
                self.zone_occupancy_table, row,
                [
                    snapshot.zone_names.get(zone_id, zone_id),
                    "-" if observation.occupant_count is None else f"{observation.occupant_count:g}",
                    "-" if observation.confidence is None else f"{observation.confidence:.2f}",
                ],
            )

    # =====================================================

    def _refresh_camera_table(self, snapshot):

        state = snapshot.state
        cameras = sorted(state.camera_observations.values(), key=lambda asset: asset.status.camera_id)

        self.camera_table.setRowCount(len(cameras))

        for row, asset in enumerate(cameras):

            observation = asset.frame_observation
            zone_names = ", ".join(self._zone_name(snapshot, zone_id) for zone_id in asset.status.zone_ids) or "(none)"

            self._set_row(
                self.camera_table, row,
                [
                    asset.status.camera_id,
                    "yes" if asset.status.active else "no",
                    asset.status.mode,
                    zone_names,
                    "-" if observation is None or observation.estimated_occupant_count is None
                    else f"{observation.estimated_occupant_count:g}",
                    "-" if observation is None or observation.confidence is None
                    else f"{observation.confidence:.2f}",
                ],
                color=None if asset.status.active else _STATUS_WARN,
            )

    # =====================================================

    def _refresh_detector_table(self, table, detector_states, snapshot):

        assets = sorted(detector_states.values(), key=lambda asset: asset.status.sensor_id)

        table.setRowCount(len(assets))

        for row, asset in enumerate(assets):

            reading = asset.reading
            zone_names = ", ".join(self._zone_name(snapshot, zone_id) for zone_id in asset.status.zone_ids) or "(none)"

            self._set_row(
                table, row,
                [
                    asset.status.sensor_id,
                    "yes" if asset.status.active else "no",
                    asset.status.health_status,
                    zone_names,
                    "-" if reading is None else ("ALARM" if reading.alarm_active else "clear"),
                    "-" if reading is None or reading.confidence is None else f"{reading.confidence:.2f}",
                ],
                color=None if asset.status.active else _STATUS_WARN,
            )

    # =====================================================

    def _refresh_hazard_alarm_tab(self, snapshot):

        state = snapshot.state

        self.alarm_status_label.setText(f"Building Alarm Status: {state.building_alarm_status.name}")
        self.alarm_status_label.setStyleSheet(
            f"color: {_ALARM_COLOR.get(state.building_alarm_status, _STATUS_OK).name()};"
        )

        self.overall_severity_label.setText(
            f"Overall Hazard Severity: {state.hazard_summary.overall_severity.name}"
            + (
                f" (worst zone: {self._zone_name(snapshot, state.hazard_summary.worst_zone_id)})"
                if state.hazard_summary.worst_zone_id else ""
            )
        )

        zone_ids = sorted(
            state.hazard_summary.zone_severities.keys(),
            key=lambda zone_id: snapshot.zone_names.get(zone_id, zone_id),
        )

        self.hazard_summary_table.setRowCount(len(zone_ids))

        for row, zone_id in enumerate(zone_ids):

            self._set_row(
                self.hazard_summary_table, row,
                [
                    self._zone_name(snapshot, zone_id),
                    state.hazard_summary.severity_for(zone_id).name,
                ],
            )

    # =====================================================

    def _refresh_facp_tab(self, snapshot):

        facp_status = snapshot.state.facp_status

        if facp_status is None:
            return

        self.facp_panel_state_label.setText(f"Panel State: {facp_status.panel_state.name}")
        self.facp_panel_state_label.setStyleSheet(
            f"color: {_PANEL_STATE_COLOR.get(facp_status.panel_state, _STATUS_OK).name()};"
        )

        self.facp_alarm_table.setRowCount(len(facp_status.active_alarm_source_ids))

        for row, asset_id in enumerate(facp_status.active_alarm_source_ids):
            self._set_row(self.facp_alarm_table, row, [asset_id], color=_STATUS_BAD)

        self.facp_fault_table.setRowCount(len(facp_status.active_fault_source_ids))

        for row, asset_id in enumerate(facp_status.active_fault_source_ids):
            self._set_row(self.facp_fault_table, row, [asset_id], color=_STATUS_WARN)

        events = facp_status.recent_events
        self.facp_event_table.setRowCount(len(events))

        for row, event in enumerate(events):

            self._set_row(
                self.facp_event_table, row,
                [
                    f"{event.timestamp:.1f}",
                    event.event_type.name,
                    event.source_asset_id or "-",
                    event.floor_id or "-",
                    ", ".join(event.zone_ids) or "-",
                    event.panel_state_after.name,
                ],
            )

    # =====================================================

    def _refresh_diagnostics_table(self, snapshot):

        warnings = snapshot.warnings

        self.diagnostics_table.setRowCount(len(warnings))

        for row, warning in enumerate(warnings):

            self._set_row(
                self.diagnostics_table, row,
                [
                    warning.warning_type.name,
                    warning.message,
                    self._zone_name(snapshot, warning.zone_id) if warning.zone_id else "-",
                    warning.asset_id or "-",
                ],
                color=_STATUS_BAD,
            )

    # =====================================================
    # FACP operator control handlers -- operate ONLY on this panel's
    # own BuildingStateDebugRunner's SimulatedFACP (never real
    # hardware, per Phase 7's own explicit requirement). Each one
    # re-refreshes the panel afterward so the FACP tab immediately
    # reflects the new panel state/event.
    # =====================================================

    def _on_facp_acknowledge(self):

        self._try_facp_operation(self._runner.acknowledge_facp)

    # =====================================================

    def _on_facp_silence(self):

        self._try_facp_operation(self._runner.silence_facp)

    # =====================================================

    def _on_facp_reset(self):

        self._try_facp_operation(self._runner.reset_facp)

    # =====================================================

    def _try_facp_operation(self, operation):

        try:
            operation(self._last_time)
        except InvalidPanelOperation as error:
            QMessageBox.warning(self, "Fire Alarm Control Panel", str(error))
            return

        if self._last_building is not None:
            self.refresh(self._last_building, self._last_sandbox_manager, self._last_time)
