from typing import Optional

from PyQt6.QtWidgets import QGroupBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from facp.models import PanelState


# =====================================================
# LiveStatusPanel -- Phase 8's operational asset status panel. Display
# only: no operator control of any kind (no acknowledge/silence/reset
# button, no camera/sensor enable-disable toggle) -- every value shown
# here is read straight off an already-computed building_state.models.
# BuildingState, the same "dumb widget, pushed updates only" convention
# every other Command Center panel already follows. Live-only; never
# shown/populated in Replay mode (Dashboard only calls show_building_state()
# from the Live rendering path).
# =====================================================


def _dash_join(values):

    return ", ".join(values) if values else "-"


class LiveStatusPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        cameras_group = QGroupBox("Cameras")
        cameras_layout = QVBoxLayout(cameras_group)
        self.camera_table = QTableWidget(0, 5)
        self.camera_table.setHorizontalHeaderLabels(["Camera", "Floor", "Zones", "Mode", "Active"])
        self.camera_table.horizontalHeader().setStretchLastSection(True)
        self.camera_table.verticalHeader().setVisible(False)
        cameras_layout.addWidget(self.camera_table)
        layout.addWidget(cameras_group)

        sensors_group = QGroupBox("Sensors")
        sensors_layout = QVBoxLayout(sensors_group)
        self.sensor_table = QTableWidget(0, 6)
        self.sensor_table.setHorizontalHeaderLabels(["Sensor", "Type", "Floor", "Condition", "Health", "Active"])
        self.sensor_table.horizontalHeader().setStretchLastSection(True)
        self.sensor_table.verticalHeader().setVisible(False)
        sensors_layout.addWidget(self.sensor_table)
        layout.addWidget(sensors_group)

        facp_group = QGroupBox("Fire Alarm Control Panel")
        facp_layout = QVBoxLayout(facp_group)
        self.facp_state_table = QTableWidget(0, 2)
        self.facp_state_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.facp_state_table.horizontalHeader().setStretchLastSection(True)
        self.facp_state_table.verticalHeader().setVisible(False)
        facp_layout.addWidget(self.facp_state_table)
        layout.addWidget(facp_group)

        layout.addStretch(1)

    # =====================================================

    def show_building_state(self, building_state: Optional[object]) -> None:

        if building_state is None:

            self.camera_table.setRowCount(0)
            self.sensor_table.setRowCount(0)
            self.facp_state_table.setRowCount(0)
            return

        cameras = list(building_state.camera_observations.values())
        self.camera_table.setRowCount(len(cameras))
        for row_index, asset in enumerate(cameras):

            status = asset.status
            self.camera_table.setItem(row_index, 0, QTableWidgetItem(status.name or status.camera_id))
            self.camera_table.setItem(row_index, 1, QTableWidgetItem(status.floor_id or "-"))
            self.camera_table.setItem(row_index, 2, QTableWidgetItem(_dash_join(status.zone_ids)))
            self.camera_table.setItem(row_index, 3, QTableWidgetItem(status.mode or "-"))
            self.camera_table.setItem(row_index, 4, QTableWidgetItem("Yes" if status.active else "No"))

        from live_system.live_command_center_gateway import detector_condition

        detectors = {
            **building_state.smoke_detector_states, **building_state.heat_detector_states,
        }
        detector_ids = sorted(detectors)
        self.sensor_table.setRowCount(len(detector_ids))
        for row_index, sensor_id in enumerate(detector_ids):

            asset = detectors[sensor_id]
            status = asset.status
            self.sensor_table.setItem(row_index, 0, QTableWidgetItem(status.name or status.sensor_id))
            self.sensor_table.setItem(row_index, 1, QTableWidgetItem(status.sensor_type or "-"))
            self.sensor_table.setItem(row_index, 2, QTableWidgetItem(status.floor_id or "-"))
            self.sensor_table.setItem(row_index, 3, QTableWidgetItem(detector_condition(asset)))
            self.sensor_table.setItem(row_index, 4, QTableWidgetItem(status.health_status or "-"))
            self.sensor_table.setItem(row_index, 5, QTableWidgetItem("Yes" if status.active else "No"))

        facp = building_state.facp_status

        if facp is None:

            self.facp_state_table.setRowCount(0)
            return

        rows = [
            ("Panel State", facp.panel_state.name if isinstance(facp.panel_state, PanelState) else str(facp.panel_state)),
            ("Active Alarm Sources", _dash_join(facp.active_alarm_source_ids)),
            ("Active Fault Sources", _dash_join(facp.active_fault_source_ids)),
            ("Acknowledged", "Yes" if facp.acknowledged else "No"),
            ("Silenced", "Yes" if facp.silenced else "No"),
        ]
        self.facp_state_table.setRowCount(len(rows))
        for row_index, (field, value) in enumerate(rows):
            self.facp_state_table.setItem(row_index, 0, QTableWidgetItem(field))
            self.facp_state_table.setItem(row_index, 1, QTableWidgetItem(value))
