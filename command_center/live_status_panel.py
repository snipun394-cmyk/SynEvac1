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

        # Manual Call Points & Emergency Lighting milestone, Phase 11 --
        # "MCP activations should appear through the existing FACP/alarm
        # presentation wherever possible... existing FACP UI can display
        # source id, source type, zone, state." Every alarm/fault source
        # (SmokeDetector, HeatDetector, ManualCallPoint alike -- this
        # table has no hardcoded knowledge of any specific asset type)
        # already reaches facp.active_alarm_source_ids/active_fault_
        # source_ids; this table only adds richer per-source display
        # over that SAME already-existing FACPSnapshot data (cross-
        # referenced against facp.recent_events for type/zone), never a
        # second alarm representation or a new BuildingState field.
        self.facp_sources_table = QTableWidget(0, 4)
        self.facp_sources_table.setHorizontalHeaderLabels(["Source", "Type", "Zone", "State"])
        self.facp_sources_table.horizontalHeader().setStretchLastSection(True)
        self.facp_sources_table.verticalHeader().setVisible(False)
        facp_layout.addWidget(self.facp_sources_table)

        layout.addWidget(facp_group)

        # Fire Suppression & Water-Based Safety Asset Digital Twin
        # milestone, Phase 12 -- "avoid creating four new panels": ONE
        # additive table, same Source/Type/Zone/State shape as
        # facp_sources_table above, covering all four asset types
        # (Sprinkler/FireExtinguisher/FireHydrant/HoseReel) at once.
        # AVAILABLE/ACTIVATED here is a maintenance/status fact only --
        # never a claim that a route is safe or that a fire will be
        # controlled (see docs/architecture/
        # fire_suppression_safety_assets.md).
        fire_safety_group = QGroupBox("Fire Suppression & Safety Assets")
        fire_safety_layout = QVBoxLayout(fire_safety_group)
        self.fire_safety_table = QTableWidget(0, 4)
        self.fire_safety_table.setHorizontalHeaderLabels(["Asset", "Type", "Zone", "State"])
        self.fire_safety_table.horizontalHeader().setStretchLastSection(True)
        self.fire_safety_table.verticalHeader().setVisible(False)
        fire_safety_layout.addWidget(self.fire_safety_table)
        layout.addWidget(fire_safety_group)

        layout.addStretch(1)

    # =====================================================

    def show_building_state(self, building_state: Optional[object]) -> None:

        if building_state is None:

            self.camera_table.setRowCount(0)
            self.sensor_table.setRowCount(0)
            self.facp_state_table.setRowCount(0)
            self.facp_sources_table.setRowCount(0)
            self.fire_safety_table.setRowCount(0)
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

        # Independent of FACP -- a Sprinkler/FireExtinguisher/
        # FireHydrant/HoseReel is never a FACP alarm source (see
        # models.sprinkler.Sprinkler's own docstring), so this table is
        # populated regardless of whether facp_status is configured.
        self._show_fire_safety_assets(building_state.fire_safety_status)

        facp = building_state.facp_status

        if facp is None:

            self.facp_state_table.setRowCount(0)
            self.facp_sources_table.setRowCount(0)
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

        self._show_facp_sources(facp)

    # =====================================================

    def _show_facp_sources(self, facp) -> None:

        # The most recent event mentioning each source id, purely for
        # type/zone display -- an id with no matching event in the
        # bounded recent_events window (e.g. it alarmed before the
        # window started) still shows honestly as "-" for those two
        # columns, never fabricated, and its own id/state are never
        # withheld on account of it.
        latest_event_by_source = {}
        for event in facp.recent_events:
            if event.source_asset_id is not None:
                latest_event_by_source[event.source_asset_id] = event

        rows = [(source_id, "ALARM") for source_id in facp.active_alarm_source_ids]
        rows += [(source_id, "FAULT") for source_id in facp.active_fault_source_ids]

        self.facp_sources_table.setRowCount(len(rows))

        for row_index, (source_id, state) in enumerate(rows):

            event = latest_event_by_source.get(source_id)
            source_type = event.source_asset_type if event is not None and event.source_asset_type else "-"
            zone = _dash_join(event.zone_ids) if event is not None else "-"

            self.facp_sources_table.setItem(row_index, 0, QTableWidgetItem(source_id))
            self.facp_sources_table.setItem(row_index, 1, QTableWidgetItem(source_type))
            self.facp_sources_table.setItem(row_index, 2, QTableWidgetItem(zone))
            self.facp_sources_table.setItem(row_index, 3, QTableWidgetItem(state))

    # =====================================================

    def _show_fire_safety_assets(self, fire_safety_status) -> None:

        if fire_safety_status is None:

            self.fire_safety_table.setRowCount(0)
            return

        entries = fire_safety_status.entries

        self.fire_safety_table.setRowCount(len(entries))

        for row_index, entry in enumerate(entries):

            self.fire_safety_table.setItem(row_index, 0, QTableWidgetItem(entry.name or entry.asset_id))
            self.fire_safety_table.setItem(row_index, 1, QTableWidgetItem(entry.asset_type))
            self.fire_safety_table.setItem(row_index, 2, QTableWidgetItem(_dash_join(entry.zone_ids)))
            self.fire_safety_table.setItem(row_index, 3, QTableWidgetItem(entry.state))
