from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from dataset_builder.schema import (
    ordered_cameras,
    ordered_detectors,
    ordered_doors,
    ordered_exits,
    ordered_stairs,
)


# =====================================================
# IncidentPanel -- scenario/incident metadata, current engineering
# device states (door/exit/stair/detector/camera), and GroundTruth's
# own whole-run congestion/bottleneck facts. Dumb widget, same
# convention as every other Command Center panel: set_incident() runs
# once per loaded incident (metadata + GroundTruth facts + table
# skeletons), show_frame() runs on every playback tick (current
# engineering states).
# =====================================================


def _yes_no(value):

    return "-" if value is None else ("Yes" if value else "No")


class IncidentPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        metadata_group = QGroupBox("Incident / Scenario")
        metadata_layout = QVBoxLayout(metadata_group)
        self.scenario_id_label = QLabel("Scenario: -")
        self.seed_label = QLabel("Seed: -")
        self.fire_profile_label = QLabel("Fire profile: -")
        for label in (self.scenario_id_label, self.seed_label, self.fire_profile_label):
            metadata_layout.addWidget(label)
        layout.addWidget(metadata_group)

        states_group = QGroupBox("Engineering States")
        states_layout = QVBoxLayout(states_group)

        self.door_table = self._state_table("Doors")
        self.exit_table = self._state_table("Exits")
        self.stair_table = self._state_table("Stairs")
        self.detector_table = self._state_table("Detectors")
        self.camera_table = self._state_table("Cameras")

        for label_text, table in (
            ("Doors", self.door_table),
            ("Exits", self.exit_table),
            ("Stairs", self.stair_table),
            ("Detectors", self.detector_table),
            ("Cameras", self.camera_table),
        ):
            states_layout.addWidget(QLabel(label_text))
            states_layout.addWidget(table)

        layout.addWidget(states_group)

        ground_truth_group = QGroupBox("Ground Truth -- Congestion / Bottlenecks")
        ground_truth_layout = QVBoxLayout(ground_truth_group)
        self.worst_exit_label = QLabel("Worst exit: -")
        self.worst_stair_label = QLabel("Worst stair: -")
        self.worst_door_label = QLabel("Worst door: -")
        self.peak_congestion_label = QLabel("Peak congestion location: -")
        self.congestion_duration_label = QLabel("Congestion duration: -")
        self.bottleneck_doors_label = QLabel("Doors that became bottlenecks: -")
        self.bottleneck_doors_label.setWordWrap(True)
        self.exits_underutilized_label = QLabel("Underutilized exits: -")
        self.exits_underutilized_label.setWordWrap(True)
        self.exits_exceeding_label = QLabel("Exits exceeding capacity: -")
        self.exits_exceeding_label.setWordWrap(True)
        self.stairs_exceeding_label = QLabel("Stairs exceeding capacity: -")
        self.stairs_exceeding_label.setWordWrap(True)
        for label in (
            self.worst_exit_label,
            self.worst_stair_label,
            self.worst_door_label,
            self.peak_congestion_label,
            self.congestion_duration_label,
            self.bottleneck_doors_label,
            self.exits_underutilized_label,
            self.exits_exceeding_label,
            self.stairs_exceeding_label,
        ):
            ground_truth_layout.addWidget(label)
        layout.addWidget(ground_truth_group)

        layout.addStretch(1)

    # =====================================================

    @staticmethod
    def _state_table(_caption):

        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Name", "State"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setMaximumHeight(140)
        return table

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        self._show_metadata(incident_data.scenario if incident_data is not None else None)
        self._populate_state_tables(incident_data)
        self._show_ground_truth(incident_data.ground_truth if incident_data is not None else None)

        first_frame = incident_data.frame_at_index(0) if incident_data is not None else None
        self.show_frame(first_frame)

    # =====================================================

    def show_frame(self, frame):

        if frame is None:

            for table in (
                self.door_table, self.exit_table, self.stair_table,
                self.detector_table, self.camera_table,
            ):
                for row in range(table.rowCount()):
                    table.setItem(row, 1, QTableWidgetItem("-"))
            return

        self._fill_state_column(self.door_table, frame.door_states)
        self._fill_state_column(self.exit_table, frame.exit_states)
        self._fill_state_column(self.stair_table, frame.stair_states)
        self._fill_state_column(self.detector_table, frame.detector_states)
        self._fill_state_column(self.camera_table, frame.camera_states)

    # =====================================================

    def _fill_state_column(self, table, states_by_id):

        for row in range(table.rowCount()):

            id_item = table.item(row, 0)
            object_id = id_item.data(Qt.ItemDataRole.UserRole) if id_item is not None else None
            state = states_by_id.get(object_id, "-") if object_id else "-"

            table.setItem(row, 1, QTableWidgetItem(str(state)))

    # =====================================================

    def _populate_state_tables(self, incident_data):

        for table in (
            self.door_table, self.exit_table, self.stair_table,
            self.detector_table, self.camera_table,
        ):
            table.setRowCount(0)

        if incident_data is None:
            return

        building = incident_data.building
        name = incident_data.name_for

        self._fill_object_rows(self.door_table, ordered_doors(building), name)
        self._fill_object_rows(self.exit_table, ordered_exits(building), name)
        self._fill_object_rows(self.stair_table, ordered_stairs(building), name)
        self._fill_object_rows(self.detector_table, ordered_detectors(building), name)
        self._fill_object_rows(self.camera_table, ordered_cameras(building), name)

    # =====================================================

    @staticmethod
    def _fill_object_rows(table, objects, name):

        table.setRowCount(len(objects))

        for row_index, obj in enumerate(objects):

            id_item = QTableWidgetItem(name(obj.id))
            id_item.setData(Qt.ItemDataRole.UserRole, obj.id)

            table.setItem(row_index, 0, id_item)
            table.setItem(row_index, 1, QTableWidgetItem("-"))

    # =====================================================

    def _show_metadata(self, scenario):

        if scenario is None:
            self.scenario_id_label.setText("Scenario: -")
            self.seed_label.setText("Seed: -")
            self.fire_profile_label.setText("Fire profile: -")
            return

        self.scenario_id_label.setText(f"Scenario: {scenario.metadata.scenario_id}")
        self.seed_label.setText(f"Seed: {scenario.metadata.seed}")

        fire_profile = scenario.fire.fire_profile if scenario.fire is not None else "-"
        self.fire_profile_label.setText(f"Fire profile: {fire_profile}")

    # =====================================================

    def _show_ground_truth(self, ground_truth):

        if ground_truth is None or self._incident is None:

            self.worst_exit_label.setText("Worst exit: -")
            self.worst_stair_label.setText("Worst stair: -")
            self.worst_door_label.setText("Worst door: -")
            self.peak_congestion_label.setText("Peak congestion location: -")
            self.congestion_duration_label.setText("Congestion duration: -")
            self.bottleneck_doors_label.setText("Doors that became bottlenecks: -")
            self.exits_underutilized_label.setText("Underutilized exits: -")
            self.exits_exceeding_label.setText("Exits exceeding capacity: -")
            self.stairs_exceeding_label.setText("Stairs exceeding capacity: -")
            return

        name = self._incident.name_for

        self.worst_exit_label.setText(f"Worst exit: {name(ground_truth.worst_exit)}")
        self.worst_stair_label.setText(f"Worst stair: {name(ground_truth.worst_stair)}")
        self.worst_door_label.setText(f"Worst door: {name(ground_truth.worst_door)}")

        if ground_truth.peak_congestion_location_id:
            location = (
                f"{name(ground_truth.peak_congestion_location_id)} "
                f"({ground_truth.peak_congestion_location_type or '?'}) = "
                f"{ground_truth.peak_congestion_value}"
            )
        else:
            location = "-"
        self.peak_congestion_label.setText(f"Peak congestion location: {location}")

        duration = ground_truth.congestion_duration
        self.congestion_duration_label.setText(
            f"Congestion duration: {'-' if duration is None else f'{duration:.1f}s'}",
        )

        self.bottleneck_doors_label.setText(
            "Doors that became bottlenecks: "
            + (", ".join(name(door_id) for door_id in ground_truth.doors_that_became_bottlenecks) or "-"),
        )
        self.exits_underutilized_label.setText(
            "Underutilized exits: "
            + (", ".join(name(exit_id) for exit_id in ground_truth.exits_underutilized) or "-"),
        )
        self.exits_exceeding_label.setText(
            "Exits exceeding capacity: "
            + (", ".join(name(exit_id) for exit_id in ground_truth.exits_exceeding_capacity) or "-"),
        )
        self.stairs_exceeding_label.setText(
            "Stairs exceeding capacity: "
            + (", ".join(name(stair_id) for stair_id in ground_truth.stairs_exceeding_capacity) or "-"),
        )
