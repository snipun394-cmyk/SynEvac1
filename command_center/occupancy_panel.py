from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# =====================================================
# OccupancyPanel -- current occupancy, evacuation progress (evacuated/
# remaining/trapped), live congestion/queueing readouts, and
# GroundTruth's own whole-run evacuation metrics. Dumb widget, same
# convention as every other Command Center panel: set_incident() runs
# once per loaded incident (zone list + GroundTruth summary),
# show_frame() runs on every playback tick (current occupancy +
# congestion).
# =====================================================


def _format_number(value):

    return "-" if value is None else str(value)


def _format_float(value, digits=2):

    return "-" if value is None else f"{value:.{digits}f}"


class OccupancyPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        stats_group = QGroupBox("Evacuation Progress")
        stats_layout = QHBoxLayout(stats_group)
        self.evacuated_value, evacuated_tile = self._stat_tile("Evacuated")
        self.remaining_value, remaining_tile = self._stat_tile("Remaining")
        self.trapped_value, trapped_tile = self._stat_tile("Trapped")
        stats_layout.addWidget(evacuated_tile)
        stats_layout.addWidget(remaining_tile)
        stats_layout.addWidget(trapped_tile)
        layout.addWidget(stats_group)

        occupancy_group = QGroupBox("Current Occupancy by Zone")
        occupancy_layout = QVBoxLayout(occupancy_group)
        self.occupancy_table = QTableWidget(0, 2)
        self.occupancy_table.setHorizontalHeaderLabels(["Zone", "Occupants"])
        self.occupancy_table.horizontalHeader().setStretchLastSection(True)
        self.occupancy_table.verticalHeader().setVisible(False)
        occupancy_layout.addWidget(self.occupancy_table)
        layout.addWidget(occupancy_group)

        congestion_group = QGroupBox("Current Congestion")
        congestion_layout = QVBoxLayout(congestion_group)
        self.congestion_label = QLabel("Current congestion: -")
        self.max_congestion_label = QLabel("Maximum congestion so far: -")
        self.evacuation_rate_label = QLabel("Current evacuation rate: -")
        self.queue_length_label = QLabel("Current queue length: -")
        self.route_utilization_label = QLabel("Current route utilization: -")
        self.bottleneck_label = QLabel("Current bottleneck: -")
        for label in (
            self.congestion_label,
            self.max_congestion_label,
            self.evacuation_rate_label,
            self.queue_length_label,
            self.route_utilization_label,
            self.bottleneck_label,
        ):
            congestion_layout.addWidget(label)
        layout.addWidget(congestion_group)

        ground_truth_group = QGroupBox("Ground Truth -- Evacuation Metrics")
        ground_truth_layout = QVBoxLayout(ground_truth_group)
        self.total_evacuation_time_label = QLabel("Total evacuation time: -")
        self.building_cleared_label = QLabel("Building cleared: -")
        self.reachable_label = QLabel("Reachable occupants: -")
        self.unreachable_label = QLabel("Unreachable occupants: -")
        for label in (
            self.total_evacuation_time_label,
            self.building_cleared_label,
            self.reachable_label,
            self.unreachable_label,
        ):
            ground_truth_layout.addWidget(label)
        layout.addWidget(ground_truth_group)

        layout.addStretch(1)

    # =====================================================

    @staticmethod
    def _stat_tile(title):

        tile = QWidget()
        tile_layout = QVBoxLayout(tile)

        value_label = QLabel("-")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = value_label.font()
        font.setPointSizeF(font.pointSizeF() * 2.2)
        font.setBold(True)
        value_label.setFont(font)

        caption_label = QLabel(title)
        caption_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        tile_layout.addWidget(value_label)
        tile_layout.addWidget(caption_label)

        return value_label, tile

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        self._populate_zone_rows(incident_data)
        self._show_ground_truth(incident_data.ground_truth if incident_data is not None else None)

        first_frame = incident_data.frame_at_index(0) if incident_data is not None else None
        self.show_frame(first_frame)

    # =====================================================

    def show_frame(self, frame):

        if frame is None or self._incident is None:

            self.evacuated_value.setText("-")
            self.remaining_value.setText("-")
            self.trapped_value.setText("-")

            self.congestion_label.setText("Current congestion: -")
            self.max_congestion_label.setText("Maximum congestion so far: -")
            self.evacuation_rate_label.setText("Current evacuation rate: -")
            self.queue_length_label.setText("Current queue length: -")
            self.route_utilization_label.setText("Current route utilization: -")
            self.bottleneck_label.setText("Current bottleneck: -")

            for row in range(self.occupancy_table.rowCount()):
                self.occupancy_table.setItem(row, 1, QTableWidgetItem("-"))

            return

        self.evacuated_value.setText(_format_number(frame.people_evacuated))
        self.remaining_value.setText(_format_number(frame.people_remaining))
        self.trapped_value.setText(_format_number(frame.people_trapped))

        self.congestion_label.setText(f"Current congestion: {_format_number(frame.current_congestion)}")
        self.max_congestion_label.setText(
            f"Maximum congestion so far: {_format_number(frame.maximum_congestion_so_far)}",
        )
        self.evacuation_rate_label.setText(
            f"Current evacuation rate: {_format_float(frame.current_evacuation_rate)} people/s",
        )
        self.queue_length_label.setText(
            f"Current queue length: {_format_number(frame.current_queue_length)}",
        )
        self.route_utilization_label.setText(
            f"Current route utilization: {_format_float(frame.current_route_utilization)}",
        )
        self.bottleneck_label.setText(
            f"Current bottleneck: {self._incident.name_for(frame.current_bottleneck)}",
        )

        name = self._incident.name_for
        occupancy_rows = {
            name(zone_id): count for zone_id, count in frame.zone_occupancy.items()
        }

        for row in range(self.occupancy_table.rowCount()):

            zone_name_item = self.occupancy_table.item(row, 0)
            zone_name = zone_name_item.text() if zone_name_item is not None else None
            count = occupancy_rows.get(zone_name)

            self.occupancy_table.setItem(row, 1, QTableWidgetItem(_format_number(count)))

    # =====================================================

    def _populate_zone_rows(self, incident_data):

        self.occupancy_table.setRowCount(0)

        if incident_data is None:
            return

        zones = [
            zone for floor in incident_data.building.ordered_floors() for zone in floor.zones
        ]

        self.occupancy_table.setRowCount(len(zones))
        for row_index, zone in enumerate(zones):
            self.occupancy_table.setItem(row_index, 0, QTableWidgetItem(zone.name))
            self.occupancy_table.setItem(row_index, 1, QTableWidgetItem("-"))

    # =====================================================

    def _show_ground_truth(self, ground_truth):

        if ground_truth is None:

            self.total_evacuation_time_label.setText("Total evacuation time: -")
            self.building_cleared_label.setText("Building cleared: -")
            self.reachable_label.setText("Reachable occupants: -")
            self.unreachable_label.setText("Unreachable occupants: -")
            return

        self.total_evacuation_time_label.setText(
            f"Total evacuation time: {_format_float(ground_truth.total_evacuation_time)}",
        )
        self.building_cleared_label.setText(
            f"Building cleared: {'Yes' if ground_truth.building_cleared else 'No'}",
        )
        self.reachable_label.setText(f"Reachable occupants: {ground_truth.reachable_occupants}")
        self.unreachable_label.setText(f"Unreachable occupants: {ground_truth.unreachable_occupants}")
