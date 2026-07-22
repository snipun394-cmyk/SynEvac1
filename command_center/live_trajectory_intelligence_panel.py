from typing import Optional

from PyQt6.QtWidgets import QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


# =====================================================
# LiveMovementIntelligencePanel -- Live Occupant Trajectory, Movement
# Anomaly & Route-Deviation Intelligence milestone, Phase 24. Display
# only: no operator control of any kind -- every value shown here is
# read straight off an already-computed trajectory_intelligence.models.
# TrajectoryIntelligenceSnapshot, the same "dumb widget, pushed updates
# only" convention every other Command Center Live panel already
# follows (see LiveEmergencyResponsePanel/LiveEvacuationProgressPanel).
# Live-only; harmlessly empty in Replay mode.
#
# This is DECISION SUPPORT ONLY, geometry only -- the panel never
# displays "PANICKING"/"CONFUSED"/"DISOBEDIENT" or any other
# psychological/compliance label; only the neutral, deterministic
# movement-status/route-progress/anomaly vocabulary trajectory_
# intelligence.models itself defines (Phase 24's own explicit
# requirement).
# =====================================================


def _format_distance(value):

    return "-" if value is None else f"{value:.1f} m"


class LiveMovementIntelligencePanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        note = QLabel(
            "Decision support only -- movement geometry, not a judgment of intent. This panel does not "
            "dispatch personnel, broadcast voice messages, or execute building controls."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        table_group = QGroupBox("Live Movement Intelligence")
        table_layout = QVBoxLayout(table_group)
        self.occupant_table = QTableWidget(0, 8)
        self.occupant_table.setHorizontalHeaderLabels(
            ["Occupant", "Floor", "Zone", "Movement", "Route Progress", "Route Distance", "Anomaly", "Position"]
        )
        self.occupant_table.horizontalHeader().setStretchLastSection(True)
        self.occupant_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.occupant_table)
        layout.addWidget(table_group)

        detail_group = QGroupBox("Selected Occupant -- Evidence")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_label = QLabel("-")
        self.detail_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_label)
        layout.addWidget(detail_group)

        self.occupant_table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addStretch(1)

    # =====================================================

    def show_trajectory_intelligence(self, snapshot: Optional[object]) -> None:

        self._snapshot = snapshot

        if snapshot is None or not snapshot.occupants:

            self.occupant_table.setRowCount(0)
            self.detail_label.setText("-")
            return

        occupant_ids = sorted(snapshot.occupants.keys())
        self.occupant_table.setRowCount(len(occupant_ids))

        for row_index, occupant_id in enumerate(occupant_ids):

            result = snapshot.occupants[occupant_id]

            self.occupant_table.setItem(row_index, 0, QTableWidgetItem(occupant_id))
            self.occupant_table.setItem(row_index, 1, QTableWidgetItem(result.floor_id or "-"))
            self.occupant_table.setItem(row_index, 2, QTableWidgetItem(result.zone_id or "-"))
            self.occupant_table.setItem(row_index, 3, QTableWidgetItem(result.movement_status))
            self.occupant_table.setItem(row_index, 4, QTableWidgetItem(result.route_progress_status))
            self.occupant_table.setItem(row_index, 5, QTableWidgetItem(_format_distance(result.route_distance_m)))
            self.occupant_table.setItem(row_index, 6, QTableWidgetItem(result.anomaly_severity))
            self.occupant_table.setItem(
                row_index, 7, QTableWidgetItem("Available" if result.position_available else "Unavailable"),
            )

    # =====================================================

    def _on_selection_changed(self):

        selected_items = self.occupant_table.selectedItems()

        if not selected_items or getattr(self, "_snapshot", None) is None:
            self.detail_label.setText("-")
            return

        row_index = selected_items[0].row()
        occupant_id = self.occupant_table.item(row_index, 0).text()
        result = self._snapshot.occupant(occupant_id)

        if result is None:
            self.detail_label.setText("-")
            return

        self.detail_label.setText(_format_occupant_detail(result))


# =====================================================


def _format_occupant_detail(result) -> str:

    lines = [
        f"{result.occupant_id} -- {result.movement_status}, {result.route_progress_status}",
        f"Nearest safe exit: {result.nearest_safe_exit_id or 'unknown'}",
        f"Route distance: {_format_distance(result.route_distance_m)} (trend: {result.route_distance_trend})",
        f"Anomaly severity: {result.anomaly_severity}",
    ]

    if result.anomaly_flags:

        lines.append("Evidence:")
        for flag in result.anomaly_flags:
            lines.append(f"  - {flag}")

    else:
        lines.append("Evidence: none currently observed.")

    if result.stale:
        lines.append("(stale -- no recent observation)")

    return "\n".join(lines)
