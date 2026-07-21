from typing import Optional

from PyQt6.QtWidgets import QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


# =====================================================
# LiveEmergencyResponsePanel -- Live Emergency Response & Rescue
# Priority Intelligence milestone, Phase 21. Display only: no operator
# control of any kind, no automatic dispatch -- every value shown here
# is read straight off an already-computed emergency_response.models.
# EmergencyResponseSnapshot, the same "dumb widget, pushed updates
# only" convention every other Command Center Live panel already
# follows (see LiveAIPanel/LiveEvacuationProgressPanel). Live-only;
# harmlessly empty in Replay mode.
#
# This is DECISION SUPPORT ONLY -- the panel is deliberately titled and
# worded so an operator never mistakes the response priority queue
# below for an automatic firefighter dispatch order (Phase 21's own
# explicit "make clear this is decision support ... do not call it
# automatic firefighter dispatch").
# =====================================================


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


def _format_score(value):

    return "-" if value is None else f"{value:.2f}"


class LiveEmergencyResponsePanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        note = QLabel(
            "Decision support only -- this panel does not dispatch personnel, broadcast voice "
            "messages, or execute building controls."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        queue_group = QGroupBox("Response Priority Queue (highest first)")
        queue_layout = QVBoxLayout(queue_group)
        self.queue_table = QTableWidget(0, 8)
        self.queue_table.setHorizontalHeaderLabels(
            ["Rank", "Priority", "Zone", "Floor", "Occupants", "Assistance", "Clearance", "Hazard"]
        )
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_table.verticalHeader().setVisible(False)
        queue_layout.addWidget(self.queue_table)
        layout.addWidget(queue_group)

        detail_group = QGroupBox("Selected Zone -- Reasons")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_label = QLabel("-")
        self.detail_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_label)
        layout.addWidget(detail_group)

        self.queue_table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addStretch(1)

    # =====================================================

    def show_response(self, snapshot: Optional[object]) -> None:

        self._snapshot = snapshot

        if snapshot is None or not snapshot.response_priority_order:

            self.queue_table.setRowCount(0)
            self.detail_label.setText("-")
            return

        order = snapshot.response_priority_order
        self.queue_table.setRowCount(len(order))

        for row_index, zone_id in enumerate(order):

            priority = snapshot.zone(zone_id)

            self.queue_table.setItem(row_index, 0, QTableWidgetItem(str(row_index + 1)))
            self.queue_table.setItem(row_index, 1, QTableWidgetItem(priority.priority_level))
            self.queue_table.setItem(row_index, 2, QTableWidgetItem(zone_id))
            self.queue_table.setItem(row_index, 3, QTableWidgetItem(priority.floor_id or "-"))
            self.queue_table.setItem(row_index, 4, QTableWidgetItem(str(priority.known_occupant_count)))
            self.queue_table.setItem(
                row_index, 5,
                QTableWidgetItem(str(priority.possible_assistance_count + priority.confirmed_assistance_count)),
            )
            self.queue_table.setItem(row_index, 6, QTableWidgetItem(priority.clearance_status or "-"))
            self.queue_table.setItem(row_index, 7, QTableWidgetItem(priority.hazard_severity or "-"))

    # =====================================================

    def _on_selection_changed(self):

        selected_items = self.queue_table.selectedItems()

        if not selected_items or getattr(self, "_snapshot", None) is None:
            self.detail_label.setText("-")
            return

        row_index = selected_items[0].row()
        zone_id = self.queue_table.item(row_index, 2).text()
        priority = self._snapshot.zone(zone_id)

        if priority is None:
            self.detail_label.setText("-")
            return

        self.detail_label.setText(priority.explanation)
