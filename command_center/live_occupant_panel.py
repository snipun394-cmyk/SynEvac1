from typing import Optional

from PyQt6.QtWidgets import QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


# =====================================================
# LiveOccupantPanel -- Expose Real Live Camera Occupant State In
# SynEvac UI milestone. Dumb widget, display only: no operator control,
# no tracking/fusion/lifecycle logic of its own -- every value shown
# here is read straight off an already-computed live_occupants.models.
# LiveOccupantsSnapshot, the same "pushed updates only" convention
# every other Command Center Live panel already follows (see
# LiveMovementIntelligencePanel/LiveEmergencyResponsePanel). Live-only;
# harmlessly empty in Replay mode.
#
# ANONYMOUS TRACKING, NOT IDENTITY -- the occupant_id shown here is
# whatever live_occupants.manager.LiveOccupantManager already resolved
# it to be (today: SimulationIdentityResolver passing a
# SimpleSingleCameraTracker track_id straight through, e.g.
# "<camera_id>-T3") -- never a name, never a known person. The note
# label below states this explicitly so an operator never mistakes an
# anonymous track for a confirmed identity.
#
# Headcount ("Currently Present") is read directly off
# LiveOccupantsSnapshot.current_occupant_count -- ALWAYS the value
# live_occupants.manager.LiveOccupantManager.canonical_occupancy()
# already computed (NEW/ACTIVE occupants only). This panel never counts
# len(snapshot.occupants) itself (that would include historical/
# TEMPORARILY_LOST/EXITED tracks and could overcount a single physical
# person who churned through multiple track lifecycles -- see this
# milestone's own predecessor audit, SINGLE_PERSON_TRACK_CHURN_
# HEADCOUNT_SAFE).
# =====================================================


def _format_confidence(confidence) -> str:

    return "-" if confidence is None else f"{confidence:.0%}"


def _format_optional(value) -> str:

    return "-" if value is None else str(value)


class LiveOccupantPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        note = QLabel(
            "Anonymous tracking only -- occupant/track IDs are camera-local tracker identities, "
            "never a known person's name or identity. Historical (temporarily lost / exited) tracks "
            "are shown for lifecycle visibility but are excluded from Currently Present."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        self.currently_present_label = QLabel("Currently Present: -")
        font = self.currently_present_label.font()
        font.setBold(True)
        self.currently_present_label.setFont(font)
        layout.addWidget(self.currently_present_label)

        table_group = QGroupBox("Live Occupants / Tracks")
        table_layout = QVBoxLayout(table_group)
        self.occupant_table = QTableWidget(0, 6)
        self.occupant_table.setHorizontalHeaderLabels(
            ["Occupant/Track ID", "Camera", "Status", "Confidence", "Zone", "Floor"],
        )
        self.occupant_table.horizontalHeader().setStretchLastSection(True)
        self.occupant_table.verticalHeader().setVisible(False)
        self.occupant_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table_layout.addWidget(self.occupant_table)
        layout.addWidget(table_group)

        layout.addStretch(1)

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def show_live_occupants(self, snapshot: Optional[object]) -> None:

        if snapshot is None or not snapshot.occupants:

            self.currently_present_label.setText("Currently Present: -" if snapshot is None else "Currently Present: 0")
            self.occupant_table.setRowCount(0)
            return

        self.currently_present_label.setText(f"Currently Present: {snapshot.current_occupant_count}")

        occupants = sorted(snapshot.occupants, key=lambda occupant: occupant.occupant_id)
        self.occupant_table.setRowCount(len(occupants))

        for row_index, occupant in enumerate(occupants):

            self.occupant_table.setItem(row_index, 0, QTableWidgetItem(occupant.occupant_id))
            self.occupant_table.setItem(row_index, 1, QTableWidgetItem(_format_optional(occupant.current_camera_id)))
            self.occupant_table.setItem(row_index, 2, QTableWidgetItem(occupant.status.name))
            self.occupant_table.setItem(row_index, 3, QTableWidgetItem(_format_confidence(occupant.confidence)))
            self.occupant_table.setItem(row_index, 4, QTableWidgetItem(_format_optional(occupant.current_zone_id)))
            self.occupant_table.setItem(row_index, 5, QTableWidgetItem(_format_optional(occupant.current_floor_id)))
