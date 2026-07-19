from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# =====================================================
# HazardPanel -- fire location, smoke spread, hazard growth summary,
# and risk levels. Dumb widget, same convention every other Command
# Center panel follows: set_incident() is called once per loaded
# incident (static GroundTruth-derived summary + risk tables) and
# show_frame() on every playback tick (current fire zones + smoke
# spread at that moment). No domain logic beyond formatting lives
# here -- every value is read straight off IncidentData/IncidentFrame/
# GroundTruth.
# =====================================================


class HazardPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        fire_group = QGroupBox("Current Fire Location")
        fire_layout = QVBoxLayout(fire_group)
        self.ignition_label = self._label("Ignition zone: -")
        self.current_fire_label = self._label("Currently on fire: -")
        fire_layout.addWidget(self.ignition_label)
        fire_layout.addWidget(self.current_fire_label)
        layout.addWidget(fire_group)

        smoke_group = QGroupBox("Smoke Spread")
        smoke_layout = QVBoxLayout(smoke_group)
        self.smoke_table = QTableWidget(0, 2)
        self.smoke_table.setHorizontalHeaderLabels(["Zone", "Smoke Level"])
        self.smoke_table.horizontalHeader().setStretchLastSection(True)
        self.smoke_table.verticalHeader().setVisible(False)
        smoke_layout.addWidget(self.smoke_table)
        layout.addWidget(smoke_group)

        growth_group = QGroupBox("Hazard Growth Summary (Ground Truth)")
        growth_layout = QVBoxLayout(growth_group)
        self.first_hazardous_label = self._label("First hazardous zone: -")
        self.spread_order_label = self._label("Hazard spread order: -")
        self.maximum_hazard_label = self._label("Maximum hazard zone: -")
        self.highest_risk_floor_label = self._label("Highest risk floor: -")
        for label in (
            self.first_hazardous_label,
            self.spread_order_label,
            self.maximum_hazard_label,
            self.highest_risk_floor_label,
        ):
            growth_layout.addWidget(label)
        layout.addWidget(growth_group)

        risk_group = QGroupBox("Risk Levels (Ground Truth)")
        risk_layout = QVBoxLayout(risk_group)
        self.risk_table = QTableWidget(0, 3)
        self.risk_table.setHorizontalHeaderLabels(["Object", "Type", "Risk Score"])
        self.risk_table.horizontalHeader().setStretchLastSection(True)
        self.risk_table.verticalHeader().setVisible(False)
        risk_layout.addWidget(self.risk_table)
        layout.addWidget(risk_group)

        layout.addStretch(1)

    # =====================================================

    @staticmethod
    def _label(text):

        label = QLabel(text)
        label.setWordWrap(True)
        return label

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        ground_truth = incident_data.ground_truth if incident_data is not None else None
        self._show_ground_truth(ground_truth)

        first_frame = incident_data.frame_at_index(0) if incident_data is not None else None
        self.show_frame(first_frame)

    # =====================================================

    def show_frame(self, frame):

        if frame is None or self._incident is None:

            self.ignition_label.setText("Ignition zone: -")
            self.current_fire_label.setText("Currently on fire: -")
            self.smoke_table.setRowCount(0)
            return

        name = self._incident.name_for

        self.ignition_label.setText(f"Ignition zone: {name(frame.ignition_zone_id)}")

        fire_zones = ", ".join(name(zone_id) for zone_id in sorted(frame.current_fire_zone_ids))
        self.current_fire_label.setText(f"Currently on fire: {fire_zones or 'None'}")

        smoke_rows = sorted(
            (
                (name(zone_id), level)
                for zone_id, level in frame.zone_smoke.items()
                if level is not None and level > 0
            ),
            key=lambda entry: entry[1],
            reverse=True,
        )

        self.smoke_table.setRowCount(len(smoke_rows))
        for row_index, (label, level) in enumerate(smoke_rows):
            self.smoke_table.setItem(row_index, 0, QTableWidgetItem(label))
            self.smoke_table.setItem(row_index, 1, QTableWidgetItem(f"{level:.2f}"))

    # =====================================================

    def _show_ground_truth(self, ground_truth):

        if ground_truth is None or self._incident is None:

            self.first_hazardous_label.setText("First hazardous zone: -")
            self.spread_order_label.setText("Hazard spread order: -")
            self.maximum_hazard_label.setText("Maximum hazard zone: -")
            self.highest_risk_floor_label.setText("Highest risk floor: -")
            self.risk_table.setRowCount(0)
            return

        name = self._incident.name_for

        self.first_hazardous_label.setText(
            f"First hazardous zone: {name(ground_truth.first_hazardous_zone)}",
        )

        spread = " -> ".join(name(zone_id) for zone_id in ground_truth.hazard_spread_order)
        self.spread_order_label.setText(f"Hazard spread order: {spread or '-'}")

        self.maximum_hazard_label.setText(
            f"Maximum hazard zone: {name(ground_truth.maximum_hazard_zone)}",
        )
        self.highest_risk_floor_label.setText(
            f"Highest risk floor: {name(ground_truth.highest_risk_floor)}",
        )

        rows = []
        for entry in ground_truth.zone_risk_scores:
            rows.append((name(entry["zone_id"]), "Zone", entry.get("risk_score")))
        for entry in ground_truth.exit_risk_scores:
            rows.append((name(entry["exit_id"]), "Exit", entry.get("risk_score")))
        for entry in ground_truth.stair_risk_scores:
            rows.append((name(entry["stair_id"]), "Stair", entry.get("risk_score")))

        self.risk_table.setRowCount(len(rows))
        for row_index, (label, kind, score) in enumerate(rows):
            self.risk_table.setItem(row_index, 0, QTableWidgetItem(label))
            self.risk_table.setItem(row_index, 1, QTableWidgetItem(kind))
            score_text = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
            self.risk_table.setItem(row_index, 2, QTableWidgetItem(score_text))
