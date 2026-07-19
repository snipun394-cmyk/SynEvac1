from PyQt6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# =====================================================
# RecommendationPanel -- displays decision_policy.DecisionPolicy in
# full: per-zone/exit/stair recommendations, announcements, rescue
# priorities/order, and firefighter deployment priority. DecisionPolicy
# is a whole-run, already-computed artifact (not per-tick), so this
# panel only ever needs set_incident() -- there is no show_frame().
# Dumb widget: no ranking/formatting decision here beyond turning
# already-decided plain dicts into table rows.
# =====================================================


class RecommendationPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        zone_group = QGroupBox("Recommended Zone Actions")
        zone_layout = QVBoxLayout(zone_group)
        self.zone_table = QTableWidget(0, 4)
        self.zone_table.setHorizontalHeaderLabels(
            ["Zone", "Action", "Recommended Exit", "Recommended Stair"],
        )
        self.zone_table.horizontalHeader().setStretchLastSection(True)
        self.zone_table.verticalHeader().setVisible(False)
        zone_layout.addWidget(self.zone_table)
        layout.addWidget(zone_group)

        exit_stair_group = QGroupBox("Recommended Exits and Stairs")
        exit_stair_layout = QVBoxLayout(exit_stair_group)
        self.exit_table = QTableWidget(0, 2)
        self.exit_table.setHorizontalHeaderLabels(["Exit", "Status"])
        self.exit_table.horizontalHeader().setStretchLastSection(True)
        self.exit_table.verticalHeader().setVisible(False)
        self.stair_table = QTableWidget(0, 2)
        self.stair_table.setHorizontalHeaderLabels(["Stair", "Status"])
        self.stair_table.horizontalHeader().setStretchLastSection(True)
        self.stair_table.verticalHeader().setVisible(False)
        exit_stair_layout.addWidget(self.exit_table)
        exit_stair_layout.addWidget(self.stair_table)
        layout.addWidget(exit_stair_group)

        announcement_group = QGroupBox("Recommended Announcements")
        announcement_layout = QVBoxLayout(announcement_group)
        self.announcement_table = QTableWidget(0, 3)
        self.announcement_table.setHorizontalHeaderLabels(["Priority", "Message", "Target Zones"])
        self.announcement_table.horizontalHeader().setStretchLastSection(True)
        self.announcement_table.verticalHeader().setVisible(False)
        announcement_layout.addWidget(self.announcement_table)
        layout.addWidget(announcement_group)

        rescue_group = QGroupBox("Recommended Rescue Priorities")
        rescue_layout = QVBoxLayout(rescue_group)
        self.rescue_order_label = QLabel("Rescue order: -")
        self.rescue_order_label.setWordWrap(True)
        self.rescue_table = QTableWidget(0, 3)
        self.rescue_table.setHorizontalHeaderLabels(["Zone", "Priority", "Occupants"])
        self.rescue_table.horizontalHeader().setStretchLastSection(True)
        self.rescue_table.verticalHeader().setVisible(False)
        rescue_layout.addWidget(self.rescue_order_label)
        rescue_layout.addWidget(self.rescue_table)
        layout.addWidget(rescue_group)

        firefighter_group = QGroupBox("Recommended Firefighter Deployment")
        firefighter_layout = QVBoxLayout(firefighter_group)
        self.firefighter_table = QTableWidget(0, 4)
        self.firefighter_table.setHorizontalHeaderLabels(["Rank", "Target Type", "Target", "Reason"])
        self.firefighter_table.horizontalHeader().setStretchLastSection(True)
        self.firefighter_table.verticalHeader().setVisible(False)
        firefighter_layout.addWidget(self.firefighter_table)
        layout.addWidget(firefighter_group)

        layout.addStretch(1)

    # =====================================================
    # Public API -- pushed updates only. DecisionPolicy is a whole-run
    # artifact, so a single set_incident() call is all this panel ever
    # needs.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        policy = incident_data.decision_policy if incident_data is not None else None
        name = incident_data.name_for if incident_data is not None else (lambda object_id: object_id or "-")

        if policy is None:

            for table in (
                self.zone_table,
                self.exit_table,
                self.stair_table,
                self.announcement_table,
                self.rescue_table,
                self.firefighter_table,
            ):
                table.setRowCount(0)

            self.rescue_order_label.setText("Rescue order: -")
            return

        self._fill_zone_table(policy, name)
        self._fill_exit_table(policy, name)
        self._fill_stair_table(policy, name)
        self._fill_announcement_table(policy, name)
        self._fill_rescue(policy, name)
        self._fill_firefighter_table(policy, name)

    # =====================================================

    def _fill_zone_table(self, policy, name):

        self.zone_table.setRowCount(len(policy.zone_decisions))

        for row_index, entry in enumerate(policy.zone_decisions):

            self.zone_table.setItem(row_index, 0, QTableWidgetItem(name(entry["zone_id"])))
            self.zone_table.setItem(row_index, 1, QTableWidgetItem(entry["action"]))
            self.zone_table.setItem(
                row_index, 2, QTableWidgetItem(name(entry.get("recommended_exit"))),
            )
            self.zone_table.setItem(
                row_index, 3, QTableWidgetItem(name(entry.get("recommended_stair"))),
            )

    # =====================================================

    def _fill_exit_table(self, policy, name):

        self.exit_table.setRowCount(len(policy.exit_decisions))

        for row_index, entry in enumerate(policy.exit_decisions):
            self.exit_table.setItem(row_index, 0, QTableWidgetItem(name(entry["exit_id"])))
            self.exit_table.setItem(row_index, 1, QTableWidgetItem(entry["status"]))

    # =====================================================

    def _fill_stair_table(self, policy, name):

        self.stair_table.setRowCount(len(policy.stair_decisions))

        for row_index, entry in enumerate(policy.stair_decisions):
            self.stair_table.setItem(row_index, 0, QTableWidgetItem(name(entry["stair_id"])))
            self.stair_table.setItem(row_index, 1, QTableWidgetItem(entry["status"]))

    # =====================================================

    def _fill_announcement_table(self, policy, name):

        self.announcement_table.setRowCount(len(policy.announcements))

        for row_index, entry in enumerate(policy.announcements):

            target_zones = ", ".join(name(zone_id) for zone_id in entry.get("target_zones", []))

            self.announcement_table.setItem(row_index, 0, QTableWidgetItem(entry["priority"]))
            self.announcement_table.setItem(row_index, 1, QTableWidgetItem(entry["message"]))
            self.announcement_table.setItem(row_index, 2, QTableWidgetItem(target_zones))

    # =====================================================

    def _fill_rescue(self, policy, name):

        order_text = " -> ".join(name(zone_id) for zone_id in policy.rescue_order)
        self.rescue_order_label.setText(f"Rescue order: {order_text or '-'}")

        priorities = [
            entry for entry in policy.rescue_priorities if entry["rescue_priority"] != "NONE"
        ]

        self.rescue_table.setRowCount(len(priorities))
        for row_index, entry in enumerate(priorities):
            self.rescue_table.setItem(row_index, 0, QTableWidgetItem(name(entry["zone_id"])))
            self.rescue_table.setItem(row_index, 1, QTableWidgetItem(entry["rescue_priority"]))
            self.rescue_table.setItem(
                row_index, 2, QTableWidgetItem(str(entry["occupant_count"])),
            )

    # =====================================================

    def _fill_firefighter_table(self, policy, name):

        entries = policy.firefighter_deployment_priority

        self.firefighter_table.setRowCount(len(entries))
        for row_index, entry in enumerate(entries):
            self.firefighter_table.setItem(row_index, 0, QTableWidgetItem(str(entry["rank"])))
            self.firefighter_table.setItem(row_index, 1, QTableWidgetItem(entry["target_type"]))
            self.firefighter_table.setItem(row_index, 2, QTableWidgetItem(name(entry["target_id"])))
            self.firefighter_table.setItem(row_index, 3, QTableWidgetItem(entry["reason"]))
