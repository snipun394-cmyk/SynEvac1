from functools import partial
from typing import Optional

from PyQt6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from command_center.live_operator_action_gateway import (
    PROVIDER_CAPABILITY_NO_PROVIDER, VOICE_STATUS_RECOMMENDED, VOICE_STATUS_REJECTED, VOICE_STATUS_SUPERSEDED,
)


# =====================================================
# LiveEvacuationGuidancePanel -- Live Evacuation Guidance & Zoned
# Message Planning milestone, Phase 18/19. Display + explicit operator
# Approve/Reject affordance for the PLANNED guidance voice message,
# mirroring command_center.recommendation_center.VoiceEvacuationPanel's
# own established live-mode pattern exactly (same PROVIDER_CAPABILITY_*/
# VOICE_STATUS_* vocabulary, same "gateway injected, never imported
# elsewhere" discipline -- this panel never imports voice_evacuation.
# controller/speaker_manager/evacuation_guidance.engine itself, only
# command_center.live_operator_action_gateway's own already-established
# public constants).
#
# DECISION SUPPORT + EXPLICIT OPERATOR ACTION ONLY -- every Approve
# click routes through LiveOperatorActionGateway.approve_guidance_message(),
# the ONLY seam that can ever reach VoiceEvacuationController. This
# panel never broadcasts, executes a control, or mutates FACP state
# itself.
# =====================================================


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


def _format_distance(value):

    return "-" if value is None else f"{value:.1f} m"


class LiveEvacuationGuidancePanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._snapshot = None
        self._gateway = None
        self._time = 0.0

        layout = QVBoxLayout(self)

        note = QLabel(
            "Decision support only -- a planned route and message, never an executed broadcast. Approving a "
            "message below routes it through the SAME operator-approval path as civilian announcements."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        table_group = QGroupBox("Live Evacuation Guidance")
        table_layout = QVBoxLayout(table_group)
        self.zone_table = QTableWidget(0, 11)
        self.zone_table.setHorizontalHeaderLabels([
            "Zone", "Recommended Exit", "Route Status", "Route Summary", "Required Stair(s)",
            "Route Distance", "Guidance Revision", "Speaker Coverage", "Message Status", "Last Updated", "Decision",
        ])
        self.zone_table.horizontalHeader().setStretchLastSection(True)
        self.zone_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.zone_table)
        layout.addWidget(table_group)

        detail_group = QGroupBox("Selected Zone -- Full Structured Route")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_label = QLabel("-")
        self.detail_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_label)
        layout.addWidget(detail_group)

        self.zone_table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addStretch(1)

    # =====================================================

    def show_guidance(self, snapshot: Optional[object], gateway=None, time: float = 0.0) -> None:

        self._snapshot = snapshot
        self._gateway = gateway
        self._time = time

        if snapshot is None or not snapshot.zones:

            self.zone_table.setRowCount(0)
            self.detail_label.setText("-")
            return

        zone_ids = sorted(snapshot.zones.keys())
        self.zone_table.setRowCount(len(zone_ids))

        for row_index, zone_id in enumerate(zone_ids):

            plan = snapshot.zones[zone_id]
            voice_plan = snapshot.voice_plan(zone_id)

            route_summary = " -> ".join(plan.ordered_zone_ids) if plan.ordered_zone_ids else "-"
            required_stairs = ", ".join(plan.ordered_stair_ids) or "-"
            speaker_coverage = ", ".join(voice_plan.speaker_ids) if voice_plan is not None and voice_plan.speaker_ids else "-"
            message_status = voice_plan.delivery_status if voice_plan is not None else "NOT_PLANNED"

            self.zone_table.setItem(row_index, 0, QTableWidgetItem(zone_id))
            self.zone_table.setItem(row_index, 1, QTableWidgetItem(plan.recommended_exit_id or "-"))
            self.zone_table.setItem(row_index, 2, QTableWidgetItem(plan.route_status))
            self.zone_table.setItem(row_index, 3, QTableWidgetItem(route_summary))
            self.zone_table.setItem(row_index, 4, QTableWidgetItem(required_stairs))
            self.zone_table.setItem(row_index, 5, QTableWidgetItem(_format_distance(plan.route_distance_m)))
            self.zone_table.setItem(row_index, 6, QTableWidgetItem(str(plan.revision)))
            self.zone_table.setItem(row_index, 7, QTableWidgetItem(speaker_coverage))
            self.zone_table.setItem(row_index, 8, QTableWidgetItem(message_status))
            self.zone_table.setItem(row_index, 9, QTableWidgetItem(f"{plan.evidence_timestamp:.1f}"))

            self._set_decision_cell(row_index, voice_plan)

    # =====================================================

    def _set_decision_cell(self, row_index, voice_plan) -> None:

        if voice_plan is None:

            self.zone_table.setItem(row_index, 10, QTableWidgetItem("No valid route -- nothing to send"))
            return

        status = (
            self._gateway.guidance_recommendation_status(voice_plan) if self._gateway is not None
            else VOICE_STATUS_RECOMMENDED
        )

        if status != VOICE_STATUS_RECOMMENDED:

            label = {
                VOICE_STATUS_REJECTED: "Rejected by operator",
                VOICE_STATUS_SUPERSEDED: "Superseded",
            }.get(status, status)

            self.zone_table.setItem(row_index, 10, QTableWidgetItem(label))
            return

        capability = self._gateway.voice_capability if self._gateway is not None else PROVIDER_CAPABILITY_NO_PROVIDER

        widget = QWidget()
        decision_layout = QHBoxLayout(widget)
        decision_layout.setContentsMargins(0, 0, 0, 0)

        approve_button = QPushButton("Approve / Send")
        approve_button.setEnabled(self._gateway is not None and capability != PROVIDER_CAPABILITY_NO_PROVIDER)
        approve_button.clicked.connect(partial(self._on_approve, voice_plan))
        decision_layout.addWidget(approve_button)

        reject_button = QPushButton("Reject")
        reject_button.setEnabled(self._gateway is not None)
        reject_button.clicked.connect(partial(self._on_reject, voice_plan))
        decision_layout.addWidget(reject_button)

        self.zone_table.setCellWidget(row_index, 10, widget)

    # =====================================================

    def _on_approve(self, voice_plan) -> None:

        if self._gateway is None:
            return

        self._gateway.approve_guidance_message(voice_plan, self._time)
        self.show_guidance(self._snapshot, self._gateway, self._time)

    # =====================================================

    def _on_reject(self, voice_plan) -> None:

        if self._gateway is None:
            return

        self._gateway.reject_guidance_message(voice_plan, self._time)
        self.show_guidance(self._snapshot, self._gateway, self._time)

    # =====================================================

    def _on_selection_changed(self):

        selected_items = self.zone_table.selectedItems()

        if not selected_items or self._snapshot is None:
            self.detail_label.setText("-")
            return

        row_index = selected_items[0].row()
        zone_id = self.zone_table.item(row_index, 0).text()
        plan = self._snapshot.zone(zone_id)

        if plan is None:
            self.detail_label.setText("-")
            return

        self.detail_label.setText(_format_plan_detail(plan))


# =====================================================


def _format_plan_detail(plan) -> str:

    lines = [f"{plan.zone_id} -- {plan.route_status} (revision {plan.revision})"]

    if plan.instructions:

        lines.append("Instructions:")
        for instruction in plan.instructions:
            lines.append(f"  - {instruction}")

    else:
        lines.append("No instructions available.")

    if plan.inconsistencies:

        lines.append("Diagnostics:")
        for code in plan.inconsistencies:
            lines.append(f"  - {code}")

    return "\n".join(lines)
