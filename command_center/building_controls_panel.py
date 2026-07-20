from functools import partial

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from building_control.types import RequestStatus


# =====================================================
# Building Control & Safety Systems Integration -- Command Center panel.
#
# The one interactive panel in Command Center: every other panel here
# is a "dumb widget, pushed updates only" replay of already-computed
# data (see recommendation_center.py's own module docstring). This
# panel is different by necessity -- Phase 5's human-in-the-loop
# approval model means a pending ControlRequest cannot be resolved by
# replay alone, an operator decides. APPROVE/REJECT here call straight
# into IncidentData.control_controller (the one BuildingControlController
# shared for this incident's whole viewing session) and re-render
# immediately; nothing about this ever rewrites the incident's own
# recorded simulation history.
#
# Clearly distinguishes, per this milestone's core rule (RECOMMENDATION
# != COMMAND != CONFIRMED PHYSICAL ACTION):
#   Pending Recommendations -- AWAITING_APPROVAL, not yet acted on.
#   Active / Confirmed Controls -- only entries a provider has actually
#     CONFIRMED (BuildingControlController.snapshot()); never shown for
#     a merely dispatched/pending one.
#   Control History -- every status transition, append-only.
# =====================================================


def _format_seconds(value):

    if value is None:
        return "-"

    minutes, seconds = divmod(int(value), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


class BuildingControlsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        mode_label = QLabel(
            "Digital-twin control layer -- no real building hardware is involved. "
            "\"Confirmed\" below means the configured Building Control Provider (Simulation "
            "only, in this build) reported the simulated engineering state changed. Stair "
            "Pressurization / Smoke Exhaust / Deluge confirmations are state-only: this "
            "platform's hazard/fire simulation does not model their physical effect."
        )
        mode_label.setWordWrap(True)
        layout.addWidget(mode_label)

        layout.addWidget(QLabel("Pending Recommendations (AWAITING APPROVAL)"))
        self.pending_table = QTableWidget(0, 7)
        self.pending_table.setHorizontalHeaderLabels(
            ["System", "Target", "Action", "Confidence", "Reason", "Status", "Decision"],
        )
        self.pending_table.horizontalHeader().setStretchLastSection(True)
        self.pending_table.verticalHeader().setVisible(False)
        layout.addWidget(self.pending_table, 2)

        layout.addWidget(QLabel("Active / Confirmed Controls"))
        self.active_table = QTableWidget(0, 4)
        self.active_table.setHorizontalHeaderLabels(["System", "Target", "Action", "Confirmed At"])
        self.active_table.horizontalHeader().setStretchLastSection(True)
        self.active_table.verticalHeader().setVisible(False)
        layout.addWidget(self.active_table, 1)

        layout.addWidget(QLabel("Control History"))
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(["Time", "Request", "Transition", "Actor", "Note"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.verticalHeader().setVisible(False)
        layout.addWidget(self.history_table, 2)

    # =====================================================
    # Live Command Center Integration milestone -- Phase 12's display-
    # only live rendering path. Deliberately does NOT touch
    # self._incident/BuildingControlController/SimulationControlProvider
    # at all -- there is no live BuildingControlProvider in this
    # milestone (Phase 12's own "there currently should not be one"),
    # so nothing here can execute, confirm, or even PENDING_APPROVAL-
    # queue a control action. Every row rendered is directly off
    # AdvisoryReport.building_recommendations -- inert data, never
    # submitted anywhere -- with no Approve/Reject affordance at all
    # (a disabled button would still imply an execution path exists;
    # omitting it entirely is the honest choice Phase 12 asks for).
    # =====================================================

    def show_live(self, report) -> None:

        self.pending_table.setRowCount(0)
        self.active_table.setRowCount(0)
        self.history_table.setRowCount(0)

        recommendations = report.building_recommendations if report is not None else ()

        self.pending_table.setRowCount(len(recommendations))

        for row_index, entry in enumerate(recommendations):

            self.pending_table.setItem(row_index, 0, QTableWidgetItem(entry.target_type))
            self.pending_table.setItem(row_index, 1, QTableWidgetItem(entry.target_id or "-"))
            self.pending_table.setItem(row_index, 2, QTableWidgetItem(entry.action))
            self.pending_table.setItem(row_index, 3, QTableWidgetItem(_format_percent(entry.confidence)))
            self.pending_table.setItem(row_index, 4, QTableWidgetItem(entry.reason))
            self.pending_table.setItem(row_index, 5, QTableWidgetItem("RECOMMENDED (not submitted)"))
            self.pending_table.setItem(row_index, 6, QTableWidgetItem("Execution Provider: Not Connected"))

    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        first_frame = incident_data.frame_at_index(0) if incident_data is not None else None
        first_report = incident_data.advisory_report_at_index(0) if incident_data is not None else None
        self.show_frame(first_frame, first_report)

    # =====================================================

    def show_frame(self, _frame, report):

        if self._incident is not None:
            self._incident.ingest_control_recommendations(report)

        self._refresh()

    # =====================================================

    def _refresh(self):

        self.pending_table.setRowCount(0)
        self.active_table.setRowCount(0)
        self.history_table.setRowCount(0)

        if self._incident is None:
            return

        controller = self._incident.control_controller
        name = self._incident.name_for

        self._refresh_pending(controller, name)
        self._refresh_active(controller, name)
        self._refresh_history(controller, name)

    # =====================================================

    def _refresh_pending(self, controller, name):

        pending = controller.pending_requests()
        self.pending_table.setRowCount(len(pending))

        for row_index, request in enumerate(pending):

            self.pending_table.setItem(row_index, 0, QTableWidgetItem(request.system_type.name))
            self.pending_table.setItem(row_index, 1, QTableWidgetItem(name(request.target_id)))
            self.pending_table.setItem(row_index, 2, QTableWidgetItem(request.requested_action.name))
            self.pending_table.setItem(row_index, 3, QTableWidgetItem(_format_percent(request.confidence)))
            self.pending_table.setItem(row_index, 4, QTableWidgetItem(request.reason))
            self.pending_table.setItem(row_index, 5, QTableWidgetItem("AWAITING APPROVAL"))

            decision_widget = QWidget()
            decision_layout = QHBoxLayout(decision_widget)
            decision_layout.setContentsMargins(0, 0, 0, 0)

            approve_button = QPushButton("Approve")
            approve_button.clicked.connect(partial(self._on_approve, request.request_id))
            decision_layout.addWidget(approve_button)

            reject_button = QPushButton("Reject")
            reject_button.clicked.connect(partial(self._on_reject, request.request_id))
            decision_layout.addWidget(reject_button)

            self.pending_table.setCellWidget(row_index, 6, decision_widget)

    # =====================================================

    def _refresh_active(self, controller, name):

        entries = controller.snapshot().entries
        self.active_table.setRowCount(len(entries))

        for row_index, entry in enumerate(entries):

            self.active_table.setItem(row_index, 0, QTableWidgetItem(entry.system_type.name))
            self.active_table.setItem(row_index, 1, QTableWidgetItem(name(entry.target_id)))
            self.active_table.setItem(row_index, 2, QTableWidgetItem(entry.action.name))
            self.active_table.setItem(row_index, 3, QTableWidgetItem(_format_seconds(entry.confirmed_at)))

    # =====================================================

    def _refresh_history(self, controller, name):

        events = controller.history()
        requests_by_id = {request.request_id: request for request in controller.all_requests()}

        self.history_table.setRowCount(len(events))

        for row_index, event in enumerate(events):

            request = requests_by_id.get(event.request_id)
            target_label = name(request.target_id) if request is not None else event.request_id
            system_label = request.system_type.name if request is not None else "-"

            from_label = event.from_status.name if event.from_status is not None else "(new)"
            transition = f"{system_label} {target_label}: {from_label} -> {event.to_status.name}"

            self.history_table.setItem(row_index, 0, QTableWidgetItem(_format_seconds(event.timestamp)))
            self.history_table.setItem(row_index, 1, QTableWidgetItem(event.request_id[:8]))
            self.history_table.setItem(row_index, 2, QTableWidgetItem(transition))
            self.history_table.setItem(row_index, 3, QTableWidgetItem(event.actor))
            self.history_table.setItem(row_index, 4, QTableWidgetItem(event.note))

    # =====================================================

    def _on_approve(self, request_id):

        if self._incident is None:
            return

        controller = self._incident.control_controller

        if controller.status_of(request_id) != RequestStatus.PENDING_APPROVAL:
            self._refresh()
            return

        controller.approve(request_id, actor="operator")
        self._refresh()

    # =====================================================

    def _on_reject(self, request_id):

        if self._incident is None:
            return

        controller = self._incident.control_controller

        if controller.status_of(request_id) != RequestStatus.PENDING_APPROVAL:
            self._refresh()
            return

        controller.reject(request_id, actor="operator")
        self._refresh()
