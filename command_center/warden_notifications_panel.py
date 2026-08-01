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

from recommendation_layer.models import RecommendationType

from command_center.live_operator_action_gateway import PROVIDER_CAPABILITY_NO_PROVIDER


# =====================================================
# Warden Notification -- Command Center panel. Execution Layer V1
# milestone. Mirrors building_controls_panel.py's own LIVE-mode shape
# exactly (submit-then-decide human approval, APPROVE/REJECT calling
# straight into the injected gateway, re-rendering immediately).
#
# Live-only -- the Recommendation Layer has no Replay/IncidentData
# equivalent (it is recomputed fresh every live cycle, never a stored
# per-frame replay artifact), so this panel, unlike BuildingControlsPanel,
# has no separate Replay-mode section.
# =====================================================


def _format_seconds(value):

    if value is None:
        return "-"

    minutes, seconds = divmod(int(value), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


class WardenNotificationsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._live_recommendation_set = None
        self._live_gateway = None

        layout = QVBoxLayout(self)

        mode_label = QLabel(
            "Notifies a human warden/responder of a zone that needs attention. No real SMS/push/"
            "email/webhook transport exists anywhere in this codebase -- \"Confirmed\" below means "
            "the configured Warden Notification Provider (Simulation only, in this build) recorded "
            "the notification, never that a real person was actually reached."
        )
        mode_label.setWordWrap(True)
        layout.addWidget(mode_label)

        layout.addWidget(QLabel("Pending Recommendations (AWAITING APPROVAL)"))
        self.pending_table = QTableWidget(0, 5)
        self.pending_table.setHorizontalHeaderLabels(["Zone", "Reason", "Confidence", "Status", "Decision"])
        self.pending_table.horizontalHeader().setStretchLastSection(True)
        self.pending_table.verticalHeader().setVisible(False)
        layout.addWidget(self.pending_table, 2)

        layout.addWidget(QLabel("Notification History"))
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(["Time", "Request", "Transition", "Actor"])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.verticalHeader().setVisible(False)
        layout.addWidget(self.history_table, 2)

    # =====================================================

    def show_live(self, recommendation_set, gateway=None) -> None:

        self.pending_table.setRowCount(0)
        self.history_table.setRowCount(0)

        self._live_recommendation_set = recommendation_set
        self._live_gateway = gateway

        capability = gateway.warden_capability if gateway is not None else PROVIDER_CAPABILITY_NO_PROVIDER

        if capability == PROVIDER_CAPABILITY_NO_PROVIDER:

            # Honest fallback -- no controller exists to submit a
            # WardenNotificationRequest to at all, so nothing is
            # submitted; the operator can still review the raw
            # recommendation.
            recommendations = (
                recommendation_set.by_type(RecommendationType.WARDEN_DISPATCH)
                if recommendation_set is not None else ()
            )

            self.pending_table.setRowCount(len(recommendations))

            for row_index, entry in enumerate(recommendations):

                zone_label = ", ".join(entry.affected_zones) or "-"

                self.pending_table.setItem(row_index, 0, QTableWidgetItem(zone_label))
                self.pending_table.setItem(row_index, 1, QTableWidgetItem(entry.recommended_action))
                self.pending_table.setItem(row_index, 2, QTableWidgetItem(_format_percent(entry.confidence)))
                self.pending_table.setItem(row_index, 3, QTableWidgetItem("RECOMMENDED (not submitted)"))
                self.pending_table.setItem(row_index, 4, QTableWidgetItem("Execution Provider: Not Connected"))

            return

        gateway.ingest_warden_recommendations(recommendation_set, recommendation_set.timestamp if recommendation_set is not None else 0.0)

        self._refresh_pending_live(gateway)
        self._refresh_history_live(gateway)

    # =====================================================

    def _refresh_pending_live(self, gateway) -> None:

        pending = gateway.pending_warden_notifications()
        self.pending_table.setRowCount(len(pending))

        for row_index, request in enumerate(pending):

            self.pending_table.setItem(row_index, 0, QTableWidgetItem(request.zone_id or "-"))
            self.pending_table.setItem(row_index, 1, QTableWidgetItem(request.reason))
            self.pending_table.setItem(row_index, 2, QTableWidgetItem(_format_percent(request.confidence)))
            self.pending_table.setItem(row_index, 3, QTableWidgetItem("AWAITING APPROVAL"))

            decision_widget = QWidget()
            decision_layout = QHBoxLayout(decision_widget)
            decision_layout.setContentsMargins(0, 0, 0, 0)

            approve_button = QPushButton("Approve")
            approve_button.clicked.connect(partial(self._on_approve_live, request.request_id))
            decision_layout.addWidget(approve_button)

            reject_button = QPushButton("Reject")
            reject_button.clicked.connect(partial(self._on_reject_live, request.request_id))
            decision_layout.addWidget(reject_button)

            self.pending_table.setCellWidget(row_index, 4, decision_widget)

    # =====================================================

    def _refresh_history_live(self, gateway) -> None:

        events = gateway.warden_notification_history()
        requests_by_id = {request.request_id: request for request in gateway.all_warden_notifications()}

        self.history_table.setRowCount(len(events))

        for row_index, event in enumerate(events):

            request = requests_by_id.get(event.request_id)
            zone_label = (request.zone_id or "-") if request is not None else event.request_id

            from_label = event.from_status.name if event.from_status is not None else "(new)"
            transition = f"zone {zone_label}: {from_label} -> {event.to_status.name}"

            self.history_table.setItem(row_index, 0, QTableWidgetItem(_format_seconds(event.timestamp)))
            self.history_table.setItem(row_index, 1, QTableWidgetItem(event.request_id[:8]))
            self.history_table.setItem(row_index, 2, QTableWidgetItem(transition))
            self.history_table.setItem(row_index, 3, QTableWidgetItem(event.actor))

    # =====================================================

    def _on_approve_live(self, request_id) -> None:

        if self._live_gateway is None:
            return

        self._live_gateway.approve_warden_notification(request_id)
        self.show_live(self._live_recommendation_set, self._live_gateway)

    # =====================================================

    def _on_reject_live(self, request_id) -> None:

        if self._live_gateway is None:
            return

        self._live_gateway.reject_warden_notification(request_id)
        self.show_live(self._live_recommendation_set, self._live_gateway)
