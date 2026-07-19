from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


# =====================================================
# RecommendationTimelinePanel -- Phase 4's "display every recommendation
# ever issued." Reads command_center.incident_data.IncidentData.
# recommendation_history once (the full, already-accumulated
# Tuple[advisory_system.RecommendationChange, ...] the running
# AdvisoryOrchestrator recorded across every frame during IncidentData.
# __post_init__ -- see that module's own docstring) -- nothing here
# recomputes or stages a recommendation change of its own. Same "dumb
# widget, pushed updates only" convention every other Command Center
# panel follows: set_incident() builds the table once, show_frame()
# only ever highlights which already-built rows have happened "so far"
# at the currently displayed frame's time -- it never adds, removes, or
# reorders a row.
# =====================================================

_REACHED_BACKGROUND = QBrush(QColor(45, 62, 50))
_PENDING_BACKGROUND = QBrush(QColor(255, 255, 255, 0))
_PENDING_FOREGROUND = QBrush(QColor(120, 120, 120))
_DEFAULT_FOREGROUND = QBrush(QColor(220, 220, 220))


def _format_seconds(value):

    if value is None:
        return "-"

    minutes, seconds = divmod(int(value), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


def _format_action(action):

    return "-" if not action else str(action).replace("_", " ").title()


class RecommendationTimelinePanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None
        self._row_timestamps = []

        layout = QVBoxLayout(self)

        group = QGroupBox("Recommendation Timeline")
        group_layout = QVBoxLayout(group)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Timestamp", "Zone", "Recommendation Change", "Reason for Change", "Confidence", "Status"],
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        group_layout.addWidget(self.table)

        layout.addWidget(group)

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        self.table.setRowCount(0)
        self._row_timestamps = []

        if incident_data is None:
            return

        history = sorted(incident_data.recommendation_history, key=lambda change: change.timestamp)

        latest_zone_timestamp = {}
        for change in history:
            latest_zone_timestamp[change.zone_id] = change.timestamp

        name = incident_data.name_for

        self.table.setRowCount(len(history))
        for row_index, change in enumerate(history):

            is_current = latest_zone_timestamp.get(change.zone_id) == change.timestamp
            status = "Current" if is_current else "Superseded"

            change_text = (
                f"{_format_action(change.previous_recommendation)} -> "
                f"{_format_action(change.new_recommendation)}"
            )
            confidence_text = (
                f"{_format_percent(change.confidence_before)} -> {_format_percent(change.confidence_after)}"
            )

            self.table.setItem(row_index, 0, QTableWidgetItem(_format_seconds(change.timestamp)))
            self.table.setItem(row_index, 1, QTableWidgetItem(name(change.zone_id)))
            self.table.setItem(row_index, 2, QTableWidgetItem(change_text))
            self.table.setItem(row_index, 3, QTableWidgetItem(change.reason_for_change or "-"))
            self.table.setItem(row_index, 4, QTableWidgetItem(confidence_text))
            self.table.setItem(row_index, 5, QTableWidgetItem(status))

            self._row_timestamps.append(change.timestamp)

        self.show_frame(incident_data.frame_at_index(0))

    # =====================================================

    def show_frame(self, frame):

        current_time = frame.time if frame is not None else 0.0

        for row_index, timestamp in enumerate(self._row_timestamps):

            reached = timestamp <= current_time
            background = _REACHED_BACKGROUND if reached else _PENDING_BACKGROUND
            foreground = _DEFAULT_FOREGROUND if reached else _PENDING_FOREGROUND

            for column in range(self.table.columnCount()):

                item = self.table.item(row_index, column)
                if item is None:
                    continue

                item.setBackground(background)
                item.setForeground(foreground)
