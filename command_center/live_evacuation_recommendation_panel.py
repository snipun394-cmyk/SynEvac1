from typing import Optional

from PyQt6.QtWidgets import QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


# =====================================================
# LiveEvacuationRecommendationPanel -- Live Dynamic Evacuation
# Recommendation Engine milestone, Phase 12. Display only: no operator
# control of any kind, no automatic voice broadcast, no automatic
# building control -- every value shown here is read straight off an
# already-computed evacuation_recommendation.models.
# EvacuationRecommendationSnapshot, the same "dumb widget, pushed
# updates only" convention every other Command Center Live panel
# already follows (see LiveMovementIntelligencePanel/
# LiveEmergencyResponsePanel). Live-only; harmlessly empty in Replay
# mode.
#
# DECISION SUPPORT ONLY -- this is a ranked, explainable recommendation,
# never an executed evacuation order.
# =====================================================


def _format_confidence(value):

    return "-" if value is None else f"{value:.0%}"


def _format_coverage(value):

    return "-" if value is None else f"{value:.0%}"


class LiveEvacuationRecommendationPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        note = QLabel(
            "Decision support only -- a ranked, explainable exit recommendation, never an executed evacuation "
            "order. This panel does not broadcast voice messages or execute building controls."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        table_group = QGroupBox("Live Evacuation Recommendations")
        table_layout = QVBoxLayout(table_group)
        self.zone_table = QTableWidget(0, 7)
        self.zone_table.setHorizontalHeaderLabels(
            ["Zone", "Recommended Exit", "Alternative Exits", "Reason", "Confidence", "Coverage", "Last Updated"]
        )
        self.zone_table.horizontalHeader().setStretchLastSection(True)
        self.zone_table.verticalHeader().setVisible(False)
        table_layout.addWidget(self.zone_table)
        layout.addWidget(table_group)

        detail_group = QGroupBox("Selected Zone -- Full Explanation")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_label = QLabel("-")
        self.detail_label.setWordWrap(True)
        detail_layout.addWidget(self.detail_label)
        layout.addWidget(detail_group)

        self.zone_table.itemSelectionChanged.connect(self._on_selection_changed)

        layout.addStretch(1)

    # =====================================================

    def show_recommendations(self, snapshot: Optional[object]) -> None:

        self._snapshot = snapshot

        if snapshot is None or not snapshot.zones:

            self.zone_table.setRowCount(0)
            self.detail_label.setText("-")
            return

        zone_ids = sorted(snapshot.zones.keys())
        self.zone_table.setRowCount(len(zone_ids))

        for row_index, zone_id in enumerate(zone_ids):

            recommendation = snapshot.zones[zone_id]

            reason_summary = recommendation.reason_codes[0] if recommendation.reason_codes else "-"

            self.zone_table.setItem(row_index, 0, QTableWidgetItem(zone_id))
            self.zone_table.setItem(row_index, 1, QTableWidgetItem(recommendation.recommended_exit_id or "-"))
            self.zone_table.setItem(row_index, 2, QTableWidgetItem(", ".join(recommendation.alternative_exit_ids) or "-"))
            self.zone_table.setItem(row_index, 3, QTableWidgetItem(reason_summary))
            self.zone_table.setItem(row_index, 4, QTableWidgetItem(_format_confidence(recommendation.confidence)))
            self.zone_table.setItem(row_index, 5, QTableWidgetItem(_format_coverage(recommendation.coverage_fraction)))
            self.zone_table.setItem(row_index, 6, QTableWidgetItem(f"{recommendation.timestamp:.1f}"))

    # =====================================================

    def _on_selection_changed(self):

        selected_items = self.zone_table.selectedItems()

        if not selected_items or getattr(self, "_snapshot", None) is None:
            self.detail_label.setText("-")
            return

        row_index = selected_items[0].row()
        zone_id = self.zone_table.item(row_index, 0).text()
        recommendation = self._snapshot.zone(zone_id)

        if recommendation is None:
            self.detail_label.setText("-")
            return

        self.detail_label.setText(recommendation.explanation)
