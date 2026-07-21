from typing import Optional

from PyQt6.QtWidgets import QGroupBox, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget


# =====================================================
# LiveEvacuationProgressPanel -- Live Evacuation Progress, Flow &
# Clearance Intelligence milestone, Phase 17. Display only: no operator
# control of any kind, no automatic action -- every value shown here is
# read straight off an already-computed evacuation_progress.models.
# EvacuationProgressSnapshot, the same "dumb widget, pushed updates
# only" convention every other Command Center Live panel already
# follows (see LiveAIPanel/LiveStatusPanel). Live-only; harmlessly
# empty in Replay mode.
#
# Wording is deliberately explicit throughout that every number here is
# OBSERVED/TRACKED, never the true building population (Phase 17's own
# explicit "never display '82% of building evacuated' unless total-
# building population is actually known" requirement) -- this platform
# never claims to know that number.
# =====================================================


def _format_percent(value):

    return "-" if value is None else f"{value:.0%}"


def _format_rate(value):

    return "-" if value is None else f"{value:.1f}/min"


def _dash_join(values):

    return ", ".join(values) if values else "-"


class LiveEvacuationProgressPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        overview_group = QGroupBox("Observed Evacuation Progress")
        overview_layout = QVBoxLayout(overview_group)
        self.active_label = QLabel("Observed active occupants: -")
        self.exited_label = QLabel("Observed exited occupants: -")
        self.progress_label = QLabel("Observed progress: -")
        self.trend_label = QLabel("Overall progress trend: -")
        self.coverage_label = QLabel("Position coverage: -")
        for label in (self.active_label, self.exited_label, self.progress_label, self.trend_label, self.coverage_label):
            overview_layout.addWidget(label)
        layout.addWidget(overview_group)

        zone_group = QGroupBox("Zone Clearance")
        zone_layout = QVBoxLayout(zone_group)
        self.zone_table = QTableWidget(0, 5)
        self.zone_table.setHorizontalHeaderLabels(["Zone", "Status", "Active", "Clearance", "Trend"])
        self.zone_table.horizontalHeader().setStretchLastSection(True)
        self.zone_table.verticalHeader().setVisible(False)
        zone_layout.addWidget(self.zone_table)
        layout.addWidget(zone_group)

        exit_group = QGroupBox("Exit Flow -- Queue vs. Throughput")
        exit_layout = QVBoxLayout(exit_group)
        self.exit_table = QTableWidget(0, 6)
        self.exit_table.setHorizontalHeaderLabels(["Exit", "Queue", "Approaching", "Flow Rate", "Active", "Trend"])
        self.exit_table.horizontalHeader().setStretchLastSection(True)
        self.exit_table.verticalHeader().setVisible(False)
        exit_layout.addWidget(self.exit_table)
        layout.addWidget(exit_group)

        flags_group = QGroupBox("Flags")
        flags_layout = QVBoxLayout(flags_group)
        self.stalled_label = QLabel("Stalled zones: -")
        self.stalled_label.setWordWrap(True)
        self.low_flow_label = QLabel("High-queue/low-flow exits: -")
        self.low_flow_label.setWordWrap(True)
        self.uncovered_label = QLabel("Zones without camera coverage: -")
        self.uncovered_label.setWordWrap(True)
        for label in (self.stalled_label, self.low_flow_label, self.uncovered_label):
            flags_layout.addWidget(label)
        layout.addWidget(flags_group)

        layout.addStretch(1)

    # =====================================================

    def show_progress(self, snapshot: Optional[object]) -> None:

        if snapshot is None:

            self.active_label.setText("Observed active occupants: -")
            self.exited_label.setText("Observed exited occupants: -")
            self.progress_label.setText("Observed progress: - (no evacuation_progress_gateway configured, or no cycle run yet)")
            self.trend_label.setText("Overall progress trend: -")
            self.coverage_label.setText("Position coverage: -")
            self.zone_table.setRowCount(0)
            self.exit_table.setRowCount(0)
            self.stalled_label.setText("Stalled zones: -")
            self.low_flow_label.setText("High-queue/low-flow exits: -")
            self.uncovered_label.setText("Zones without camera coverage: -")
            return

        self.active_label.setText(f"Observed active occupants: {snapshot.known_active_occupants}")
        self.exited_label.setText(f"Observed exited occupants: {snapshot.known_exited_occupants}")

        if snapshot.evacuation_progress_fraction is None:
            self.progress_label.setText("Observed progress: - (no occupants observed yet)")
        else:
            self.progress_label.setText(
                f"Observed progress: {_format_percent(snapshot.evacuation_progress_fraction)} of "
                f"{snapshot.known_total_observed_occupants} tracked/observed occupant(s) cleared"
            )

        self.trend_label.setText(f"Overall progress trend: {snapshot.overall_progress_trend}")
        self.coverage_label.setText(f"Position coverage: {_format_percent(snapshot.observability.position_coverage_fraction)}")

        zone_ids = sorted(snapshot.zones.keys())
        self.zone_table.setRowCount(len(zone_ids))
        for row_index, zone_id in enumerate(zone_ids):

            clearance = snapshot.zones[zone_id]
            self.zone_table.setItem(row_index, 0, QTableWidgetItem(zone_id))
            self.zone_table.setItem(row_index, 1, QTableWidgetItem(clearance.status))
            self.zone_table.setItem(row_index, 2, QTableWidgetItem(str(clearance.current_active_count)))
            self.zone_table.setItem(row_index, 3, QTableWidgetItem(_format_percent(clearance.clearance_fraction)))
            self.zone_table.setItem(row_index, 4, QTableWidgetItem(clearance.trend))

        exit_ids = sorted(snapshot.exits.keys())
        self.exit_table.setRowCount(len(exit_ids))
        for row_index, exit_id in enumerate(exit_ids):

            flow = snapshot.exits[exit_id]
            self.exit_table.setItem(row_index, 0, QTableWidgetItem(exit_id))
            self.exit_table.setItem(row_index, 1, QTableWidgetItem(str(flow.queue_candidate_count)))
            self.exit_table.setItem(row_index, 2, QTableWidgetItem(str(flow.approaching_count)))
            self.exit_table.setItem(row_index, 3, QTableWidgetItem(_format_rate(flow.recent_flow_per_minute)))
            self.exit_table.setItem(row_index, 4, QTableWidgetItem("Yes" if flow.flow_active else "No"))
            self.exit_table.setItem(row_index, 5, QTableWidgetItem(flow.trend))

        self.stalled_label.setText(f"Stalled zones: {_dash_join(snapshot.stalled_zone_ids)}")
        self.low_flow_label.setText(f"High-queue/low-flow exits: {_dash_join(snapshot.low_flow_exit_ids)}")
        self.uncovered_label.setText(f"Zones without camera coverage: {_dash_join(snapshot.observability.zones_without_camera_coverage)}")
