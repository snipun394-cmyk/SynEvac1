from collections import Counter

from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from navigation.edge import Edge


# =====================================================
# StatisticsPanel -- Simulation Replay Studio V1's per-scenario time-
# series charts. Every series is read directly off already-computed
# IncidentFrame fields or derived by a plain arithmetic rollup over
# IncidentData.occupant_routes (bucketed hop counts/speeds) -- nothing
# here re-simulates or estimates a value the completed run did not
# already produce. A run that never populated a given field (e.g.
# zone_smoke, left None/blank by a Timeline Dataset row that never
# computed one -- see dataset_builder/timeline.py's own disclosed gap)
# simply renders that series empty, never fabricated.
#
# Same lightweight, dependency-free custom-painted line chart approach
# command_center.timeline_panel._TrendChart already establishes --
# restated here as a small, reusable multi-series widget rather than
# introducing a new charting library dependency this codebase does not
# otherwise have.
# =====================================================

_SERIES_COLORS = (
    QColor(214, 91, 45),
    QColor(90, 140, 200),
    QColor(110, 190, 110),
    QColor(200, 200, 90),
    QColor(180, 110, 200),
)


class _SeriesChart(QWidget):

    def __init__(self, title, parent=None):
        super().__init__(parent)

        self.setMinimumHeight(120)
        self._title = title
        self._series = []  # list of (label, values)

    def set_series(self, series):

        self._series = [(label, tuple(values)) for label, values in series]
        self.update()

    def paintEvent(self, _event):

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(32, 34, 38))

        painter.setPen(QPen(QColor(220, 220, 220)))
        painter.drawText(4, 14, self._title)

        legend_x = 4
        for index, (label, values) in enumerate(self._series):

            color = _SERIES_COLORS[index % len(_SERIES_COLORS)]
            painter.setPen(QPen(color, 2))
            painter.drawText(legend_x, 28, label)
            legend_x += 10 * len(label) + 20

            self._draw_series(painter, values, color)

        painter.end()

    def _draw_series(self, painter, values, color):

        points_with_value = [
            (index, value) for index, value in enumerate(values) if value is not None
        ]

        if len(points_with_value) < 2:
            return

        maximum = max(value for _, value in points_with_value) or 1.0

        width = max(self.width(), 1)
        height = max(self.height() - 32, 1)
        count = max(len(values) - 1, 1)

        painter.setPen(QPen(color, 2))

        screen_points = [
            QPointF(
                width * index / count,
                32 + height - 4 - (min(value / maximum, 1.0) * (height - 8)),
            )
            for index, value in points_with_value
        ]

        for start, end in zip(screen_points, screen_points[1:]):
            painter.drawLine(start, end)


class StatisticsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        self.evacuation_chart = _SeriesChart("Occupants Evacuated / Remaining vs. Time")
        self.congestion_chart = _SeriesChart("Congestion / Smoke vs. Time")
        self.utilization_chart = _SeriesChart("Door / Exit / Stair Utilization vs. Time")
        self.speed_chart = _SeriesChart("Average Occupant Speed vs. Time")

        for chart in (
            self.evacuation_chart, self.congestion_chart, self.utilization_chart, self.speed_chart,
        ):
            layout.addWidget(chart)

        profile_group = QGroupBox("Behavior Profile Distribution")
        profile_layout = QVBoxLayout(profile_group)
        self.profile_table = QTableWidget(0, 2)
        self.profile_table.setHorizontalHeaderLabels(["Behavior Profile", "Occupant Count"])
        self.profile_table.horizontalHeader().setStretchLastSection(True)
        self.profile_table.verticalHeader().setVisible(False)
        profile_layout.addWidget(self.profile_table)
        layout.addWidget(profile_group)

        layout.addStretch(1)

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        if incident_data is None:

            for chart in (
                self.evacuation_chart, self.congestion_chart, self.utilization_chart, self.speed_chart,
            ):
                chart.set_series(())

            self.profile_table.setRowCount(0)
            return

        frames = incident_data.frames

        self.evacuation_chart.set_series(
            (
                ("Evacuated", [frame.people_evacuated for frame in frames]),
                ("Remaining", [frame.people_remaining for frame in frames]),
            )
        )

        self.congestion_chart.set_series(
            (
                ("Congestion", [frame.current_congestion for frame in frames]),
                ("Avg. Smoke", [_average_smoke(frame) for frame in frames]),
            )
        )

        self.utilization_chart.set_series(self._utilization_series(incident_data))
        self.speed_chart.set_series((("Avg. Speed (m/s)", self._average_speed_series(incident_data)),))

        self._populate_profile_table(incident_data)

    # =====================================================

    def show_frame(self, frame):

        # Every chart above is a whole-run time series, computed once in
        # set_incident() -- nothing here varies per displayed frame.
        return

    # =====================================================

    def _utilization_series(self, incident_data):

        frames = incident_data.frames
        frame_times = [frame.time for frame in frames]

        counts_by_type = {Edge.DOOR: [], Edge.EXIT: [], Edge.STAIR: []}

        for frame_time in frame_times:

            active_by_type = Counter()

            for record in incident_data.occupant_routes:
                for hop in record.hops:
                    if hop.start_time <= frame_time < hop.end_time:
                        active_by_type[hop.edge_type] += 1

            for edge_type in counts_by_type:
                counts_by_type[edge_type].append(active_by_type.get(edge_type, 0))

        return [
            ("Doors In Use", counts_by_type[Edge.DOOR]),
            ("Exits In Use", counts_by_type[Edge.EXIT]),
            ("Stairs In Use", counts_by_type[Edge.STAIR]),
        ]

    # =====================================================

    def _average_speed_series(self, incident_data):

        speeds = []

        for frame in incident_data.frames:

            values = [
                position.speed for position in frame.occupant_positions.values()
                if position.speed is not None and position.speed > 0
            ]
            speeds.append(sum(values) / len(values) if values else None)

        return speeds

    # =====================================================

    def _populate_profile_table(self, incident_data):

        counts = Counter(
            occupant.behaviour_profile_id for occupant in incident_data.scenario.occupants
        )
        counts.update(
            firefighter.behaviour_profile_id for firefighter in incident_data.scenario.firefighters
        )

        rows = sorted(counts.items())

        self.profile_table.setRowCount(len(rows))
        for row_index, (profile_id, count) in enumerate(rows):
            self.profile_table.setItem(row_index, 0, QTableWidgetItem(profile_id))
            self.profile_table.setItem(row_index, 1, QTableWidgetItem(str(count)))


# =====================================================


def _average_smoke(frame):

    levels = [level for level in frame.zone_smoke.values() if level is not None]

    return sum(levels) / len(levels) if levels else None
