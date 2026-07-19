from bisect import bisect_right

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


# =====================================================
# TimelinePanel -- playback controls and a trend preview of the
# incident's own already-computed hazard/congestion/occupancy/smoke
# trajectory. Same "dumb widget, pushed updates only" convention
# designer/widgets/simulation_panel.py already establishes: this widget
# owns no QTimer and no domain logic. MainWindow owns the one clock (a
# QTimer), reading speed_multiplier() each tick and pushing the
# resulting frame index back in via set_frame_index() -- exactly the
# shape MainWindow.on_simulation_tick() already uses for the Manual
# Simulation Sandbox.
# =====================================================

_SPEED_OPTIONS = ("0.25x", "0.5x", "1x", "2x", "4x", "8x")


class _TrendChart(QWidget):

    # Small painted line chart of current_hazard_score, current_
    # congestion, remaining occupancy, and average zone smoke across
    # every already-computed IncidentFrame -- a preview of the
    # incident's own trajectory, not a prediction: every point plotted
    # is a value some IncidentFrame already carries (occupancy/smoke are
    # a plain per-frame aggregation of frame.people_remaining/
    # frame.zone_smoke, nothing re-derived beyond that).

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumHeight(70)
        self._hazard = ()
        self._congestion = ()
        self._occupancy = ()
        self._smoke = ()

    def set_series(self, hazard, congestion, occupancy, smoke):

        self._hazard = tuple(hazard)
        self._congestion = tuple(congestion)
        self._occupancy = tuple(occupancy)
        self._smoke = tuple(smoke)
        self.update()

    def paintEvent(self, _event):

        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(32, 34, 38))

        self._draw_series(painter, self._hazard, QColor(214, 91, 45), maximum=1.0)
        self._draw_series(painter, self._congestion, QColor(90, 140, 200), maximum=None)
        self._draw_series(painter, self._occupancy, QColor(110, 190, 110), maximum=None)
        self._draw_series(painter, self._smoke, QColor(200, 200, 90), maximum=1.0)

        painter.end()

    def _draw_series(self, painter, values, color, maximum):

        points_with_value = [
            (index, value) for index, value in enumerate(values) if value is not None
        ]

        if len(points_with_value) < 2:
            return

        scale = maximum if maximum else max(value for _, value in points_with_value) or 1.0

        width = max(self.width(), 1)
        height = max(self.height(), 1)
        count = max(len(values) - 1, 1)

        painter.setPen(QPen(color, 2))

        screen_points = [
            QPointF(
                width * index / count,
                height - 4 - (min(value / scale, 1.0) * (height - 8)),
            )
            for index, value in points_with_value
        ]

        for start, end in zip(screen_points, screen_points[1:]):
            painter.drawLine(start, end)


class TimelinePanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self._incident = None

        layout = QVBoxLayout(self)

        self.trend_chart = _TrendChart()
        layout.addWidget(self.trend_chart)

        controls = QHBoxLayout()

        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")

        controls.addWidget(self.play_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.reset_button)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Speed:"))

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(_SPEED_OPTIONS)
        self.speed_combo.setCurrentIndex(_SPEED_OPTIONS.index("1x"))
        controls.addWidget(self.speed_combo)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Jump to:"))

        self.jump_to_time_spinbox = QDoubleSpinBox()
        self.jump_to_time_spinbox.setDecimals(1)
        self.jump_to_time_spinbox.setSuffix("s")
        self.jump_to_time_spinbox.setMaximum(0.0)
        controls.addWidget(self.jump_to_time_spinbox)

        self.jump_to_time_button = QPushButton("Go")
        self.jump_to_time_button.clicked.connect(self._on_jump_to_time)
        controls.addWidget(self.jump_to_time_button)

        controls.addStretch(1)

        layout.addLayout(controls)

        slider_row = QHBoxLayout()

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)

        self.time_label = QLabel("t = 0.0s / 0.0s")

        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.time_label)

        layout.addLayout(slider_row)

    # =====================================================
    # Public API -- pushed updates only.
    # =====================================================

    def set_incident(self, incident_data):

        self._incident = incident_data

        frames = incident_data.frames if incident_data is not None else ()

        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(len(frames) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)

        self.trend_chart.set_series(
            (frame.current_hazard_score for frame in frames),
            (frame.current_congestion for frame in frames),
            (frame.people_remaining for frame in frames),
            (_average_smoke(frame) for frame in frames),
        )

        self.jump_to_time_spinbox.setMaximum(incident_data.duration if incident_data is not None else 0.0)

        self.set_frame_index(0)

    # =====================================================

    def set_frame_index(self, index):

        if self._incident is None or self._incident.frame_count == 0:
            self.time_label.setText("t = 0.0s / 0.0s")
            return

        frame = self._incident.frame_at_index(index)

        self.slider.blockSignals(True)
        self.slider.setValue(index)
        self.slider.blockSignals(False)

        self.time_label.setText(f"t = {frame.time:.1f}s / {self._incident.duration:.1f}s")

    # =====================================================

    def speed_multiplier(self) -> float:

        return float(self.speed_combo.currentText().rstrip("x"))

    # =====================================================

    @property
    def frame_index(self) -> int:

        return self.slider.value()

    # =====================================================
    # Internal wiring
    # =====================================================

    def _on_jump_to_time(self):

        if self._incident is None or self._incident.frame_count == 0:
            return

        # Same "nearest preceding frame" convention IncidentData.
        # frame_at() itself already documents -- recomputed here from
        # the public frames/time properties rather than reaching into
        # IncidentData's own internals. Setting the slider (rather than
        # calling set_frame_index() directly) reuses the exact same
        # valueChanged -> Dashboard._on_frame_index_changed() path every
        # other slider interaction already goes through.
        times = [frame.time for frame in self._incident.frames]
        index = bisect_right(times, self.jump_to_time_spinbox.value()) - 1
        index = max(0, min(index, len(times) - 1))

        self.slider.setValue(index)


# =====================================================


def _average_smoke(frame):

    levels = [level for level in frame.zone_smoke.values() if level is not None]

    return sum(levels) / len(levels) if levels else None
