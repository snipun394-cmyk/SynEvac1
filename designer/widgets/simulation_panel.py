from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class SimulationPanel(QWidget):

    # A dumb presentational widget, same convention MainToolbar/
    # BottomInfoBar already follow: buttons/slider/label are exposed
    # as plain public attributes for MainWindow to wire up and drive
    # (this class owns no SandboxManager, no QTimer, no occupant
    # loop of its own) -- it only ever displays whatever MainWindow
    # tells it to via update_time()/set_running_state(), and only
    # ever reports the user's chosen speed via speed_multiplier().

    MIN_SPEED_STEP = 1       # 0.1x
    MAX_SPEED_STEP = 50      # 5.0x
    DEFAULT_SPEED_STEP = 10  # 1.0x
    SPEED_STEP_DIVISOR = 10.0

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        # =====================================================
        # Playback Controls
        # =====================================================

        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset")
        self.step_button = QPushButton("Step Forward")

        button_row = QHBoxLayout()

        for button in (
            self.start_button,
            self.pause_button,
            self.stop_button,
            self.reset_button,
            self.step_button,
        ):
            button_row.addWidget(button)

        layout.addLayout(button_row)

        # =====================================================
        # Simulation Speed
        # =====================================================

        speed_row = QHBoxLayout()

        speed_row.addWidget(QLabel("Speed"))

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(self.MIN_SPEED_STEP)
        self.speed_slider.setMaximum(self.MAX_SPEED_STEP)
        self.speed_slider.setValue(self.DEFAULT_SPEED_STEP)

        self.speed_value_label = QLabel(
            f"{self._step_to_multiplier(self.DEFAULT_SPEED_STEP):.1f}x"
        )

        speed_row.addWidget(self.speed_slider)
        speed_row.addWidget(self.speed_value_label)

        layout.addLayout(speed_row)

        # =====================================================
        # Current Simulation Time
        # =====================================================

        self.time_label = QLabel("Time : 0.00 s")

        layout.addWidget(self.time_label)

        self.setLayout(layout)

        self.speed_slider.valueChanged.connect(self._on_speed_changed)

        self.pause_button.setEnabled(False)

    # =====================================================

    def _step_to_multiplier(self, step):

        return step / self.SPEED_STEP_DIVISOR

    # =====================================================

    def _on_speed_changed(self, step):

        self.speed_value_label.setText(
            f"{self._step_to_multiplier(step):.1f}x"
        )

    # =====================================================

    def speed_multiplier(self):

        return self._step_to_multiplier(self.speed_slider.value())

    # =====================================================

    def update_time(self, seconds):

        self.time_label.setText(
            f"Time : {seconds:.2f} s"
        )

    # =====================================================
    # Toggles which controls make sense while running vs. not --
    # Stop/Reset/Step remain available either way (Step Mode is
    # explicitly meant to work on its own, without Start ever having
    # been pressed).
    # =====================================================

    def set_running_state(self, running):

        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
