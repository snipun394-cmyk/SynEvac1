from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class EngineeringCategoryEditor(QGroupBox):

    # One category's (Doors/Exits/Stairs/Obstacles/Cameras/Detectors)
    # engineering-state authoring widget, replacing the single
    # "Use Building Default / Always X / Random Mix" combo box that
    # produced empty EngineeringConstraints distributions by default.
    # Three explicit, mutually exclusive modes -- never a blended
    # combo-box choice -- so a fire engineer can see at a glance
    # whether a category is left to Building Defaults, pinned to one
    # Fixed State, or driven by an editable Random Distribution over
    # every state, with a live running total that must reach exactly
    # 100%.

    MODE_BUILDING_DEFAULT = "Use Building Default"
    MODE_FIXED = "Fixed State"
    MODE_RANDOM = "Random Distribution"

    MODES = (MODE_BUILDING_DEFAULT, MODE_FIXED, MODE_RANDOM)

    changed = pyqtSignal()

    def __init__(self, title, state_names, parent=None):

        super().__init__(title, parent)

        self.state_names = tuple(state_names)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(self.MODES)
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_combo)

        self.fixed_container = QWidget()
        fixed_form = QFormLayout()
        fixed_form.setContentsMargins(0, 0, 0, 0)
        self.fixed_container.setLayout(fixed_form)

        self.fixed_state_combo = QComboBox()
        self.fixed_state_combo.addItems(self.state_names)
        self.fixed_state_combo.currentTextChanged.connect(lambda _: self.changed.emit())
        fixed_form.addRow("Fixed State", self.fixed_state_combo)

        layout.addWidget(self.fixed_container)

        self.random_container = QWidget()
        random_form = QFormLayout()
        random_form.setContentsMargins(0, 0, 0, 0)
        self.random_container.setLayout(random_form)

        # Even split across every state, remainder folded into the
        # last one -- an always-valid (sums to exactly 100) starting
        # point, rather than leaving a fresh Random Distribution
        # selection at 0%/invalid until the user manually balances it.
        even_share, remainder = divmod(100, len(self.state_names))

        self.percentage_spins = {}

        for index, name in enumerate(self.state_names):

            spin = QSpinBox()
            spin.setRange(0, 100)
            spin.setSuffix("%")
            spin.setValue(even_share + (remainder if index == len(self.state_names) - 1 else 0))
            spin.valueChanged.connect(self._on_percentage_changed)

            self.percentage_spins[name] = spin
            random_form.addRow(name, spin)

        self.total_label = QLabel()
        random_form.addRow("Total", self.total_label)

        layout.addWidget(self.random_container)

        self._on_percentage_changed()
        self._on_mode_changed(self.mode_combo.currentText())

    # =====================================================

    def _on_mode_changed(self, mode):

        self.fixed_container.setVisible(mode == self.MODE_FIXED)
        self.random_container.setVisible(mode == self.MODE_RANDOM)

        self.changed.emit()

    # =====================================================

    def _on_percentage_changed(self, *_):

        total = self.total_percentage()

        self.total_label.setText(f"{total}%")
        self.total_label.setStyleSheet(
            "color: #2f7d32; font-weight: bold;" if total == 100
            else "color: #a83232; font-weight: bold;"
        )

        self.changed.emit()

    # =====================================================
    # Accessors
    # =====================================================

    def mode(self) -> str:

        return self.mode_combo.currentText()

    def set_mode(self, mode: str):

        self.mode_combo.setCurrentText(mode)

    def fixed_state(self) -> str:

        return self.fixed_state_combo.currentText()

    def set_fixed_state(self, state_name: str):

        self.fixed_state_combo.setCurrentText(state_name)

    def percentages(self) -> dict:

        return {name: spin.value() for name, spin in self.percentage_spins.items()}

    def set_percentages(self, values: dict):

        for name, value in values.items():

            if name in self.percentage_spins:
                self.percentage_spins[name].setValue(value)

    def total_percentage(self) -> int:

        return sum(spin.value() for spin in self.percentage_spins.values())

    def weights(self) -> dict:

        # Zero-weight states are dropped -- an author who leaves a
        # state at 0% means "never", identical in meaning to simply
        # excluding it from the WeightedOptions map.
        return {
            name: value / 100.0
            for name, value in self.percentages().items()
            if value > 0
        }

    def is_valid(self) -> bool:

        return self.mode() != self.MODE_RANDOM or self.total_percentage() == 100

    def summary_lines(self) -> list:

        mode = self.mode()

        if mode == self.MODE_BUILDING_DEFAULT:
            return ["Building Defaults"]

        if mode == self.MODE_FIXED:
            return [f"Fixed: {self.fixed_state()}"]

        return [f"{name} {value}%" for name, value in self.percentages().items()]
