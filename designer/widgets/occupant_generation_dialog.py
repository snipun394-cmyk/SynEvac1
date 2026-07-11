from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
)

from sandbox.occupant import SandboxDistribution


class OccupantGenerationDialog(QDialog):

    # Shown once, after the user releases the Occupant Tool's
    # drag-rectangle -- mirrors QInputDialog.getItem()'s role for the
    # Stair Tool's destination-floor picker (MainWindow owns the
    # dialog, GraphicsScene never shows one itself). Only "Number of
    # Occupants" and "Distribution" actually do anything in V0; the
    # two fields below them are deliberately present but disabled,
    # per the milestone's own "optional future fields may remain
    # disabled" -- they document where this dialog is headed without
    # implementing any behavior behind them yet.

    DEFAULT_COUNT = 5
    MIN_COUNT = 1
    MAX_COUNT = 500

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Generate Occupants")

        layout = QFormLayout()

        self.count_spinbox = QSpinBox()
        self.count_spinbox.setMinimum(self.MIN_COUNT)
        self.count_spinbox.setMaximum(self.MAX_COUNT)
        self.count_spinbox.setValue(self.DEFAULT_COUNT)

        self.distribution_combo = QComboBox()

        for distribution in SandboxDistribution.OPTIONS:
            self.distribution_combo.addItem(distribution)

        layout.addRow("Number of Occupants", self.count_spinbox)
        layout.addRow("Distribution", self.distribution_combo)

        # Future fields -- disabled placeholders only, no behavior yet.
        self.behavior_profile_combo = QComboBox()
        self.behavior_profile_combo.addItem("Default")
        self.behavior_profile_combo.setEnabled(False)

        self.mobility_combo = QComboBox()
        self.mobility_combo.addItem("Default")
        self.mobility_combo.setEnabled(False)

        layout.addRow("Behavior Profile (future)", self.behavior_profile_combo)
        layout.addRow("Mobility (future)", self.mobility_combo)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addRow(buttons)

        self.setLayout(layout)

    # =====================================================

    def occupant_count(self):

        return self.count_spinbox.value()

    # =====================================================

    def distribution(self):

        return self.distribution_combo.currentText()
