from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QToolBar


class MainToolbar(QToolBar):

    def __init__(self):
        super().__init__("Main Toolbar")

        self.setMovable(False)
        self.setFloatable(False)

        self.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )

        # =====================================================
        # File
        # =====================================================

        self.new_action = QAction("New", self)
        self.open_action = QAction("Open", self)
        self.save_action = QAction("Save", self)

        # =====================================================
        # Edit
        # =====================================================

        self.undo_action = QAction("Undo", self)
        self.redo_action = QAction("Redo", self)

        # =====================================================
        # Navigation
        # =====================================================

        self.select_action = QAction("Select", self)

        # =====================================================
        # Drawing Tools
        # =====================================================

        self.zone_action = QAction("Zone", self)
        self.exit_action = QAction("Exit", self)
        self.door_action = QAction("Door", self)
        self.stair_action = QAction("Stair", self)
        self.elevator_action = QAction("Elevator", self)
        self.obstacle_action = QAction("Obstacle", self)

        # =====================================================
        # Devices
        # =====================================================

        self.camera_action = QAction("Camera", self)
        self.detector_action = QAction("Detector", self)

        # =====================================================
        # Safety
        # =====================================================

        self.assembly_point_action = QAction("Assembly Point", self)

        # =====================================================
        # Manual Simulation Sandbox (Simulation V0)
        # =====================================================

        self.occupant_action = QAction("Occupant", self)
        self.simulation_action = QAction("Simulation", self)

        # =====================================================
        # View
        # =====================================================

        self.zoom_in_action = QAction("Zoom +", self)
        self.zoom_out_action = QAction("Zoom -", self)
        self.reset_view_action = QAction("Reset", self)

        # =====================================================
        # Toolbar Layout
        # =====================================================

        self.addAction(self.new_action)
        self.addAction(self.open_action)
        self.addAction(self.save_action)

        self.addSeparator()

        self.addAction(self.undo_action)
        self.addAction(self.redo_action)

        self.addSeparator()

        self.addAction(self.select_action)

        self.addSeparator()

        self.addAction(self.zone_action)
        self.addAction(self.exit_action)
        self.addAction(self.door_action)
        self.addAction(self.stair_action)
        self.addAction(self.elevator_action)
        self.addAction(self.obstacle_action)

        self.addSeparator()

        self.addAction(self.camera_action)
        self.addAction(self.detector_action)

        self.addSeparator()

        self.addAction(self.assembly_point_action)

        self.addSeparator()

        self.addAction(self.occupant_action)
        self.addAction(self.simulation_action)

        self.addSeparator()

        self.addAction(self.zoom_in_action)
        self.addAction(self.zoom_out_action)
        self.addAction(self.reset_view_action)