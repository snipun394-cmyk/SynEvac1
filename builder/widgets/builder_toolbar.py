from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QToolBar


class BuilderToolbar(QToolBar):

    # A new, small, Builder-only action registry -- deliberately NOT a
    # reuse of designer.widgets.toolbar.MainToolbar. MainToolbar is
    # Studio's own registry for its full ~25-asset palette (including
    # nine fire-safety/water-infrastructure tools, Sign, Manual Call
    # Point, Occupant, Simulation); this milestone's brief scopes
    # Builder's authoring palette to exactly nine asset types (Zones,
    # Doors, Exits, Stairs, Cameras, Smoke Detectors, Heat Detectors,
    # Speakers, Obstacles). Trimming MainToolbar down to that set would
    # mean editing a Studio-owned widget for a Builder-only reason,
    # or shipping Builder with visible-but-unwired buttons for tools
    # (Occupant, Simulation, Sign, ...) explicitly out of scope
    # (Simulation especially -- "Builder MUST NOT include Simulation"
    # is a hard requirement, not just an unused import). A QAction
    # declaration carries no logic to duplicate; this is a scoped
    # surface, not a fork.
    #
    # Every drawing-tool QAction here is wired the exact same way
    # MainToolbar's already are -- BuilderMainWindow.change_tool()
    # calls the SAME GraphicsScene.set_tool()/mousePressEvent dispatch
    # chain Studio uses, unmodified. Only the set of tools offered
    # differs, never how a tool places an object.

    def __init__(self):
        super().__init__("Builder Toolbar")

        self.setMovable(False)
        self.setFloatable(False)

        self.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextUnderIcon
        )

        # =====================================================
        # Project Management
        # =====================================================

        self.new_action = QAction("New", self)
        self.open_action = QAction("Open", self)
        self.save_action = QAction("Save", self)
        self.save_as_action = QAction("Save As", self)

        # =====================================================
        # Edit -- disabled, tooltip-disclosed, same convention as
        # designer/widgets/toolbar.py's own Undo/Redo: there is no
        # undo/redo command stack anywhere in the code this milestone
        # reuses (designer/scene/graphics_scene.py, designer/items/*),
        # and building one is a substantial new subsystem this
        # milestone's investigation never scoped. Matching Studio's
        # own current, honest "not implemented yet" state rather than
        # silently pretending it exists.
        # =====================================================

        self.undo_action = QAction("Undo", self)
        self.undo_action.setEnabled(False)
        self.undo_action.setToolTip("Undo is not implemented yet.")

        self.redo_action = QAction("Redo", self)
        self.redo_action.setEnabled(False)
        self.redo_action.setToolTip("Redo is not implemented yet.")

        # =====================================================
        # Navigation
        # =====================================================

        self.select_action = QAction("Select", self)

        # =====================================================
        # Floor Plan / Scale Calibration
        # =====================================================

        self.import_floor_plan_action = QAction("Import Floor Plan", self)

        self.calibrate_scale_action = QAction("Calibrate Scale", self)
        self.calibrate_scale_action.setToolTip(
            "Click two points on the imported floor plan, then enter "
            "the real-world distance between them."
        )

        # =====================================================
        # Structural Drawing Tools
        # =====================================================

        self.zone_action = QAction("Zone", self)
        self.door_action = QAction("Door", self)
        self.exit_action = QAction("Exit", self)
        self.stair_action = QAction("Stair", self)
        self.obstacle_action = QAction("Obstacle", self)

        # =====================================================
        # Asset Placement Tools
        # =====================================================

        self.camera_action = QAction("Camera", self)
        self.smoke_detector_action = QAction("Smoke Detector", self)
        self.heat_detector_action = QAction("Heat Detector", self)
        self.speaker_action = QAction("Speaker", self)

        # =====================================================
        # View
        # =====================================================

        self.zoom_in_action = QAction("Zoom +", self)
        self.zoom_out_action = QAction("Zoom -", self)
        self.reset_view_action = QAction("Reset", self)

        # =====================================================
        # Validation
        # =====================================================

        self.validate_action = QAction("Validate", self)

        # =====================================================
        # Layout
        # =====================================================

        self.addAction(self.new_action)
        self.addAction(self.open_action)
        self.addAction(self.save_action)
        self.addAction(self.save_as_action)

        self.addSeparator()

        self.addAction(self.undo_action)
        self.addAction(self.redo_action)

        self.addSeparator()

        self.addAction(self.select_action)

        self.addSeparator()

        self.addAction(self.import_floor_plan_action)
        self.addAction(self.calibrate_scale_action)

        self.addSeparator()

        # ---- Structure ----
        self.addAction(self.zone_action)
        self.addAction(self.door_action)
        self.addAction(self.exit_action)
        self.addAction(self.stair_action)
        self.addAction(self.obstacle_action)

        self.addSeparator()

        # ---- Assets ----
        self.addAction(self.camera_action)
        self.addAction(self.smoke_detector_action)
        self.addAction(self.heat_detector_action)
        self.addAction(self.speaker_action)

        self.addSeparator()

        self.addAction(self.validate_action)

        self.addSeparator()

        self.addAction(self.zoom_in_action)
        self.addAction(self.zoom_out_action)
        self.addAction(self.reset_view_action)
