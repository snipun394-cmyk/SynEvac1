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

        # Disabled, not wired to a handler: there is no undo/redo
        # command stack anywhere in Designer to back these actions.
        # Tooltip-disclosed rather than left silently clickable-but-
        # inert (see designer/windows/main_window.py's own toolbar
        # wiring for the actions that ARE backed).
        self.undo_action = QAction("Undo", self)
        self.undo_action.setEnabled(False)
        self.undo_action.setToolTip("Undo is not implemented yet.")
        self.undo_action.setStatusTip("Undo is not implemented yet.")

        self.redo_action = QAction("Redo", self)
        self.redo_action.setEnabled(False)
        self.redo_action.setToolTip("Redo is not implemented yet.")
        self.redo_action.setStatusTip("Redo is not implemented yet.")

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

        # Disabled, not wired to a handler: no "elevator" drawing tool
        # exists anywhere in GraphicsScene.set_tool()/GraphicsView.
        # set_tool() to back this button (models.elevator.Elevator is
        # a data model with no authoring tool yet).
        self.elevator_action = QAction("Elevator", self)
        self.elevator_action.setEnabled(False)
        self.elevator_action.setToolTip("Elevator authoring is not implemented yet.")
        self.elevator_action.setStatusTip("Elevator authoring is not implemented yet.")

        self.obstacle_action = QAction("Obstacle", self)

        # =====================================================
        # Devices
        # =====================================================

        self.camera_action = QAction("Camera", self)
        self.detector_action = QAction("Detector", self)

        # Building Sensor Network Framework -- first-class Smoke/Heat
        # Detector tools, alongside (never replacing) the pre-existing
        # generic Detector tool above.
        self.smoke_detector_action = QAction("Smoke Detector", self)
        self.heat_detector_action = QAction("Heat Detector", self)

        # Zoned Voice Evacuation & Speaker Network Framework -- a
        # first-class Speaker placement tool, same "additive, alongside
        # the existing device tools" convention as Smoke/Heat Detector
        # above.
        self.speaker_action = QAction("Speaker", self)

        # Live Dynamic Evacuation Signage milestone -- a first-class
        # Dynamic Sign placement tool, same "additive, alongside the
        # existing device tools" convention as Speaker above.
        self.sign_action = QAction("Dynamic Sign", self)

        # Manual Call Points & Emergency Lighting milestone -- same
        # "additive, alongside the existing device tools" convention.
        self.manual_call_point_action = QAction("Manual Call Point", self)
        self.emergency_light_action = QAction("Emergency Light", self)

        # Fire Suppression & Water-Based Safety Asset Digital Twin
        # milestone -- same "additive, alongside the existing device
        # tools" convention.
        self.sprinkler_action = QAction("Sprinkler", self)
        self.fire_extinguisher_action = QAction("Fire Extinguisher", self)
        self.fire_hydrant_action = QAction("Fire Hydrant", self)
        self.hose_reel_action = QAction("Hose Reel", self)

        # Fire Water Supply & Suppression Infrastructure milestone --
        # same "additive, alongside the existing device tools"
        # convention.
        self.fire_water_tank_action = QAction("Fire Water Tank", self)
        self.fire_pump_action = QAction("Fire Pump", self)
        self.jockey_pump_action = QAction("Jockey Pump", self)
        self.fire_service_inlet_action = QAction("Fire Service Inlet", self)

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

        # Camera Coverage & Visibility Engine -- toggles the
        # occlusion-aware coverage overlay (visible/blind/overlapping
        # zone tint + each camera's true visibility polygon) on the
        # current floor. See designer/scene/graphics_scene.py::
        # set_show_camera_coverage()/refresh_camera_coverage() and
        # visibility/engine.py.
        self.coverage_action = QAction("Coverage", self)
        self.coverage_action.setCheckable(True)
        self.coverage_action.setToolTip(
            "Toggle camera coverage & blind-spot visualization"
        )

        # =====================================================
        # Toolbar Layout
        #
        # SynEvac Designer Simplification & Product-Boundary Cleanup
        # milestone -- the main toolbar is now grouped around the
        # assets that actually participate in evacuation intelligence
        # (docs/architecture/designer_asset_connectivity_audit.md is
        # the authoritative source for this grouping), rather than one
        # long undifferentiated list of every asset ever added.
        #
        # Three groups of actions are DELIBERATELY constructed above
        # but never added to this toolbar's own visible layout below:
        #   - the generic "Detector" tool (superseded by Smoke/Heat for
        #     any type that has real behavior; Flame/Gas do nothing at
        #     all -- see models/detector_migration.py) is not offered
        #     for new authoring at all, in any menu;
        #   - "Elevator" was already disabled/unconnected before this
        #     milestone (no authoring tool backs it -- see its own
        #     comment above) and stays exactly that way;
        #   - the nine fire-safety/water-infrastructure asset actions
        #     (Emergency Light through Fire Service Inlet) are moved to
        #     a secondary surface -- MainWindow adds them to an
        #     "Advanced Fire-Safety Tools" submenu (Insert menu) instead
        #     of this toolbar, via the SAME QAction objects constructed
        #     here (a QAction is not tied to one widget -- adding it to
        #     a menu instead of a toolbar changes nothing about how it
        #     is triggered or what it is connected to).
        # Nothing is deleted: every QAction above still exists, is still
        # constructed the same way, and is still wired to the same
        # MainWindow.change_tool() handler in connect_toolbar() --
        # existing/legacy projects containing any of these asset types
        # still load, render, and remain editable in the Property Panel.
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

        # ---- Building ----
        self.addAction(self.zone_action)
        self.addAction(self.door_action)
        self.addAction(self.exit_action)
        self.addAction(self.stair_action)
        self.addAction(self.obstacle_action)
        self.addAction(self.assembly_point_action)

        self.addSeparator()

        # ---- Perception & Alarm ----
        self.addAction(self.camera_action)
        self.addAction(self.smoke_detector_action)
        self.addAction(self.heat_detector_action)
        self.addAction(self.manual_call_point_action)

        self.addSeparator()

        # ---- Guidance & Output ----
        self.addAction(self.speaker_action)
        self.addAction(self.sign_action)

        self.addSeparator()

        # ---- Simulation ----
        self.addAction(self.occupant_action)
        self.addAction(self.simulation_action)

        self.addSeparator()

        # ---- View ----
        self.addAction(self.zoom_in_action)
        self.addAction(self.zoom_out_action)
        self.addAction(self.reset_view_action)

        self.addSeparator()

        self.addAction(self.coverage_action)