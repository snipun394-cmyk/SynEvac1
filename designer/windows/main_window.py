from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QDockWidget,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from designer.campaign import CampaignController, CampaignWindow
from designer.items.assembly_point_item import AssemblyPointItem
from designer.items.camera_item import CameraItem
from designer.items.detector_item import DetectorItem
from designer.items.door_item import DoorItem
from designer.items.exit_item import ExitItem
from designer.items.heat_detector_item import HeatDetectorItem
from designer.items.obstacle_item import ObstacleItem
from designer.items.occupant_item import OccupantItem
from designer.items.smoke_detector_item import SmokeDetectorItem
from designer.items.speaker_item import SpeakerItem
from designer.items.sign_item import SignItem
from designer.items.manual_call_point_item import ManualCallPointItem
from designer.items.emergency_light_item import EmergencyLightItem
from designer.items.sprinkler_item import SprinklerItem
from designer.items.fire_extinguisher_item import FireExtinguisherItem
from designer.items.fire_hydrant_item import FireHydrantItem
from designer.items.hose_reel_item import HoseReelItem
from designer.items.stair_item import StairItem
from designer.items.zone_rectangle import ZoneRectangle
from designer.scene.graphics_view import GraphicsView
from designer.widgets.bottom_info_bar import BottomInfoBar
from designer.widgets.building_state_debug_panel import BuildingStateDebugPanel
from designer.widgets.camera_manager_panel import CameraManagerPanel
from designer.widgets.speaker_manager_panel import SpeakerManagerPanel
from designer.widgets.camera_validation_panel import CameraValidationPanel
from designer.widgets.floor_list import FloorList
from designer.widgets.occupant_generation_dialog import OccupantGenerationDialog
from designer.widgets.perception_debug_panel import PerceptionDebugPanel
from designer.widgets.project_tree import ProjectTree
from designer.widgets.property_panel import PropertyPanel
from designer.widgets.simulation_panel import SimulationPanel
from designer.widgets.toolbar import MainToolbar
from designer.validation import validate_building_authoring

from models.project import Project

from serialization.serializer import Serializer

from credential_store.local_file_store import LocalFileCredentialStore


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SynEvac Studio")
        self.resize(1600, 900)

        # One store for the whole session -- every camera's password
        # (present or future) is captured/resolved through this, never
        # written into a saved .syn project file. See
        # credential_store/local_file_store.py; constructing it does
        # no disk I/O until a password actually needs saving.
        self._credential_store = LocalFileCredentialStore()

        # =====================================================
        # Central Widget
        # =====================================================

        central = QWidget()
        self.setCentralWidget(central)

        self.layout = QVBoxLayout()

        self.layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self.layout.setSpacing(0)

        central.setLayout(self.layout)

        # =====================================================
        # Graphics View
        # =====================================================

        self.canvas = GraphicsView()

        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.layout.addWidget(
            self.canvas,
            1,
        )

        # =====================================================
        # Bottom Info Bar
        # =====================================================

        # BottomInfoBar owns its own fixed height internally
        # (tall enough to avoid clipping if its labels wrap).
        self.info_bar = BottomInfoBar()

        self.layout.addWidget(
            self.info_bar
        )

        # =====================================================
        # Property Panel
        # =====================================================

        self.property_panel = PropertyPanel()

        # =====================================================
        # Simulation Panel (Manual Simulation Sandbox / Simulation V0)
        # =====================================================

        self.simulation_panel = SimulationPanel()

        # QTimer drives continuous playback (Start/Pause); Step
        # Forward and Reset never touch it. MainWindow owns this loop
        # -- GraphicsScene/SandboxManager stay unaware of wall-clock
        # time entirely, they only ever see the dt each tick/step
        # hands them.
        self.simulation_timer = QTimer(self)
        self.simulation_timer.setInterval(50)
        self.simulation_timer.timeout.connect(self.on_simulation_tick)

        self.simulation_time = 0.0

        # =====================================================
        # Perception Debug Panel (verification/visualization only --
        # see designer/widgets/perception_debug_panel.py and
        # designer/perception_debug_runner.py. Never touched by, and
        # never touches, the Rule-Based Engine or any perception
        # algorithm's own logic -- it only displays what the existing
        # Phase 1-4 Perception classes already compute.)
        # =====================================================

        self.perception_debug_panel = PerceptionDebugPanel()

        # =====================================================
        # Building State Debug Panel (verification/visualization only --
        # see designer/widgets/building_state_debug_panel.py and
        # designer/building_state_debug_runner.py. Displays the Building
        # State Estimator's fused BuildingState snapshot -- occupant
        # tracks, zone occupancy, camera/detector asset state, hazard
        # summary, building-wide alarm status, and consistency
        # diagnostics. Performs no AI reasoning or decision-making of
        # any kind; purely additive alongside the existing Perception
        # Debug Panel.)
        # =====================================================

        self.building_state_debug_panel = BuildingStateDebugPanel()

        # =====================================================
        # Camera Manager Panel (Building-wide Camera Asset
        # administration -- see designer/widgets/camera_manager_panel.py
        # and camera_manager/manager.py. Never performs computer vision
        # or visibility computation itself; only manages Camera Assets
        # and routes to whatever DetectionProvider is registered for
        # each one's current mode.)
        # =====================================================

        self.camera_manager_panel = CameraManagerPanel()

        self.camera_manager_panel.on_camera_changed = self._on_camera_manager_changed

        # =====================================================
        # Speaker Manager Panel (Zoned Voice Evacuation & Speaker
        # Network Framework -- Building-wide Speaker Asset administration,
        # see designer/widgets/speaker_manager_panel.py and
        # speaker_manager/manager.py. Never performs recommendation
        # reasoning or broadcasts anything itself; only manages Speaker
        # Assets.)
        # =====================================================

        self.speaker_manager_panel = SpeakerManagerPanel()

        # =====================================================
        # Camera Calibration & Placement Validation Panel -- an
        # on-demand engineering report (see designer/widgets/
        # camera_validation_panel.py and camera_validation/validator.py).
        # Unlike Camera Manager/Perception Debug, this is never
        # refreshed on every simulation tick -- only once when a
        # project first loads, and whenever the engineer presses its
        # own Run Validation button.
        # =====================================================

        self.camera_validation_panel = CameraValidationPanel()

        # =====================================================
        # Scenario Campaign Studio -- lazily created on first open
        # (open_campaign_studio()), same "built once, reused, refreshed
        # with current state on reopen" convention nothing else in
        # MainWindow currently needs since every other dock is built
        # up front; this one is a heavy, occasional-use tool window
        # rather than part of the default Designer layout, so it is
        # not built until the user actually asks for it.
        # =====================================================

        self.campaign_window = None
        self.campaign_controller = None

        # =====================================================
        # Project Tree
        # =====================================================

        self.project_tree = ProjectTree()

        # =====================================================
        # Floor List
        # =====================================================

        self.floor_list = FloorList()

        # =====================================================
        # Unsaved-Changes Tracking
        #
        # Deliberately coarse: hooks Qt's own built-in QGraphicsScene.
        # changed signal (fires whenever the scene's rendered content
        # changes -- an item added, moved, or removed) rather than
        # threading a dirty flag through every individual mutation
        # method across items/controllers/scene. This is a real,
        # working "was anything touched since the last save/open/new"
        # signal, not a fully precise semantic change tracker -- see
        # _mark_dirty()/_confirm_discard_unsaved_changes() below.
        # =====================================================

        self._dirty = False

        # =====================================================
        # Build UI
        # =====================================================

        self.create_actions()

        self.create_menu()

        self.create_toolbar()

        self.create_docks()

        self.connect_toolbar()

        self.connect_signals()

        # =====================================================
        # Initial Project
        # =====================================================

        self.project_tree.set_project(
            self.canvas.scene_obj.project
        )

        self.floor_list.set_building(
            self.canvas.scene_obj.project.building
        )

        self.property_panel.set_building(
            self.canvas.scene_obj.project.building
        )

        self._refresh_perception_debug_panel()
        self._refresh_building_state_debug_panel()
        self._refresh_camera_manager_panel()
        self._refresh_speaker_manager_panel()

        # Camera Validation is deliberately NOT refreshed on every
        # simulation tick (see CameraValidationPanel's own docstring)
        # -- only once here, at initial project load, and otherwise on
        # its own Run Validation button / whenever its dock is shown.
        self.camera_validation_panel.refresh(
            self.canvas.scene_obj.project.building
        )

    # =====================================================

    def create_actions(self):

        self.new_action = QAction(
            "New Project",
            self,
        )

        self.open_action = QAction(
            "Open Project",
            self,
        )

        self.save_action = QAction(
            "Save Project",
            self,
        )

        self.import_floor_action = QAction(
            "Import Floor Plan",
            self,
        )

        self.new_action.triggered.connect(
            self.new_project
        )

        self.open_action.triggered.connect(
            self.open_project
        )

        self.save_action.triggered.connect(
            self.save_project
        )

        self.import_floor_action.triggered.connect(
            self.import_floor_plan
        )

        self.toggle_simulation_panel_action = QAction(
            "Simulation Panel",
            self,
        )

        self.toggle_simulation_panel_action.triggered.connect(
            self.toggle_simulation_panel
        )

        self.validate_project_action = QAction(
            "Validate Project",
            self,
        )

        self.validate_project_action.triggered.connect(
            self.validate_project
        )

        self.toggle_perception_debug_panel_action = QAction(
            "Perception Debug Panel",
            self,
        )

        self.toggle_perception_debug_panel_action.triggered.connect(
            self.toggle_perception_debug_panel
        )

        self.toggle_building_state_debug_panel_action = QAction(
            "Building State Debug Panel",
            self,
        )

        self.toggle_building_state_debug_panel_action.triggered.connect(
            self.toggle_building_state_debug_panel
        )

        self.toggle_camera_manager_panel_action = QAction(
            "Camera Manager Panel",
            self,
        )

        self.toggle_camera_manager_panel_action.triggered.connect(
            self.toggle_camera_manager_panel
        )

        self.toggle_speaker_manager_panel_action = QAction(
            "Speaker Manager Panel",
            self,
        )

        self.toggle_speaker_manager_panel_action.triggered.connect(
            self.toggle_speaker_manager_panel
        )

        self.toggle_camera_validation_panel_action = QAction(
            "Camera Validation Panel",
            self,
        )

        self.toggle_camera_validation_panel_action.triggered.connect(
            self.toggle_camera_validation_panel
        )

        self.open_campaign_studio_action = QAction(
            "Scenario Campaign Studio...",
            self,
        )

        self.open_campaign_studio_action.triggered.connect(
            self.open_campaign_studio
        )

    # =====================================================

    def create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        menubar.addMenu("Edit")

        view_menu = menubar.addMenu("View")

        view_menu.addAction(
            self.toggle_perception_debug_panel_action
        )

        view_menu.addAction(
            self.toggle_building_state_debug_panel_action
        )

        view_menu.addAction(
            self.toggle_camera_manager_panel_action
        )

        view_menu.addAction(
            self.toggle_speaker_manager_panel_action
        )

        view_menu.addAction(
            self.toggle_camera_validation_panel_action
        )

        menubar.addMenu("Insert")

        simulation_menu = menubar.addMenu("Simulation")

        simulation_menu.addAction(
            self.toggle_simulation_panel_action
        )

        menubar.addMenu("AI")

        campaign_menu = menubar.addMenu("Campaign")

        campaign_menu.addAction(
            self.open_campaign_studio_action
        )

        tools_menu = menubar.addMenu("Tools")

        tools_menu.addAction(
            self.validate_project_action
        )

        menubar.addMenu("Help")

        file_menu.addAction(
            self.new_action
        )

        file_menu.addAction(
            self.open_action
        )

        file_menu.addAction(
            self.save_action
        )

        file_menu.addSeparator()

        file_menu.addAction(
            self.import_floor_action
        )

    # =====================================================

    def create_toolbar(self):

        self.toolbar = MainToolbar()

        self.addToolBar(
            self.toolbar
        )

    # =====================================================

    def create_docks(self):

        project_dock = QDockWidget(
            "Project Explorer",
            self,
        )

        project_dock.setWidget(
            self.project_tree
        )

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            project_dock,
        )

        floor_dock = QDockWidget(
            "Floors",
            self,
        )

        floor_dock.setWidget(
            self.floor_list
        )

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            floor_dock,
        )

        property_dock = QDockWidget(
            "Properties",
            self,
        )

        property_dock.setWidget(
            self.property_panel
        )

        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            property_dock,
        )

        self.simulation_dock = QDockWidget(
            "Simulation",
            self,
        )

        self.simulation_dock.setWidget(
            self.simulation_panel
        )

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.simulation_dock,
        )

        # Hidden until the user opens it via the toolbar/menu action --
        # Simulation V0 is opt-in, not part of the default Designer
        # layout.
        self.simulation_dock.hide()

        self.perception_debug_dock = QDockWidget(
            "Perception Debug",
            self,
        )

        self.perception_debug_dock.setWidget(
            self.perception_debug_panel
        )

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.perception_debug_dock,
        )

        self.tabifyDockWidget(
            self.simulation_dock,
            self.perception_debug_dock,
        )

        self.building_state_debug_dock = QDockWidget(
            "Building State Debug",
            self,
        )

        self.building_state_debug_dock.setWidget(
            self.building_state_debug_panel
        )

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.building_state_debug_dock,
        )

        self.tabifyDockWidget(
            self.perception_debug_dock,
            self.building_state_debug_dock,
        )

        # Hidden by default, same opt-in convention as
        # simulation_dock -- verification/debug tooling, not part of
        # the default Designer layout.
        self.perception_debug_dock.hide()

        # Hidden by default, same opt-in convention as
        # perception_debug_dock above.
        self.building_state_debug_dock.hide()

        self.camera_manager_dock = QDockWidget(
            "Camera Manager",
            self,
        )

        self.camera_manager_dock.setWidget(
            self.camera_manager_panel
        )

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.camera_manager_dock,
        )

        self.tabifyDockWidget(
            self.perception_debug_dock,
            self.camera_manager_dock,
        )

        # Hidden by default, same opt-in convention as the other
        # bottom docks -- not part of the default Designer layout.
        self.camera_manager_dock.hide()

        self.speaker_manager_dock = QDockWidget(
            "Speaker Manager",
            self,
        )

        self.speaker_manager_dock.setWidget(
            self.speaker_manager_panel
        )

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.speaker_manager_dock,
        )

        self.tabifyDockWidget(
            self.camera_manager_dock,
            self.speaker_manager_dock,
        )

        # Hidden by default, same opt-in convention as the other
        # bottom docks -- not part of the default Designer layout.
        self.speaker_manager_dock.hide()

        self.camera_validation_dock = QDockWidget(
            "Camera Validation",
            self,
        )

        self.camera_validation_dock.setWidget(
            self.camera_validation_panel
        )

        self.addDockWidget(
            Qt.DockWidgetArea.BottomDockWidgetArea,
            self.camera_validation_dock,
        )

        self.tabifyDockWidget(
            self.speaker_manager_dock,
            self.camera_validation_dock,
        )

        # Hidden by default, same opt-in convention as the other
        # bottom docks -- not part of the default Designer layout.
        self.camera_validation_dock.hide()

    # =====================================================

    def connect_toolbar(self):

        # File/View actions: MainToolbar constructs its own separate
        # QAction instances from the File menu's (same convention the
        # drawing-tool buttons below already follow -- one QAction per
        # widget, wired to the same MainWindow handler). Undo/Redo/
        # Elevator are deliberately left unconnected -- see toolbar.py,
        # they are disabled at construction time with an explanatory
        # tooltip instead of wired to a handler that doesn't exist.

        self.toolbar.new_action.triggered.connect(
            self.new_project
        )

        self.toolbar.open_action.triggered.connect(
            self.open_project
        )

        self.toolbar.save_action.triggered.connect(
            self.save_project
        )

        self.toolbar.zoom_in_action.triggered.connect(
            self.canvas.zoom_in
        )

        self.toolbar.zoom_out_action.triggered.connect(
            self.canvas.zoom_out
        )

        self.toolbar.reset_view_action.triggered.connect(
            self.canvas.reset_view
        )

        self.toolbar.coverage_action.toggled.connect(
            self.canvas.scene_obj.set_show_camera_coverage
        )

        self.property_panel.on_visual_change = (
            self.canvas.scene_obj.refresh_camera_coverage
        )

        self.toolbar.select_action.triggered.connect(
            lambda: self.change_tool(
                "select"
            )
        )

        self.toolbar.zone_action.triggered.connect(
            lambda: self.change_tool(
                "zone"
            )
        )

        self.toolbar.exit_action.triggered.connect(
            lambda: self.change_tool(
                "exit"
            )
        )

        self.toolbar.stair_action.triggered.connect(
            lambda: self.change_tool(
                "stair"
            )
        )

        self.toolbar.camera_action.triggered.connect(
            lambda: self.change_tool(
                "camera"
            )
        )

        self.toolbar.detector_action.triggered.connect(
            lambda: self.change_tool(
                "detector"
            )
        )

        self.toolbar.smoke_detector_action.triggered.connect(
            lambda: self.change_tool(
                "smoke_detector"
            )
        )

        self.toolbar.heat_detector_action.triggered.connect(
            lambda: self.change_tool(
                "heat_detector"
            )
        )

        self.toolbar.speaker_action.triggered.connect(
            lambda: self.change_tool(
                "speaker"
            )
        )

        self.toolbar.sign_action.triggered.connect(
            lambda: self.change_tool(
                "sign"
            )
        )

        self.toolbar.manual_call_point_action.triggered.connect(
            lambda: self.change_tool(
                "manual_call_point"
            )
        )

        self.toolbar.emergency_light_action.triggered.connect(
            lambda: self.change_tool(
                "emergency_light"
            )
        )

        self.toolbar.sprinkler_action.triggered.connect(
            lambda: self.change_tool(
                "sprinkler"
            )
        )

        self.toolbar.fire_extinguisher_action.triggered.connect(
            lambda: self.change_tool(
                "fire_extinguisher"
            )
        )

        self.toolbar.fire_hydrant_action.triggered.connect(
            lambda: self.change_tool(
                "fire_hydrant"
            )
        )

        self.toolbar.hose_reel_action.triggered.connect(
            lambda: self.change_tool(
                "hose_reel"
            )
        )

        self.toolbar.assembly_point_action.triggered.connect(
            lambda: self.change_tool(
                "assembly_point"
            )
        )

        self.toolbar.obstacle_action.triggered.connect(
            lambda: self.change_tool(
                "obstacle"
            )
        )

        self.toolbar.door_action.triggered.connect(
            lambda: self.change_tool(
                "door"
            )
        )

        self.toolbar.occupant_action.triggered.connect(
            lambda: self.change_tool(
                "occupant"
            )
        )

        self.toolbar.simulation_action.triggered.connect(
            self.toggle_simulation_panel
        )

    # =====================================================

    def connect_signals(self):

        # Unsaved-Changes Tracking -- QGraphicsScene's own built-in
        # `changed` signal, see the field's own comment in __init__.
        self.canvas.scene_obj.changed.connect(
            self._mark_dirty
        )

        # Bottom Information Bar
        self.canvas.status_callback = (
            self.info_bar.update_coordinates
        )

        # Property Panel
        self.canvas.scene_obj.selection_changed_callback = (
            self.on_selection_changed
        )

        # Floor List
        self.floor_list.active_floor_changed_callback = (
            self.on_active_floor_changed
        )

        self.floor_list.floors_changed_callback = (
            self.project_tree.refresh
        )

        # Property Panel -> Floor List / Project Tree, so edits to
        # Name/Elevation/Height made in the Properties dock are
        # reflected back in the Floors dock and the tree.
        self.property_panel.floor_updated_callback = (
            self.on_floor_properties_changed
        )

        # Stair Tool -- GraphicsScene never shows dialogs or
        # switches its own floor (same convention FloorList
        # already follows); it asks MainWindow to do both.
        self.canvas.scene_obj.floor_picker_callback = (
            self.choose_stair_destination_floor
        )

        self.canvas.scene_obj.floor_switch_requested_callback = (
            self.switch_to_floor
        )

        self.canvas.scene_obj.duplicate_stair_confirmation_callback = (
            self.confirm_duplicate_stair_connection
        )

        # Occupant Tool -- GraphicsScene never shows the generation
        # dialog itself, same convention as the Stair Tool's
        # floor_picker_callback above.
        self.canvas.scene_obj.occupant_generation_callback = (
            self.request_occupant_generation
        )

        # Manual Simulation Sandbox
        self.property_panel.set_sandbox_manager(
            self.canvas.scene_obj.sandbox_manager
        )

        self.property_panel.occupant_route_changed_callback = (
            self.canvas.scene_obj.refresh_occupant_route_highlight
        )

        self.simulation_panel.start_button.clicked.connect(
            self.start_simulation
        )

        self.simulation_panel.pause_button.clicked.connect(
            self.pause_simulation
        )

        self.simulation_panel.stop_button.clicked.connect(
            self.stop_simulation
        )

        self.simulation_panel.reset_button.clicked.connect(
            self.reset_simulation
        )

        self.simulation_panel.step_button.clicked.connect(
            self.step_simulation
        )

    # =====================================================

    def on_active_floor_changed(self, floor):

        # Floor List -> MainWindow -> Building/Project -> GraphicsScene.
        # MainWindow is the coordinator; GraphicsScene only ever
        # receives an already-decided floor to render.

        self.info_bar.update_floor(
            floor.name if floor else "-"
        )

        if floor is not None:

            self.canvas.scene_obj.set_current_floor(floor)

            self.property_panel.show_floor(floor)

        else:

            self.property_panel.clear()

    # =====================================================

    def on_floor_properties_changed(self, floor):

        self.floor_list.refresh()

        self.project_tree.refresh()

        self.info_bar.update_floor(floor.name)

    # =====================================================

    def choose_stair_destination_floor(self, candidate_floors):

        names = [
            floor.name
            for floor in candidate_floors
        ]

        choice, ok = QInputDialog.getItem(
            self,
            "Stair Destination",
            "Connect to floor:",
            names,
            0,
            False,
        )

        if not ok:
            return None

        return candidate_floors[
            names.index(choice)
        ]

    # =====================================================

    def confirm_duplicate_stair_connection(self, floor_a, floor_b):

        # Asked by GraphicsScene only when a Stair already connects
        # this exact floor pair -- a real building can legitimately
        # have more than one physical staircase between the same two
        # floors, so this warns rather than blocks outright. Defaults
        # to No (the safer choice when a dialog is dismissed).

        choice = QMessageBox.question(
            self,
            "Possible Duplicate Stair",
            (
                f"A Stair already connects {floor_a.name!r} and "
                f"{floor_b.name!r}.\n\n"
                f"Create another one anyway?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return choice == QMessageBox.StandardButton.Yes

    # =====================================================

    def validate_project(self):

        # Designer-level authoring completeness, deliberately separate
        # from GraphicsScene.rebuild_scene()/NavigationGraph.validate()
        # (both frozen/unmodified) -- see designer/validation.py for
        # why a missing Door/Exit/Stair Zone reference is an ERROR
        # here even though the Navigation Graph itself only ever
        # treats it as a WARNING.

        report = validate_building_authoring(
            self.canvas.scene_obj.project.building
        )

        if not report.issues:

            QMessageBox.information(
                self,
                "Validate Project",
                "No issues found -- every Door, Exit, and Stair is "
                "fully connected.",
            )

            return

        lines = [
            f"[{issue.severity.upper()}] {issue.message}"
            for issue in report.issues
        ]

        message = (
            f"{len(report.errors)} error(s), {len(report.warnings)} "
            f"warning(s) found."
        )

        detail_box = QMessageBox(self)
        detail_box.setWindowTitle("Validate Project")
        detail_box.setText(message)
        detail_box.setDetailedText("\n".join(lines))

        detail_box.setIcon(
            QMessageBox.Icon.Critical
            if report.errors
            else QMessageBox.Icon.Warning
        )

        detail_box.exec()

    # =====================================================

    def request_occupant_generation(self):

        # Asked by GraphicsScene once the Occupant Tool's drag-
        # rectangle is released -- same "Scene asks, MainWindow shows
        # the dialog" shape as choose_stair_destination_floor above.
        # None (cancelled) tells the Scene to generate nothing.

        dialog = OccupantGenerationDialog(self)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return dialog.occupant_count(), dialog.distribution()

    # =====================================================

    def switch_to_floor(self, floor):

        # Reuses the exact same path FloorList's own selection
        # takes (set_active_floor -> active_floor_changed_callback
        # -> on_active_floor_changed), so the Floors dock, canvas,
        # and Property Panel all stay in sync with a Tool-driven
        # switch exactly as they would with a user click.
        self.floor_list.set_active_floor(floor)

        self.floor_list.refresh()

    # =====================================================

    def on_selection_changed(self, item):

        if item is None:

            self.property_panel.clear()

            self.project_tree.refresh()

            return

        if isinstance(item, ZoneRectangle):
            self.property_panel.show_rectangle(item)

        elif isinstance(item, ExitItem):
            self.property_panel.show_line(item)

        elif isinstance(item, StairItem):
            self.property_panel.show_stair(item)

        elif isinstance(item, CameraItem):
            self.property_panel.show_camera(item)

        elif isinstance(item, DetectorItem):
            self.property_panel.show_detector(item)

        elif isinstance(item, SmokeDetectorItem):
            self.property_panel.show_smoke_detector(item)

        elif isinstance(item, HeatDetectorItem):
            self.property_panel.show_heat_detector(item)

        elif isinstance(item, SpeakerItem):
            self.property_panel.show_speaker(item)

        elif isinstance(item, SignItem):
            self.property_panel.show_sign(item)

        elif isinstance(item, ManualCallPointItem):
            self.property_panel.show_manual_call_point(item)

        elif isinstance(item, EmergencyLightItem):
            self.property_panel.show_emergency_light(item)

        elif isinstance(item, SprinklerItem):
            self.property_panel.show_sprinkler(item)

        elif isinstance(item, FireExtinguisherItem):
            self.property_panel.show_fire_extinguisher(item)

        elif isinstance(item, FireHydrantItem):
            self.property_panel.show_fire_hydrant(item)

        elif isinstance(item, HoseReelItem):
            self.property_panel.show_hose_reel(item)

        elif isinstance(item, AssemblyPointItem):
            self.property_panel.show_assembly_point(item)

        elif isinstance(item, ObstacleItem):
            self.property_panel.show_obstacle(item)

        elif isinstance(item, DoorItem):
            self.property_panel.show_door(item)

        elif isinstance(item, OccupantItem):
            self.property_panel.show_occupant(item)

        item.geometry_changed_callback = (
            lambda i: self.refresh_ui()
        )

        self.project_tree.refresh()

    # =====================================================

    def refresh_ui(self):

        self.property_panel.refresh()

        self.project_tree.refresh()

        self.canvas.scene_obj.refresh_camera_coverage()

    # =====================================================

    def change_tool(self, tool):

        self.canvas.set_tool(tool)

        self.info_bar.update_tool(tool)

    # =====================================================

    def import_floor_plan(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Floor Plan",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )

        if filename:

            self.canvas.load_floor_plan(filename)

    # =====================================================

    def save_project(self) -> bool:

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "SynEvac Project (*.syn)",
        )

        if not filename:
            return False

        if not filename.endswith(".syn"):

            filename += ".syn"

        Serializer.save(
            self.canvas.scene_obj.get_project(),
            filename,
            credential_store=self._credential_store,
        )

        self._dirty = False

        return True

    # =====================================================

    def new_project(self):

        if not self._confirm_discard_unsaved_changes():
            return

        self.stop_simulation()

        self.canvas.scene_obj.sandbox_manager.clear()

        self.canvas.scene_obj.project = Project.new_default()

        if self.canvas.scene_obj.project.building.floor_count > 0:

            self.canvas.scene_obj.current_floor = (
                self.canvas.scene_obj.project.building.ordered_floors()[0]
            )

        self.project_tree.set_project(self.canvas.scene_obj.project)

        self.floor_list.set_building(self.canvas.scene_obj.project.building)

        self.property_panel.set_building(self.canvas.scene_obj.project.building)

        self.canvas.scene_obj.rebuild_scene()

        self._dirty = False

    # =====================================================

    def open_project(self):

        if not self._confirm_discard_unsaved_changes():
            return

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "SynEvac Project (*.syn)",
        )

        if not filename:
            return

        project = Serializer.load(filename, credential_store=self._credential_store)

        # Occupants belong to whichever project placed them -- never
        # saved, never reloaded, and meaningless (dangling floor_id/
        # zone_id references) once that project is replaced. Stopping
        # the simulation loop first means no in-flight tick can touch
        # an occupant belonging to the project about to disappear.
        self.stop_simulation()

        self.canvas.scene_obj.sandbox_manager.clear()

        self.canvas.scene_obj.project = project

        if project.building and project.building.floor_count > 0:

            self.canvas.scene_obj.current_floor = (
                project.building.ordered_floors()[0]
            )

        self.project_tree.set_project(project)

        self.floor_list.set_building(project.building)

        self.property_panel.set_building(project.building)

        self.canvas.scene_obj.rebuild_scene()

        self._dirty = False

    # =====================================================

    def _mark_dirty(self, *_args):

        self._dirty = True

    # =====================================================

    def _confirm_discard_unsaved_changes(self) -> bool:

        # Gate before any destructive action (new/open/close). Returns
        # True when it is safe to proceed (nothing unsaved, the user
        # chose Discard, or the user chose Save and it actually
        # completed), False when the caller should abort (Cancel, or
        # Save was chosen but the file dialog itself was cancelled).

        if not self._dirty:
            return True

        choice = QMessageBox.question(
            self,
            "Unsaved Changes",
            "This project has unsaved changes. Save before continuing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        if choice == QMessageBox.StandardButton.Cancel:
            return False

        if choice == QMessageBox.StandardButton.Save:
            return self.save_project()

        return True  # Discard

    # =====================================================
    # Manual Simulation Sandbox (Simulation V0)
    #
    # MainWindow owns the only clock in this tool -- SandboxManager
    # never reads wall-clock time itself, it only ever receives a dt
    # (continuous playback, scaled by the speed slider) or a bare
    # "advance one node" request (Step Mode). GraphicsScene/
    # OccupantItem stay exactly as passive as every other item: they
    # render whatever SandboxManager's occupants currently say.
    # =====================================================

    def toggle_simulation_panel(self):

        self.simulation_dock.setVisible(
            not self.simulation_dock.isVisible()
        )

    # =====================================================

    def toggle_perception_debug_panel(self):

        self.perception_debug_dock.setVisible(
            not self.perception_debug_dock.isVisible()
        )

    # =====================================================

    def toggle_building_state_debug_panel(self):

        self.building_state_debug_dock.setVisible(
            not self.building_state_debug_dock.isVisible()
        )

    # =====================================================

    def toggle_camera_manager_panel(self):

        visible = not self.camera_manager_dock.isVisible()

        self.camera_manager_dock.setVisible(visible)

        if visible:
            self._refresh_camera_manager_panel()

    # =====================================================

    def toggle_speaker_manager_panel(self):

        visible = not self.speaker_manager_dock.isVisible()

        self.speaker_manager_dock.setVisible(visible)

        if visible:
            self._refresh_speaker_manager_panel()

    # =====================================================

    def toggle_camera_validation_panel(self):

        visible = not self.camera_validation_dock.isVisible()

        self.camera_validation_dock.setVisible(visible)

        if visible:
            self.camera_validation_panel.refresh(
                self.canvas.scene_obj.project.building
            )

    # =====================================================
    # Scenario Campaign Studio -- the graphical orchestration layer
    # over the already-frozen scenario_definition/scenario_generator/
    # scenario_validator/scenario_pipeline/scenario_storage/
    # scenario_runner/behaviour_profile_resolver/simulation_runtime/
    # ai_decision pipeline (designer/campaign/). MainWindow's only
    # responsibility here is the same one it already has for every
    # other tool: own the window, hand it the current Building, do
    # nothing else -- CampaignController/CampaignWorker own everything
    # about how a campaign actually runs.
    # =====================================================

    def open_campaign_studio(self):

        if self.campaign_window is None:

            self.campaign_window = CampaignWindow()

            self.campaign_controller = CampaignController(
                self.campaign_window,
                lambda: self.canvas.scene_obj.project.building,
            )

        self.campaign_window.set_building(
            self.canvas.scene_obj.project.building
        )

        self.campaign_window.show()
        self.campaign_window.raise_()
        self.campaign_window.activateWindow()

    # =====================================================

    def start_simulation(self):

        scene = self.canvas.scene_obj

        scene.sandbox_manager.ensure_routes_current(scene.project.building)

        self._sync_occupant_items()

        if scene.selected_item is not None:
            self.property_panel.refresh()

        self._refresh_perception_debug_panel()
        self._refresh_building_state_debug_panel()
        self._refresh_camera_manager_panel()
        self._refresh_speaker_manager_panel()

        self.simulation_timer.start()

        self.simulation_panel.set_running_state(True)

    # =====================================================

    def pause_simulation(self):

        self.simulation_timer.stop()

        self.simulation_panel.set_running_state(False)

    # =====================================================

    def stop_simulation(self):

        # Halts playback and zeroes the displayed clock -- unlike
        # Pause, this is not meant to be resumed from where it left
        # off. Deliberately leaves occupants exactly where they
        # currently are; Reset (below) is the one that sends them
        # back to the start of their route.
        self.simulation_timer.stop()

        self.simulation_panel.set_running_state(False)

        self.simulation_time = 0.0

        self.simulation_panel.update_time(self.simulation_time)

    # =====================================================

    def reset_simulation(self):

        self.stop_simulation()

        scene = self.canvas.scene_obj

        for occupant in scene.sandbox_manager.occupants:

            scene.sandbox_manager.reset_occupant(
                occupant, scene.project.building,
            )

        self._sync_occupant_items()

        if scene.selected_item is not None:
            self.property_panel.refresh()

        self._refresh_perception_debug_panel()
        self._refresh_building_state_debug_panel()
        self._refresh_camera_manager_panel()
        self._refresh_speaker_manager_panel()

    # =====================================================

    def step_simulation(self):

        scene = self.canvas.scene_obj

        scene.sandbox_manager.ensure_routes_current(scene.project.building)

        for occupant in scene.sandbox_manager.occupants:
            scene.sandbox_manager.step(occupant)

        self._sync_occupant_items()

        self.property_panel.refresh()

        self._refresh_perception_debug_panel()
        self._refresh_building_state_debug_panel()
        self._refresh_camera_manager_panel()
        self._refresh_speaker_manager_panel()

    # =====================================================

    def on_simulation_tick(self):

        dt = (
            (self.simulation_timer.interval() / 1000.0)
            * self.simulation_panel.speed_multiplier()
        )

        self.simulation_time += dt

        self.simulation_panel.update_time(self.simulation_time)

        scene = self.canvas.scene_obj

        for occupant in scene.sandbox_manager.occupants:
            scene.sandbox_manager.tick(occupant, dt)

        self._sync_occupant_items()

        self.property_panel.refresh()

        self._refresh_perception_debug_panel()
        self._refresh_building_state_debug_panel()
        self._refresh_camera_manager_panel()
        self._refresh_speaker_manager_panel()

    # =====================================================

    def _refresh_perception_debug_panel(self):

        # Only visible/active dock this touches -- see
        # perception_debug_panel.py's own "dumb widget, MainWindow
        # pushes updates in" convention. Kept as its own method (rather
        # than inlined at each call site) so the panel could later be
        # made toggle-aware (e.g. skip recomputation while hidden)
        # without touching every call site.

        scene = self.canvas.scene_obj

        self.perception_debug_panel.refresh(
            scene.project.building,
            scene.sandbox_manager,
            self.simulation_time,
        )

    # =====================================================

    def _refresh_building_state_debug_panel(self):

        # Same "own method, so it can be made toggle-aware later
        # without touching every call site" reasoning as
        # _refresh_perception_debug_panel() above.

        scene = self.canvas.scene_obj

        self.building_state_debug_panel.refresh(
            scene.project.building,
            scene.sandbox_manager,
            self.simulation_time,
        )

    # =====================================================

    def _refresh_camera_manager_panel(self):

        # Same "own method, so it can be made toggle-aware later
        # without touching every call site" reasoning as
        # _refresh_perception_debug_panel() above.

        self.camera_manager_panel.refresh(
            self.canvas.scene_obj.project.building
        )

    # =====================================================

    def _refresh_speaker_manager_panel(self):

        # Same "own method, so it can be made toggle-aware later
        # without touching every call site" reasoning as
        # _refresh_perception_debug_panel() above.

        self.speaker_manager_panel.refresh(
            self.canvas.scene_obj.project.building
        )

    # =====================================================

    def _on_camera_manager_changed(self):

        # An enable/disable or mode change made through the Camera
        # Manager Panel mutates the same Camera object the Property
        # Panel (if that camera happens to be selected) and the Camera
        # Coverage overlay both already read -- refresh both exactly
        # the way editing those same fields through the Property Panel
        # itself already does (see PropertyPanel.on_visual_change's own
        # wiring below).

        if self.canvas.scene_obj.selected_item is not None:
            self.property_panel.refresh()

        self.canvas.scene_obj.refresh_camera_coverage()

    # =====================================================

    def _sync_occupant_items(self):

        # Delegates to GraphicsScene.sync_occupants(), which
        # reconciles occupant_items with the displayed floor every
        # call -- not just a position refresh. An occupant that
        # crossed a Stair onto a different floor than the one on
        # screen gets its stale item removed here (see that method's
        # own comment for why a blind per-item sync_from_occupant()
        # loop was not enough).
        self.canvas.scene_obj.sync_occupants()

    # =====================================================

    def closeEvent(self, event):

        if not self._confirm_discard_unsaved_changes():
            event.ignore()
            return

        event.accept()