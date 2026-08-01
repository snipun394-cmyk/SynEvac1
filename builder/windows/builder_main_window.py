from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from designer.items.camera_item import CameraItem
from designer.items.door_item import DoorItem
from designer.items.exit_item import ExitItem
from designer.items.heat_detector_item import HeatDetectorItem
from designer.items.obstacle_item import ObstacleItem
from designer.items.smoke_detector_item import SmokeDetectorItem
from designer.items.speaker_item import SpeakerItem
from designer.items.stair_item import StairItem
from designer.items.zone_rectangle import ZoneRectangle
from designer.scene.graphics_view import GraphicsView
from designer.widgets.bottom_info_bar import BottomInfoBar
from designer.widgets.dock_manager import DockManager
from designer.widgets.floor_list import FloorList
from designer.widgets.project_tree import ProjectTree

from models.project import Project

from serialization.serializer import Serializer

from builder.scale_calibration import (
    ScaleCalibrationController, ScaleCalibrationError, compute_scale_pixels_per_meter,
)
from builder.widgets.builder_property_panel import BuilderPropertyPanel
from builder.widgets.builder_toolbar import BuilderToolbar
from builder.widgets.navigation_preview_panel import NavigationPreviewPanel
from builder.widgets.project_summary_panel import ProjectSummaryPanel
from builder.widgets.validation_panel import ValidationPanel


MAX_RECENT_PROJECTS = 8


class BuilderMainWindow(QMainWindow):

    # SynEvac Builder's own main window -- a NEW, small file (per the
    # feasibility investigation's Phase 2/8 recommendation), not a
    # stripped-down copy of designer/windows/main_window.py. It wires
    # together only already-independent pieces confirmed reusable by
    # that investigation: designer.scene.graphics_view.GraphicsView
    # (and, transitively, GraphicsScene/designer.items.*), designer.
    # widgets.{floor_list,project_tree,bottom_info_bar}, models.
    # project.Project, serialization.serializer.Serializer -- plus
    # this milestone's own new Builder-only widgets (toolbar, property
    # panel, validation/summary/navigation-preview panels, scale
    # calibration). It never imports designer.windows.main_window.
    # MainWindow, designer.campaign.*, or any of the eight heavy
    # Studio-only panels that file constructs -- that is what makes
    # "no Simulation/AI/Perception/Live-Runtime" a structural
    # guarantee here rather than a runtime convention.

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SynEvac Builder")
        self.resize(1500, 900)

        self._settings = QSettings("SynEvac", "Builder")
        self._current_filename = None
        self._dirty = False

        # =====================================================
        # Central Widget
        # =====================================================

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        central.setLayout(layout)

        self.canvas = GraphicsView()
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas, 1)

        self.info_bar = BottomInfoBar()
        layout.addWidget(self.info_bar)

        # =====================================================
        # Panels
        # =====================================================

        self.project_tree = ProjectTree()
        self.floor_list = FloorList()
        self.property_panel = BuilderPropertyPanel()
        self.validation_panel = ValidationPanel()
        self.summary_panel = ProjectSummaryPanel()
        self.navigation_preview_panel = NavigationPreviewPanel()

        # =====================================================
        # Scale Calibration
        # =====================================================

        self._calibration_controller = ScaleCalibrationController(self.canvas)
        self._calibration_controller.on_points_chosen = self._on_calibration_points_chosen

        # =====================================================
        # Build UI
        # =====================================================

        self.create_actions()
        self.create_toolbar()

        # Dock Management Refactor V1 -- create_docks() must run before
        # create_menu() now: the View menu's dock entries are each
        # dock's own real toggleViewAction() (designer.widgets.
        # dock_manager.DockManager), which doesn't exist until the dock
        # itself has been created and registered. Same reordering
        # SynEvac Studio's own MainWindow needed for the same reason.
        self.create_docks()
        self.create_menu()

        self.connect_toolbar()
        self.connect_signals()

        # =====================================================
        # Initial Project -- GraphicsScene() already constructs a
        # Project.new_default() in its own __init__ (see designer/
        # scene/graphics_scene.py); BuilderMainWindow only needs to
        # bind the other panels to it, the same "Scene owns the
        # default, MainWindow-equivalent only binds" split Studio's
        # own MainWindow already follows.
        # =====================================================

        self._bind_project(self.canvas.scene_obj.project)

        # Same "GraphicsScene already resolved a current_floor in its
        # own __init__, FloorList itself starts with no selection"
        # state Studio's own MainWindow.__init__ leaves things in --
        # matched here rather than introduced. Property Panel/info bar
        # are seeded directly from the Scene's already-current floor,
        # not through on_active_floor_changed() (which would fire with
        # None since FloorList has no selection yet).
        initial_floor = self.canvas.scene_obj.current_floor

        if initial_floor is not None:

            self.property_panel.show_floor(initial_floor)

            self.info_bar.update_floor(initial_floor.name)

            self.info_bar.update_scale(
                f"{initial_floor.floor_plan_scale:.1f} px = 1 m"
                if initial_floor.is_scale_calibrated
                else "Not Calibrated"
            )

    # =====================================================

    def create_actions(self):

        self.new_action = QAction("New Project", self)
        self.open_action = QAction("Open Project", self)
        self.save_action = QAction("Save Project", self)
        self.save_as_action = QAction("Save Project As...", self)
        self.import_floor_action = QAction("Import Floor Plan", self)
        self.calibrate_scale_action = QAction("Calibrate Scale", self)
        self.validate_project_action = QAction("Validate Project", self)

        self.new_action.triggered.connect(self.new_project)
        self.open_action.triggered.connect(self.open_project)
        self.save_action.triggered.connect(self.save_project)
        self.save_as_action.triggered.connect(self.save_project_as)
        self.import_floor_action.triggered.connect(self.import_floor_plan)
        self.calibrate_scale_action.triggered.connect(self.calibrate_scale)
        self.validate_project_action.triggered.connect(self.validate_project)

    # =====================================================

    def create_toolbar(self):

        self.toolbar = BuilderToolbar()
        self.addToolBar(self.toolbar)

    # =====================================================

    def create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        file_menu.addAction(self.new_action)
        file_menu.addAction(self.open_action)

        self.recent_menu = file_menu.addMenu("Open Recent")
        self.recent_menu.aboutToShow.connect(self._populate_recent_menu)

        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_floor_action)

        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction(self.calibrate_scale_action)
        tools_menu.addAction(self.validate_project_action)

        # Dock Management Refactor V1 -- Builder previously had no View
        # menu, and no dock (Project Explorer/Floors/Properties/
        # Validation/Project Summary/Navigation Preview) had any reopen
        # action at all: each was a local variable inside _add_dock(),
        # discarded the moment addDockWidget() returned. Since
        # DockWidgetClosable is on by default, closing any of them via
        # its own native close button left no way back, ever. Every
        # dock's own real toggleViewAction() (collected by create_docks(),
        # which now runs before this method) is added here -- one View
        # menu, same mechanism SynEvac Studio uses, making every Builder
        # dock recoverable for the first time.
        view_menu = menubar.addMenu("View")

        for action in self._view_menu_actions:
            view_menu.addAction(action)

    # =====================================================

    def create_docks(self):

        self.dock_manager = DockManager(self)
        self._view_menu_actions = []

        self._add_dock("Project Explorer", self.project_tree, Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock("Floors", self.floor_list, Qt.DockWidgetArea.LeftDockWidgetArea)
        self._add_dock("Properties", self.property_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock("Validation", self.validation_panel, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._add_dock("Project Summary", self.summary_panel, Qt.DockWidgetArea.RightDockWidgetArea)
        self._add_dock("Navigation Preview", self.navigation_preview_panel, Qt.DockWidgetArea.BottomDockWidgetArea)

    # =====================================================

    def _add_dock(self, title, widget, area):

        dock = QDockWidget(title, self)
        dock.setWidget(widget)

        action = self.dock_manager.register(dock, area, title=title)
        self._view_menu_actions.append(action)

        return dock

    # =====================================================

    def connect_toolbar(self):

        self.toolbar.new_action.triggered.connect(self.new_project)
        self.toolbar.open_action.triggered.connect(self.open_project)
        self.toolbar.save_action.triggered.connect(self.save_project)
        self.toolbar.save_as_action.triggered.connect(self.save_project_as)

        self.toolbar.import_floor_plan_action.triggered.connect(self.import_floor_plan)
        self.toolbar.calibrate_scale_action.triggered.connect(self.calibrate_scale)
        self.toolbar.validate_action.triggered.connect(self.validate_project)

        self.toolbar.zoom_in_action.triggered.connect(self.canvas.zoom_in)
        self.toolbar.zoom_out_action.triggered.connect(self.canvas.zoom_out)
        self.toolbar.reset_view_action.triggered.connect(self.canvas.reset_view)

        self.toolbar.select_action.triggered.connect(lambda: self.change_tool("select"))
        self.toolbar.zone_action.triggered.connect(lambda: self.change_tool("zone"))
        self.toolbar.door_action.triggered.connect(lambda: self.change_tool("door"))
        self.toolbar.exit_action.triggered.connect(lambda: self.change_tool("exit"))
        self.toolbar.stair_action.triggered.connect(lambda: self.change_tool("stair"))
        self.toolbar.obstacle_action.triggered.connect(lambda: self.change_tool("obstacle"))
        self.toolbar.camera_action.triggered.connect(lambda: self.change_tool("camera"))
        self.toolbar.smoke_detector_action.triggered.connect(lambda: self.change_tool("smoke_detector"))
        self.toolbar.heat_detector_action.triggered.connect(lambda: self.change_tool("heat_detector"))
        self.toolbar.speaker_action.triggered.connect(lambda: self.change_tool("speaker"))

    # =====================================================

    def connect_signals(self):

        self.canvas.scene_obj.changed.connect(self._mark_dirty)

        self.canvas.status_callback = self.info_bar.update_coordinates

        self.canvas.scene_obj.selection_changed_callback = self.on_selection_changed

        self.floor_list.active_floor_changed_callback = self.on_active_floor_changed
        self.floor_list.floors_changed_callback = self._on_floors_changed

        self.property_panel.floor_updated_callback = self.on_floor_properties_changed
        self.property_panel.item_changed_callback = self._on_authoring_changed

        self.canvas.scene_obj.floor_picker_callback = self.choose_stair_destination_floor
        self.canvas.scene_obj.floor_switch_requested_callback = self.switch_to_floor
        self.canvas.scene_obj.duplicate_stair_confirmation_callback = self.confirm_duplicate_stair_connection

    # =====================================================
    # Project binding
    # =====================================================

    def _bind_project(self, project):

        self.project_tree.set_project(project)
        self.floor_list.set_building(project.building)
        self.property_panel.set_building(project.building)

        self._refresh_authoring_panels()

    # =====================================================

    def _on_floors_changed(self):

        self.project_tree.refresh()

        self._refresh_authoring_panels()
        self._mark_dirty()

    # =====================================================

    def _on_authoring_changed(self):

        self._refresh_authoring_panels()
        self._mark_dirty()

    # =====================================================

    def _refresh_authoring_panels(self):

        building = self.canvas.scene_obj.project.building

        self.validation_panel.refresh(building)

        self.summary_panel.refresh(
            building,
            self.validation_panel.is_valid,
            self.validation_panel.summary_label.text(),
        )

        self.navigation_preview_panel.refresh(building, self.canvas.scene_obj.current_floor)

        self.project_tree.refresh()

    # =====================================================
    # Floors
    # =====================================================

    def on_active_floor_changed(self, floor):

        self.info_bar.update_floor(floor.name if floor else "-")

        if floor is not None:

            self.canvas.scene_obj.set_current_floor(floor)
            self.property_panel.show_floor(floor)

            self.info_bar.update_scale(
                f"{floor.floor_plan_scale:.1f} px = 1 m"
                if floor.is_scale_calibrated
                else "Not Calibrated"
            )

        else:

            self.property_panel.clear()
            self.info_bar.update_scale("Not Calibrated")

        self._refresh_authoring_panels()

    # =====================================================

    def on_floor_properties_changed(self, floor):

        self.floor_list.refresh()

        self._refresh_authoring_panels()
        self._mark_dirty()

        self.info_bar.update_floor(floor.name)

    # =====================================================

    def choose_stair_destination_floor(self, candidate_floors):

        names = [floor.name for floor in candidate_floors]

        choice, ok = QInputDialog.getItem(
            self, "Stair Destination", "Connect to floor:", names, 0, False,
        )

        if not ok:
            return None

        return candidate_floors[names.index(choice)]

    # =====================================================

    def confirm_duplicate_stair_connection(self, floor_a, floor_b):

        choice = QMessageBox.question(
            self,
            "Possible Duplicate Stair",
            (
                f"A Stair already connects {floor_a.name!r} and "
                f"{floor_b.name!r}.\n\nCreate another one anyway?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        return choice == QMessageBox.StandardButton.Yes

    # =====================================================

    def switch_to_floor(self, floor):

        self.floor_list.set_active_floor(floor)
        self.floor_list.refresh()

    # =====================================================
    # Selection
    # =====================================================

    def on_selection_changed(self, item):

        if item is None:

            self.property_panel.clear()
            self.project_tree.refresh()

            return

        if isinstance(item, ZoneRectangle):
            self.property_panel.show_zone(item)

        elif isinstance(item, DoorItem):
            self.property_panel.show_door(item)

        elif isinstance(item, ExitItem):
            self.property_panel.show_exit(item)

        elif isinstance(item, StairItem):
            self.property_panel.show_stair(item)

        elif isinstance(item, CameraItem):
            self.property_panel.show_camera(item)

        elif isinstance(item, SmokeDetectorItem):
            self.property_panel.show_smoke_detector(item)

        elif isinstance(item, HeatDetectorItem):
            self.property_panel.show_heat_detector(item)

        elif isinstance(item, SpeakerItem):
            self.property_panel.show_speaker(item)

        elif isinstance(item, ObstacleItem):
            self.property_panel.show_obstacle(item)

        item.geometry_changed_callback = lambda i: self.refresh_ui()

        self.project_tree.refresh()

    # =====================================================

    def refresh_ui(self):

        self.property_panel.refresh()

        self._refresh_authoring_panels()

    # =====================================================

    def change_tool(self, tool):

        self.canvas.set_tool(tool)
        self.info_bar.update_tool(tool)

    # =====================================================
    # Floor Plan / Scale Calibration
    # =====================================================

    def import_floor_plan(self):

        filename, _ = QFileDialog.getOpenFileName(
            self, "Import Floor Plan", "", "Images (*.png *.jpg *.jpeg)",
        )

        if not filename:
            return

        self.canvas.load_floor_plan(filename)

        self._refresh_authoring_panels()
        self._mark_dirty()

    # =====================================================

    def calibrate_scale(self):

        if self.canvas.scene_obj.floor_plan_item is None:

            QMessageBox.warning(
                self, "Calibrate Scale",
                "Import a floor plan for this floor before calibrating its scale.",
            )

            return

        QMessageBox.information(
            self, "Calibrate Scale",
            "Click two points on the floor plan, then enter the real-world "
            "distance between them.",
        )

        self._calibration_controller.start()

    # =====================================================

    def _on_calibration_points_chosen(self, scene_point_a, scene_point_b):

        floor_plan_item = self.canvas.scene_obj.floor_plan_item

        if floor_plan_item is None:
            return

        # mapFromScene() reverses whatever scale is CURRENTLY applied
        # to the item (zero, on a first calibration; the previous
        # scale, on a recalibration) -- so these are always the raw
        # floor plan IMAGE's own pixel coordinates, exactly what
        # Floor.floor_plan_scale is defined against, regardless of how
        # many times this floor has been calibrated before.
        image_point_a = floor_plan_item.mapFromScene(scene_point_a)
        image_point_b = floor_plan_item.mapFromScene(scene_point_b)

        floor = self.canvas.scene_obj.current_floor

        default_distance = floor.floor_plan_calibration_distance_m or 1.0

        distance_m, ok = QInputDialog.getDouble(
            self, "Scale Calibration",
            "Real-world distance between the two points (meters):",
            default_distance, 0.001, 100000.0, 3,
        )

        if not ok:
            return

        try:

            pixels_per_meter = compute_scale_pixels_per_meter(
                (image_point_a.x(), image_point_a.y()),
                (image_point_b.x(), image_point_b.y()),
                distance_m,
            )

        except ScaleCalibrationError as error:

            QMessageBox.warning(self, "Calibrate Scale", str(error))

            return

        floor.set_scale_calibration(
            (image_point_a.x(), image_point_a.y()),
            (image_point_b.x(), image_point_b.y()),
            distance_m,
            pixels_per_meter,
        )

        self.canvas.scene_obj.rebuild_scene()

        self.info_bar.update_scale(f"{pixels_per_meter:.1f} px = 1 m")

        self._refresh_authoring_panels()
        self._mark_dirty()

    # =====================================================
    # Validation
    # =====================================================

    def validate_project(self):

        self._refresh_authoring_panels()

        if self.validation_panel.is_valid:

            QMessageBox.information(
                self, "Validate Project",
                "No critical errors found. See the Validation panel for "
                "any warnings/notes.",
            )

        else:

            QMessageBox.critical(
                self, "Validate Project",
                "Critical validation errors were found -- see the "
                "Validation panel. These must be resolved before this "
                "project can be saved.",
            )

    # =====================================================
    # Project Management
    # =====================================================

    def new_project(self):

        if not self._confirm_discard_unsaved_changes():
            return

        self.canvas.scene_obj.project = Project.new_default()

        if self.canvas.scene_obj.project.building.floor_count > 0:

            self.canvas.scene_obj.current_floor = (
                self.canvas.scene_obj.project.building.ordered_floors()[0]
            )

        self._current_filename = None

        self._bind_project(self.canvas.scene_obj.project)

        self.canvas.scene_obj.rebuild_scene()

        self._dirty = False

    # =====================================================

    def open_project(self):

        if not self._confirm_discard_unsaved_changes():
            return

        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "SynEvac Project (*.syn)",
        )

        if not filename:
            return

        self._open_project_file(filename)

    # =====================================================

    def _open_project_file(self, filename):

        try:
            project = Serializer.load(filename)
        except (OSError, ValueError, KeyError) as error:

            QMessageBox.critical(
                self, "Open Project", f"Could not open {filename!r}:\n{error}",
            )

            return

        self.canvas.scene_obj.project = project

        if project.building and project.building.floor_count > 0:

            self.canvas.scene_obj.current_floor = project.building.ordered_floors()[0]

        self._current_filename = filename

        self._bind_project(project)

        self.canvas.scene_obj.rebuild_scene()

        self._dirty = False

        self._add_recent_project(filename)

    # =====================================================
    # Recent Projects -- persisted via QSettings, same mechanism any
    # desktop Qt app uses for this; a plain, capped, most-recent-first
    # list of file paths, never anything project-content-related.
    # =====================================================

    def _add_recent_project(self, filename):

        recent = self._settings.value("recent_projects", [], type=list)

        recent = [path for path in recent if path != filename]

        recent.insert(0, filename)

        self._settings.setValue("recent_projects", recent[:MAX_RECENT_PROJECTS])

    # =====================================================

    def _populate_recent_menu(self):

        self.recent_menu.clear()

        recent = self._settings.value("recent_projects", [], type=list)

        if not recent:

            empty_action = QAction("(No Recent Projects)", self)
            empty_action.setEnabled(False)
            self.recent_menu.addAction(empty_action)

            return

        for filename in recent:

            action = QAction(filename, self)
            action.triggered.connect(
                lambda checked=False, path=filename: self._open_recent_project(path)
            )
            self.recent_menu.addAction(action)

    # =====================================================

    def _open_recent_project(self, filename):

        if not self._confirm_discard_unsaved_changes():
            return

        self._open_project_file(filename)

    # =====================================================

    def save_project(self) -> bool:

        if self._current_filename:
            return self._save_to(self._current_filename)

        return self.save_project_as()

    # =====================================================

    def save_project_as(self) -> bool:

        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", "SynEvac Project (*.syn)",
        )

        if not filename:
            return False

        if not filename.endswith(".syn"):
            filename += ".syn"

        return self._save_to(filename)

    # =====================================================

    def _save_to(self, filename) -> bool:

        # "Critical validation errors should prevent export" (milestone
        # brief) -- Save IS export, there is no separate export action
        # (see docs/architecture/synevac_builder_feasibility_
        # investigation.md, Phase 3: Serializer.save() already produces
        # the exact .syn file Studio reads).
        self._refresh_authoring_panels()

        if not self.validation_panel.is_valid:

            QMessageBox.critical(
                self, "Save Project",
                "This project has critical validation errors and cannot "
                "be saved until they are resolved -- see the Validation "
                "panel.",
            )

            return False

        Serializer.save(self.canvas.scene_obj.get_project(), filename)

        self._current_filename = filename
        self._dirty = False

        self._add_recent_project(filename)

        return True

    # =====================================================

    def _mark_dirty(self, *_args):

        self._dirty = True

    # =====================================================

    def _confirm_discard_unsaved_changes(self) -> bool:

        if not self._dirty:
            return True

        choice = QMessageBox.question(
            self, "Unsaved Changes",
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

        return True

    # =====================================================

    def closeEvent(self, event):

        if not self._confirm_discard_unsaved_changes():
            event.ignore()
            return

        event.accept()
