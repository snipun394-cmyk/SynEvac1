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

from designer.items.assembly_point_item import AssemblyPointItem
from designer.items.camera_item import CameraItem
from designer.items.detector_item import DetectorItem
from designer.items.door_item import DoorItem
from designer.items.exit_item import ExitItem
from designer.items.obstacle_item import ObstacleItem
from designer.items.occupant_item import OccupantItem
from designer.items.stair_item import StairItem
from designer.items.zone_rectangle import ZoneRectangle
from designer.scene.graphics_view import GraphicsView
from designer.widgets.bottom_info_bar import BottomInfoBar
from designer.widgets.floor_list import FloorList
from designer.widgets.occupant_generation_dialog import OccupantGenerationDialog
from designer.widgets.project_tree import ProjectTree
from designer.widgets.property_panel import PropertyPanel
from designer.widgets.simulation_panel import SimulationPanel
from designer.widgets.toolbar import MainToolbar
from designer.validation import validate_building_authoring

from serialization.serializer import Serializer


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SynEvac Studio")
        self.resize(1600, 900)

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
        # Project Tree
        # =====================================================

        self.project_tree = ProjectTree()

        # =====================================================
        # Floor List
        # =====================================================

        self.floor_list = FloorList()

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

    # =====================================================

    def create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        menubar.addMenu("Edit")
        menubar.addMenu("View")
        menubar.addMenu("Insert")

        simulation_menu = menubar.addMenu("Simulation")

        simulation_menu.addAction(
            self.toggle_simulation_panel_action
        )

        menubar.addMenu("AI")

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

    # =====================================================

    def connect_toolbar(self):

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

    def save_project(self):

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            "",
            "SynEvac Project (*.syn)",
        )

        if not filename:
            return

        if not filename.endswith(".syn"):

            filename += ".syn"

        Serializer.save(
            self.canvas.scene_obj.get_project(),
            filename,
        )

    # =====================================================

    def open_project(self):

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            "",
            "SynEvac Project (*.syn)",
        )

        if not filename:
            return

        project = Serializer.load(filename)

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

    def start_simulation(self):

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

    # =====================================================

    def step_simulation(self):

        scene = self.canvas.scene_obj

        for occupant in scene.sandbox_manager.occupants:
            scene.sandbox_manager.step(occupant)

        self._sync_occupant_items()

        self.property_panel.refresh()

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

        event.accept()