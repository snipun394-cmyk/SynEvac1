from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QFileDialog,
    QDockWidget,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from designer.items.exit_item import ExitItem
from designer.items.zone_rectangle import ZoneRectangle
from designer.scene.graphics_view import GraphicsView
from designer.widgets.bottom_info_bar import BottomInfoBar
from designer.widgets.floor_list import FloorList
from designer.widgets.project_tree import ProjectTree
from designer.widgets.property_panel import PropertyPanel
from designer.widgets.toolbar import MainToolbar

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

        self.info_bar = BottomInfoBar()

        self.info_bar.setFixedHeight(30)

        self.layout.addWidget(
            self.info_bar
        )

        # =====================================================
        # Property Panel
        # =====================================================

        self.property_panel = PropertyPanel()

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

    # =====================================================

    def create_menu(self):

        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        menubar.addMenu("Edit")
        menubar.addMenu("View")
        menubar.addMenu("Insert")
        menubar.addMenu("Simulation")
        menubar.addMenu("AI")
        menubar.addMenu("Tools")
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
        )    # =====================================================

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

        self.canvas.scene_obj.project = project

        if project.building and project.building.floor_count > 0:

            self.canvas.scene_obj.current_floor = (
                project.building.ordered_floors()[0]
            )

        self.project_tree.set_project(project)

        self.floor_list.set_building(project.building)

        self.canvas.scene_obj.rebuild_scene()

    # =====================================================

    def closeEvent(self, event):

        event.accept()